"""Reward program -> task latent z, plus CEM refinement in cloned worlds.

B and the policy are plain callables so this module does not depend on
skyhand.train:  B(g [M, G_DIM]) -> [M, d];  policy(obs, z [N, d]) -> actions [N, A, ACTION_DIM].
All z live on the sphere of radius sqrt(d), as in FB pretraining.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .. import spec as S
from .dsl import CompiledReward, compile_reward

RewardLike = Union[str, Callable[[torch.Tensor], torch.Tensor]]


def _as_reward(r: Optional[RewardLike]):
    if r is None or callable(r):
        return r
    return compile_reward(r)


def project_sphere(z: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
    """Scale z (any leading dims) to norm sqrt(d)."""
    d = z.shape[-1] if d is None else d
    return math.sqrt(d) * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def _encode(B: Callable, g: torch.Tensor, batch_size: int) -> torch.Tensor:
    outs = [B(g[i:i + batch_size]) for i in range(0, g.shape[0], batch_size)]
    return torch.cat(outs, 0).float()


@torch.no_grad()
def reward_to_z(reward_fn: RewardLike, B: Callable, g_samples: torch.Tensor, d: Optional[int] = None,
                center: bool = False, batch_size: int = 16384) -> torch.Tensor:
    """z_r = sqrt(d) * normalize(mean_i r(g_i) B(g_i)) over replay states g_samples [..., G_DIM].

    center=True subtracts the mean reward first (removes the component along mean B, which only
    reflects the state distribution, not the task). Returns z [d] on g_samples' device.
    """
    r_fn = _as_reward(reward_fn)
    g = g_samples.reshape(-1, g_samples.shape[-1]).float()
    r = torch.cat([r_fn(g[i:i + batch_size]) for i in range(0, g.shape[0], batch_size)]).float()
    if center:
        r = r - r.mean()
    b = _encode(B, g, batch_size)
    z = (r.unsqueeze(-1) * b).mean(0)
    return project_sphere(z, d or b.shape[-1])


@torch.no_grad()
def rollout_best_state(reward_fn: RewardLike, g_samples: torch.Tensor,
                       batch_size: int = 16384) -> Tuple[torch.Tensor, float, int]:
    """Find (g*, max_r, best_idx) maximizing reward_fn over rollout states g_samples [..., G_DIM]."""
    r_fn = _as_reward(reward_fn)
    g = g_samples.reshape(-1, g_samples.shape[-1]).float()
    r = torch.cat([r_fn(g[i:i + batch_size]) for i in range(0, g.shape[0], batch_size)]).float()
    r = torch.nan_to_num(r, nan=-1e9, posinf=1e9, neginf=-1e9)
    best_idx = int(torch.argmax(r).item())
    return g[best_idx], float(r[best_idx].item()), best_idx


@torch.no_grad()
def argmax_rollout_z(reward_fn: RewardLike, B: Callable, g_samples: torch.Tensor,
                     d: Optional[int] = None, batch_size: int = 16384) -> torch.Tensor:
    """Compute z = B(g*) where g* = argmax_{g in g_samples} reward_fn(g).

    Evaluates the Gemini/user reward on all states in the rollout, identifies the
    state g* that achieves maximum reward, and maps it via the backward representation B
    to the goal-directed FB task vector z, scaled to radius sqrt(d).
    """
    g_star, _, _ = rollout_best_state(reward_fn, g_samples, batch_size=batch_size)
    b = B(g_star.unsqueeze(0)).float().reshape(-1)
    return project_sphere(b, d or b.shape[-1])


@torch.no_grad()
def goal_z(B: Callable, g_goal: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
    """Goal reaching: z = sqrt(d) * normalize(B(g_goal)); several goals are averaged."""
    g = g_goal.reshape(-1, g_goal.shape[-1]).float()
    b = B(g).float().mean(0)
    return project_sphere(b, d or b.shape[-1])


@torch.no_grad()
def avoid_z(B: Callable, g_goal: torch.Tensor, g_forbidden: torch.Tensor, lam: float = 1.0,
            d: Optional[int] = None) -> torch.Tensor:
    """Reach while avoiding (FB paper eq. 7): z = B(g_goal) - lam * sum_j B(g_forbidden_j), on the sphere."""
    g0 = g_goal.reshape(-1, g_goal.shape[-1]).float()
    gf = g_forbidden.reshape(-1, g_forbidden.shape[-1]).float()
    z = B(g0).float().mean(0) - lam * B(gf).float().sum(0)
    return project_sphere(z, d or z.shape[-1])


def mission_z(mission, B: Callable, g_samples: torch.Tensor, step: Optional[int] = None,
              center: bool = False, agent: Optional[int] = None,
              method: str = "argmax") -> torch.Tensor:
    """Compute task vector z for a MissionSpec (or one of its steps/segments).

    method: "argmax" (default) uses z = B(argmax_{g in rollout} r(g));
            "expectation" uses linear FB projection z_r = mean r(g) B(g).
    """
    expr = mission.stage(step, agent).reward if hasattr(mission, "stage") else mission
    if method == "argmax":
        return argmax_rollout_z(expr, B, g_samples)
    return reward_to_z(expr, B, g_samples, center=center)


# ------------------------------------------------------------------ CEM
def _current_obs(env) -> Dict[str, torch.Tensor]:
    for name in ("observe", "get_obs", "_obs"):
        fn = getattr(env, name, None)
        if callable(fn):
            return fn()
    raise AttributeError("env needs observe()/_obs() to read observations after copy_state")


@torch.no_grad()
def cem_refine(env, policy: Callable, z0: torch.Tensor, reward_fn: RewardLike, src_env: int = 0,
               n_candidates: int = 64, horizon: int = 60, iters: int = 3, elite_frac: float = 0.125,
               sigma: float = 0.3, bindings=None, done_fn: Optional[RewardLike] = None,
               dst_ids: Optional[Sequence[int]] = None, generator: Optional[torch.Generator] = None,
               min_sigma: float = 0.02, z_per_agent: bool = False) -> Tuple[torch.Tensor, Dict]:
    """Refine z with the cross-entropy method in clones of world src_env.

    Requires env.num_envs >= n_candidates (asserted). Candidate worlds dst_ids default to the first
    n_candidates worlds other than src_env (src_env itself is used only if num_envs == n_candidates).
    Each iteration: env.copy_state(src_env, dst_ids) clones the source world, optional ``bindings``
    are bound in the clones (a List[AnchorBinding] for all agents, or {agent: List[AnchorBinding]}),
    the policy flies candidate k in world dst_ids[k] for ``horizon`` steps, and its score is the
    sum over steps of reward_fn(obs["g"]) averaged over agents. A world stops accumulating once
    env reports done or done_fn(g) > 0.5 for every agent. Candidate 0 is always the current mean,
    so the returned z scores at least as well as z0 did in iteration 1.

    State: every env.step advances ALL worlds, so the source world is snapshotted (env.get_state)
    before the first rollout and written back (env.set_state(snapshot, [src_env])) before each
    clone and at the end. Nothing else is restored: candidate worlds are left in their rolled-out
    state, and any other world (neither source nor candidate) is advanced by the rollouts with the
    policy flying the current CEM mean. Reset or restore those yourself if they matter.

    policy(obs, z [N, d]) -> [N, A, ACTION_DIM]  (z [N, A, d] if z_per_agent).
    Returns (best z [d], info) with info = {"z0_score", "best_score", "history": [per-iteration stats]}.
    """
    N = env.num_envs
    assert N >= n_candidates, f"cem_refine needs env.num_envs ({N}) >= n_candidates ({n_candidates})"
    dev = env.device
    r_fn = _as_reward(reward_fn)
    d_fn = _as_reward(done_fn)
    d = z0.shape[-1]
    if dst_ids is None:
        others = [i for i in range(N) if i != src_env]
        ids = (others + [src_env])[:n_candidates]
    else:
        ids = [int(i) for i in dst_ids]
        assert len(ids) == n_candidates, "dst_ids must have n_candidates entries"
    dst = torch.tensor(ids, dtype=torch.long, device=dev)
    snapshot = env.get_state()
    src_t = torch.tensor([src_env], dtype=torch.long, device=dev)

    def clone():
        env.set_state(snapshot, src_t)   # the live source row was advanced by the previous rollout
        env.copy_state(src_env, dst)

    mu = project_sphere(z0.detach().float().to(dev).reshape(d))
    std = torch.full((d,), float(sigma), device=dev)
    n_elite = max(2, int(round(elite_frac * n_candidates)))
    best_z, best_score, z0_score = mu.clone(), -float("inf"), None
    history: List[Dict] = []
    for it in range(iters):
        eps = torch.randn(n_candidates, d, generator=generator).to(dev) if generator is not None \
            else torch.randn(n_candidates, d, device=dev)
        cand = mu + std * eps
        cand[0] = mu
        cand = project_sphere(cand)
        clone()
        if bindings is not None:
            if isinstance(bindings, dict):
                for ag, bl in bindings.items():
                    env.bind_anchors(dst, bl, agent=int(ag))
            else:
                env.bind_anchors(dst, bindings)
        obs = _current_obs(env)
        z_all = mu.expand(N, d).clone()
        z_all[dst] = cand
        scores = _rollout(env, policy, obs, z_all, dst, r_fn, d_fn, horizon, z_per_agent)
        if it == 0:
            z0_score = float(scores[0])
        order = torch.argsort(scores, descending=True)
        elites = cand[order[:n_elite]]
        if float(scores[order[0]]) > best_score:
            best_score = float(scores[order[0]])
            best_z = cand[order[0]].clone()
        history.append({"iter": it, "best": float(scores[order[0]]), "mean": float(scores.mean()),
                        "elite_mean": float(scores[order[:n_elite]].mean()), "mu_score": float(scores[0]),
                        "sigma": float(std.mean())})
        mu = project_sphere(elites.mean(0))
        std = elites.std(0).clamp_min(min_sigma)
    env.set_state(snapshot, src_t)
    return best_z, {"z0_score": z0_score, "best_score": best_score, "history": history,
                    "dst_ids": ids}


@torch.no_grad()
def _rollout(env, policy, obs, z_all, dst, r_fn, d_fn, horizon, z_per_agent) -> torch.Tensor:
    N, A = env.num_envs, env.num_agents
    total = torch.zeros(N, device=env.device)
    active = torch.ones(N, dtype=torch.bool, device=env.device)
    z_in = z_all.unsqueeze(1).expand(N, A, z_all.shape[-1]) if z_per_agent else z_all
    for _ in range(horizon):
        act = policy(obs, z_in)
        out = env.step(act)
        obs = out.obs
        g = obs["g"]
        r = r_fn(g).reshape(N, A).mean(-1)
        total += torch.where(active, r, torch.zeros_like(r))
        fin = out.done.reshape(N).bool()
        if d_fn is not None:
            fin = fin | (d_fn(g).reshape(N, A) > 0.5).all(-1)
        active = active & ~fin
        if not active[dst].any():
            break
    return total[dst]
