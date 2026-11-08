"""Mission success eval: does a checkpoint fly the missions the demo will show?

For every mission it runs many worlds at once, the same way the demo server runs one:
compile the English text (offline compiler, no network, unless --gemini), bind the anchors,
infer z from the reward program (scripted rollouts with those anchors bound, then
z = sqrt(d) * normalize(mean r(g) B(g))), fly the frozen policy, and score each world with
the server's rules:

  success   done_when holds (shared mission: any drone; per-drone jobs: every drone with a
            done_when has met it, and finished drones hold position), or a carry mission's
            "delivered" event fires
  failure   the episode ends early (crash, out of bounds)
  timeout   the scene's episode length runs out

Each world compiles against its own layout, since entity names such as item_3 mean
different things in different worlds. --baseline scripted scores the demo's scripted
controller on the same worlds, the ceiling to compare a checkpoint against.

    python -m demo.mission_eval --checkpoint runs/job/agent.pt
    python -m demo.mission_eval --checkpoint runs/job/agent.pt --scenes trash delivery --envs 32 --rounds 2
    python -m demo.mission_eval --baseline scripted --scenes trash
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from skyhand import spec

from . import pipeline as P

# The missions the demo shows, per scene. The first of each is the scene's headline example.
MISSIONS: Dict[str, List[str]] = {
    "trash": ["pick up the trash bag and put it in the bin without hitting the cars",
              "fly to the bin",
              "pick up the pizza box and put it in the bin"],
    "delivery": ["deliver the organ cooler to the drop-off pad smoothly and stay out of the no-fly zones",
                 "fly to the pickup pad"],
    "ocean": ["net the school closest to the boat without catching any whales"],
    "world": ["drone 0 pick up the pizza box and drop it in the garbage truck, drone 1 deliver the organ cooler "
              "to the clinic, drone 2 fly to the boat",
              "collect the trash and put it in the garbage truck without hitting the cars"],
}


TRIVIAL_STEPS = 10      # a mission "done" within 0.5 s was already done when it started (e.g. item next to the bin)


@dataclass
class MissionResult:
    scene: str
    mission: str
    controller: str
    worlds: int
    success: float
    failure: float
    timeout: float
    mean_success_s: Optional[float]       # mean time to success over successful worlds
    mean_reward: float                    # mean per-step reward of the compiled program
    seconds: float                        # wall-clock for this mission
    notes: List[str] = field(default_factory=list)
    trivial: float = 0.0                  # worlds already done before the first step (excluded from the rates)


class _WorldView:
    """One world of a batched env, as the compiler sees it: entity descriptions of world i."""

    def __init__(self, env, i: int):
        self._env, self._i = env, int(i)

    def entity_descriptions(self, env_id: Optional[int] = None) -> Dict[str, str]:
        return self._env.entity_descriptions(self._i if env_id is None else env_id)

    def __getattr__(self, name):
        return getattr(self._env, name)


def offline_api() -> Dict[str, Callable]:
    """The mission package without the Gemini call (deterministic, no network)."""
    api = P.load_mission_api()
    api.pop("compile_mission", None)
    return api


def check_api(api: Dict[str, Callable]) -> Optional[str]:
    """A warning when the mission package is missing, so missions would compile with the demo's
    last-resort keyword parser instead of the compiler the demo uses."""
    if "offline_compile" in api or "compile_mission" in api:
        return None
    try:
        import skyhand.mission  # noqa: F401
        why = "it has no compiler"
    except Exception as e:
        why = f"{type(e).__name__}: {e}"
    return (f"WARNING: skyhand.mission is unavailable ({why}); missions compile with the keyword parser, "
            "not the demo's compiler. Install pydantic (pip install pydantic) for comparable numbers.")


def _fixed_z(z: Optional[torch.Tensor], a: int) -> Optional[torch.Tensor]:
    """compute_z gives [d] (team) or [A, d] (per drone); return [A, d]."""
    if z is None:
        return None
    z = torch.as_tensor(z).float()
    return z.unsqueeze(0).expand(a, -1).clone() if z.dim() == 1 else z


def _done_now(m, g: torch.Tensor, latched: set) -> bool:
    """The server's done rule for one world. g [A, G_DIM]; latched: per-drone jobs already met."""
    if not m.agent_done_fns:
        return m.done_fn is not None and bool((m.done_fn(g).reshape(-1) > 0.5).any())
    need = {ag for ag, fn in m.agent_done_fns.items() if fn is not None and ag < g.shape[0]}
    for ag in need - latched:
        if bool(m.agent_done_fns[ag](g[ag:ag + 1]).reshape(-1)[0] > 0.5):
            latched.add(ag)
    return bool(need) and need <= latched


def _reward_now(m, g: torch.Tensor) -> float:
    if m.agent_reward_fns:
        rs = [float(fn(g[ag:ag + 1]).reshape(-1)[0]) for ag, fn in m.agent_reward_fns.items() if ag < g.shape[0]]
        return sum(rs) / max(len(rs), 1)
    return float(m.reward_fn(g[0:1]).reshape(-1)[0])


def _is_carry(m) -> bool:
    if m.agents:
        return False
    a0, a1 = m.anchors.get("a0"), m.anchors.get("a1")
    return isinstance(a0, str) and a0.startswith("item_") and a1 in ("bin", "pad_dropoff")


def run_mission(scene: str, text: str, agent, *, envs: int = 16, seed: int = 0, device: str = "cpu",
                lidar: bool = True, max_steps: Optional[int] = None, z_samples_envs: int = 32,
                z_samples_steps: int = 60, mission_api: Optional[Dict[str, Callable]] = None,
                make_env=None) -> MissionResult:
    """Fly one mission in `envs` worlds. agent=None scores the scripted controller instead."""
    t0 = time.time()
    if make_env is None:
        from skyhand.env.core import make_env as _mk

        def make_env(sc, n, **kw):
            kw.setdefault("lidar", lidar)
            kw.setdefault("random_anchors", False)
            return _mk(sc, num_envs=n, device=device, **kw)
    sample_env = lambda sc, n: make_env(sc, n, lidar=False, seed=seed + 7)   # noqa: E731  z sampling needs g only
    pipe = P.Pipeline(agent=agent, mission_api=offline_api() if mission_api is None else mission_api,
                      device=device, z_samples_envs=z_samples_envs, z_samples_steps=z_samples_steps)
    env = make_env(scene, envs, seed=seed)
    n, a = env.num_envs, env.num_agents
    steps = int(max_steps or env.episode_steps)
    notes: List[str] = []
    missions, zs = [], []
    for i in range(n):
        m = pipe.compile(text, scene, _WorldView(env, i))
        pipe.bind(env, m, torch.tensor([i], device=env.device))
        missions.append(m)
        zs.append(_fixed_z(pipe.compute_z(m.reward_fn, scene, m.anchors, sample_env, mission=m), a))
        if m.warnings:
            notes += [f"world {i}: {w}" for w in m.warnings]
    if missions[0].source:
        notes.insert(0, f"compiled by: {missions[0].source}")
    ctrl = pipe.make_controller(env)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    z = torch.stack(zs).to(getattr(agent, "device", "cpu")) if agent is not None else None   # [N, A, d]
    obs = P.observe(env)
    status = ["flying"] * n
    for i in range(n):            # already done at the start (e.g. the item spawned next to the bin): not a success
        try:
            if _done_now(missions[i], obs["g"][i], set()):
                status[i] = "trivial"
        except Exception:
            pass
    t_succ: List[float] = []
    latched = [set() for _ in range(n)]
    rew_sum, rew_cnt = 0.0, 0
    for step in range(steps):
        act = ctrl(obs, z)
        for i in range(n):        # per-drone jobs: a drone that met its goal holds position
            if status[i] == "flying" and latched[i] and missions[i].agents:
                for ag in latched[i]:
                    act[i, ag, 0:4] = 0.0
        out = env.step(act)
        obs = out.obs
        g = out.obs["g"]
        for i in range(n):
            if status[i] != "flying":
                continue
            m = missions[i]
            try:
                rew_sum += _reward_now(m, g[i])
                rew_cnt += 1
                ok = _done_now(m, g[i], latched[i])
            except Exception as e:   # a bad program counts as not done, like the server
                notes.append(f"world {i}: reward evaluation failed: {e}")
                ok = False
            delivered = _is_carry(m) and bool(out.events["delivered"][i].any())
            if (ok or delivered) and step + 1 < TRIVIAL_STEPS:   # met before the drone could do anything
                status[i] = "trivial"
            elif ok or delivered:
                status[i] = "success"
                t_succ.append((step + 1) * spec.POLICY_DT)
            elif bool(out.done[i]):
                status[i] = "failure"
        if all(s != "flying" for s in status):
            break
    status = [s if s != "flying" else "timeout" for s in status]
    scored = max(sum(s != "trivial" for s in status), 1)
    frac = lambda k: sum(s == k for s in status) / scored   # noqa: E731
    n_triv = sum(s == "trivial" for s in status)
    if n_triv:
        notes.insert(0, f"{n_triv} world(s) met done_when within {TRIVIAL_STEPS} steps (already done at the start); excluded from the rates")
    return MissionResult(scene=scene, mission=text, controller="policy" if agent is not None else "scripted",
                         worlds=n, success=frac("success"), failure=frac("failure"), timeout=frac("timeout"),
                         trivial=n_triv / n,
                         mean_success_s=(sum(t_succ) / len(t_succ)) if t_succ else None,
                         mean_reward=rew_sum / max(rew_cnt, 1), seconds=time.time() - t0, notes=notes[:10])


def evaluate(agent, scenes: Sequence[str], *, missions: Optional[Dict[str, List[str]]] = None, envs: int = 16,
             rounds: int = 1, seed: int = 0, device: str = "cpu", lidar: bool = True,
             max_steps: Optional[int] = None, verbose: bool = True, **kw) -> List[MissionResult]:
    """Every mission of every scene, `rounds` times over fresh layouts (results merged per mission)."""
    missions = missions or MISSIONS
    out: List[MissionResult] = []
    for sc in scenes:
        for text in missions.get(sc, []):
            parts = [run_mission(sc, text, agent, envs=envs, seed=seed + 1000 * r, device=device, lidar=lidar,
                                 max_steps=max_steps, **kw) for r in range(rounds)]
            res = _merge(parts)
            out.append(res)
            if verbose:
                print(_row(res), flush=True)
    return out


def _merge(parts: List[MissionResult]) -> MissionResult:
    if len(parts) == 1:
        return parts[0]
    w = sum(p.worlds for p in parts)
    avg = lambda k: sum(getattr(p, k) * p.worlds for p in parts) / w   # noqa: E731
    succ = [(p.mean_success_s, p.success * p.worlds) for p in parts if p.mean_success_s is not None]
    ns = sum(c for _, c in succ)
    return MissionResult(scene=parts[0].scene, mission=parts[0].mission, controller=parts[0].controller, worlds=w,
                         success=avg("success"), failure=avg("failure"), timeout=avg("timeout"),
                         mean_success_s=(sum(t * c for t, c in succ) / ns) if ns else None,
                         mean_reward=avg("mean_reward"), seconds=sum(p.seconds for p in parts), trivial=avg("trivial"),
                         notes=[x for p in parts for x in p.notes][:10])


def _row(r: MissionResult) -> str:
    t = f"{r.mean_success_s:5.1f}s" if r.mean_success_s is not None else "   -  "
    return (f"[{r.controller:8s}] {r.scene:8s} success {100 * r.success:5.1f}%  crash {100 * r.failure:5.1f}%  "
            f"timeout {100 * r.timeout:5.1f}%  t_success {t}  reward {r.mean_reward:7.3f}  "
            f"({r.worlds} worlds{f', {round(r.trivial * r.worlds)} trivial' if r.trivial else ''}, {r.seconds:.0f}s)  "
            f"{r.mission[:70]}")


def main(argv: Optional[Sequence[str]] = None) -> List[MissionResult]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default=None, help="agent.pt to evaluate")
    p.add_argument("--baseline", choices=["scripted"], default=None,
                   help="also (or, without --checkpoint, only) score the demo's scripted controller")
    p.add_argument("--scenes", nargs="+", default=list(spec.SCENES), choices=list(spec.SCENES))
    p.add_argument("--mission", action="append", default=None,
                   help="evaluate this text instead of the built-in list (repeatable; needs one --scenes)")
    p.add_argument("--envs", type=int, default=16, help="worlds per mission per round")
    p.add_argument("--rounds", type=int, default=1, help="repeat over fresh layouts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default: cuda if available)")
    p.add_argument("--max-steps", type=int, default=None, help="cap the episode length (default: the scene's)")
    p.add_argument("--z-samples-envs", type=int, default=32)
    p.add_argument("--z-samples-steps", type=int, default=60)
    p.add_argument("--gemini", action="store_true", help="compile with Gemini when a key is set (default: offline)")
    p.add_argument("--out", default=None, help="write results as JSON here")
    args = p.parse_args(argv)
    if args.checkpoint is None and args.baseline is None:
        p.error("give --checkpoint, --baseline scripted, or both")
    if args.mission and len(args.scenes) != 1:
        p.error("--mission needs exactly one --scenes")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    missions = {args.scenes[0]: args.mission} if args.mission else None
    api = P.load_mission_api() if args.gemini else offline_api()
    warn = check_api(api)
    if warn:
        print(f"[mission_eval] {warn}", flush=True)
    common = dict(missions=missions, envs=args.envs, rounds=args.rounds, seed=args.seed, device=device,
                  max_steps=args.max_steps, z_samples_envs=args.z_samples_envs,
                  z_samples_steps=args.z_samples_steps, mission_api=api)
    results: List[MissionResult] = []
    if args.checkpoint:
        from skyhand.train.agent import Agent
        if not os.path.exists(args.checkpoint):
            p.error(f"checkpoint {args.checkpoint} not found")
        agent = Agent.load(args.checkpoint, device=device)
        lidar = bool(getattr(agent.cfg, "lidar", True))
        print(f"[mission_eval] {args.checkpoint}: method={agent.method} d={agent.d} "
              f"updates={agent.learner.num_updates} lidar={lidar} device={device}", flush=True)
        results += evaluate(agent, args.scenes, lidar=lidar, **common)
    if args.baseline == "scripted":
        results += evaluate(None, args.scenes, lidar=True, **common)
    if results:
        pol = [r for r in results if r.controller == "policy"]
        if pol:
            print(f"[mission_eval] policy mean success {100 * sum(r.success for r in pol) / len(pol):.1f}% "
                  f"over {len(pol)} missions", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=1)
        print(f"[mission_eval] wrote {args.out}", flush=True)
    return results


if __name__ == "__main__":
    main()
