"""FB-CPR / SF pretraining entry point (PLAN.md 7.3).

    python -m skyhand.train.train --scene trash|delivery|ocean|all --backend torch|genesis \
        --num-envs 1024 --device cuda --steps 100000 --out runs/NAME [--method sf] \
        [--set key=value ...]

Loop: warm start (expert + random actions), then each iteration: act with
exploration noise using each world's team z, env.step, store transitions,
reset done|truncated worlds, advance z timers, run G gradient updates.
Periodically: checkpoint (agent.pt), metrics to stdout and metrics.csv,
collect expert demo segments from a separate small expert env.

Never imports genesis (the genesis backend is loaded only through make_env).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

from .. import spec
from ..env.core import make_env
from . import dist as D
from .agent import Agent
from .buffer import DemoStore, ReplayBuffer
from .config import Config, tiny_config
from .nets import sphere
from .zsample import WorldZ

EXPERT_ERRORS = (NotImplementedError,)      # "no expert for this env"; real bugs propagate


def scene_list(scene: str) -> List[str]:
    """Scenes to train on: one, or every scene for "all" (and the older alias "both")."""
    return list(spec.SCENES) if scene in ("all", "both") else [scene]


def try_expert(env) -> Optional[torch.Tensor]:
    try:
        return env.expert_action()
    except EXPERT_ERRORS:
        return None


def random_actions(n: int, a: int, device) -> torch.Tensor:
    return torch.rand(n, a, spec.ACTION_DIM, device=device) * 2.0 - 1.0


def first_worlds(n: int, frac: float, device) -> torch.Tensor:
    """Fixed mask of the first round(frac * n) worlds. Fixed, so the expert's phase
    machine drives the same worlds every step and its episodes stay coherent."""
    k = int(round(max(0.0, min(1.0, frac)) * n))
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[:k] = True
    return mask


class _Stream:
    def __init__(self, idx: int, env, world_offset: int, wz: WorldZ, warm_frac: float, expert_frac: float):
        self.idx, self.env, self.world_offset, self.wz = idx, env, world_offset, wz
        self.obs = env.reset()
        self.warm_expert = first_worlds(env.num_envs, warm_frac, env.device)
        self.train_expert = first_worlds(env.num_envs, expert_frac, env.device)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rank, self.world = D.rank(), D.world_size()
        torch.manual_seed(cfg.seed)                    # identical network init on every rank
        self.device = torch.device(cfg.device)
        self.agent = Agent(cfg, self.device)
        if cfg.init_from:
            self.agent.load_weights(cfg.init_from)
            print(f"[train] continuing from {cfg.init_from} (updates so far: {self.agent.learner.num_updates})",
                  flush=True)
        D.broadcast_module(self.agent.learner)         # data-parallel: every rank starts from rank 0's weights
        if self.world > 1:
            torch.manual_seed(cfg.seed + 7919 * self.rank)   # but each rank sees different worlds and noise
            print(f"[train] data-parallel: {self.world} ranks, {cfg.batch_size} samples per rank per update "
                  f"({cfg.batch_size * self.world} per update)", flush=True)
        d = self.agent.d
        bdev = cfg.buffer_device or cfg.device
        self.buffer = ReplayBuffer(cfg.buffer_capacity, d, bdev)
        self.demo = DemoStore(cfg.demo_capacity, cfg.demo_seg_len, self.device)
        scenes = scene_list(cfg.scene)
        self.streams: List[_Stream] = []
        off = 0
        for i, sc in enumerate(scenes):
            env = make_env(sc, cfg.num_envs, cfg.device, backend=cfg.backend, lidar=cfg.lidar,
                           seed=cfg.seed + i + 1000 * self.rank,
                           domain_rand=cfg.domain_rand, rand_drone_dynamics=cfg.rand_drone_dynamics,
                           enable_wind=cfg.enable_wind, dynamic_wind=cfg.dynamic_wind,
                           enable_gusts=cfg.enable_gusts)
            wz = WorldZ(cfg.num_envs, self.agent.learner.zs, cfg.z_resample_steps, self.device)
            self.streams.append(_Stream(i, env, off, wz, cfg.warm_expert_frac, cfg.expert_world_frac))
            off += cfg.num_envs
        self.demo_envs = {}            # scene -> expert env that keeps running between collections
        self.expert_ok = True
        self.iter = 0
        self.env_steps = 0
        if D.is_main():
            os.makedirs(cfg.out, exist_ok=True)
            with open(os.path.join(cfg.out, "config.json"), "w") as f:
                json.dump(cfg.to_dict(), f, indent=2)
        self._csv_path = os.path.join(cfg.out, "metrics.csv")
        self._csv_fields: Optional[List[str]] = None
        self._acc: Dict[str, float] = {}       # learner metrics summed since the last log
        self._acc_n = 0
        self._env_acc: Dict[str, float] = {}   # env rates summed since the last log
        self._env_n = 0
        self._pending_eval: Dict[str, float] = {}   # eval metrics waiting for the next log row
        self._t0 = time.time()

    # -------------------------------------------------- data collection
    def _encode(self, g):
        return self.agent.learner.encode_g(g.to(self.device))

    def env_step(self, st: _Stream, warm: bool) -> Dict[str, float]:
        env, cfg = st.env, self.cfg
        n, a = env.num_envs, env.num_agents
        use = st.warm_expert if warm else st.train_expert
        if warm:
            act = random_actions(n, a, env.device)
        else:
            act = self.agent.act(st.obs, st.wz.z, noise=cfg.expl_noise).to(env.device)
        expert = torch.zeros(n, dtype=torch.bool, device=env.device)
        if self.expert_ok and use.any():
            ex = try_expert(env)
            if ex is None:
                self.expert_ok = False
                print("[train] env.expert_action() unavailable: policy actions only", flush=True)
            else:
                act = torch.where(use.view(n, 1, 1), ex.to(env.device).float(), act)
                expert = use
        out = env.step(act)
        reset = out.done | out.truncated
        done = out.info.get("agent_done", out.done) if out.info else out.done   # per drone: teammates truncate
        self.buffer.add(st.obs, act, out.obs, done, reset, st.wz.z, st.world_offset, stream=st.idx,
                        expert=expert)
        if reset.any():
            st.obs = env.reset(torch.nonzero(reset).squeeze(-1))
        else:
            st.obs = out.obs
        st.wz.step(self._encode, self.buffer, self.demo)
        self.env_steps += n * a
        ev = out.events
        pol = ~expert                                  # rates for policy-flown worlds only
        n_pol = max(int(pol.sum()), 1)
        return {"collision_rate": float(ev["collision"][pol].float().sum()) / (n_pol * a),
                "shield_rate": float(ev["shield"][pol].float().sum()) / (n_pol * a),
                "done_rate": float(out.done[pol].float().sum()) / n_pol,
                "grasp_rate": float(ev["grasp"][pol].float().sum()) / (n_pol * a),
                "delivered_rate": float(ev["delivered"][pol].float().sum()) / (n_pol * a)}

    @torch.no_grad()
    def collect_demos(self) -> int:
        """Roll the scripted expert in small separate envs and store g segments.

        The demo env is created and reset once, then keeps running between calls, so
        episodes progress through approach, grasp, carry, and release. The first call
        runs demo_initial_steps (a full episode); later calls run demo_steps."""
        if not self.expert_ok:
            return 0
        cfg = self.cfg
        added = 0
        for st in self.streams:
            sc = st.env.scene
            steps = cfg.demo_steps
            if sc not in self.demo_envs:
                self.demo_envs[sc] = make_env(sc, cfg.demo_envs, cfg.device, backend=cfg.backend,
                                              lidar=False, seed=cfg.seed + 100 + st.idx + 1000 * self.rank,
                                              domain_rand=cfg.domain_rand,
                                              rand_drone_dynamics=cfg.rand_drone_dynamics,
                                              enable_wind=cfg.enable_wind, dynamic_wind=cfg.dynamic_wind,
                                              enable_gusts=cfg.enable_gusts)
                steps = max(cfg.demo_steps, cfg.demo_initial_steps)
            env = self.demo_envs[sc]
            gs, cuts, bads = [], [], []
            for _ in range(steps):
                ex = try_expert(env)
                if ex is None:
                    self.expert_ok = False
                    print("[train] env.expert_action() unavailable: random warm start only, no demos",
                          flush=True)
                    return added
                out = env.step(ex)
                reset = out.done | out.truncated
                g_now = out.obs["g"].reshape(env.num_envs * env.num_agents, -1)
                gs.append(g_now)
                cuts.append(reset.view(-1, 1).expand(-1, env.num_agents).reshape(-1))
                # never show the discriminator bycatch or crashes as expert behaviour
                bads.append((spec.g_field(g_now, "bycatch").squeeze(-1) > 0.5) |
                            (spec.g_field(g_now, "collision").squeeze(-1) > 0.5))
                if reset.any():
                    env.reset(torch.nonzero(reset).squeeze(-1))
            added += self.demo.add_rollout(torch.stack(gs), torch.stack(cuts), torch.stack(bads))
        return added

    # -------------------------------------------------- evaluation
    def evaluate(self, steps: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation rollouts on clean/configured eval environments."""
        if not hasattr(self, "eval_envs"):
            self.eval_envs = {}
        return evaluate(self.agent, self.cfg, steps, self.eval_envs)

    # -------------------------------------------------- learning
    def learn(self) -> Optional[Dict[str, float]]:
        cfg = self.cfg
        start = cfg.learning_starts or cfg.batch_size
        if len(self.buffer) < start:
            return None
        acc: Dict[str, float] = {}
        for _ in range(cfg.updates_per_step):
            batch = self.buffer.sample(cfg.batch_size, self.device)
            m = self.agent.update(batch, self.demo)
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v / cfg.updates_per_step
        return acc

    # -------------------------------------------------- logging / checkpoints
    def _log(self) -> None:
        """One metrics row: env rates averaged over the window, learner losses averaged over
        the updates in it, and any eval metrics produced since the last row."""
        if not D.is_main():                            # only rank 0 writes metrics
            self._acc, self._acc_n, self._env_acc, self._env_n, self._pending_eval = {}, 0, {}, 0, {}
            return
        row = {"iter": self.iter, "env_steps": self.env_steps, "updates": self.agent.learner.num_updates,
               "buffer": len(self.buffer), "demo_segs": len(self.demo),
               "sps": round(self.env_steps / max(time.time() - self._t0, 1e-6), 1)}
        if self._env_n:
            row.update({k: v / self._env_n for k, v in self._env_acc.items()})
        if self._acc_n:
            row.update({k: v / self._acc_n for k, v in self._acc.items()})
        row.update(self._pending_eval)
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, self._csv_fields).writeheader()
        elif any(k not in self._csv_fields for k in row):
            # new columns (first losses, first eval): rewrite the header, keep old rows
            with open(self._csv_path, newline="") as f:
                old = list(csv.DictReader(f))
            self._csv_fields += [k for k in row if k not in self._csv_fields]
            with open(self._csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, self._csv_fields)
                w.writeheader()
                w.writerows(old)
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, self._csv_fields, extrasaction="ignore").writerow(row)
        print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in row.items()), flush=True)
        self._acc, self._acc_n = {}, 0
        self._env_acc, self._env_n = {}, 0
        self._pending_eval = {}

    @torch.no_grad()
    def g_pool(self) -> Optional[torch.Tensor]:
        """cfg.z_infer_samples states for z inference: buffer states plus ~10% demo states."""
        n = self.cfg.z_infer_samples
        if self.buffer.obs_size == 0 or n <= 0:
            return None
        nd = n // 10 if len(self.demo) else 0
        parts = [self.buffer.sample_g(n - nd, "cpu")]
        if nd:
            parts.append(self.demo.sample_g(nd, "cpu"))
        return torch.cat(parts)

    def save(self, name: str = "agent.pt") -> str:
        path = os.path.join(self.cfg.out, name)
        if not D.is_main():                            # every rank holds the same weights; rank 0 saves
            return path
        self.agent.g_pool = self.g_pool()
        self.agent.save(path, with_optim=True)        # optimizers too, so --init-from can continue the run
        return path

    # -------------------------------------------------- main loop
    def run(self) -> str:
        cfg = self.cfg
        t0 = time.time()
        print(f"[train] collecting expert demos ({len(self.streams)} scenes, first collection is a full episode)",
              flush=True)
        self.collect_demos()
        print(f"[train] demos collected: demo_segs={len(self.demo)} in {time.time() - t0:.0f}s", flush=True)
        t1 = time.time()
        for w in range(1, cfg.warm_steps + 1):
            for st in self.streams:
                self.env_step(st, warm=True)
            if w % 25 == 0 or w == cfg.warm_steps:
                el = time.time() - t1
                print(f"[train] warm start {w}/{cfg.warm_steps} buffer={len(self.buffer)} "
                      f"{el:.0f}s elapsed, ~{el / w * (cfg.warm_steps - w):.0f}s left", flush=True)
        print(f"[train] warm start done: buffer={len(self.buffer)} demo_segs={len(self.demo)} "
              f"expert={'yes' if self.expert_ok else 'no'} ({time.time() - t0:.0f}s since start)", flush=True)
        for it in range(1, cfg.steps + 1):
            self.iter = it
            st = self.streams[(it - 1) % len(self.streams)]
            for k, v in self.env_step(st, warm=False).items():
                self._env_acc[k] = self._env_acc.get(k, 0.0) + v
            self._env_n += 1
            m = self.learn()
            if m:
                for k, v in m.items():
                    self._acc[k] = self._acc.get(k, 0.0) + v
                self._acc_n += 1
            if cfg.demo_every and it % cfg.demo_every == 0:
                self.collect_demos()
            if cfg.eval_every and it % cfg.eval_every == 0 and D.is_main():
                self._pending_eval.update(self.evaluate())
            if cfg.log_every and it % cfg.log_every == 0:
                self._log()
            if cfg.save_every and it % cfg.save_every == 0:
                self.save()
        return self.save()


# ============================================================ evaluation
@torch.no_grad()
def evaluate(agent: Agent, cfg: Config, steps: Optional[int] = None,
             envs: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Evaluation rollouts of `agent` with a fixed random z per world; envs caches
    the eval envs across calls (dict scene -> env). Needs no Trainer or replay buffer."""
    steps = steps or cfg.eval_steps
    envs = {} if envs is None else envs
    tot_col, tot_sh, tot_done, count = 0.0, 0.0, 0.0, 0
    for i, sc in enumerate(scene_list(cfg.scene)):
        if sc not in envs:
            envs[sc] = make_env(sc, cfg.eval_envs, cfg.device, backend=cfg.backend, lidar=cfg.lidar,
                                seed=cfg.seed + 500 + i, domain_rand=cfg.eval_domain_rand,
                                rand_drone_dynamics=cfg.eval_rand_drone_dynamics,
                                enable_wind=cfg.eval_enable_wind, dynamic_wind=cfg.eval_dynamic_wind,
                                enable_gusts=cfg.eval_enable_gusts)
        env = envs[sc]
        obs = env.reset()
        z = sphere(torch.randn(cfg.eval_envs, agent.d, device=agent.device))
        for _ in range(steps):
            act = agent.act(obs, z, noise=0.0).to(env.device)
            out = env.step(act)
            reset = out.done | out.truncated
            obs = env.reset(torch.nonzero(reset).squeeze(-1)) if reset.any() else out.obs
            tot_col += out.events["collision"].float().mean().item()
            tot_sh += out.events["shield"].float().mean().item()
            tot_done += out.done.float().mean().item()
            count += 1
    return {"eval_collision_rate": tot_col / max(count, 1),
            "eval_shield_rate": tot_sh / max(count, 1),
            "eval_done_rate": tot_done / max(count, 1)}


EVAL_FIELDS = ("eval_envs", "eval_steps", "eval_domain_rand", "eval_rand_drone_dynamics",
               "eval_enable_wind", "eval_dynamic_wind", "eval_enable_gusts", "backend", "device", "seed")


def eval_checkpoint(path: str, cli: Config) -> Dict[str, float]:
    """Evaluate a saved agent: scene, lidar and networks come from the checkpoint's config,
    the eval_* settings, backend, device and seed from the command line."""
    agent = Agent.load(path, device=cli.device)
    cfg = Config.from_dict(agent.cfg.to_dict())
    for k in EVAL_FIELDS:
        setattr(cfg, k, getattr(cli, k))
    return evaluate(agent, cfg)


# ============================================================ CLI
def _coerce(field: dataclasses.Field, s: str):
    t = field.type
    if t in (bool, "bool"):
        return s.lower() in ("1", "true", "yes", "on")
    if t in (int, "int"):
        return int(float(s))
    if t in (float, "float"):
        return float(s)
    return s


def parse_args(argv: Optional[List[str]] = None, with_eval: bool = False):
    """Config from the command line; with with_eval=True returns (Config, --eval path or None)."""
    p = argparse.ArgumentParser(prog="python -m skyhand.train.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", choices=list(spec.SCENES) + ["all", "both"], default="trash",
                   help="one scene, or all of them ('both' is kept as an alias for all)")
    p.add_argument("--backend", choices=["torch", "genesis"], default="torch")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu | mps")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--init-from", default=None, metavar="CHECKPOINT",
                   help="continue training from this agent.pt (same architecture)")
    p.add_argument("--out", default=None)
    p.add_argument("--method", choices=["fb", "sf"], default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--updates-per-step", type=int, default=None)
    p.add_argument("--warm-steps", type=int, default=None)
    p.add_argument("--buffer-capacity", type=int, default=None)
    p.add_argument("--buffer-device", default=None)
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="tiny networks for smoke tests")
    p.add_argument("--no-domain-rand", action="store_true", help="globally disable domain randomisation")
    p.add_argument("--no-rand-dynamics", action="store_true", help="disable drone dynamics (mass/drag) randomisation")
    p.add_argument("--no-wind", action="store_true", help="disable ambient wind")
    p.add_argument("--no-dynamic-wind", action="store_true", help="disable stochastic dynamic wind turbulence")
    p.add_argument("--no-gusts", action="store_true", help="disable dynamic gusts")
    p.add_argument("--eval-every", type=int, default=None, help="evaluation interval in iterations")
    p.add_argument("--eval-envs", type=int, default=None, help="evaluation environments count")
    p.add_argument("--eval-steps", type=int, default=None, help="evaluation steps count")
    p.add_argument("--eval-domain-rand", action="store_true", help="enable domain randomisation during evaluation")
    p.add_argument("--eval-rand-dynamics", action="store_true", help="enable drone dynamics randomisation during evaluation")
    p.add_argument("--eval-wind", action="store_true", help="enable wind during evaluation")
    p.add_argument("--eval-dynamic-wind", action="store_true", help="enable dynamic wind turbulence during evaluation")
    p.add_argument("--eval-gusts", action="store_true", help="enable gusts during evaluation")
    p.add_argument("--eval", default=None, metavar="CHECKPOINT", help="evaluate a trained agent checkpoint and exit")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any Config field, repeatable")
    a = p.parse_args(argv)
    cfg = tiny_config() if a.tiny else Config()
    cfg.scene, cfg.backend = a.scene, a.backend
    direct = {"num_envs": a.num_envs, "device": a.device, "steps": a.steps, "out": a.out,
              "method": a.method, "batch_size": a.batch_size, "updates_per_step": a.updates_per_step,
              "warm_steps": a.warm_steps, "buffer_capacity": a.buffer_capacity,
              "buffer_device": a.buffer_device, "seed": a.seed, "init_from": a.init_from}
    for k, v in direct.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.no_lidar:
        cfg.lidar = False
    if a.no_domain_rand:
        cfg.domain_rand = False
    if a.no_rand_dynamics:
        cfg.rand_drone_dynamics = False
    if a.no_wind:
        cfg.enable_wind = False
    if a.no_dynamic_wind:
        cfg.dynamic_wind = False
    if a.no_gusts:
        cfg.enable_gusts = False

    if a.eval_every is not None:
        cfg.eval_every = a.eval_every
    if a.eval_envs is not None:
        cfg.eval_envs = a.eval_envs
    if a.eval_steps is not None:
        cfg.eval_steps = a.eval_steps
    if a.eval_domain_rand:
        cfg.eval_domain_rand = True
    if a.eval_rand_dynamics:
        cfg.eval_rand_drone_dynamics = True
    if a.eval_wind:
        cfg.eval_enable_wind = True
    if a.eval_dynamic_wind:
        cfg.eval_dynamic_wind = True
    if a.eval_gusts:
        cfg.eval_enable_gusts = True
    fmap = {f.name: f for f in dataclasses.fields(Config)}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in fmap:
            p.error(f"unknown config field {k!r}")
        setattr(cfg, k, _coerce(fmap[k], v))
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available, using cpu", file=sys.stderr)
        cfg.device = "cpu"
    return (cfg, a.eval) if with_eval else cfg


def main(argv: Optional[List[str]] = None) -> str:
    cfg, eval_ckpt = parse_args(argv, with_eval=True)
    if eval_ckpt is not None:
        print(f"[eval] evaluating checkpoint: {eval_ckpt}", flush=True)
        metrics = eval_checkpoint(eval_ckpt, cfg)
        print("[eval] " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()), flush=True)
        return eval_ckpt
    rank, world, _ = D.init_from_env(cfg.device)
    if rank != 0:                                      # one log stream: other ranks only print errors
        sys.stdout = open(os.devnull, "w")
    print(f"[train] method={cfg.method} scene={cfg.scene} backend={cfg.backend} envs={cfg.num_envs} "
          f"device={cfg.device} out={cfg.out}", flush=True)
    path = Trainer(cfg).run()
    print(f"[train] saved {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
