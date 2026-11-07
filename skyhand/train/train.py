"""FB-CPR / SF pretraining entry point (PLAN.md 7.3).

    python -m skyhand.train.train --scene trash|ocean|both --backend torch|genesis \
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
from .agent import Agent
from .buffer import DemoStore, ReplayBuffer
from .config import Config, tiny_config
from .zsample import WorldZ

EXPERT_ERRORS = (NotImplementedError, AttributeError, ImportError)


def try_expert(env) -> Optional[torch.Tensor]:
    try:
        return env.expert_action()
    except EXPERT_ERRORS:
        return None


def random_actions(n: int, a: int, device) -> torch.Tensor:
    return torch.rand(n, a, spec.ACTION_DIM, device=device) * 2.0 - 1.0


class _Stream:
    def __init__(self, idx: int, env, world_offset: int, wz: WorldZ):
        self.idx, self.env, self.world_offset, self.wz = idx, env, world_offset, wz
        self.obs = env.reset()


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.agent = Agent(cfg, self.device)
        d = self.agent.d
        bdev = cfg.buffer_device or cfg.device
        self.buffer = ReplayBuffer(cfg.buffer_capacity, d, bdev)
        self.demo = DemoStore(cfg.demo_capacity, cfg.demo_seg_len, self.device)
        scenes = ["trash", "ocean"] if cfg.scene == "both" else [cfg.scene]
        self.streams: List[_Stream] = []
        off = 0
        for i, sc in enumerate(scenes):
            env = make_env(sc, cfg.num_envs, cfg.device, backend=cfg.backend, lidar=cfg.lidar, seed=cfg.seed + i)
            wz = WorldZ(cfg.num_envs, self.agent.learner.zs, cfg.z_resample_steps, self.device)
            self.streams.append(_Stream(i, env, off, wz))
            off += cfg.num_envs
        self.demo_envs = {}
        self.expert_ok = True
        self.iter = 0
        self.env_steps = 0
        os.makedirs(cfg.out, exist_ok=True)
        with open(os.path.join(cfg.out, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        self._csv_path = os.path.join(cfg.out, "metrics.csv")
        self._csv_fields: Optional[List[str]] = None
        self._acc: Dict[str, float] = {}
        self._acc_n = 0
        self._t0 = time.time()

    # -------------------------------------------------- data collection
    def _encode(self, g):
        return self.agent.learner.encode_g(g.to(self.device))

    def env_step(self, st: _Stream, warm: bool) -> Dict[str, float]:
        env, cfg = st.env, self.cfg
        n, a = env.num_envs, env.num_agents
        if warm:
            act = random_actions(n, a, env.device)
            ex = try_expert(env) if self.expert_ok else None
            if ex is not None:
                use = torch.rand(n, device=env.device) < cfg.warm_expert_frac
                act = torch.where(use.view(n, 1, 1), ex.to(env.device).float(), act)
        else:
            act = self.agent.act(st.obs, st.wz.z, noise=cfg.expl_noise).to(env.device)
        out = env.step(act)
        reset = out.done | out.truncated
        self.buffer.add(st.obs, act, out.obs, out.done, reset, st.wz.z, st.world_offset, stream=st.idx)
        if reset.any():
            st.obs = env.reset(torch.nonzero(reset).squeeze(-1))
        else:
            st.obs = out.obs
        st.wz.step(self._encode, self.buffer, self.demo)
        self.env_steps += n * a
        ev = out.events
        return {"collision_rate": ev["collision"].float().mean().item(),
                "shield_rate": ev["shield"].float().mean().item(),
                "done_rate": out.done.float().mean().item()}

    @torch.no_grad()
    def collect_demos(self) -> int:
        """Roll the scripted expert in small separate envs and store g segments."""
        if not self.expert_ok:
            return 0
        cfg = self.cfg
        added = 0
        for st in self.streams:
            sc = st.env.scene
            if sc not in self.demo_envs:
                self.demo_envs[sc] = make_env(sc, cfg.demo_envs, cfg.device, backend=cfg.backend,
                                              lidar=False, seed=cfg.seed + 100 + st.idx)
            env = self.demo_envs[sc]
            obs = env.reset()
            gs, cuts = [], []
            for _ in range(cfg.demo_steps):
                ex = try_expert(env)
                if ex is None:
                    self.expert_ok = False
                    print("[train] env.expert_action() unavailable: random warm start only, no demos",
                          flush=True)
                    return added
                out = env.step(ex)
                reset = out.done | out.truncated
                gs.append(out.obs["g"].reshape(env.num_envs * env.num_agents, -1))
                cuts.append(reset.view(-1, 1).expand(-1, env.num_agents).reshape(-1))
                if reset.any():
                    env.reset(torch.nonzero(reset).squeeze(-1))
            added += self.demo.add_rollout(torch.stack(gs), torch.stack(cuts))
        return added

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
    def _log(self, extra: Dict[str, float]) -> None:
        row = {"iter": self.iter, "env_steps": self.env_steps, "updates": self.agent.learner.num_updates,
               "buffer": len(self.buffer), "demo_segs": len(self.demo),
               "sps": round(self.env_steps / max(time.time() - self._t0, 1e-6), 1)}
        if self._acc_n:
            row.update({k: v / self._acc_n for k, v in self._acc.items()})
        row.update(extra)
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, self._csv_fields).writeheader()
        elif any(k not in self._csv_fields for k in row) and self._acc_n:
            # first log after learning starts adds loss columns: rewrite header
            old = list(csv.DictReader(open(self._csv_path)))
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
        self.agent.g_pool = self.g_pool()
        self.agent.save(path)
        return path

    # -------------------------------------------------- main loop
    def run(self) -> str:
        cfg = self.cfg
        self.collect_demos()
        for _ in range(cfg.warm_steps):
            for st in self.streams:
                self.env_step(st, warm=True)
        print(f"[train] warm start done: buffer={len(self.buffer)} demo_segs={len(self.demo)} "
              f"expert={'yes' if self.expert_ok else 'no'}", flush=True)
        ev: Dict[str, float] = {}
        for it in range(1, cfg.steps + 1):
            self.iter = it
            st = self.streams[(it - 1) % len(self.streams)]
            ev = self.env_step(st, warm=False)
            m = self.learn()
            if m:
                for k, v in m.items():
                    self._acc[k] = self._acc.get(k, 0.0) + v
                self._acc_n += 1
            if cfg.demo_every and it % cfg.demo_every == 0:
                self.collect_demos()
            if cfg.log_every and it % cfg.log_every == 0:
                self._log(ev)
            if cfg.save_every and it % cfg.save_every == 0:
                self.save()
        return self.save()


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


def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(prog="python -m skyhand.train.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", choices=["trash", "ocean", "both"], default="trash")
    p.add_argument("--backend", choices=["torch", "genesis"], default="torch")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu | mps")
    p.add_argument("--steps", type=int, default=None)
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
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any Config field, repeatable")
    a = p.parse_args(argv)
    cfg = tiny_config() if a.tiny else Config()
    cfg.scene, cfg.backend = a.scene, a.backend
    direct = {"num_envs": a.num_envs, "device": a.device, "steps": a.steps, "out": a.out,
              "method": a.method, "batch_size": a.batch_size, "updates_per_step": a.updates_per_step,
              "warm_steps": a.warm_steps, "buffer_capacity": a.buffer_capacity,
              "buffer_device": a.buffer_device, "seed": a.seed}
    for k, v in direct.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.no_lidar:
        cfg.lidar = False
    fmap = {f.name: f for f in dataclasses.fields(Config)}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in fmap:
            p.error(f"unknown config field {k!r}")
        setattr(cfg, k, _coerce(fmap[k], v))
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available, using cpu", file=sys.stderr)
        cfg.device = "cpu"
    return cfg


def main(argv: Optional[List[str]] = None) -> str:
    cfg = parse_args(argv)
    print(f"[train] method={cfg.method} scene={cfg.scene} backend={cfg.backend} envs={cfg.num_envs} "
          f"device={cfg.device} out={cfg.out}", flush=True)
    path = Trainer(cfg).run()
    print(f"[train] saved {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
