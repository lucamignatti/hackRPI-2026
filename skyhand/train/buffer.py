"""Replay buffer and demo store (PLAN.md 7.3, 10.3).

ReplayBuffer stores per-agent transitions flattened from [N, A]. Observations
live in a ring "obs table" (LiDAR kept as uint8). Each transition points at the
row of its obs and of its next obs, so a state is written once even though it
is both the next obs of step t and the obs of step t+1 (halves LiDAR memory).
Rows carry a write generation; a transition whose row was overwritten is
invalid and never sampled.

The caller guarantees, per `stream`, that obs passed to add() at step t+1
equals next_obs from step t for worlds that were not reset. This is checked on
proprio; rows that fail the check are simply written again.

Torch and the standard library only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .. import spec

OBS_KEYS = spec.CRITIC_KEYS + ("g",)
_DT = {"float32": torch.float32, "bool": torch.bool, "uint8": torch.uint8}


def flatten_obs(obs: Dict[str, torch.Tensor], keys: Sequence[str] = OBS_KEYS) -> Dict[str, torch.Tensor]:
    """[N, A, ...] -> [N*A, ...] for the given keys."""
    return {k: obs[k].reshape((-1,) + tuple(obs[k].shape[2:])) for k in keys}


class ReplayBuffer:
    def __init__(self, capacity: int, d: int, device="cpu", obs_capacity: Optional[int] = None):
        self.capacity = int(capacity)
        self.d = d
        self.device = torch.device(device)
        co = obs_capacity or self.capacity + max(self.capacity // 8, 1024)
        self.obs_capacity = co
        dev = self.device
        self.obs = {k: torch.zeros((co,) + spec.OBS_SPEC[k][1], dtype=_DT[spec.OBS_SPEC[k][0]], device=dev)
                    for k in OBS_KEYS}
        self.obs_gen = torch.full((co,), -1, dtype=torch.long, device=dev)
        self._obs_written = 0                     # monotone generation counter
        c = self.capacity
        self.obs_idx = torch.zeros(c, dtype=torch.long, device=dev)
        self.next_idx = torch.zeros(c, dtype=torch.long, device=dev)
        self.obs_g = torch.full((c,), -2, dtype=torch.long, device=dev)
        self.next_g = torch.full((c,), -2, dtype=torch.long, device=dev)
        self.action = torch.zeros(c, spec.ACTION_DIM, device=dev)
        self.done = torch.zeros(c, dtype=torch.bool, device=dev)
        self.z = torch.zeros(c, d, dtype=torch.float16, device=dev)   # half: 512 B at d=256
        self.world = torch.zeros(c, dtype=torch.long, device=dev)
        self.expert = torch.zeros(c, dtype=torch.bool, device=dev)     # action came from the scripted expert
        self.ptr = 0
        self.size = 0
        self._streams: Dict[object, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.size

    @property
    def obs_size(self) -> int:
        return min(self._obs_written, self.obs_capacity)

    # ------------------------------------------------------------ writing
    def _write_obs(self, flat: Dict[str, torch.Tensor], sel: torch.Tensor):
        """Write rows flat[sel] into the obs table; returns (row idx, gens) for them."""
        k = int(sel.shape[0])
        gens = self._obs_written + torch.arange(k, device=self.device)
        rows = gens % self.obs_capacity
        for key in OBS_KEYS:
            self.obs[key][rows] = flat[key][sel].to(self.device)
        self.obs_gen[rows] = gens
        self._obs_written += k
        return rows, gens

    @torch.no_grad()
    def add(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, next_obs: Dict[str, torch.Tensor],
            done: torch.Tensor, reset: torch.Tensor, z: torch.Tensor,
            world_offset: int = 0, stream: object = 0, expert: Optional[torch.Tensor] = None) -> None:
        """obs/next_obs: [N, A, ...] dicts; action [N, A, 5]; done, reset [N] bool
        (done masks the bootstrap; reset = done | truncated, worlds reset after this call);
        z [N, d] (team z) or [N, A, d]; expert [N] bool, True where the scripted expert acted."""
        n, a = action.shape[:2]
        m = n * a
        dev = self.device
        fo, fn = flatten_obs(obs), flatten_obs(next_obs)
        all_rows = torch.arange(m, device=fo["proprio"].device)
        # obs rows: reuse the previous call's next rows when possible
        st = self._streams.get(stream)
        if st is not None and st["rows"].shape[0] == m:
            rows, gens = st["rows"].clone(), st["gens"].clone()
            same = torch.isclose(self.obs["proprio"][rows], fo["proprio"].to(dev)).all(-1)
            ok = (~st["reset"]) & same & (self.obs_gen[rows] == gens)
            need = torch.nonzero(~ok).squeeze(-1)
            if need.numel():
                r2, g2 = self._write_obs(fo, need.to(fo["proprio"].device))
                rows[need], gens[need] = r2, g2
        else:
            rows, gens = self._write_obs(fo, all_rows)
        nrows, ngens = self._write_obs(fn, all_rows)
        rs = reset.to(dev).bool().reshape(n, 1).expand(n, a).reshape(m)
        self._streams[stream] = {"rows": nrows, "gens": ngens, "reset": rs}
        # transitions
        t = (self.ptr + torch.arange(m, device=dev)) % self.capacity
        self.obs_idx[t], self.obs_g[t] = rows, gens
        self.next_idx[t], self.next_g[t] = nrows, ngens
        self.action[t] = action.reshape(m, -1).to(dev).float()
        self.done[t] = done.to(dev).bool().reshape(n, 1).expand(n, a).reshape(m)
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(n, a, z.shape[-1])
        self.z[t] = z.reshape(m, -1).to(dev).half()
        w = world_offset + torch.arange(n, device=dev)
        self.world[t] = w.unsqueeze(1).expand(n, a).reshape(m)
        ex = torch.zeros(n, dtype=torch.bool, device=dev) if expert is None else expert.to(dev).bool().reshape(n)
        self.expert[t] = ex.reshape(n, 1).expand(n, a).reshape(m)
        self.ptr = (self.ptr + m) % self.capacity
        self.size = min(self.size + m, self.capacity)

    # ------------------------------------------------------------ reading
    def _valid(self, t: torch.Tensor) -> torch.Tensor:
        return (self.obs_gen[self.obs_idx[t]] == self.obs_g[t]) & (self.obs_gen[self.next_idx[t]] == self.next_g[t])

    def _gather(self, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[rows] for k, v in self.obs.items()}

    @torch.no_grad()
    def sample(self, b: int, device=None) -> Dict[str, object]:
        """Random batch: obs, next_obs (dicts with CRITIC_KEYS + g, LiDAR uint8),
        action [b, 5], g, next_g [b, G], done [b] float, z [b, d], world [b], expert [b] bool."""
        assert self.size > 0, "empty buffer"
        dev = self.device
        t = torch.randint(0, self.size, (b,), device=dev)
        for _ in range(16):
            bad = ~self._valid(t)
            if not bad.any():
                break
            t[bad] = torch.randint(0, self.size, (int(bad.sum()),), device=dev)
        obs, nobs = self._gather(self.obs_idx[t]), self._gather(self.next_idx[t])
        out = {"obs": obs, "next_obs": nobs, "action": self.action[t], "g": obs["g"],
               "next_g": nobs["g"], "done": self.done[t].float(), "z": self.z[t].float(), "world": self.world[t],
               "expert": self.expert[t]}
        if device is not None:
            out = _to(out, torch.device(device))
        return out

    @torch.no_grad()
    def sample_g(self, n: int, device=None) -> torch.Tensor:
        """n random stored states' g, [n, G_DIM] (for z inference and B(g) z samples)."""
        assert self.obs_size > 0, "empty buffer"
        rows = torch.randint(0, self.obs_size, (n,), device=self.device)
        g = self.obs["g"][rows]
        return g.to(device) if device is not None else g


def _to(x, dev):
    if isinstance(x, dict):
        return {k: _to(v, dev) for k, v in x.items()}
    return x.to(dev, non_blocking=True)


class DemoStore:
    """State-only expert segments of g, [S, L, G_DIM], ring buffer."""

    def __init__(self, capacity: int, seg_len: int = 8, device="cpu"):
        self.capacity, self.seg_len = int(capacity), int(seg_len)
        self.device = torch.device(device)
        self.segs = torch.zeros(self.capacity, self.seg_len, spec.G_DIM, device=self.device)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add_segments(self, segs: torch.Tensor) -> None:
        segs = segs.to(self.device).float()
        assert segs.shape[1:] == (self.seg_len, spec.G_DIM), segs.shape
        k = segs.shape[0]
        if k > self.capacity:
            segs, k = segs[-self.capacity:], self.capacity
        if k == 0:
            return
        idx = (self.ptr + torch.arange(k, device=self.device)) % self.capacity
        self.segs[idx] = segs
        self.ptr = (self.ptr + k) % self.capacity
        self.size = min(self.size + k, self.capacity)

    @torch.no_grad()
    def add_rollout(self, g_seq: torch.Tensor, cut: Optional[torch.Tensor] = None,
                    bad: Optional[torch.Tensor] = None) -> int:
        """g_seq [T, M, G]; cut [T, M] bool, True where the episode ended after step t;
        bad [T, M] bool, states that must not appear in demos (segments containing one
        are dropped). Cuts into non-overlapping length-L segments that do not cross an
        episode end."""
        T, M = g_seq.shape[:2]
        L = self.seg_len
        if T < L:
            return 0
        nw = T // L
        segs = g_seq[: nw * L].reshape(nw, L, M, -1).permute(0, 2, 1, 3)      # [nw, M, L, G]
        if cut is not None:
            c = cut[: nw * L].reshape(nw, L, M).permute(0, 2, 1)               # [nw, M, L]
            ok = ~c[..., :-1].any(-1)
        else:
            ok = torch.ones(nw, M, dtype=torch.bool, device=g_seq.device)
        if bad is not None:
            b = bad[: nw * L].reshape(nw, L, M).permute(0, 2, 1)
            ok = ok & ~b.any(-1)
        segs = segs[ok]
        self.add_segments(segs)
        return int(segs.shape[0])

    @torch.no_grad()
    def sample_segments(self, n: int, device=None) -> torch.Tensor:
        assert self.size > 0, "empty demo store"
        idx = torch.randint(0, self.size, (n,), device=self.device)
        s = self.segs[idx]
        return s.to(device) if device is not None else s

    @torch.no_grad()
    def sample_g(self, n: int, device=None) -> torch.Tensor:
        s = self.sample_segments(n, device)
        j = torch.randint(0, self.seg_len, (n,), device=s.device)
        return s[torch.arange(n, device=s.device), j]
