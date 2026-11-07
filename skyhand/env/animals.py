"""Marine animals for the ocean scene: whales, dolphins, turtles (pure torch).

They wander independently and do not avoid the net, so the team has to keep the
net away from them. An animal is tangled when it touches the net while the net is
held; it then rides with the net until the net is let go (bycatch).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .. import spec

#            radius, speed (m/s), preferred depth (m below the surface)
PARAMS = {"whale": (1.2, 0.6, 2.5), "dolphin": (0.45, 2.0, 0.8), "turtle": (0.35, 0.35, 1.0)}


class MarineAnimals:
    def __init__(self, n: int, device=None):
        self.n, self.device = n, device
        types = []
        for name, cnt in zip(spec.ANIMAL_TYPES, spec.ANIMAL_COUNTS):
            types += [name] * cnt
        self.types = types
        self.k = len(types)
        self.names = []
        counts = {}
        for t in types:
            self.names.append(f"{t}_{counts.get(t, 0)}")
            counts[t] = counts.get(t, 0) + 1
        p = torch.tensor([PARAMS[t] for t in types], dtype=torch.float32, device=device)
        self.radius, self.speed, self.depth = p[:, 0], p[:, 1], p[:, 2]
        self.is_dolphin = torch.tensor([t == "dolphin" for t in types], device=device)
        self.pos = torch.zeros(n, self.k, 3, device=device)
        self.vel = torch.zeros(n, self.k, 3, device=device)
        self.target = torch.zeros(n, self.k, 2, device=device)
        self.tangled = torch.zeros(n, self.k, dtype=torch.bool, device=device)
        self.tangle_off = torch.zeros(n, self.k, 3, device=device)
        self.phase = torch.zeros(n, self.k, device=device)
        self.home_c = torch.zeros(n, 2, device=device)
        self.home_r = torch.full((n,), spec.ARENA_HALF - 3.0, device=device)

    def _in_home(self, ids: torch.Tensor, r) -> torch.Tensor:
        m = len(ids)
        ang = r(m, self.k) * 2 * math.pi
        rad = torch.sqrt(r(m, self.k)) * self.home_r[ids, None] * 0.85
        return self.home_c[ids, None] + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * rad.unsqueeze(-1)

    def reset(self, ids: torch.Tensor, gen: Optional[torch.Generator] = None,
              home_c: Optional[torch.Tensor] = None, home_r: Optional[torch.Tensor] = None) -> None:
        m = len(ids)
        if m == 0:
            return
        r = lambda *s: torch.rand(*s, generator=gen).to(self.device)
        if home_c is not None:
            self.home_c[ids] = home_c
            self.home_r[ids] = home_r
        xy = self._in_home(ids, r)
        self.pos[ids] = torch.cat([xy, -self.depth.expand(m, -1).unsqueeze(-1)], -1)
        self.target[ids] = self._in_home(ids, r)
        self.vel[ids] = 0.0
        self.tangled[ids] = False
        self.phase[ids] = r(m, self.k) * 2 * math.pi

    def step(self, dt: float, t: torch.Tensor, surface: torch.Tensor,
             gen: Optional[torch.Generator] = None) -> None:
        """surface [N, K]: water height above each animal."""
        to = self.target - self.pos[..., :2]
        dist = to.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        v_xy = to / dist * self.speed[None, :, None]
        # dolphins porpoise: dip and rise; whales surface now and then
        self.phase = self.phase + dt * torch.where(self.is_dolphin, 2.0, 0.15)[None]
        bob = torch.where(self.is_dolphin[None], 0.9 * torch.sin(self.phase), 1.5 * torch.sin(self.phase).clamp_min(0.6) - 0.6)
        z_t = surface - self.depth[None] + bob
        vz = (z_t - self.pos[..., 2]) * 1.5
        v = torch.cat([v_xy, vz.unsqueeze(-1)], -1)
        free = ~self.tangled.unsqueeze(-1)
        self.vel = torch.where(free, v, self.vel)
        self.pos = torch.where(free, self.pos + dt * self.vel, self.pos)
        off = self.pos[..., :2] - self.home_c[:, None]
        dist_h = off.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        self.pos[..., :2] = torch.where(dist_h > self.home_r[:, None, None],
                                        self.home_c[:, None] + off / dist_h * self.home_r[:, None, None], self.pos[..., :2])
        arrived = dist.squeeze(-1) < 1.5
        if arrived.any():
            r = lambda *s: torch.rand(*s, generator=gen).to(self.device)
            new = self._in_home(torch.arange(self.n, device=self.device), r)
            self.target = torch.where(arrived.unsqueeze(-1), new, self.target)

    def update_tangle(self, net_pos: torch.Tensor, net_vel: torch.Tensor, net_held: torch.Tensor) -> torch.Tensor:
        """Tangle animals touching the held net. Returns newly tangled [N, K] bool."""
        d = (self.pos[:, :, None] - net_pos[:, None]).norm(dim=-1)            # [N, K, P]
        dmin, near = d.min(-1)
        touch = (dmin < self.radius[None] + 0.25) & net_held[:, None]
        new = touch & ~self.tangled
        if new.any():
            node = torch.gather(net_pos, 1, near.unsqueeze(-1).expand(-1, -1, 3))
            self.tangle_off = torch.where(new.unsqueeze(-1), self.pos - node, self.tangle_off)
        self.tangled = (self.tangled | new) & net_held[:, None]
        if self.tangled.any():
            node = torch.gather(net_pos, 1, near.unsqueeze(-1).expand(-1, -1, 3))
            nvel = torch.gather(net_vel, 1, near.unsqueeze(-1).expand(-1, -1, 3))
            c = self.tangled.unsqueeze(-1)
            self.pos = torch.where(c, node + self.tangle_off * 0.5, self.pos)
            self.vel = torch.where(c, nvel, self.vel)
        return new

    def net_distance(self, net_pos: torch.Tensor) -> torch.Tensor:
        """Smallest gap between any animal's body and any net node [N]."""
        d = (self.pos[:, :, None] - net_pos[:, None]).norm(dim=-1) - self.radius[None, :, None]
        return d.amin((1, 2)).clamp_min(0.0)

    def state(self) -> dict:
        return {"pos": self.pos, "vel": self.vel, "target": self.target, "tangled": self.tangled,
                "tangle_off": self.tangle_off, "phase": self.phase, "home_c": self.home_c, "home_r": self.home_r}

    def load(self, st: dict, env_ids: torch.Tensor, src: Optional[torch.Tensor] = None) -> None:
        src = env_ids if src is None else src
        for k, v in st.items():
            getattr(self, k)[env_ids] = v[src]
