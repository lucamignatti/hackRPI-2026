"""Fish schools as batched boids under the wave surface (pure torch)."""
from __future__ import annotations

import math
from typing import Optional

import torch

from .. import spec


class FishSchools:
    def __init__(self, n: int, device=None, n_schools: int = spec.N_FISH_SCHOOLS,
                 per_school: int = spec.FISH_PER_SCHOOL, radius: float = 0.15):
        self.n, self.s, self.k, self.device = n, n_schools, per_school, device
        self.f = n_schools * per_school
        self.radius = radius
        self.school = torch.arange(self.f, device=device) // per_school       # [F]
        self.same = self.school[:, None] == self.school[None, :]              # [F, F]
        self.pos = torch.zeros(n, self.f, 3, device=device)
        self.vel = torch.zeros(n, self.f, 3, device=device)
        self.target = torch.zeros(n, n_schools, 3, device=device)
        self.caught = torch.zeros(n, self.f, dtype=torch.bool, device=device)
        self.caught_off = torch.zeros(n, self.f, 3, device=device)
        self.flee_gain = torch.ones(n, device=device)
        self.home_c = torch.zeros(n, 2, device=device)                     # region the schools stay in
        self.home_r = torch.full((n,), spec.ARENA_HALF - 4.0, device=device)
        # counter-based randomness: a per-world key and step counter (both part of the state)
        # drive the darting and course changes, so a copied world replays the same future
        self.key = torch.zeros(n, dtype=torch.long, device=device)
        self.tick = torch.zeros(n, dtype=torch.long, device=device)

    _M = 0xFFFFFFFF

    def _rand(self, env_ids: torch.Tensor, salt: int, *shape: int) -> torch.Tensor:
        """Uniform [0, 1) of shape [len(env_ids), *shape] hashed from (key, tick, salt, index)."""
        M = self._M
        num = 1
        for d in shape:
            num *= d
        idx = torch.arange(num, device=self.device, dtype=torch.long)
        k = (self.key[env_ids] & 0x7FFFFFFF)[:, None]
        t = (self.tick[env_ids] & 0x7FFFFFFF)[:, None]
        x = ((k * 0x9E3779B1) & M) ^ ((t * 0x85EBCA77) & M) ^ ((idx[None] * 0xC2B2AE3D) & M) ^ ((salt * 0x27D4EB2F) & M)
        for mul in (0x7FEB352D, 0x446F86B3):                                  # integer avalanche
            x = x ^ (x >> 16)
            x = (x * mul) & M
        x = x ^ (x >> 15)
        return ((x.float() + 0.5) / 4294967296.0).view(len(env_ids), *shape)

    def _randn(self, env_ids: torch.Tensor, salt: int, *shape: int) -> torch.Tensor:
        u1, u2 = self._rand(env_ids, salt, *shape), self._rand(env_ids, salt + 7919, *shape)
        return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2 * math.pi * u2)

    def _in_home(self, env_ids: torch.Tensor, k: int, r) -> torch.Tensor:
        """k random xy points [m, k, 2] inside each world's home disc."""
        m = len(env_ids)
        ang = r(m, k) * 2 * math.pi
        rad = torch.sqrt(r(m, k)) * self.home_r[env_ids, None] * 0.85
        return self.home_c[env_ids, None] + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * rad.unsqueeze(-1)

    def reset(self, env_ids: torch.Tensor, gen: Optional[torch.Generator] = None,
              home_c: Optional[torch.Tensor] = None, home_r: Optional[torch.Tensor] = None) -> None:
        m = len(env_ids)
        if m == 0:
            return
        r = lambda *s: torch.rand(*s, generator=gen).to(self.device)
        if home_c is not None:
            self.home_c[env_ids] = home_c
            self.home_r[env_ids] = home_r
        centers = torch.cat([self._in_home(env_ids, self.s, r), -(0.5 + 0.8 * r(m, self.s, 1))], -1)
        jitter = (r(m, self.f, 3) - 0.5) * torch.tensor([2.0, 2.0, 0.5], device=self.device)
        self.pos[env_ids] = centers[:, self.school] + jitter
        ang = r(m, self.s) * 2 * math.pi
        v = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * 0.6
        self.vel[env_ids] = v[:, self.school]
        self.caught[env_ids] = False
        self.key[env_ids] = torch.randint(0, 2 ** 31 - 1, (m,), generator=gen).to(self.device)
        self.tick[env_ids] = 0
        self._new_targets(env_ids)

    def _new_targets(self, env_ids, salt: int = 1):
        m = len(env_ids)
        cnt = [0]

        def r(*s):
            cnt[0] += 1
            return self._rand(env_ids, salt * 16 + cnt[0], *s[1:])
        self.target[env_ids] = torch.cat([self._in_home(env_ids, self.s, r), -(0.5 + 0.8 * r(m, self.s, 1))], -1)

    def school_centers(self) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(self.school, self.s).float()          # [F, S]
        return torch.einsum("nfd,fs->nsd", self.pos, oh) / oh.sum(0)[None, :, None]

    def step(self, dt: float, drones: torch.Tensor, surface: torch.Tensor,
             net_nodes: Optional[torch.Tensor] = None, gen: Optional[torch.Generator] = None,
             jitter: float = 0.9, turn_prob: float = 0.01) -> None:
        """drones [N, A, 3]; surface [N, F] wave height above each fish.
        jitter: per-fish random acceleration (m/s^2); turn_prob: chance per step that a school
        picks a new destination, so schools wander unpredictably."""
        p, v = self.pos, self.vel
        centers = self.school_centers()
        oh = torch.nn.functional.one_hot(self.school, self.s).float()
        mean_v = torch.einsum("nfd,fs->nsd", v, oh) / oh.sum(0)[None, :, None]
        acc = 0.6 * (centers[:, self.school] - p) + 0.5 * (mean_v[:, self.school] - v)
        acc += 0.1 * (self.target[:, self.school] - p) / (self.target[:, self.school] - p).norm(
            dim=-1, keepdim=True).clamp_min(1.0)
        d = p[:, :, None] - p[:, None]                                         # [N, F, F, 3]
        dist = d.norm(dim=-1).clamp_min(1e-3)
        close = (dist < 0.6) & self.same[None] & (dist > 1e-3)
        acc += (torch.where(close.unsqueeze(-1), d / dist.unsqueeze(-1) ** 2, torch.zeros_like(d))).sum(2) * 0.3
        # flee drones that come near the surface above them
        dd = p[:, :, None] - drones[:, None]                                   # [N, F, A, 3]
        ddist = dd[..., :2].norm(dim=-1).clamp_min(0.1)
        scare = (ddist < 2.5) & (drones[:, None, :, 2] < 2.0)
        flee = torch.where(scare.unsqueeze(-1), dd / ddist.unsqueeze(-1) ** 2, torch.zeros_like(dd)).sum(2)
        flee[..., 2] = 0.0
        acc += 0.8 * flee * self.flee_gain[:, None, None]
        if net_nodes is not None:   # weak avoidance of the net mesh
            dn = p[:, :, None] - net_nodes[:, None]
            nd = dn.norm(dim=-1).clamp_min(0.05)
            near = nd < 0.5
            acc += 0.2 * torch.where(near.unsqueeze(-1), dn / nd.unsqueeze(-1) ** 2, torch.zeros_like(dn)).sum(2)
        # depth band below the surface
        top = surface - 0.3                  # schools feed near the surface
        bottom = surface - 2.0
        acc[..., 2] += 4.0 * (top - p[..., 2]).clamp(max=0.0) + 4.0 * (bottom - p[..., 2]).clamp(min=0.0)
        # stay inside the home disc (open water)
        off = p[..., :2] - self.home_c[:, None]
        dist = off.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        acc[..., :2] -= 2.0 * (dist - self.home_r[:, None, None]).clamp(min=0.0) * off / dist
        all_ids = torch.arange(self.n, device=p.device)
        self.tick += 1
        noise = self._randn(all_ids, 3, self.f, 3) * torch.tensor([1.0, 1.0, 0.4], device=p.device)
        acc = acc + jitter * noise                                             # individual darting
        v = v + dt * acc
        speed = v.norm(dim=-1, keepdim=True)
        v = v / speed.clamp_min(1e-6) * speed.clamp(0.1, 1.2)
        turn = self._rand(all_ids, 5, self.s) < turn_prob
        if turn.any():                                                         # sudden change of course
            nt = self.target.clone()
            self._new_targets(all_ids, salt=2)
            self.target = torch.where(turn.unsqueeze(-1), self.target, nt)
        free = ~self.caught.unsqueeze(-1)
        self.vel = torch.where(free, v, self.vel)
        self.pos = torch.where(free, p + dt * v, self.pos)
        reached = (centers - self.target).norm(dim=-1).max(-1).values < 1.5
        if reached.any():
            self._new_targets(torch.nonzero(reached).squeeze(-1), salt=3)

    def update_capture(self, inside: torch.Tensor, net_pos: torch.Tensor, net_vel: torch.Tensor,
                       net_held: torch.Tensor) -> None:
        """Fish inside the net bag are caught and ride with the net's lowest node until
        the net is let go. inside [N, F]; net_pos/vel [N, P, 3]; net_held [N] bool."""
        new = inside & ~self.caught & net_held.unsqueeze(-1)
        if new.any():
            self.caught_off = torch.where(new.unsqueeze(-1), (torch.rand_like(self.pos) - 0.5) * 0.6, self.caught_off)
        self.caught = (self.caught | new) & net_held.unsqueeze(-1)
        low = net_pos[..., 2].argmin(-1)
        ar = torch.arange(net_pos.shape[0], device=net_pos.device)
        anchor = net_pos[ar, low] + torch.tensor([0.0, 0.0, 0.2], device=net_pos.device)
        c = self.caught.unsqueeze(-1)
        self.pos = torch.where(c, anchor[:, None] + self.caught_off * torch.tensor([1.0, 1.0, 0.3], device=net_pos.device), self.pos)
        self.vel = torch.where(c, net_vel[ar, low][:, None].expand_as(self.vel), self.vel)

    def state(self) -> dict:
        return {"pos": self.pos, "vel": self.vel, "target": self.target, "caught": self.caught, "caught_off": self.caught_off,
                "home_c": self.home_c, "home_r": self.home_r, "key": self.key, "tick": self.tick}

    def load(self, st: dict, env_ids: torch.Tensor, src: Optional[torch.Tensor] = None) -> None:
        src = env_ids if src is None else src
        for k in ("pos", "vel", "target", "caught", "caught_off", "home_c", "home_r", "key", "tick"):
            getattr(self, k)[env_ids] = st[k][src]
