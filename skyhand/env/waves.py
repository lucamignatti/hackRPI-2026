"""Ocean surface as a sum of linear (Airy) deep-water waves, batched per world.

The same parameters are sent to the web viewer, so the rendered surface matches
the physics exactly.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .. import spec

G = 9.81


class WaveField:
    def __init__(self, n: int, device=None, n_waves: int = spec.N_WAVES):
        self.n, self.w, self.device = n, n_waves, device
        z = lambda: torch.zeros(n, n_waves, device=device)
        self.amp, self.k, self.dir, self.omega, self.phase = z(), z(), z(), z(), z()
        self.scale = torch.ones(n, device=device)      # live multiplier (demo slider)
        self.randomize(torch.arange(n, device=device))

    def randomize(self, env_ids: torch.Tensor, gen: Optional[torch.Generator] = None,
                  max_total_amp: float = 0.35) -> None:
        m = len(env_ids)
        if m == 0:
            return
        r = lambda *s: torch.rand(*s, generator=gen, device="cpu").to(self.device)
        wavelength = 4.0 + 16.0 * r(m, self.w)
        k = 2.0 * math.pi / wavelength
        amp = r(m, self.w)
        amp = amp / amp.sum(-1, keepdim=True) * (max_total_amp * (0.3 + 0.7 * r(m, 1)))
        main = r(m, 1) * 2.0 * math.pi
        self.k[env_ids] = k
        self.amp[env_ids] = amp
        self.dir[env_ids] = main + (r(m, self.w) - 0.5) * 1.2
        self.omega[env_ids] = torch.sqrt(G * k)
        self.phase[env_ids] = r(m, self.w) * 2.0 * math.pi
        self.scale[env_ids] = 1.0

    def _theta(self, xy: torch.Tensor, t: torch.Tensor):
        """xy [N, M, 2], t [N] -> theta [N, M, W], direction unit vectors [N, 1, W, 2]."""
        d = torch.stack([torch.cos(self.dir), torch.sin(self.dir)], dim=-1).unsqueeze(1)
        proj = (xy.unsqueeze(2) * d).sum(-1)                     # [N, M, W]
        theta = self.k.unsqueeze(1) * proj - (self.omega * t.unsqueeze(-1)).unsqueeze(1) \
            + self.phase.unsqueeze(1)
        return theta, d

    def height(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta, _ = self._theta(xy, t)
        a = (self.amp * self.scale.unsqueeze(-1)).unsqueeze(1)
        return (a * torch.cos(theta)).sum(-1)

    def water_velocity(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Orbital velocity [N, M, 3] at points (decays as exp(k z) below the surface)."""
        theta, d = self._theta(xyz[..., :2], t)
        a = (self.amp * self.scale.unsqueeze(-1)).unsqueeze(1)
        decay = torch.exp((self.k.unsqueeze(1) * xyz[..., 2:3]).clamp(max=0.0))
        mag = a * self.omega.unsqueeze(1) * decay                  # [N, M, W]
        horiz = (mag * torch.cos(theta)).unsqueeze(-1) * d         # [N, M, W, 2]
        vert = mag * torch.sin(theta)
        return torch.cat([horiz.sum(2), vert.sum(-1, keepdim=True)], dim=-1)

    def ray_hit(self, origin: torch.Tensor, direction: torch.Tensor, t: torch.Tensor,
                iters: int = 3) -> torch.Tensor:
        """Distance along rays to the surface. origin/direction [N, R, 3] -> [N, R] (inf if none)."""
        eta = self.height(origin[..., :2], t)
        dz = direction[..., 2]
        s = torch.full_like(dz, float("inf"))
        above = origin[..., 2] > eta
        going = torch.where(above, dz < -1e-6, dz > 1e-6)
        s0 = torch.where(going, (eta - origin[..., 2]) / torch.where(going, dz, torch.ones_like(dz)),
                         torch.zeros_like(dz)).clamp_min(0.0)
        for _ in range(iters):   # fixed-point refinement on the moving surface height
            p = origin + s0.unsqueeze(-1) * direction
            eta = self.height(p[..., :2], t)
            s0 = torch.where(going, (eta - origin[..., 2]) / torch.where(going, dz, torch.ones_like(dz)),
                             s0).clamp_min(0.0)
        return torch.where(going, s0, s)

    def params(self, i: int) -> dict:
        a = (self.amp[i] * self.scale[i]).tolist()
        return {"amp": a, "k": self.k[i].tolist(), "dir": self.dir[i].tolist(),
                "omega": self.omega[i].tolist(), "phase": self.phase[i].tolist()}

    def state(self) -> dict:
        return {"amp": self.amp, "k": self.k, "dir": self.dir, "omega": self.omega,
                "phase": self.phase, "scale": self.scale}

    def load(self, st: dict, env_ids: torch.Tensor, src: Optional[torch.Tensor] = None) -> None:
        src = env_ids if src is None else src
        for name, val in st.items():
            getattr(self, name)[env_ids] = val[src]
