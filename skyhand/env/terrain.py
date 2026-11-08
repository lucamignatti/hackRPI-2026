"""Per-world heightfield terrain with lakes and a tide (pure torch, batched).

Terrain is a TERRAIN_RES x TERRAIN_RES grid over the arena, sampled bilinearly.
Water is a flat level per world (lakes wherever the terrain is lower) that rises
and falls with a slow tide. Flat worlds (city, ocean seabed) skip ray marching.

Styles
------
flat:    constant height (city streets, ocean seabed).
rolling: gentle hills and one or two lake basins with beaches (trash litter).
rugged:  mountains, hills, valleys and lakes.
remote:  large-scale rural landscape for long-range delivery: hills, mesas, a
         meandering river valley and lakes (sizes scale with the arena).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .. import spec

NO_WATER = -1.0e4


class Terrain:
    def __init__(self, n: int, device=None, res: int = spec.TERRAIN_RES, half: float = spec.ARENA_HALF):
        self.n, self.res, self.half, self.device = n, res, half, device
        self.h = torch.zeros(n, res, res, device=device)
        self.flat = torch.ones(n, dtype=torch.bool, device=device)
        self.flat_z = torch.zeros(n, device=device)
        self.water_base = torch.full((n,), NO_WATER, device=device)
        self.tide_amp = torch.zeros(n, device=device)
        self.tide_omega = torch.full((n,), 2 * math.pi / 60.0, device=device)
        self.tide_phase = torch.zeros(n, device=device)
        self.tide_offset = torch.zeros(n, device=device)     # demo slider
        self.version = torch.zeros(n, dtype=torch.long, device=device)
        self._ray_hmax = torch.zeros(n, device=device)
        self._ray_lip = torch.full((n,), 0.05, device=device)
        lin = torch.linspace(-half, half, res, device=device)
        self.gy, self.gx = torch.meshgrid(lin, lin, indexing="ij")   # [res, res]: row = y, col = x

    def _refresh_ray_bounds(self, ids: torch.Tensor) -> None:
        """Cache LiDAR ray-march bounds whenever terrain changes, not once per scan."""
        h = self.h[ids]
        self._ray_hmax[ids] = h.flatten(1).amax(1)
        gy = (h[:, 1:, :] - h[:, :-1, :]).abs().flatten(1).amax(1)
        gx = (h[:, :, 1:] - h[:, :, :-1]).abs().flatten(1).amax(1)
        self._ray_lip[ids] = 1.5 * torch.maximum(gx, gy) / (2 * self.half / (self.res - 1)) + 0.05

    # ------------------------------------------------------------ generation
    def set_flat(self, ids: torch.Tensor, z: float = 0.0, water: Optional[float] = None) -> None:
        self.h[ids] = z
        self.flat[ids] = True
        self.flat_z[ids] = z
        self.water_base[ids] = NO_WATER if water is None else water
        self.tide_amp[ids] = 0.0
        self.tide_offset[ids] = 0.0
        self.version[ids] += 1
        self._refresh_ray_bounds(ids)

    def set_heights(self, ids: torch.Tensor, h: torch.Tensor, water: float, tide_amp: float,
                    gen: Optional[torch.Generator] = None) -> None:
        """Load externally generated heightfields h [m, res, res] (rows = y, cols = x)."""
        m = len(ids)
        self.h[ids] = h.to(self.h.device)
        self.flat[ids] = False
        self.water_base[ids] = water
        self.tide_amp[ids] = tide_amp
        self.tide_phase[ids] = torch.rand(m, generator=gen).to(self.h.device) * 2 * math.pi
        self.tide_offset[ids] = 0.0
        self.version[ids] += 1
        self._refresh_ray_bounds(ids)

    def generate(self, ids: torch.Tensor, style: str, gen: Optional[torch.Generator] = None) -> None:
        m = len(ids)
        if m == 0:
            return
        r = lambda *s: torch.rand(*s, generator=gen).to(self.device)
        H = self.half
        h = torch.zeros(m, self.res, self.res, device=self.device)

        def bumps(k, amp_lo, amp_hi, sig_lo, sig_hi, sign=1.0, edge=4.0):
            nonlocal h
            c = (r(m, k, 2) - 0.5) * 2 * (H - edge)
            a = sign * (amp_lo + (amp_hi - amp_lo) * r(m, k))
            s = sig_lo + (sig_hi - sig_lo) * r(m, k)
            dx = self.gx[None, None] - c[..., 0, None, None]
            dy = self.gy[None, None] - c[..., 1, None, None]
            h = h + (a[..., None, None] * torch.exp(-(dx * dx + dy * dy) / (2 * s[..., None, None] ** 2))).sum(1)

        if style == "remote":
            bumps(10, 6.0, 25.0, 30.0, 70.0, edge=20.0)          # hills
            bumps(2, 35.0, 65.0, 35.0, 70.0, edge=40.0)          # mesas and mountains
            bumps(3, 10.0, 22.0, 40.0, 80.0, sign=-1.0, edge=30.0)   # valleys and lake basins
            # a meandering river channel across the map
            th = r(m) * 2 * math.pi
            c, s_ = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
            u = c * self.gx + s_ * self.gy
            w = -s_ * self.gx + c * self.gy
            k = 2 * math.pi / (150.0 + 150.0 * r(m))
            centre = 40.0 * r(m)[:, None, None] * torch.sin(k[:, None, None] * u + r(m)[:, None, None] * 6.3) \
                + (r(m)[:, None, None] - 0.5) * H
            h = h - 9.0 * torch.exp(-((w - centre) ** 2) / (2 * 9.0 ** 2))
            water = -4.0 - 2.0 * r(m)
            amp = 0.05 + 0.05 * r(m)
            noise = 2.0
            lo, hi = -20.0, 90.0
        elif style == "rolling":
            bumps(7, 0.4, 2.5, 3.0, 8.0)
            bumps(2, 1.5, 3.0, 4.0, 8.0, sign=-1.0)          # lake basins
            water = -0.6 - 0.4 * r(m)
            amp = 0.15 + 0.15 * r(m)
            noise = 0.25
            lo, hi = -6.0, 16.0
        elif style == "rugged":
            bumps(2, 7.0, 13.0, 4.5, 8.0)                    # mountains
            bumps(5, 1.0, 3.5, 3.0, 7.0)                     # hills
            bumps(2, 2.5, 5.0, 4.0, 8.0, sign=-1.0)          # valleys and lakes
            water = -0.8 - 0.8 * r(m)
            amp = 0.2 + 0.2 * r(m)
            noise = 0.5
            lo, hi = -6.0, 16.0
        else:
            raise ValueError(f"unknown terrain style {style!r}")
        for _ in range(3):                                   # low-frequency roughness
            k = 2 * math.pi / ((8.0 + 12.0 * r(m)) * (H / 25.0))
            th = r(m) * 2 * math.pi
            ph = r(m) * 2 * math.pi
            proj = torch.cos(th)[:, None, None] * self.gx + torch.sin(th)[:, None, None] * self.gy
            h = h + noise * r(m)[:, None, None] * torch.sin(k[:, None, None] * proj + ph[:, None, None])
        self.h[ids] = h.clamp(lo, hi)
        self.flat[ids] = False
        self.water_base[ids] = water
        self.tide_amp[ids] = amp
        self.tide_phase[ids] = r(m) * 2 * math.pi
        self.tide_offset[ids] = 0.0
        self.version[ids] += 1
        self._refresh_ray_bounds(ids)

    # ------------------------------------------------------------ queries
    def _rows(self, ids):
        return slice(None) if ids is None else ids

    def height(self, xy: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Bilinear terrain height at xy [M?, K, 2] for worlds ids (default all) -> [.., K]."""
        rows = self._rows(ids)
        h = self.h[rows]
        n = h.shape[0]
        flat = self.flat[rows]
        u = ((xy[..., 0] + self.half) / (2 * self.half) * (self.res - 1)).clamp(0, self.res - 1 - 1e-4)
        v = ((xy[..., 1] + self.half) / (2 * self.half) * (self.res - 1)).clamp(0, self.res - 1 - 1e-4)
        j0, i0 = u.floor().long(), v.floor().long()
        fu, fv = u - j0, v - i0
        flatg = h.reshape(n, -1)
        sh = xy.shape[:-1]
        idx = lambda i, j: (i * self.res + j).reshape(n, -1)
        g = lambda i, j: torch.gather(flatg, 1, idx(i, j)).reshape(sh)
        val = (g(i0, j0) * (1 - fu) * (1 - fv) + g(i0, j0 + 1) * fu * (1 - fv)
               + g(i0 + 1, j0) * (1 - fu) * fv + g(i0 + 1, j0 + 1) * fu * fv)
        fz = self.flat_z[rows].reshape((n,) + (1,) * (len(sh) - 1))
        return torch.where(flat.reshape((n,) + (1,) * (len(sh) - 1)), fz.expand(sh), val)

    def water_level(self, t: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        rows = self._rows(ids)
        lvl = self.water_base[rows] + self.tide_offset[rows] + \
            self.tide_amp[rows] * torch.sin(self.tide_omega[rows] * t + self.tide_phase[rows])
        return torch.where(self.water_base[rows] <= NO_WATER / 2, torch.full_like(lvl, NO_WATER), lvl)

    def max_height(self, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.h[self._rows(ids)].flatten(1).amax(1)

    def _height_rays(self, hf: torch.Tensor, w: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear height for flattened rays: hf [m, res*res], w [K] world row, xy [K, 2]."""
        r = self.res
        u = ((xy[:, 0] + self.half) / (2 * self.half) * (r - 1)).clamp(0, r - 1 - 1e-4)
        v = ((xy[:, 1] + self.half) / (2 * self.half) * (r - 1)).clamp(0, r - 1 - 1e-4)
        j0, i0 = u.floor().long(), v.floor().long()
        fu, fv = u - j0, v - i0
        base = w * (r * r)
        flat = hf.reshape(-1)
        g = lambda i, j: flat[base + i * r + j]
        return (g(i0, j0) * (1 - fu) * (1 - fv) + g(i0, j0 + 1) * fu * (1 - fv)
                + g(i0 + 1, j0) * (1 - fu) * fv + g(i0 + 1, j0 + 1) * fu * fv)

    def ray_hit(self, o: torch.Tensor, d: torch.Tensor, max_range: float, ids: Optional[torch.Tensor] = None,
                min_step: float = 0.2, max_iters: int = 48, refine: int = 4) -> torch.Tensor:
        """Distance along rays o, d [n, R, 3] to the terrain (inf if none within max_range).

        Flat worlds use an exact plane intersection. Others use Lipschitz-bounded marching:
        a ray at height gap above the terrain can advance gap / (slope*|d_xy| + max(-d_z, 0))
        without passing through it; once it does cross, bisection refines the hit."""
        rows = self._rows(ids)
        inf = float("inf")
        out = torch.full(o.shape[:2], inf, device=o.device)
        flat = self.flat[rows]
        dz = d[..., 2]
        fz = self.flat_z[rows]
        tp = (fz[:, None] - o[..., 2]) / torch.where(dz.abs() < 1e-9, torch.full_like(dz, -1e-9), dz)
        out = torch.where(flat[:, None] & (dz < -1e-9) & (tp > 0) & (tp < max_range), tp, out)
        if bool(flat.all()):
            return out
        sel = torch.nonzero(~flat).squeeze(-1)
        sub_ids = sel if ids is None else ids[sel]
        hf = self.h[sub_ids].reshape(len(sel), -1)
        hmax = self._ray_hmax[sub_ids]
        lip = self._ray_lip[sub_ids]
        os, ds = o[sel], d[sel]
        m, R = os.shape[:2]
        O, D = os.reshape(-1, 3), ds.reshape(-1, 3)
        w = torch.arange(m, device=o.device).repeat_interleave(R)
        dxy = D[:, :2].norm(dim=-1)
        down = (-D[:, 2]).clamp_min(0.0)
        # start where the ray can first reach the highest peak; skip rays that never can
        above_top = O[:, 2] - hmax[w]
        t0 = torch.where(above_top > 0, above_top / down.clamp_min(1e-6), torch.zeros_like(above_top))
        live = (t0 < max_range) & ~((above_top > 0) & (down < 1e-6))
        res = torch.full((m * R,), inf, device=o.device)
        idx = torch.nonzero(live).squeeze(-1)
        t = t0[idx]
        gap_prev = O[idx, 2] + D[idx, 2] * t - self._height_rays(hf, w[idx], O[idx, :2] + D[idx, :2] * t[:, None])
        t_prev = t.clone()
        for _ in range(max_iters):
            if idx.numel() == 0:
                break
            p = O[idx] + D[idx] * t[:, None]
            gap = p[:, 2] - self._height_rays(hf, w[idx], p[:, :2])
            crossed = (gap <= 0) & (gap_prev > 0)
            if crossed.any():
                ci = idx[crossed]
                lo, hi = t_prev[crossed], t[crossed]
                for _ in range(refine):
                    mid = 0.5 * (lo + hi)
                    pm = O[ci] + D[ci] * mid[:, None]
                    am = (pm[:, 2] - self._height_rays(hf, w[ci], pm[:, :2])) > 0
                    lo, hi = torch.where(am, mid, lo), torch.where(am, hi, mid)
                res[ci] = hi
            step = (gap / (lip[w[idx]] * dxy[idx] + down[idx] + 1e-3)).clamp(min_step, 4.0)
            keep = ~crossed & (gap > 0) & (t + step < max_range)
            idx, t_prev, gap_prev = idx[keep], t[keep], gap[keep]
            t = t_prev + step[keep]
        out[sel] = res.reshape(m, R)
        return out

    # ------------------------------------------------------------ state
    def state(self) -> dict:
        return {"h": self.h, "flat": self.flat, "flat_z": self.flat_z, "water_base": self.water_base,
                "tide_amp": self.tide_amp, "tide_omega": self.tide_omega, "tide_phase": self.tide_phase,
                "tide_offset": self.tide_offset, "version": self.version}

    def load(self, st: dict, env_ids: torch.Tensor, src: Optional[torch.Tensor] = None) -> None:
        src = env_ids if src is None else src
        for k, v in st.items():
            getattr(self, k)[env_ids] = v[src]
        self.version[env_ids] += 1
        self._refresh_ray_bounds(env_ids)


class LakeWater:
    """Flat water at the tide level wherever the terrain is lower; the LiDAR's water model."""

    def __init__(self, terrain: Terrain, ids: torch.Tensor):
        self.terrain, self.ids = terrain, ids

    def height(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lvl = self.terrain.water_level(t, self.ids)
        return lvl.reshape((-1,) + (1,) * (xy.dim() - 2)).expand(xy.shape[:-1])

    def ray_hit(self, o: torch.Tensor, d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lvl = self.terrain.water_level(t, self.ids)[:, None]
        dz = d[..., 2]
        above = o[..., 2] > lvl
        s = (lvl - o[..., 2]) / torch.where(dz.abs() < 1e-9, torch.full_like(dz, 1e-9), dz)
        ok = torch.where(above, dz < -1e-9, dz > 1e-9) & (s > 0)
        p = o + d * s.clamp_min(0).unsqueeze(-1)
        wet = self.terrain.height(p[..., :2], self.ids) < lvl                  # lake, not dry land
        return torch.where(ok & wet & (lvl > NO_WATER / 2), s, torch.full_like(s, float("inf")))
