"""Dual-wavelength LiDAR model (pure torch), shared by both backends.

NIR channel: first hit including the water surface.
Green channel: passes the water surface; underwater hits are attenuated by
exp(-2 * k * depth) with per-world turbidity k and vanish below GREEN_MAX_DEPTH.

Scene geometry is passed as boxes, spheres, and a ground plane. The Genesis
backend can pass its own raycaster distances as `solid_range` instead; water,
fish, and net hits are still computed here.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .. import spec
from .math3d import quat_rotate

NO_RETURN = float("inf")


def spherical_dirs(device=None) -> torch.Tensor:
    """Body-frame unit ray directions [V, H, 3] for the spinning pattern."""
    el = torch.linspace(math.radians(spec.LIDAR_V_FOV[0]), math.radians(spec.LIDAR_V_FOV[1]),
                        spec.LIDAR_V, device=device)
    az = torch.arange(spec.LIDAR_H, device=device) * (2 * math.pi / spec.LIDAR_H)
    el, az = torch.meshgrid(el, az, indexing="ij")
    return torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)


def down_dirs(device=None) -> torch.Tensor:
    """Body-frame unit ray directions [DV, DH, 3] for the downward grid."""
    u = torch.linspace(-spec.DOWN_HALF_WIDTH, spec.DOWN_HALF_WIDTH, spec.DOWN_V, device=device)
    w = torch.linspace(-spec.DOWN_HALF_WIDTH, spec.DOWN_HALF_WIDTH, spec.DOWN_H, device=device)
    u, w = torch.meshgrid(u, w, indexing="ij")
    d = torch.stack([u, w, -torch.ones_like(u)], -1)
    return d / d.norm(dim=-1, keepdim=True)


def ray_aabb(o: torch.Tensor, d: torch.Tensor, c: torch.Tensor, h: torch.Tensor,
             active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """o, d [N, R, 3]; c, h [N, B, 3] -> distance to the nearest box [N, R]."""
    if c.shape[1] == 0:
        return torch.full(o.shape[:2], NO_RETURN, device=o.device)
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    lo = (c[:, None] - h[:, None] - o[:, :, None]) * inv[:, :, None]       # [N, R, B, 3]
    hi = (c[:, None] + h[:, None] - o[:, :, None]) * inv[:, :, None]
    tmin = torch.minimum(lo, hi).max(-1).values
    tmax = torch.maximum(lo, hi).min(-1).values
    hit = (tmax >= tmin.clamp_min(0.0)) & (tmax > 0)
    if active is not None:
        hit &= active[:, None]
    t = torch.where(hit, tmin.clamp_min(0.0), torch.full_like(tmin, NO_RETURN))
    return t.min(-1).values


def ray_spheres(o: torch.Tensor, d: torch.Tensor, c: torch.Tensor, r: torch.Tensor,
                active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """o, d [N, R, 3]; c [N, S, 3]; r [N, S] -> distance to the nearest sphere [N, R]."""
    if c.shape[1] == 0:
        return torch.full(o.shape[:2], NO_RETURN, device=o.device)
    oc = o[:, :, None] - c[:, None]                                        # [N, R, S, 3]
    b = (oc * d[:, :, None]).sum(-1)
    cc = (oc * oc).sum(-1) - r[:, None] ** 2
    disc = b * b - cc
    t = -b - torch.sqrt(disc.clamp_min(0.0))
    hit = (disc >= 0) & (t > 0)
    if active is not None:
        hit &= active[:, None]
    return torch.where(hit, t, torch.full_like(t, NO_RETURN)).min(-1).values


def ray_plane_z(o: torch.Tensor, d: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Distance to the horizontal plane z = z0 [N] from above. [N, R]."""
    dz = d[..., 2]
    t = (z0[:, None] - o[..., 2]) / torch.where(dz.abs() < 1e-9, torch.full_like(dz, -1e-9), dz)
    return torch.where((dz < -1e-9) & (t > 0), t, torch.full_like(t, NO_RETURN))


def encode_range(r: torch.Tensor) -> torch.Tensor:
    q = torch.round(255.0 * r.clamp(0.0, spec.LIDAR_MAX_RANGE) / spec.LIDAR_MAX_RANGE)
    q = torch.where(torch.isfinite(r) & (r < spec.LIDAR_MAX_RANGE), q, torch.full_like(q, 255.0))
    return q.to(torch.uint8)


def decode_range(q: torch.Tensor) -> torch.Tensor:
    """uint8 -> metres (no return = LIDAR_MAX_RANGE)."""
    return q.float() * (spec.LIDAR_MAX_RANGE / 255.0)


def encode_intensity(i: torch.Tensor) -> torch.Tensor:
    return torch.round(255.0 * i.clamp(0.0, 1.0)).to(torch.uint8)


class LidarModel:
    def __init__(self, device=None, noise_std: float = 0.02, dropout: float = 0.02,
                 chunk_rays: int = 2_000_000):
        self.device = device
        self.dirs = spherical_dirs(device).reshape(-1, 3)        # [V*H, 3]
        self.ddirs = down_dirs(device).reshape(-1, 3)            # [DV*DH, 3]
        self.noise_std, self.dropout, self.chunk_rays = noise_std, dropout, chunk_rays

    def scan(self, origin: torch.Tensor, quat: torch.Tensor, geom: Dict[str, torch.Tensor],
             waves=None, t: Optional[torch.Tensor] = None, turbidity: Optional[torch.Tensor] = None,
             solid_range: Optional[torch.Tensor] = None, solid_range_down: Optional[torch.Tensor] = None,
             noise: bool = True):
        """origin [N, A, 3], quat [N, A, 4].

        geom keys (all optional): box_c, box_h [N, B, 3], box_active [N, B];
        sph_c [N, S, 3], sph_r [N, S], sph_active [N, S]; ground_z [N];
        terrain: callable (o [N, R, 3], d [N, R, 3]) -> distance [N, R] (heightfield).
        waves: any water model with height(xy, t) and ray_hit(o, d, t) (ocean waves or lakes).
        solid_range: [N, A, V*H] distances from an external raycaster (Genesis),
        used in place of boxes/ground; spheres here are still tested.

        Returns (lidar uint8 [N, A, 3, V, H], lidar_down uint8 [N, A, 3, DV, DH],
        points dict with world-frame hit points for agent 0 of every world).
        """
        n, a = origin.shape[:2]
        out, pts = [], {}
        for dirs, shape, ext in ((self.dirs, (spec.LIDAR_V, spec.LIDAR_H), solid_range),
                                 (self.ddirs, (spec.DOWN_V, spec.DOWN_H), solid_range_down)):
            r = dirs.shape[0]
            d_world = quat_rotate(quat[:, :, None].expand(n, a, r, 4),
                                  dirs[None, None].expand(n, a, r, 3))       # [N, A, R, 3]
            o = origin[:, :, None].expand(n, a, r, 3)
            nir, green, inten = self._cast(o.reshape(n, a * r, 3), d_world.reshape(n, a * r, 3),
                                           geom, waves, t, turbidity,
                                           None if ext is None else ext.reshape(n, a * r))
            nir, green, inten = (x.reshape(n, a, r) for x in (nir, green, inten))
            if noise and self.noise_std > 0:
                nir = nir + torch.randn_like(nir) * self.noise_std
                green = green + torch.randn_like(green) * self.noise_std
                drop = torch.rand_like(nir) < self.dropout
                nir = torch.where(drop, torch.full_like(nir, NO_RETURN), nir)
                green = torch.where(drop, torch.full_like(green, NO_RETURN), green)
            img = torch.stack([encode_range(nir), encode_range(green), encode_intensity(inten)], 2)
            out.append(img.reshape(n, a, 3, *shape))
            if not pts:
                o0, d0 = origin[:, 0], d_world[:, 0]
                pts["nir"] = torch.where(torch.isfinite(nir[:, 0]).unsqueeze(-1),
                                         o0[:, None] + d0 * nir[:, 0].clamp(max=spec.LIDAR_MAX_RANGE).unsqueeze(-1),
                                         torch.full_like(d0, float("nan")))
                gre = torch.where(torch.isfinite(green[:, 0]) & (green[:, 0] < nir[:, 0] - 1e-3), green[:, 0],
                                  torch.full_like(green[:, 0], NO_RETURN))
                pts["green"] = torch.where(torch.isfinite(gre).unsqueeze(-1),
                                           o0[:, None] + d0 * gre.clamp(max=spec.LIDAR_MAX_RANGE).unsqueeze(-1),
                                           torch.full_like(d0, float("nan")))
        return out[0], out[1], pts

    def _cast(self, o, d, geom, waves, t, turbidity, ext):
        n = o.shape[0]
        solid = torch.full(o.shape[:2], NO_RETURN, device=o.device)
        if ext is not None:
            solid = torch.minimum(solid, torch.where(ext > 0, ext, torch.full_like(ext, NO_RETURN)))
        else:
            if "box_c" in geom:
                solid = torch.minimum(solid, ray_aabb(o, d, geom["box_c"], geom["box_h"], geom.get("box_active")))
            if "ground_z" in geom:
                solid = torch.minimum(solid, ray_plane_z(o, d, geom["ground_z"]))
            if "terrain" in geom:
                solid = torch.minimum(solid, geom["terrain"](o, d))
        if "sph_c" in geom:
            solid = torch.minimum(solid, ray_spheres(o, d, geom["sph_c"], geom["sph_r"], geom.get("sph_active")))
        solid = torch.where(solid > spec.LIDAR_MAX_RANGE, torch.full_like(solid, NO_RETURN), solid)
        if waves is None:
            inten = torch.where(torch.isfinite(solid), 0.9 / (1.0 + 0.03 * solid.clamp(max=1e3)),
                                torch.zeros_like(solid))
            return solid, solid, inten
        water = waves.ray_hit(o, d, t)
        nir = torch.minimum(solid, water)
        hit_p = o + d * solid.clamp(max=spec.LIDAR_MAX_RANGE).unsqueeze(-1)
        surf = waves.height(hit_p[..., :2], t)
        depth = (surf - hit_p[..., 2]).clamp_min(0.0)
        k = turbidity if turbidity is not None else torch.full((n,), 0.3, device=o.device)
        under = torch.isfinite(solid) & (depth > 0)
        inten = torch.where(under, 0.9 * torch.exp(-2.0 * k[:, None] * depth),
                            0.9 / (1.0 + 0.03 * solid.clamp(max=1e3)))
        visible = torch.isfinite(solid) & (depth < spec.GREEN_MAX_DEPTH) & (inten > 0.02)
        green = torch.where(visible, solid, torch.full_like(solid, NO_RETURN))
        inten = torch.where(visible, inten, torch.zeros_like(inten))
        return nir, green, inten
