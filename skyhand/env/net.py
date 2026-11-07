"""Fishing net as a batched position-based-dynamics spring mesh (pure torch).

Nodes form a rows x cols grid. Edges resist stretching only (rope-like). Held
corners are pinned to grippers; their rope tension is returned so the backend can
apply it to the holding drone. The net never tears: a corner pulled so far that its
ropes stretch past slip_ratio slips out of the gripper instead (see `slip`). The perimeter floats on buoys and the interior is
slightly heavier than water, so a resting net sags into a bag under the surface.
"""
from __future__ import annotations

from typing import Optional

import torch

from .. import spec

GRAVITY = 9.81


def _grid_edges(rows: int, cols: int):
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                e.append((idx(r, c), idx(r + 1, c)))
            if r + 1 < rows and c + 1 < cols:
                e.append((idx(r, c), idx(r + 1, c + 1)))
                e.append((idx(r, c + 1), idx(r + 1, c)))
    return torch.tensor(e, dtype=torch.long)


class NetSim:
    def __init__(self, n: int, device=None, rows: int = spec.NET_ROWS, cols: int = spec.NET_COLS,
                 spacing: float = 0.9, node_mass: float = 0.02, iters: int = 12,
                 slip_ratio: float = 0.6, k_tension: float = 120.0):
        self.n, self.rows, self.cols, self.device = n, rows, cols, device
        self.p = rows * cols
        self.spacing, self.node_mass, self.iters = spacing, node_mass, iters
        self.slip_ratio, self.k_tension = slip_ratio, k_tension
        e = _grid_edges(rows, cols).to(device)
        self.ei, self.ej = e[:, 0], e[:, 1]
        r = torch.arange(self.p, device=device) // cols
        c = torch.arange(self.p, device=device) % cols
        self.rest = (torch.stack([r, c], -1)[self.ei] - torch.stack([r, c], -1)[self.ej]).float().norm(dim=-1) * spacing
        self.corners = torch.tensor([0, cols - 1, (rows - 1) * cols + cols - 1, (rows - 1) * cols],
                                    device=device)          # convex order around the net
        self.is_top = r == 0
        self.is_bottom = r == rows - 1
        self.is_edge = (r == 0) | (r == rows - 1) | (c == 0) | (c == cols - 1)
        self.pos = torch.zeros(n, self.p, 3, device=device)
        self.vel = torch.zeros(n, self.p, 3, device=device)
        self.slip = torch.zeros(n, self.p, dtype=torch.bool, device=device)   # pinned nodes pulled free this step
        self.grid_rc = torch.stack([r, c], -1).float()

    def reset(self, env_ids: torch.Tensor, center: torch.Tensor, yaw: torch.Tensor) -> None:
        """Lay the net flat on the water at center [M, 3] with heading yaw [M]."""
        rc = self.grid_rc.to(center.device)
        local = torch.stack([(rc[:, 1] - (self.cols - 1) / 2) * self.spacing,
                             (rc[:, 0] - (self.rows - 1) / 2) * self.spacing], -1)   # [P, 2]
        cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        x = center[:, None, 0] + cy * local[None, :, 0] - sy * local[None, :, 1]
        y = center[:, None, 1] + sy * local[None, :, 0] + cy * local[None, :, 1]
        z = center[:, None, 2].expand_as(x)
        self.pos[env_ids] = torch.stack([x, y, z], -1)
        self.vel[env_ids] = 0.0
        self.slip[env_ids] = False

    def step(self, dt: float, pinned: torch.Tensor, pin_pos: torch.Tensor,
             water_height: Optional[torch.Tensor] = None, water_vel: Optional[torch.Tensor] = None,
             floor: Optional[torch.Tensor] = None, boxes=None):
        """pinned [N, P] bool, pin_pos [N, P, 3]; water_height [N, P], water_vel [N, P, 3].
        floor [N, P]: lowest allowed node height (seabed or shore). boxes: (centre [N, B, 3],
        half extents [N, B, 3], active [N, B]) solid boxes the net cannot pass through (the boat).

        Returns (pin_force [N, P, 3] force the net exerts on each pinned point,
        stretch [N] max edge stretch ratio minus 1). Also sets self.slip [N, P]: pinned nodes
        whose ropes are stretched past slip_ratio, which the caller lets go of.
        """
        f = torch.zeros_like(self.pos)
        f[..., 2] -= GRAVITY * self.node_mass
        if water_height is not None:
            depth = (water_height - self.pos[..., 2]).clamp(0.0, 0.3)
            buoy = torch.where(self.is_edge, 3.0, 0.7) * GRAVITY * self.node_mass * depth / 0.3
            f[..., 2] += buoy
            wet = (self.pos[..., 2] < water_height).unsqueeze(-1)
            u = water_vel if water_vel is not None else torch.zeros_like(self.vel)
            f += torch.where(wet, 0.1 * (u - self.vel), 0.01 * -self.vel)
        vel = self.vel + dt * f / self.node_mass
        pred = self.pos + dt * vel
        pred = torch.where(pinned.unsqueeze(-1), pin_pos, pred)
        inv_m = (~pinned).float()
        for _ in range(self.iters):
            d = pred[:, self.ej] - pred[:, self.ei]                       # [N, E, 3]
            L = d.norm(dim=-1).clamp_min(1e-6)
            err = (L - self.rest).clamp_min(0.0)                          # rope: only stretch
            wi, wj = inv_m[:, self.ei], inv_m[:, self.ej]
            w = (wi + wj).clamp_min(1e-6)
            corr = (err / L / w).unsqueeze(-1) * d                        # [N, E, 3]
            delta = torch.zeros_like(pred)
            delta.index_add_(1, self.ei, corr * wi.unsqueeze(-1))
            delta.index_add_(1, self.ej, -corr * wj.unsqueeze(-1))
            pred = pred + 0.5 * delta                                     # Jacobi damping
            pred = self._collide(pred, pinned, floor, boxes)
        new_vel = (pred - self.pos) / dt
        self.vel = torch.where(pinned.unsqueeze(-1), torch.zeros_like(new_vel), new_vel * 0.995)
        self.pos = pred
        # tension on pinned points: stretched edges pull them toward their neighbours
        d = self.pos[:, self.ej] - self.pos[:, self.ei]
        L = d.norm(dim=-1).clamp_min(1e-6)
        stretch_e = L / self.rest - 1.0
        t = (self.k_tension * stretch_e.clamp_min(0.0) * self.rest).unsqueeze(-1) * d / L.unsqueeze(-1)
        pin_force = torch.zeros_like(self.pos)
        pin_force.index_add_(1, self.ei, t)
        pin_force.index_add_(1, self.ej, -t)
        # weight of the net is shared by the pins
        n_pins = pinned.float().sum(-1, keepdim=True).clamp_min(1.0)
        total_w = GRAVITY * self.node_mass * self.p
        lifted = (~self._wet_mask(water_height)).float().mean(-1, keepdim=True) if water_height is not None else torch.ones_like(n_pins)
        pin_force[..., 2] -= (pinned.float() * total_w * lifted / n_pins)
        pin_force = torch.where(pinned.unsqueeze(-1), pin_force, torch.zeros_like(pin_force))
        stretch = stretch_e.max(-1).values
        node_s = torch.zeros_like(self.pos[..., 0])
        node_s.scatter_reduce_(1, self.ei.expand(self.n, -1), stretch_e, reduce="amax")
        node_s.scatter_reduce_(1, self.ej.expand(self.n, -1), stretch_e, reduce="amax")
        self.slip = pinned & (node_s > self.slip_ratio)
        return pin_force, stretch

    def _collide(self, pred, pinned, floor, boxes, skin: float = 0.08):
        """Project free nodes out of solid boxes and above the floor."""
        free = ~pinned.unsqueeze(-1)
        if floor is not None:
            up = pred.clone()
            up[..., 2] = torch.maximum(up[..., 2], floor + skin)
            pred = torch.where(free, up, pred)
        if boxes is not None:
            c, h, act = boxes
            if c.shape[1]:
                rel = pred[:, :, None] - c[:, None]                         # [N, P, B, 3]
                half = h[:, None] + skin
                pen = half - rel.abs()                                      # > 0 on every axis: inside
                inside = (pen > 0).all(-1) & act[:, None]
                axis = pen.argmin(-1)                                       # least-penetration axis
                push = torch.nn.functional.one_hot(axis, 3).to(pred.dtype) * torch.sign(rel) * pen
                push = torch.where(inside.unsqueeze(-1), push, torch.zeros_like(push)).sum(2)
                pred = torch.where(free, pred + push, pred)
        return pred

    def segment_distance(self, pts: torch.Tensor) -> torch.Tensor:
        """Distance from each point [N, A, 3] to the nearest net rope segment [N, A]."""
        p0, p1 = self.pos[:, self.ei], self.pos[:, self.ej]                 # [N, E, 3]
        seg = p1 - p0
        L2 = (seg * seg).sum(-1).clamp_min(1e-9)
        t = (((pts[:, :, None] - p0[:, None]) * seg[:, None]).sum(-1) / L2[:, None]).clamp(0.0, 1.0)
        closest = p0[:, None] + seg[:, None] * t.unsqueeze(-1)
        return (pts[:, :, None] - closest).norm(dim=-1).amin(-1)

    def _wet_mask(self, water_height: torch.Tensor) -> torch.Tensor:
        return self.pos[..., 2] < water_height

    def centroid(self) -> torch.Tensor:
        return self.pos.mean(1)

    def corner_pos(self) -> torch.Tensor:
        return self.pos[:, self.corners]

    def contains(self, pts: torch.Tensor) -> torch.Tensor:
        """Which points [N, F, 3] are in the net's bag: inside the corner quad in xy,
        above the net sheet beneath them, and below the corners. Returns [N, F] bool."""
        c = self.corner_pos()                                            # [N, 4, 3]
        inside = torch.ones(pts.shape[:2], dtype=torch.bool, device=pts.device)
        sign = None
        for k in range(4):
            a, b = c[:, k, :2], c[:, (k + 1) % 4, :2]
            cross = (b - a)[:, None, 0] * (pts[..., 1] - a[:, None, 1]) - \
                    (b - a)[:, None, 1] * (pts[..., 0] - a[:, None, 0])
            s = cross > 0
            sign = s if sign is None else sign
            inside &= s == sign
        dxy = (pts[:, :, None, :2] - self.pos[:, None, :, :2]).norm(dim=-1)   # [N, F, P]
        nearest = dxy.argmin(-1)
        sheet_z = torch.gather(self.pos[..., 2], 1, nearest)
        top = c[..., 2].max(-1, keepdim=True).values
        return inside & (pts[..., 2] > sheet_z) & (pts[..., 2] < top) & ((top - sheet_z) > 0.5)

    def state(self) -> dict:
        return {"pos": self.pos, "vel": self.vel}

    def load(self, st: dict, env_ids: torch.Tensor, src: Optional[torch.Tensor] = None) -> None:
        src = env_ids if src is None else src
        self.pos[env_ids] = st["pos"][src]
        self.vel[env_ids] = st["vel"][src]
