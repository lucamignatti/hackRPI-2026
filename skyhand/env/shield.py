"""Safety shield: project velocity commands that would enter forbidden regions.

Applied after the policy and before the controller. Guarantees hold regardless of
what the network outputs: commanded velocity never points into a no-fly cylinder
or an obstacle margin once the drone is within the look-ahead horizon.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def shield_velocity(pos: torch.Tensor, v: torch.Tensor,
                    cyl_center: Optional[torch.Tensor] = None, cyl_radius: Optional[torch.Tensor] = None,
                    cyl_active: Optional[torch.Tensor] = None,
                    box_center: Optional[torch.Tensor] = None, box_half: Optional[torch.Tensor] = None,
                    box_active: Optional[torch.Tensor] = None,
                    margin: float = 0.6, horizon: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """pos, v [N, A, 3]. Cylinders: center [N, Z, 3] (xy used), radius [N, Z], active [N, Z].
    Boxes: center/half [N, B, 3], active [N, B]. Returns (v_safe [N, A, 3], hit [N, A] bool)."""
    v = v.clone()
    hit = torch.zeros(pos.shape[:2], dtype=torch.bool, device=pos.device)
    if cyl_center is not None and cyl_center.shape[1] > 0:
        pred = pos + v * horizon
        dxy = pred[..., None, :2] - cyl_center[:, None, :, :2]                # [N, A, Z, 2]
        dist = dxy.norm(dim=-1)
        r = (cyl_radius + margin)[:, None, :]
        act = cyl_active[:, None, :] if cyl_active is not None else torch.ones_like(dist, dtype=torch.bool)
        inside = (dist < r) & act
        cur = pos[..., None, :2] - cyl_center[:, None, :, :2]
        n = cur / cur.norm(dim=-1, keepdim=True).clamp_min(1e-6)               # outward normal
        vr = (v[..., None, :2] * n).sum(-1)                                     # outward speed
        fix = inside & (vr < 0)
        corr = torch.where(fix.unsqueeze(-1), -vr.unsqueeze(-1) * n, torch.zeros_like(n)).sum(2)
        v[..., :2] = v[..., :2] + corr
        hit |= fix.any(-1)
    if box_center is not None and box_center.shape[1] > 0:
        pred = pos + v * horizon
        half = box_half[:, None] + margin
        rel_pred = pred[..., None, :] - box_center[:, None]
        act = box_active[:, None, :] if box_active is not None else torch.ones(rel_pred.shape[:-1], dtype=torch.bool, device=pos.device)
        inside = (rel_pred.abs() < half).all(-1) & act                        # [N, A, B]
        rel = pos[..., None, :] - box_center[:, None]
        closest = torch.maximum(torch.minimum(rel, half), -half)
        n = rel - closest
        nn_ = n.norm(dim=-1, keepdim=True)
        # inside the expanded box already: push out along the axis of least penetration
        pen = half - rel.abs()
        axis = torch.nn.functional.one_hot(pen.argmin(-1), 3).to(rel.dtype) * torch.sign(rel)
        n = torch.where(nn_ > 1e-6, n / nn_.clamp_min(1e-6), axis)
        vn = (v[..., None, :] * n).sum(-1)
        fix = inside & (vn < 0)
        corr = torch.where(fix.unsqueeze(-1), -vn.unsqueeze(-1) * n, torch.zeros_like(n)).sum(2)
        v = v + corr
        hit |= fix.any(-1)
    return v, hit
