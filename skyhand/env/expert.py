"""Scripted privileged expert: demonstrations for the policy regulariser.

Trash: if anchor 0 is the payload handle, pick it up and place it at anchor 1
(or the bin when anchor 1 is unset); otherwise fly to anchor 0 and hover.
Ocean: drone i grabs net corner i (or its anchor-0 corner), lifts, and once the
whole team holds, flies in formation to the boat.

Phases live in env.exp_phase and reset with the world. Actions pass through the
env's safety shield like any policy action.
"""
from __future__ import annotations

import torch

from .. import spec

KP = 1.2
CRUISE = 4.0
GENTLE = 2.0


def _to_action(v: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
    scale = v.new_tensor([spec.MAX_SPEED, spec.MAX_SPEED, spec.MAX_VZ])
    a = torch.zeros(v.shape[:-1] + (spec.ACTION_DIM,), device=v.device)
    a[..., :3] = (v / scale).clamp(-1.0, 1.0)
    a[..., 4] = grip
    return a


def _goto(pos: torch.Tensor, target: torch.Tensor, max_speed, ff: torch.Tensor = None) -> torch.Tensor:
    v = KP * (target - pos)
    if ff is not None:
        v = v + ff
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ms = max_speed if torch.is_tensor(max_speed) else torch.full_like(n, float(max_speed))
    if ms.dim() == v.dim() - 1:
        ms = ms.unsqueeze(-1)
    return v * torch.minimum(torch.ones_like(n), ms / n)


def _avoid(env, v: torch.Tensor) -> torch.Tensor:
    """Repel from nearby active boxes (moving obstacles included)."""
    rel = env.pos[:, :, None] - env.box_c[:, None]
    closest = torch.maximum(torch.minimum(rel, env.box_h[:, None]), -env.box_h[:, None])
    vec = rel - closest
    d = vec.norm(dim=-1).clamp_min(1e-3)
    near = (d < 1.8) & env.box_active[:, None]
    push = torch.where(near.unsqueeze(-1), vec / d.unsqueeze(-1) * (1.8 - d).unsqueeze(-1) * 2.5, torch.zeros_like(vec))
    v = v + push.sum(2)
    if env.num_agents > 1:   # keep clear of teammates
        dm = env.pos[:, :, None] - env.pos[:, None]
        dd = dm.norm(dim=-1).clamp_min(1e-3)
        eye = torch.eye(env.num_agents, dtype=torch.bool, device=env.device)
        close = (dd < 1.6) & ~eye & env.alive[:, None]
        v = v + torch.where(close.unsqueeze(-1), dm / dd.unsqueeze(-1) * (1.6 - dd).unsqueeze(-1) * 3.0,
                            torch.zeros_like(dm)).sum(2)
    return v


def expert_action(env) -> torch.Tensor:
    if env.scene == "trash":
        return _trash(env)
    return _ocean(env)


def _trash(env) -> torch.Tensor:
    n, a = env.num_envs, env.num_agents
    dev = env.device
    ph = env.exp_phase
    pos = env.pos
    gp = env.gripper_pos()
    off = torch.tensor(spec.GRIPPER_OFFSET, device=dev)
    handle = env.payload_handle()[:, None].expand(n, a, 3)
    apos, _, akind = env.anchor_positions()
    handle_idx = env.entity_names().index("payload.handle")
    pick = env.anc_src[..., 0] == handle_idx
    dest = torch.where((akind[..., 1] == 0).unsqueeze(-1), env.zone_c[:, None, 0].expand(n, a, 3), apos[:, :, 1])
    holding = env.holding >= 0
    speed = env.vel.norm(dim=-1)
    gentle = (env.pl_kind == 1)[:, None].expand(n, a)
    vmax = torch.where(gentle & holding, torch.full_like(speed, GENTLE), torch.full_like(speed, CRUISE))
    up = lambda z: torch.tensor([0.0, 0.0, z], device=dev)

    # transitions
    above = handle + up(1.2)
    at_above = ((pos - above)[..., :2].norm(dim=-1) < 0.2) & ((pos - above)[..., 2].abs() < 0.3)
    near_handle = ((gp - handle).norm(dim=-1) < 0.1) & (speed < 0.4)
    ph = torch.where((ph == 0) & at_above, torch.ones_like(ph), ph)
    ph = torch.where((ph == 1) & near_handle, torch.full_like(ph, 2), ph)
    failed = (ph == 2) & env.grip & ~holding
    ph = torch.where(failed, torch.ones_like(ph), ph)
    ph = torch.where((ph == 2) & holding, torch.full_like(ph, 3), ph)
    ph = torch.where((ph == 3) & (pos[..., 2] > 2.6), torch.full_like(ph, 4), ph)
    over_dest = (pos - dest)[..., :2].norm(dim=-1) < 0.3
    ph = torch.where((ph == 4) & over_dest, torch.full_like(ph, 5), ph)
    low = ((pos[..., 2] - (dest[..., 2] + 1.3)).abs() < 0.2) & (speed < 0.5)
    ph = torch.where((ph == 5) & low, torch.full_like(ph, 6), ph)
    ph = torch.where((ph == 6) & ~holding, torch.full_like(ph, 7), ph)
    lost = (ph >= 3) & (ph <= 5) & ~holding
    ph = torch.where(lost, torch.zeros_like(ph), ph)
    ph = torch.where(~pick, torch.full_like(ph, 8), ph)
    env.exp_phase = ph

    target = pos.clone()
    target = torch.where((ph == 0).unsqueeze(-1), above, target)
    target = torch.where(((ph == 1) | (ph == 2)).unsqueeze(-1), handle - off + up(0.03), target)
    lift = torch.cat([pos[..., :2], torch.full_like(pos[..., :1], 3.0)], -1)
    target = torch.where((ph == 3).unsqueeze(-1), lift, target)
    target = torch.where((ph == 4).unsqueeze(-1), dest + up(3.0), target)
    target = torch.where(((ph == 5) | (ph == 6)).unsqueeze(-1), dest + up(1.3), target)
    target = torch.where((ph == 7).unsqueeze(-1), dest + up(3.0), target)
    goal = apos[:, :, 0] + torch.where((akind[..., 0] == 3).unsqueeze(-1), up(1.5), up(0.0))
    goal[..., 2] = goal[..., 2].clamp_min(0.8)
    target = torch.where((ph == 8).unsqueeze(-1), goal, target)
    v = _goto(pos, target, vmax)
    slow = (ph == 1) | (ph == 2)
    v = torch.where(slow.unsqueeze(-1), _goto(pos, target, 0.6), v)
    v = _avoid(env, v)
    grip = torch.where((ph >= 2) & (ph <= 5), torch.ones_like(speed), -torch.ones_like(speed))
    grip = torch.where(failed, -torch.ones_like(grip), grip)
    return _to_action(v, grip)


def _ocean(env) -> torch.Tensor:
    n, a = env.num_envs, env.num_agents
    dev = env.device
    ph = env.exp_phase
    pos = env.pos
    gp = env.gripper_pos()
    off = torch.tensor(spec.GRIPPER_OFFSET, device=dev)
    up = lambda z: torch.tensor([0.0, 0.0, z], device=dev)
    corners = env.net_handles()
    cvel = env.net.vel[:, env.net.corners]
    names = env.entity_names()
    first_corner = names.index("net.corner_0")
    src0 = env.anc_src[..., 0]
    is_corner = (src0 >= first_corner) & (src0 < first_corner + 4)
    default = torch.arange(a, device=dev).expand(n, a) % 4
    cidx = torch.where(is_corner, src0 - first_corner, default)
    cidx = torch.where(env.holding >= 0, env.holding, cidx)
    cpos = torch.gather(corners, 1, cidx.unsqueeze(-1).expand(n, a, 3))
    cv = torch.gather(cvel, 1, cidx.unsqueeze(-1).expand(n, a, 3))
    holding = env.holding >= 0
    surf = env.surface_height(pos[..., :2])
    speed = env.vel.norm(dim=-1)

    lane = torch.arange(a, device=dev).float() * 0.8                 # stagger transit heights per drone
    above = cpos + up(1.5) + torch.stack([torch.zeros_like(lane), torch.zeros_like(lane), lane], -1)
    above[..., 2] = torch.maximum(above[..., 2], surf + 1.5 + lane)
    at_above = (pos - above)[..., :2].norm(dim=-1) < 0.3
    ph = torch.where((ph == 0) & at_above, torch.ones_like(ph), ph)
    near = (gp - cpos).norm(dim=-1) < 0.25
    ph = torch.where((ph == 1) & near, torch.full_like(ph, 2), ph)
    failed = (ph == 2) & env.grip & ~holding
    ph = torch.where(failed, torch.ones_like(ph), ph)
    ph = torch.where((ph == 2) & holding, torch.full_like(ph, 3), ph)
    # team phases: 3 hold low, 4 drag the net under the nearest school, 5 lift, 6 fly to the boat
    holding_team = ((ph >= 3) | ~env.alive).all(-1, keepdim=True)
    ph = torch.where((ph == 3) & holding_team, torch.full_like(ph, 4), ph)
    centroid = env.net.centroid()
    schools = env.fish.school_centers()                                   # [N, S, 3]
    sd = (schools[..., :2] - centroid[:, None, :2]).norm(dim=-1)
    school = torch.gather(schools, 1, sd.argmin(-1)[:, None, None].expand(n, 1, 3))[:, 0]
    under = ((centroid[:, :2] - school[:, :2]).norm(dim=-1) < 2.0)[:, None].expand(n, a)
    ph = torch.where((ph == 4) & under, torch.full_like(ph, 5), ph)
    high = (((pos[..., 2] - surf) > 2.5) | ~env.alive).all(-1, keepdim=True).expand(n, a)
    ph = torch.where((ph == 5) & high, torch.full_like(ph, 6), ph)
    lost = (ph >= 3) & ~holding
    ph = torch.where(lost, torch.zeros_like(ph), ph)
    env.exp_phase = ph

    target = torch.where((ph == 0).unsqueeze(-1), above, pos)
    target = torch.where(((ph == 1) | (ph == 2)).unsqueeze(-1), cpos - off + up(0.05), target)
    low_alt = torch.cat([pos[..., :2], (surf + 1.2).unsqueeze(-1)], -1)
    target = torch.where((ph == 3).unsqueeze(-1), low_alt, target)
    form = (cpos - centroid[:, None]) * torch.tensor([0.8, 0.8, 0.0], device=dev)
    drag_t = school[:, None] + form
    drag_t[..., 2] = surf + 1.2
    target = torch.where((ph == 4).unsqueeze(-1), drag_t, target)
    lift_t = torch.cat([pos[..., :2], (surf + 3.2).unsqueeze(-1)], -1)
    target = torch.where((ph == 5).unsqueeze(-1), lift_t, target)
    boat = env.zone_c[:, None, 0]
    target = torch.where((ph == 6).unsqueeze(-1), boat + form + up(4.5), target)
    target[..., 2] = torch.maximum(target[..., 2], surf + 0.5)
    cv_xy = torch.cat([cv[..., :2], torch.zeros_like(cv[..., :1])], -1)
    cv_xy = cv_xy * (2.0 / cv_xy.norm(dim=-1, keepdim=True).clamp_min(2.0))
    ff = torch.where((ph <= 2).unsqueeze(-1), cv_xy, torch.zeros_like(cv))
    team_speed = torch.where(ph == 4, torch.full_like(speed, 2.0), torch.full_like(speed, 2.0))
    v = _goto(pos, target, torch.where(ph >= 4, team_speed, torch.full_like(speed, CRUISE)), ff)
    v = torch.where(((ph == 1) | (ph == 2)).unsqueeze(-1), _goto(pos, target, 1.0, ff), v)
    v = _avoid(env, v)
    grip = torch.where(ph >= 2, torch.ones_like(speed), -torch.ones_like(speed))
    grip = torch.where(failed, -torch.ones_like(grip), grip)
    return _to_action(v, grip)
