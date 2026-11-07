"""Scripted privileged expert: demonstrations for the policy regulariser.

Trash and delivery: if anchor 0 is an item (or its handle), climb to a cruise
height that clears every building, tree, and hill on the straight path, fly over
the item, descend, grab it, cruise to anchor 1 (or the drop-off zone), and
release: trash is dropped into the bin or truck from above, the organ cooler is
set down gently on the pad. Otherwise fly to anchor 0 and hover.
Ocean: drone i grabs net corner i (or its anchor-0 corner), lifts, and once the
whole team holds, flies in formation to the boat.

Phases live in env.exp_phase and reset with the world. Actions pass through the
env's safety shield like any policy action.
"""
from __future__ import annotations

import torch

from .. import spec

KP = 1.2
CRUISE = 5.0
GENTLE = 3.0


def _to_action(v: torch.Tensor, grip: torch.Tensor, env=None) -> torch.Tensor:
    ms = spec.MAX_SPEED if env is None else env.max_speed
    mz = spec.MAX_VZ if env is None else env.max_vz
    scale = v.new_tensor([ms, ms, mz])
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


def _avoid(env, v: torch.Tensor, radius=1.4) -> torch.Tensor:
    """Repel from nearby boxes and spheres (moving ones included) and teammates.
    radius: float or [N, A] (smaller while grabbing or releasing next to obstacles)."""
    if not torch.is_tensor(radius):
        radius = torch.full(env.pos.shape[:2], float(radius), device=env.device)
    R = radius.unsqueeze(-1)
    rel = env.pos[:, :, None] - env.box_c[:, None]
    closest = torch.maximum(torch.minimum(rel, env.box_h[:, None]), -env.box_h[:, None])
    vec = rel - closest
    d = vec.norm(dim=-1).clamp_min(1e-3)
    near = (d < R) & env.box_active[:, None]
    push = torch.where(near.unsqueeze(-1), vec / d.unsqueeze(-1) * (R - d).unsqueeze(-1) * 2.5, torch.zeros_like(vec))
    v = v + push.sum(2)
    vs = env.pos[:, :, None] - env.sph_c[:, None]
    ds = (vs.norm(dim=-1) - env.sph_r[:, None]).clamp_min(1e-3)
    near = (ds < R + 0.4) & env.sph_active[:, None]
    dirs = vs / vs.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    v = v + torch.where(near.unsqueeze(-1), dirs * (R + 0.4 - ds).unsqueeze(-1) * 2.5, torch.zeros_like(vs)).sum(2)
    if env.num_agents > 1:
        dm = env.pos[:, :, None] - env.pos[:, None]
        dd = dm.norm(dim=-1).clamp_min(1e-3)
        eye = torch.eye(env.num_agents, dtype=torch.bool, device=env.device)
        close = (dd < 2.2) & ~eye & env.alive[:, None]
        v = v + torch.where(close.unsqueeze(-1), dm / dd.unsqueeze(-1) * (2.2 - dd).unsqueeze(-1) * 4.0,
                            torch.zeros_like(dm)).sum(2)
    return v


def _top_at(env, xy: torch.Tensor, margin: float = 1.2) -> torch.Tensor:
    """Highest thing to fly over at xy [N, A, M, 2]: terrain or water, box tops, sphere tops."""
    n, a, m, _ = xy.shape
    flat = xy.reshape(n, a * m, 2)
    top = env.surface_height(flat).reshape(n, a, m)
    inside = ((flat[:, :, None] - env.box_c[:, None, :, :2]).abs() < env.box_h[:, None, :, :2] + margin).all(-1)
    btop = torch.where(inside & env.box_active[:, None], (env.box_c[..., 2] + env.box_h[..., 2])[:, None],
                       torch.full(inside.shape, -1e4, device=env.device)).amax(-1)
    ds = (flat[:, :, None] - env.sph_c[:, None, :, :2]).norm(dim=-1)
    stop = torch.where((ds < env.sph_r[:, None] + margin) & env.sph_active[:, None],
                       (env.sph_c[..., 2] + env.sph_r)[:, None], torch.full(ds.shape, -1e4, device=env.device)).amax(-1)
    return torch.maximum(top, torch.maximum(btop, stop).reshape(n, a, m))


def _cruise(env, start: torch.Tensor, goal: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    """Cruise height [N, A] clearing everything along the straight path start -> goal (xy)."""
    long = env.arena_half > 100
    s = torch.linspace(0.0, 1.0, 64 if long else 16, device=env.device).view(1, 1, -1, 1)
    pts = start[..., None, :2] + (goal[..., None, :2] - start[..., None, :2]) * s
    return _top_at(env, pts).amax(-1) + (8.0 if long else 2.5) + extra


def _detour(env, start: torch.Tensor, goal: torch.Tensor, margin: float = 3.0) -> torch.Tensor:
    """Goal, or a waypoint beside the first no-fly cylinder the straight path start -> goal crosses."""
    out = goal.clone()
    for z in (1, 2):
        c, r = env.zone_c[:, None, z, :2], env.zone_r[:, None, z] + margin
        act = env.zone_active[:, None, z]
        seg = goal[..., :2] - start[..., :2]
        L2 = (seg * seg).sum(-1).clamp_min(1e-6)
        t = (((c - start[..., :2]) * seg).sum(-1) / L2).clamp(0.0, 1.0)
        closest = start[..., :2] + seg * t.unsqueeze(-1)
        off = closest - c
        dist = off.norm(dim=-1)
        perp = torch.stack([-seg[..., 1], seg[..., 0]], -1)
        n_ = torch.where((dist > 1e-3).unsqueeze(-1), off / dist.clamp_min(1e-3).unsqueeze(-1),
                         perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        way = c + n_ * (r + 2.0).unsqueeze(-1)
        cross = act & (dist < r) & (t > 0.0) & (t < 1.0)
        out[..., :2] = torch.where(cross.unsqueeze(-1), way, out[..., :2])
    return out


def expert_action(env) -> torch.Tensor:
    if env.scene == "ocean":
        return _ocean(env, env.alive)
    if env.scene == "world":
        # each drone follows its own anchors: net-corner drones haul as a team, the rest
        # pick and place items or fly to their anchor
        hidx = env.anchor_handle()
        team = (hidx >= env.corner_offset) | (env.holding >= env.corner_offset)
        ph0 = env.exp_phase.clone()
        act_land = _land(env)
        ph_land = env.exp_phase.clone()
        env.exp_phase = ph0
        act_sea = _ocean(env, team & env.alive)
        env.exp_phase = torch.where(team, env.exp_phase, ph_land)
        return torch.where(team.unsqueeze(-1), act_sea, act_land)
    return _land(env)


def _land(env) -> torch.Tensor:
    n, a = env.num_envs, env.num_agents
    dev = env.device
    ph = env.exp_phase
    pos, gp = env.pos, env.gripper_pos()
    off = torch.tensor(spec.GRIPPER_OFFSET, device=dev)
    up = lambda z: torch.tensor([0.0, 0.0, z], device=dev)
    speed = env.vel.norm(dim=-1)
    holding = env.holding >= 0
    apos, _, akind = env.anchor_positions()
    item = env.anchor_item()
    avail = env.it_active & ~env.it_delivered
    it = item.clamp_min(0)
    pick = (item >= 0) & (torch.gather(avail, 1, it) | holding)
    handle = torch.gather(env.item_handles(), 1, it.unsqueeze(-1).expand(n, a, 3))
    ihalf = torch.gather(env.it_half[..., 2], 1, it)
    # destination: anchor 1, else the drop-off zone
    gentle = torch.gather(env.it_type, 1, it) == env.catalog.id("cooler")
    # default drop-off: the clinic pad for the cooler, the bin / garbage truck for anything else
    zs = spec.ZONE_SLOT
    zone_bin = env.zone_c[:, None, zs["bin"]].expand(n, a, 3)
    zone_pad = env.zone_c[:, None, zs["pad_dropoff"]].expand(n, a, 3)
    pad_ok = env.zone_active[:, None, zs["pad_dropoff"]].expand(n, a)
    zone0 = torch.where((gentle & pad_ok).unsqueeze(-1), zone_pad, zone_bin)
    dest = torch.where((akind[..., 1] == 0).unsqueeze(-1), zone0, apos[:, :, 1])
    dest_top = _top_at(env, dest[..., None, :2], margin=0.3).squeeze(-1)
    # release: the cooler is set down on the pad; trash is dropped from above the bin or truck
    place_z = env.surface_height(dest[..., None, :2]).squeeze(-1) + 0.12 + 2 * ihalf + 0.2 + 0.18
    drop_z = dest_top + 1.6
    release_z = torch.where(gentle, place_z, drop_z)
    hang = torch.where(holding, 2 * ihalf + 0.6, torch.zeros_like(ihalf))
    to_item = _cruise(env, pos, handle, torch.zeros_like(ihalf))
    to_dest = _cruise(env, pos, dest, hang)
    cruise_v = CRUISE if env.arena_half < 100 else 0.8 * env.max_speed    # long-range delivery flies fast
    gentle_v = GENTLE if env.arena_half < 100 else cruise_v
    vmax = torch.where(gentle & holding, torch.full_like(speed, gentle_v), torch.full_like(speed, cruise_v))
    xy = lambda p, q: (p[..., :2] - q[..., :2]).norm(dim=-1)

    above_item = xy(pos, handle) < 0.4
    ph = torch.where((ph == 0) & above_item & (pos[..., 2] > handle[..., 2] + 0.8), torch.ones_like(ph), ph)
    near = ((gp - handle).norm(dim=-1) < 0.1) & (speed < 0.4)
    ph = torch.where((ph == 1) & near, torch.full_like(ph, 2), ph)
    failed = (ph == 2) & env.grip & ~holding
    ph = torch.where(failed, torch.ones_like(ph), ph)
    ph = torch.where((ph == 2) & holding, torch.full_like(ph, 3), ph)
    ph = torch.where((ph == 3) & (pos[..., 2] > to_dest - 0.5), torch.full_like(ph, 4), ph)
    ph = torch.where((ph == 4) & (xy(pos, dest) < 0.4), torch.full_like(ph, 5), ph)
    ph = torch.where((ph == 5) & ((pos[..., 2] - release_z).abs() < 0.15) & (speed < 0.4), torch.full_like(ph, 6), ph)
    ph = torch.where((ph == 6) & ~holding, torch.full_like(ph, 7), ph)
    # a selector anchor ("nearest:trash") has moved on to the next item: start the next pickup
    ph = torch.where((ph == 7) & pick & ~holding, torch.zeros_like(ph), ph)
    lost = (ph >= 3) & (ph <= 5) & ~holding
    ph = torch.where(lost, torch.zeros_like(ph), ph)
    ph = torch.where(~pick & (ph < 6), torch.full_like(ph, 8), ph)
    ph = torch.where(pick & (ph == 8), torch.zeros_like(ph), ph)
    env.exp_phase = ph

    level = lambda goal, z: torch.cat([goal[..., :2], z.unsqueeze(-1)], -1)
    climb = lambda z: torch.cat([pos[..., :2], z.unsqueeze(-1)], -1)
    high_enough = pos[..., 2] > to_item - 0.5
    t0 = torch.where(high_enough.unsqueeze(-1), level(_detour(env, pos, handle), to_item), climb(to_item))
    target = torch.where((ph == 0).unsqueeze(-1), t0, pos)
    target = torch.where(((ph == 1) | (ph == 2)).unsqueeze(-1), handle - off + up(0.03), target)
    target = torch.where((ph == 3).unsqueeze(-1), climb(to_dest), target)
    target = torch.where((ph == 4).unsqueeze(-1), level(_detour(env, pos, dest), to_dest), target)
    target = torch.where(((ph == 5) | (ph == 6)).unsqueeze(-1), level(dest, release_z), target)
    target = torch.where((ph == 7).unsqueeze(-1), level(dest, to_dest), target)
    goal = apos[:, :, 0]
    goal_z = torch.maximum(goal[..., 2], env.surface_height(goal[..., None, :2]).squeeze(-1) + 1.0)
    cr = _cruise(env, pos, goal, torch.zeros_like(ihalf))
    far = xy(pos, goal) > 1.0
    t8 = torch.where(far.unsqueeze(-1), torch.where((pos[..., 2] > cr - 0.5).unsqueeze(-1), level(goal, cr), climb(cr)),
                     level(goal, goal_z))
    target = torch.where((ph == 8).unsqueeze(-1), t8, target)
    fine = (ph == 1) | (ph == 2) | (ph == 5) | (ph == 6)
    dist = (target - pos).norm(dim=-1)
    vlim = torch.where(fine, torch.minimum(0.5 + 0.8 * dist, vmax), vmax)   # slow only near the target
    v = _goto(pos, target, vlim)
    # descend gently near the target height: vertical speed shrinks with the remaining drop
    vz_lim = 0.4 + 0.6 * (pos[..., 2] - target[..., 2]).clamp_min(0.0)
    v[..., 2] = torch.where(fine, torch.maximum(v[..., 2], -vz_lim), v[..., 2])
    v = _avoid(env, v, torch.where(fine & (dist < 2.0), torch.full_like(dist, 0.6), torch.full_like(dist, 1.4)))
    grip = torch.where((ph >= 2) & (ph <= 5), torch.ones_like(speed), -torch.ones_like(speed))
    grip = torch.where(failed, -torch.ones_like(grip), grip)
    return _to_action(v, grip, env)


def _ocean(env, team: torch.Tensor) -> torch.Tensor:
    """team [N, A]: drones taking part in the net haul (the others are ignored here)."""
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
    src0 = env.anchor_src()[..., 0]
    is_corner = (src0 >= first_corner) & (src0 < first_corner + 4)
    default = torch.arange(a, device=dev).expand(n, a) % 4
    cidx = torch.where(is_corner, src0 - first_corner, default)
    coff = env.corner_offset                                       # handle index of net corner 0
    held_corner = env.holding >= coff
    cidx = torch.where(held_corner, env.holding - coff, cidx)
    # corners are assigned greedily by distance (each drone takes the nearest free corner),
    # so the team's paths never cross; a held corner stays with its holder
    dcorner = (pos[:, :, None, :2] - corners[:, None, :, :2]).norm(dim=-1)        # [N, A, 4]
    taken = torch.zeros(n, 4, dtype=torch.bool, device=dev)
    held_c = held_corner
    taken.scatter_(1, torch.where(held_c, env.holding - coff, torch.zeros_like(env.holding)), held_c)
    for ag in range(a):
        free_d = torch.where(taken, torch.full_like(dcorner[:, ag], 1e4), dcorner[:, ag])
        pick = torch.where(held_c[:, ag], env.holding[:, ag] - coff, free_d.argmin(-1))
        cidx[:, ag] = pick
        taken.scatter_(1, pick.unsqueeze(-1), team[:, ag:ag + 1])
    cpos = torch.gather(corners, 1, cidx.unsqueeze(-1).expand(n, a, 3))
    cv = torch.gather(cvel, 1, cidx.unsqueeze(-1).expand(n, a, 3))
    holding = env.holding >= coff
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
    holding_team = ((ph >= 3) | ~team).all(-1, keepdim=True)
    ph = torch.where((ph == 3) & holding_team, torch.full_like(ph, 4), ph)
    centroid = env.net.centroid()
    schools = env.fish.school_centers()                                   # [N, S, 3]
    sd = (schools[..., :2] - centroid[:, None, :2]).norm(dim=-1)
    # prefer schools with no animal nearby (bycatch risk)
    an = env.animals.pos
    risk = ((schools[:, :, None, :2] - an[:, None, :, :2]).norm(dim=-1) < 5.0).any(-1)
    school = torch.gather(schools, 1, (sd + 30.0 * risk.float()).argmin(-1)[:, None, None].expand(n, 1, 3))[:, 0]
    under = ((centroid[:, :2] - school[:, :2]).norm(dim=-1) < 2.0)[:, None].expand(n, a)
    ph = torch.where((ph == 4) & under, torch.full_like(ph, 5), ph)
    high = (((pos[..., 2] - surf) > 2.5) | ~team).all(-1, keepdim=True).expand(n, a)
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
    # an animal near the net: steer the whole net away from it until it passes
    d_an = (an[..., :2] - centroid[:, None, :2]).norm(dim=-1)
    near_an = d_an.min(-1).values < 5.0
    away = centroid[:, :2] - torch.gather(an[..., :2], 1, d_an.argmin(-1)[:, None, None].expand(n, 1, 2))[:, 0]
    away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    flee = torch.cat([centroid[:, :2] + away * 4.0, centroid[:, 2:3]], -1)
    drag_t = torch.where(near_an.view(n, 1, 1), flee[:, None] + form, drag_t)
    lim = env.arena_half - 3.0                                         # never drag the team out of the arena
    drag_t = torch.cat([drag_t[..., :2].clamp(-lim, lim), drag_t[..., 2:]], -1)
    drag_t[..., 2] = surf + 1.2
    target = torch.where((ph == 4).unsqueeze(-1), drag_t, target)
    lift_t = torch.cat([pos[..., :2], (surf + 3.2).unsqueeze(-1)], -1)
    target = torch.where((ph == 5).unsqueeze(-1), lift_t, target)
    boat = env.zone_c[:, None, spec.ZONE_SLOT["boat"]]
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
    return _to_action(v, grip, env)
