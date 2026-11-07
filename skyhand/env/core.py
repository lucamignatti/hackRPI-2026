"""SkyEnv: batched environment with a pure-torch physics backend.

This class runs anywhere PyTorch runs, with no Genesis install. The Genesis
backend subclasses it and overrides only the rigid-body hooks
(`_integrate_drones`, `_integrate_free_payload`, `_solid_ranges`).

Scenes
------
trash: one drone, a payload (bag or cooler) with a handle, a bin, six box
       obstacles (two moving on loops), an optional no-fly cylinder, gusty wind.
ocean: OCEAN_AGENTS drones, Airy wave field, a floating net with grabbable
       corners, fish schools, a bobbing boat, an optional no-fly cylinder.

Contract: see skyhand/spec.py. step() does not auto-reset; call reset(env_ids)
for worlds that are done or truncated after storing their transitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .. import spec
from ..spec import AnchorBinding, StepOut
from .controller import GRAVITY, VelocityController
from .fish import FishSchools
from .lidar import LidarModel, decode_range
from .math3d import quat_from_yaw, quat_integrate, quat_rotate, quat_rotate_inv, tilt_of
from .net import NetSim
from .shield import shield_velocity
from .waves import WaveField

NET_GRASP_RADIUS = 0.3
NET_HANDLE_HEIGHT = 0.6          # buoy line above each net corner, grabbed by the drone
VEL_KI = 1.0                     # integral gain on velocity error (cancels steady wind)
PAYLOAD_GRASP_RADIUS = 0.15
HANDLE_HEIGHT = 0.10
EXTRA_OBSTACLE_SLOTS = 1          # inactive slot used by the demo "drop obstacle"
BIN_RADIUS = 0.7
BOAT_HALF = (2.0, 1.0, 0.4)
BOAT_RADIUS = 2.5
ANCHOR_KIND = {k: i for i, k in enumerate(spec.ANCHOR_TYPES)}
PRIV_TYPE = {k: i for i, k in enumerate(spec.PRIV_TYPES)}


@dataclass
class EnvConfig:
    scene: str = "trash"
    num_envs: int = 64
    device: str = "cpu"
    seed: int = 0
    lidar: bool = True              # False: skip scans, LiDAR images stay at "no return"
    shield: bool = True
    domain_rand: bool = True
    random_anchors: bool = True     # pretraining: randomise anchors on reset
    cooler_prob: float = 0.3        # trash scene: payload is a cooler with this probability
    nofly_prob: float = 0.5
    max_wind: float = 3.0
    expert_mode: str = "task"       # "task": pick-and-place / net haul when anchors allow
    genesis_lidar: bool = False     # Genesis backend only: use Genesis' raycaster (self-hit issue, see backend)


class SkyEnv:
    def __init__(self, cfg: Optional[EnvConfig] = None, **kw):
        cfg = cfg or EnvConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        assert cfg.scene in spec.SCENES, cfg.scene
        self.cfg = cfg
        self.scene = cfg.scene
        self.num_envs = n = cfg.num_envs
        self.num_agents = a = spec.TRASH_AGENTS if cfg.scene == "trash" else spec.OCEAN_AGENTS
        self.device = dev = torch.device(cfg.device)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.ctrl = VelocityController()
        self.lidar_model = LidarModel(dev)
        zf = lambda *s: torch.zeros(*s, device=dev)
        # time
        self.t = zf(n)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        # drones
        self.pos, self.vel, self.ang = zf(n, a, 3), zf(n, a, 3), zf(n, a, 3)
        self.quat = zf(n, a, 4)
        self.quat[..., 0] = 1.0
        self.mass = torch.full((n, a), spec.DRONE_MASS, device=dev)
        self.drag = torch.full((n, a), 0.35, device=dev)
        self.alive = torch.ones(n, a, dtype=torch.bool, device=dev)
        self.grip = torch.zeros(n, a, dtype=torch.bool, device=dev)
        self.holding = torch.full((n, a), -1, dtype=torch.long, device=dev)   # handle index or -1
        self.last_action = zf(n, a, spec.ACTION_DIM)
        self.hover = torch.ones(n, a, device=dev)
        self.acc_prev, self.vel_prev, self.jerk = zf(n, a, 3), zf(n, a, 3), zf(n, a)
        self.ext_force = zf(n, a, 3)        # payload / net reaction, applied each substep
        self.v_int = zf(n, a, 3)            # velocity-error integral of the controller
        self.collided = torch.zeros(n, a, dtype=torch.bool, device=dev)
        self.shield_hit = torch.zeros(n, a, dtype=torch.bool, device=dev)
        self.shield_count = torch.zeros(n, dtype=torch.long, device=dev)
        # wind
        self.wind, self.wind_mean = zf(n, 3), zf(n, 3)
        self.gust, self.gust_time = zf(n, 3), zf(n)
        # obstacles (boxes); trash: 6 + extra slot, ocean: boat box + extra slot
        nb = (spec.TRASH_N_OBSTACLES if cfg.scene == "trash" else 1) + EXTRA_OBSTACLE_SLOTS
        self.nb = nb
        self.box_c, self.box_h, self.box_v = zf(n, nb, 3), zf(n, nb, 3), zf(n, nb, 3)
        self.box_active = torch.zeros(n, nb, dtype=torch.bool, device=dev)
        self.box_moving = torch.zeros(n, nb, dtype=torch.bool, device=dev)
        self.orbit_c, self.orbit_r, self.orbit_w, self.orbit_ph = zf(n, nb, 2), zf(n, nb), zf(n, nb), zf(n, nb)
        # zones: index 0 = bin (trash) or boat (ocean), index 1 = no-fly
        self.zone_c, self.zone_r = zf(n, 2, 3), zf(n, 2)
        self.zone_active = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        # payload (trash)
        self.pl_pos, self.pl_vel, self.pl_size = zf(n, 3), zf(n, 3), zf(n, 3)
        self.pl_mass = torch.ones(n, device=dev)
        self.pl_kind = torch.zeros(n, dtype=torch.long, device=dev)       # 0 bag, 1 cooler
        self.pl_held_by = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.swing = zf(n)
        # ocean
        self.waves = WaveField(n, dev) if cfg.scene == "ocean" else None
        self.net = NetSim(n, dev) if cfg.scene == "ocean" else None
        self.fish = FishSchools(n, dev) if cfg.scene == "ocean" else None
        self.turbidity = torch.full((n,), 0.3, device=dev)
        self.net_stretch, self.fish_frac = zf(n), zf(n)
        self.corner_held_by = torch.full((n, 4), -1, dtype=torch.long, device=dev)
        # anchors
        K = spec.N_ANCHORS
        self.anc_kind = torch.zeros(n, a, K, dtype=torch.long, device=dev)
        self.anc_src = torch.full((n, a, K), -1, dtype=torch.long, device=dev)
        self.anc_point, self.anc_radius = zf(n, a, K, 3), zf(n, a, K)
        # expert
        self.exp_phase = torch.zeros(n, a, dtype=torch.long, device=dev)
        # LiDAR cache
        self._lidar = torch.full((n, a, spec.LIDAR_CH, spec.LIDAR_V, spec.LIDAR_H), 255, dtype=torch.uint8, device=dev)
        self._lidar[:, :, 2] = 0
        self._lidar_down = torch.full((n, a, spec.LIDAR_CH, spec.DOWN_V, spec.DOWN_H), 255, dtype=torch.uint8, device=dev)
        self._lidar_down[:, :, 2] = 0
        self._lidar_pts: Dict[str, torch.Tensor] = {}
        self._drone_spheres_first = True    # _geometry() lists one sphere per drone first
        self._build_entities()
        self._post_init()
        self.reset()

    # ------------------------------------------------------------------ hooks
    def _post_init(self) -> None:
        """Backend hook called before the first reset."""

    def _integrate_drones(self, force: torch.Tensor, torque: torch.Tensor, dt: float) -> None:
        """Advance drone rigid bodies one physics step. force world [N, A, 3], torque body."""
        alive = self.alive.unsqueeze(-1)
        inertia = self.ctrl.inertia.to(self.device)
        acc = force / self.mass.unsqueeze(-1)
        self.vel = torch.where(alive, self.vel + dt * acc, self.vel)
        self.pos = torch.where(alive, self.pos + dt * self.vel, self.pos)
        gyro = torch.cross(self.ang, self.ang * inertia, dim=-1)
        self.ang = torch.where(alive, self.ang + dt * (torque - gyro) / inertia, self.ang)
        self.ang = self.ang.clamp(-20.0, 20.0)
        self.quat = torch.where(alive, quat_integrate(self.quat, self.ang, dt), self.quat)

    def _integrate_free_payload(self, dt: float, free: torch.Tensor) -> None:
        """Advance the payload when nobody holds it. free [N] bool."""
        if not free.any():
            return
        v = self.pl_vel.clone()
        v[:, 2] -= GRAVITY * dt
        p = self.pl_pos + dt * v
        floor = self.pl_size[:, 2] / 2
        on_ground = p[:, 2] <= floor
        p[:, 2] = torch.maximum(p[:, 2], floor)
        v[:, 2] = torch.where(on_ground, torch.zeros_like(v[:, 2]), v[:, 2])
        v[:, :2] = torch.where(on_ground.unsqueeze(-1), v[:, :2] * (1.0 - 8.0 * dt), v[:, :2])
        f = free.unsqueeze(-1)
        self.pl_pos = torch.where(f, p, self.pl_pos)
        self.pl_vel = torch.where(f, v, self.pl_vel)

    def _solid_ranges(self):
        """External raycaster distances (Genesis). None = use analytic boxes/ground."""
        return None, None

    def _sync_kinematics(self) -> None:
        """Backend hook: push kinematic poses (obstacles, held payload, boat) to the simulator."""

    def _reset_backend(self, env_ids: torch.Tensor) -> None:
        """Backend hook: push freshly reset state to the simulator."""

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def gripper_pos(self) -> torch.Tensor:
        off = torch.tensor(spec.GRIPPER_OFFSET, device=self.device).expand_as(self.pos)
        return self.pos + quat_rotate(self.quat, off)

    def payload_handle(self) -> torch.Tensor:
        h = self.pl_pos.clone()
        h[:, 2] += self.pl_size[:, 2] / 2 + HANDLE_HEIGHT
        held = self.pl_held_by >= 0
        if held.any():
            gp = self.gripper_pos()
            idx = self.pl_held_by.clamp_min(0)
            h = torch.where(held.unsqueeze(-1), gp[torch.arange(self.num_envs, device=self.device), idx], h)
        return h

    def net_handles(self) -> torch.Tensor:
        """Grab points of the four net corners [N, 4, 3]: top of each buoy line."""
        h = self.net.corner_pos().clone()
        h[..., 2] += NET_HANDLE_HEIGHT
        return h

    def surface_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Ground (0) or wave height at xy [N, M, 2] -> [N, M]."""
        if self.waves is None:
            return torch.zeros(xy.shape[:-1], device=self.device)
        return self.waves.height(xy, self.t)

    # ------------------------------------------------------------------ entities
    def _build_entities(self) -> None:
        if self.scene == "trash":
            names = ["payload.handle", "payload", "bin", "nofly_0"] + \
                    [f"obstacle_{i}" for i in range(spec.TRASH_N_OBSTACLES)] + ["drone_0"]
            kinds = ["handle", "point", "zone", "zone"] + ["point"] * spec.TRASH_N_OBSTACLES + ["point"]
        else:
            names = [f"net.corner_{i}" for i in range(4)] + ["net.center", "boat"] + \
                    [f"fish_school_{i}" for i in range(spec.N_FISH_SCHOOLS)] + ["nofly_0"] + \
                    [f"drone_{i}" for i in range(self.num_agents)]
            kinds = ["handle"] * 4 + ["point", "zone"] + ["point"] * spec.N_FISH_SCHOOLS + ["zone"] + \
                    ["point"] * self.num_agents
        self._ent_names: List[str] = names
        self._ent_kind = torch.tensor([ANCHOR_KIND[k] for k in kinds], device=self.device)

    def entity_names(self) -> List[str]:
        return list(self._ent_names)

    def entity_descriptions(self) -> Dict[str, str]:
        """Human-readable descriptions for the language layer (world 0)."""
        d = {}
        pos = self._entity_positions()[0]
        for i, name in enumerate(self._ent_names):
            p = [round(x, 1) for x in pos[i].tolist()]
            if name.startswith("payload"):
                kind = "cooler" if int(self.pl_kind[0]) == 1 else "trash bag"
                d[name] = f"{kind}{' handle (grab here)' if name.endswith('handle') else ''} at {p}"
            elif name == "bin":
                d[name] = f"trash bin, drop zone radius {BIN_RADIUS} m at {p}"
            elif name.startswith("nofly"):
                act = bool(self.zone_active[0, 1])
                d[name] = f"no-fly cylinder radius {float(self.zone_r[0, 1]):.1f} m at {p}" + ("" if act else " (inactive)")
            elif name.startswith("obstacle"):
                i_b = int(name.split("_")[1])
                mv = bool(self.box_moving[0, i_b])
                d[name] = f"{'moving' if mv else 'static'} box obstacle at {p}"
            elif name.startswith("net.corner"):
                d[name] = f"net corner (grab here) at {p}"
            elif name == "net.center":
                d[name] = f"centre of the fishing net at {p}"
            elif name == "boat":
                d[name] = f"boat deck, delivery zone radius {BOAT_RADIUS} m at {p}"
            elif name.startswith("fish_school"):
                d[name] = f"school of fish under water near {p}"
            elif name.startswith("drone"):
                d[name] = f"drone at {p}"
            else:
                d[name] = f"{name} at {p}"
        return d

    def _entity_positions(self) -> torch.Tensor:
        n = self.num_envs
        if self.scene == "trash":
            parts = [self.payload_handle()[:, None], self.pl_pos[:, None], self.zone_c[:, 0:1], self.zone_c[:, 1:2],
                     self.box_c[:, :spec.TRASH_N_OBSTACLES], self.pos[:, :1]]
        else:
            parts = [self.net_handles(), self.net.centroid()[:, None], self.zone_c[:, 0:1],
                     self.fish.school_centers(), self.zone_c[:, 1:2], self.pos]
        out = torch.cat(parts, 1)
        assert out.shape[:2] == (n, len(self._ent_names))
        return out

    def _entity_radius(self) -> torch.Tensor:
        r = torch.zeros(self.num_envs, len(self._ent_names), device=self.device)
        if self.scene == "trash":
            r[:, 2] = self.zone_r[:, 0]
            r[:, 3] = self.zone_r[:, 1]
        else:
            r[:, 5] = self.zone_r[:, 0]
            r[:, 6 + spec.N_FISH_SCHOOLS] = self.zone_r[:, 1]
        return r

    # ------------------------------------------------------------------ anchors
    def randomize_anchors(self, env_ids: torch.Tensor) -> None:
        m, a, K = len(env_ids), self.num_agents, spec.N_ANCHORS
        if m == 0:
            return
        kinds = self._ent_kind
        handles = torch.nonzero(kinds == ANCHOR_KIND["handle"]).squeeze(-1)
        zones = torch.nonzero(kinds == ANCHOR_KIND["zone"]).squeeze(-1)
        points = torch.nonzero(kinds == ANCHOR_KIND["point"]).squeeze(-1)
        src = torch.full((m, a, K), -1, dtype=torch.long, device=self.device)
        pick = lambda pool: pool[(self._rand(m, a) * len(pool)).long().clamp(max=len(pool) - 1)]
        use = lambda p: self._rand(m, a) < p
        src[..., 0] = torch.where(use(0.7), pick(handles), src[..., 0])
        if self.scene == "ocean":   # drone i usually gets corner i, as in the team task
            corner = torch.arange(a, device=self.device).expand(m, a) % 4
            src[..., 0] = torch.where(use(0.6), corner, src[..., 0])
        src[..., 1] = torch.where(use(0.5), pick(zones), src[..., 1])
        src[..., 2] = torch.where(use(0.5), pick(zones), src[..., 2])
        src[..., 3] = torch.where(use(0.4), pick(points), src[..., 3])
        pt = torch.cat([(self._rand(m, a, K, 2) - 0.5) * 2 * (spec.ARENA_HALF - 2),
                        1.0 + 3.0 * self._rand(m, a, K, 1)], -1)
        self.anc_src[env_ids] = src
        self.anc_point[env_ids] = pt
        kind = torch.where(src >= 0, kinds[src.clamp_min(0)], torch.full_like(src, ANCHOR_KIND["point"]))
        self.anc_kind[env_ids] = kind
        self.anc_radius[env_ids] = 0.0

    def bind_anchors(self, env_ids: torch.Tensor, bindings: List[AnchorBinding],
                     agent: Optional[int] = None) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        agents = range(self.num_agents) if agent is None else [agent]
        for k in range(spec.N_ANCHORS):
            b = bindings[k] if k < len(bindings) else None
            for ag in agents:
                if b is None or b.kind == "none":
                    self.anc_src[env_ids, ag, k] = -1
                    self.anc_kind[env_ids, ag, k] = ANCHOR_KIND["none"]
                    self.anc_point[env_ids, ag, k] = 0.0
                    self.anc_radius[env_ids, ag, k] = 0.0
                    continue
                if b.entity is not None:
                    if b.entity not in self._ent_names:
                        raise KeyError(f"unknown entity {b.entity!r}; known: {self._ent_names}")
                    idx = self._ent_names.index(b.entity)
                    self.anc_src[env_ids, ag, k] = idx
                    kind = b.kind if b.kind in ANCHOR_KIND else spec.ANCHOR_TYPES[int(self._ent_kind[idx])]
                else:
                    assert b.point is not None, "binding needs entity or point"
                    self.anc_src[env_ids, ag, k] = -1
                    self.anc_point[env_ids, ag, k] = torch.tensor(list(b.point), dtype=torch.float32, device=self.device)
                    kind = b.kind
                self.anc_kind[env_ids, ag, k] = ANCHOR_KIND[kind]
                self.anc_radius[env_ids, ag, k] = float(b.radius)

    def anchor_positions(self):
        """(pos [N, A, K, 3], radius [N, A, K], kind [N, A, K])."""
        ent = self._entity_positions()
        er = self._entity_radius()
        n, a, K = self.anc_src.shape
        src = self.anc_src.clamp_min(0)
        gpos = torch.gather(ent[:, None].expand(n, a, -1, 3), 2, src.unsqueeze(-1).expand(n, a, K, 3))
        grad = torch.gather(er[:, None].expand(n, a, -1), 2, src)
        bound = (self.anc_src >= 0)
        pos = torch.where(bound.unsqueeze(-1), gpos, self.anc_point)
        rad = torch.where(bound & (self.anc_radius == 0), grad, self.anc_radius)
        none = self.anc_kind == ANCHOR_KIND["none"]
        pos = torch.where(none.unsqueeze(-1), self.pos.unsqueeze(2).expand_as(pos), pos)
        return pos, rad, self.anc_kind

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        m, a = len(env_ids), self.num_agents
        if m > 0:
            self._reset_worlds(env_ids, m, a)
            self._reset_backend(env_ids)
            if self.cfg.lidar:
                self._scan(env_ids)
        return self._obs()

    def _reset_worlds(self, ids: torch.Tensor, m: int, a: int) -> None:
        dr = self.cfg.domain_rand
        H = spec.ARENA_HALF
        self.t[ids] = 0.0
        self.steps[ids] = 0
        self.vel[ids] = 0.0
        self.ang[ids] = 0.0
        self.quat[ids] = quat_from_yaw((self._rand(m, a) - 0.5) * 2 * math.pi)
        self.mass[ids] = spec.DRONE_MASS * (1.0 + (0.2 * (self._rand(m, a) - 0.5) if dr else 0.0))
        self.drag[ids] = 0.35 * (1.0 + (0.4 * (self._rand(m, a) - 0.5) if dr else 0.0))
        self.alive[ids] = True
        self.grip[ids] = False
        self.holding[ids] = -1
        self.last_action[ids] = 0.0
        self.hover[ids] = 1.0
        self.acc_prev[ids] = 0.0
        self.vel_prev[ids] = 0.0
        self.jerk[ids] = 0.0
        self.ext_force[ids] = 0.0
        self.v_int[ids] = 0.0
        self.collided[ids] = False
        self.shield_hit[ids] = False
        self.shield_count[ids] = 0
        ang = self._rand(m) * 2 * math.pi
        self.wind_mean[ids] = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * \
            (self._rand(m, 1) * self.cfg.max_wind if dr else 0.0)
        self.wind[ids] = self.wind_mean[ids]
        self.gust[ids] = 0.0
        self.gust_time[ids] = 0.0
        self.exp_phase[ids] = 0
        # no-fly zone
        self.zone_c[ids, 1] = torch.cat([(self._rand(m, 2) - 0.5) * 2 * (H - 4), torch.zeros(m, 1, device=self.device)], -1)
        self.zone_r[ids, 1] = 1.5 + 1.5 * self._rand(m)
        self.zone_active[ids, 1] = self._rand(m) < self.cfg.nofly_prob
        self.box_active[ids] = False
        self.box_moving[ids] = False
        if self.scene == "trash":
            self._reset_trash(ids, m, a)
        else:
            self._reset_ocean(ids, m, a)
        if self.cfg.random_anchors:
            self.randomize_anchors(ids)

    def _free_xy(self, m: int, avoid: List[torch.Tensor], min_d: float, half: float) -> torch.Tensor:
        """Sample xy [m, 2] at least min_d from every point in avoid (list of [m, k, 2])."""
        xy = (self._rand(m, 2) - 0.5) * 2 * half
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=self.device)
            for pts in avoid:
                if pts.shape[1]:
                    bad |= ((pts - xy[:, None]).norm(dim=-1) < min_d).any(-1)
            if not bad.any():
                break
            xy = torch.where(bad.unsqueeze(-1), (self._rand(m, 2) - 0.5) * 2 * half, xy)
        return xy

    def _reset_trash(self, ids, m, a) -> None:
        H = spec.ARENA_HALF
        nob = spec.TRASH_N_OBSTACLES
        half = torch.stack([0.4 + 1.1 * self._rand(m, nob), 0.4 + 1.1 * self._rand(m, nob),
                            0.75 + 1.25 * self._rand(m, nob)], -1)
        cxy = (self._rand(m, nob, 2) - 0.5) * 2 * (H - 2)
        self.box_h[ids, :nob] = half
        self.box_c[ids, :nob] = torch.cat([cxy, half[..., 2:3]], -1)
        self.box_active[ids, :nob] = True
        mv = torch.zeros(m, self.nb, dtype=torch.bool, device=self.device)
        mv[:, :spec.TRASH_N_MOVING] = True
        self.box_moving[ids] = mv
        self.orbit_c[ids] = torch.cat([cxy, torch.zeros(m, self.nb - nob, 2, device=self.device)], 1)
        self.orbit_r[ids] = 2.0 + 3.0 * self._rand(m, self.nb)
        self.orbit_w[ids] = (0.15 + 0.25 * self._rand(m, self.nb)) * torch.sign(self._rand(m, self.nb) - 0.5)
        self.orbit_ph[ids] = self._rand(m, self.nb) * 2 * math.pi
        self.box_h[ids, nob:] = 0.6
        self.box_c[ids, nob:] = 0.0
        self.box_v[ids] = 0.0
        moving_paths = self.orbit_c[ids, :, :] + self.orbit_r[ids].unsqueeze(-1) * torch.stack(
            [torch.cos(self.orbit_ph[ids]), torch.sin(self.orbit_ph[ids])], -1)
        self.box_c[ids, :spec.TRASH_N_MOVING, :2] = moving_paths[:, :spec.TRASH_N_MOVING]
        obst_xy = self.box_c[ids, :nob, :2]
        # bin
        bin_xy = self._free_xy(m, [obst_xy], 3.5, H - 3)
        self.zone_c[ids, 0] = torch.cat([bin_xy, torch.zeros(m, 1, device=self.device)], -1)
        self.zone_r[ids, 0] = BIN_RADIUS
        self.zone_active[ids, 0] = True
        # payload
        cooler = self._rand(m) < self.cfg.cooler_prob
        self.pl_kind[ids] = cooler.long()
        size = torch.where(cooler.unsqueeze(-1), torch.tensor([0.45, 0.3, 0.3], device=self.device),
                           torch.tensor([0.35, 0.35, 0.45], device=self.device))
        self.pl_size[ids] = size
        self.pl_mass[ids] = torch.where(cooler, 1.0 + 0.5 * self._rand(m), 0.3 + 0.7 * self._rand(m))
        pl_xy = self._free_xy(m, [obst_xy, bin_xy[:, None]], 3.0, H - 3)
        self.pl_pos[ids] = torch.cat([pl_xy, size[:, 2:3] / 2], -1)
        self.pl_vel[ids] = 0.0
        self.pl_held_by[ids] = -1
        self.swing[ids] = 0.0
        d_xy = self._free_xy(m, [obst_xy, pl_xy[:, None]], 3.0, H - 3)
        self.pos[ids] = torch.cat([d_xy[:, None].expand(m, a, 2), 1.5 + 1.5 * self._rand(m, a, 1)], -1)

    def _reset_ocean(self, ids, m, a) -> None:
        H = spec.ARENA_HALF
        self.waves.randomize(ids, self.gen)
        self.turbidity[ids] = 0.1 + 0.7 * self._rand(m)
        boat_xy = (self._rand(m, 2) - 0.5) * 2 * (H - 4)
        self.zone_c[ids, 0] = torch.cat([boat_xy, torch.zeros(m, 1, device=self.device)], -1)
        self.zone_r[ids, 0] = BOAT_RADIUS
        self.zone_active[ids, 0] = True
        self.box_h[ids, 0] = torch.tensor(BOAT_HALF, device=self.device)
        self.box_c[ids, 0] = self.zone_c[ids, 0]
        self.box_active[ids, 0] = True
        self.box_h[ids, 1:] = 0.6
        self.box_v[ids] = 0.0
        net_xy = self._free_xy(m, [boat_xy[:, None]], 9.0, H - 5)
        self.net.reset(ids, torch.cat([net_xy, torch.zeros(m, 1, device=self.device)], -1), self._rand(m) * 2 * math.pi)
        self.fish.reset(ids, self.gen)
        self.corner_held_by[ids] = -1
        self.net_stretch[ids] = 0.0
        self.fish_frac[ids] = 0.0
        # drone i spawns just outside net corner i (its default corner), so the
        # team's paths to their corners never cross
        k = torch.arange(a, device=self.device) % 4
        cxy = self.net.pos[ids][:, self.net.corners[k], :2]
        out = cxy - net_xy[:, None]
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_xy = cxy + out * (1.5 + 1.5 * self._rand(m, a, 1))
        self.pos[ids] = torch.cat([d_xy, 2.0 + 2.0 * self._rand(m, a, 1)], -1)
        self.t[ids] = self._rand(m) * 20.0   # random wave phase

    # ------------------------------------------------------------------ step
    def step(self, actions: torch.Tensor) -> StepOut:
        n, a = self.num_envs, self.num_agents
        actions = actions.to(self.device).float().clamp(-1.0, 1.0)
        assert actions.shape == (n, a, spec.ACTION_DIM), actions.shape
        actions = torch.where(self.alive.unsqueeze(-1), actions, torch.zeros_like(actions))
        events = {k: torch.zeros(n, a, dtype=torch.bool, device=self.device) for k in spec.EVENT_KEYS}
        self._update_grip(actions, events)
        v_cmd, _ = VelocityController.command(actions)
        if self.cfg.shield:
            v_cmd, hit = self._shield(v_cmd)
        else:
            hit = torch.zeros(n, a, dtype=torch.bool, device=self.device)
        self.shield_hit = hit & self.alive
        self.shield_count += self.shield_hit.sum(-1)
        self.collided = torch.zeros(n, a, dtype=torch.bool, device=self.device)
        crash = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(spec.SUBSTEPS):
            self._move_obstacles(spec.PHYS_DT)
            self._update_wind(spec.PHYS_DT)
            mass = self.mass + self._held_mass()
            self.v_int = (self.v_int + spec.PHYS_DT * (v_cmd - self.vel)).clamp(-2.0, 2.0)
            v_eff = v_cmd + VEL_KI * self.v_int
            thrust, torque, hover = self.ctrl(actions, self.quat, self.vel, self.ang, mass=mass, v_des=v_eff)
            self.hover = hover
            bz = quat_rotate(self.quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.pos))
            rel_air = self.vel - (self.wind + self.gust).unsqueeze(1)
            force = thrust.unsqueeze(-1) * bz - self.drag.unsqueeze(-1) * rel_air * rel_air.norm(dim=-1, keepdim=True).clamp_min(1.0)
            force[..., 2] -= self.mass * GRAVITY
            force = force + self.ext_force
            force = torch.where(self.alive.unsqueeze(-1), force, torch.zeros_like(force))
            self._integrate_drones(force, torque, spec.PHYS_DT)
            if self.scene == "trash":
                self._payload_substep(spec.PHYS_DT)
            self._sync_kinematics()
            crash |= self._collisions()
        if self.scene == "ocean":
            self._ocean_policy_step(events)
        self.t += spec.POLICY_DT
        self.steps += 1
        # jerk at the policy rate
        acc = (self.vel - self.vel_prev) / spec.POLICY_DT
        self.jerk = (acc - self.acc_prev).norm(dim=-1) / spec.POLICY_DT
        self.acc_prev, self.vel_prev = acc, self.vel.clone()
        self.last_action = actions
        if self.cfg.lidar:
            due = (self.steps % spec.LIDAR_EVERY) == 0
            if due.any():
                self._scan(torch.nonzero(due).squeeze(-1))
        events["collision"] = self.collided
        events["shield"] = self.shield_hit
        if self.scene == "trash":
            events["delivered"] |= self._trash_delivered().unsqueeze(-1).expand(n, a)
        torn = self.net.torn if self.net is not None else torch.zeros(n, dtype=torch.bool, device=self.device)
        events["tear"] |= torn.unsqueeze(-1).expand(n, a)
        out_of_bounds = ((self.pos[..., :2].abs() > spec.ARENA_HALF + 5).any(-1) | (self.pos[..., 2] > 25)) & self.alive
        done = crash | torn | out_of_bounds.any(-1) | ~self.alive.any(-1)
        truncated = (self.steps >= spec.EPISODE_STEPS) & ~done
        return StepOut(obs=self._obs(), done=done, truncated=truncated, events=events,
                       info={"shield_count": self.shield_count.clone()})

    # ------------------------------------------------------------------ dynamics pieces
    def _held_mass(self) -> torch.Tensor:
        if self.scene != "trash":
            return torch.zeros_like(self.mass)
        held = self.holding == 0
        return torch.where(held, self.pl_mass.unsqueeze(-1).expand_as(self.mass), torch.zeros_like(self.mass))

    def _move_obstacles(self, dt: float) -> None:
        if not self.box_moving.any():
            return
        self.orbit_ph = self.orbit_ph + self.orbit_w * dt
        tgt = self.orbit_c + self.orbit_r.unsqueeze(-1) * torch.stack([torch.cos(self.orbit_ph), torch.sin(self.orbit_ph)], -1)
        vel = torch.cat([(tgt - self.box_c[..., :2]) / dt, torch.zeros_like(tgt[..., :1])], -1)
        mv = self.box_moving.unsqueeze(-1)
        self.box_v = torch.where(mv, vel, torch.zeros_like(vel))
        self.box_c[..., :2] = torch.where(mv, tgt, self.box_c[..., :2])

    def _update_wind(self, dt: float) -> None:
        theta, sigma = 0.5, 0.8 if self.cfg.domain_rand else 0.0
        self.wind = self.wind + theta * (self.wind_mean - self.wind) * dt + \
            sigma * math.sqrt(dt) * self._randn(self.num_envs, 3) * torch.tensor([1.0, 1.0, 0.2], device=self.device)
        self.gust_time = (self.gust_time - dt).clamp_min(0.0)
        self.gust = torch.where((self.gust_time > 0).unsqueeze(-1), self.gust, torch.zeros_like(self.gust))

    def _payload_substep(self, dt: float) -> None:
        held = self.pl_held_by >= 0
        free = ~held
        self._integrate_free_payload(dt, free)
        self.ext_force = torch.zeros_like(self.ext_force)
        if not held.any():
            self.swing = torch.zeros_like(self.swing)
            return
        # spherical pendulum hanging from the gripper (position-based)
        ar = torch.arange(self.num_envs, device=self.device)
        holder = self.pl_held_by.clamp_min(0)
        gp = self.gripper_pos()[ar, holder]
        gv = self.vel[ar, holder]
        L = (self.pl_size[:, 2] / 2 + HANDLE_HEIGHT).unsqueeze(-1)
        v = self.pl_vel.clone()
        v[:, 2] -= GRAVITY * dt
        rel_air = v - (self.wind + self.gust)
        v = v - dt * 0.4 * rel_air / self.pl_mass.unsqueeze(-1)
        p = self.pl_pos + dt * v
        d = p - gp
        dn = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        p_new = gp + d / dn * L
        floor = self.pl_size[:, 2] / 2
        p_new[:, 2] = torch.maximum(p_new[:, 2], floor)
        v_new = (p_new - self.pl_pos) / dt
        acc = (v_new - self.pl_vel) / dt
        tension = self.pl_mass.unsqueeze(-1) * (acc - torch.tensor([0.0, 0.0, -GRAVITY], device=self.device))
        on_ground = p_new[:, 2] <= floor + 1e-4
        tension = torch.where(on_ground.unsqueeze(-1), tension * 0.0, tension)
        tension = tension.clamp(-3 * GRAVITY * spec.DRONE_MASS, 3 * GRAVITY * spec.DRONE_MASS)
        h = held.unsqueeze(-1)
        self.pl_pos = torch.where(h, p_new, self.pl_pos)
        self.pl_vel = torch.where(h, v_new, self.pl_vel)
        rel = (self.pl_pos - gp)
        relv = self.pl_vel - gv
        self.swing = torch.where(held, torch.cross(rel, relv, dim=-1).norm(dim=-1) / (L.squeeze(-1) ** 2), torch.zeros_like(self.swing))
        f = torch.zeros_like(self.ext_force)
        f[ar, holder] = -tension
        self.ext_force = torch.where(held[:, None, None], f, self.ext_force)

    def _ocean_policy_step(self, events) -> None:
        n, a = self.num_envs, self.num_agents
        ar = torch.arange(n, device=self.device)
        P = self.net.p
        pinned = torch.zeros(n, P, dtype=torch.bool, device=self.device)
        pin_pos = self.net.pos.clone()
        gp = self.gripper_pos()
        for c in range(4):
            holder = self.corner_held_by[:, c]
            held = holder >= 0
            node = int(self.net.corners[c])
            pinned[:, node] = held
            line = gp[ar, holder.clamp_min(0)] - torch.tensor([0.0, 0.0, NET_HANDLE_HEIGHT], device=self.device)
            pin_pos[:, node] = torch.where(held.unsqueeze(-1), line, pin_pos[:, node])
        wh = self.waves.height(self.net.pos[..., :2], self.t)
        wv = self.waves.water_velocity(self.net.pos, self.t)
        pin_force, stretch = self.net.step(spec.POLICY_DT, pinned, pin_pos, wh, wv)
        self.net_stretch = stretch.clamp_min(0.0)
        f = torch.zeros(n, a, 3, device=self.device)
        for c in range(4):
            holder = self.corner_held_by[:, c]
            held = holder >= 0
            node = int(self.net.corners[c])
            fc = torch.where(held.unsqueeze(-1), pin_force[:, node], torch.zeros(n, 3, device=self.device))
            f[ar, holder.clamp_min(0)] += fc
        lim = 2.0 * GRAVITY * spec.DRONE_MASS
        self.ext_force = f.clamp(-lim, lim)
        surf = self.waves.height(self.fish.pos[..., :2], self.t)
        self.fish.step(spec.POLICY_DT, torch.where(self.alive.unsqueeze(-1), self.pos, torch.full_like(self.pos, 1e3)),
                       surf, self.net.pos)
        inside = self.net.contains(self.fish.pos)
        net_held = (self.corner_held_by >= 0).any(-1)
        self.fish.update_capture(inside, self.net.pos, self.net.vel, net_held)
        self.fish_frac = self.fish.caught.float().mean(-1)
        # boat bobs on the waves
        bz = self.waves.height(self.zone_c[:, 0:1, :2], self.t)[:, 0]
        self.zone_c[:, 0, 2] = bz
        self.box_c[:, 0, 2] = bz
        cen = self.net.centroid()
        over_boat = (cen[:, :2] - self.zone_c[:, 0, :2]).norm(dim=-1) < BOAT_RADIUS
        lifted = cen[:, 2] > bz + 0.2
        delivered = over_boat & lifted & (self.fish_frac > 0)
        events["delivered"] |= delivered.unsqueeze(-1).expand(n, a)

    def _update_grip(self, actions: torch.Tensor, events) -> None:
        closed = (actions[..., 4] > spec.GRIP_CLOSED_THRESHOLD) & self.alive
        closing = closed & ~self.grip
        opening = ~closed & self.grip
        self.grip = closed
        n, a = self.num_envs, self.num_agents
        ar = torch.arange(n, device=self.device)
        # release
        if opening.any():
            rel = opening & (self.holding >= 0)
            events["release"] |= rel
            if self.scene == "trash":
                dropped = rel[:, 0] if a == 1 else rel.any(-1)
                self.pl_held_by = torch.where(dropped, torch.full_like(self.pl_held_by, -1), self.pl_held_by)
            else:
                for c in range(4):
                    holder = self.corner_held_by[:, c]
                    rel_c = (holder >= 0) & rel[ar, holder.clamp_min(0)]
                    self.corner_held_by[:, c] = torch.where(rel_c, torch.full_like(holder, -1), holder)
            self.holding = torch.where(rel, torch.full_like(self.holding, -1), self.holding)
        if not closing.any():
            return
        gp = self.gripper_pos()
        if self.scene == "trash":
            h = self.payload_handle()
            free = self.pl_held_by < 0
            d = (gp - h[:, None]).norm(dim=-1)
            rv = (self.vel - self.pl_vel[:, None]).norm(dim=-1)
            ok = closing & free.unsqueeze(-1) & (d < PAYLOAD_GRASP_RADIUS) & (rv < spec.GRASP_MAX_REL_SPEED) & (self.holding < 0)
            # only one agent can take it: the first eligible
            first = ok & (torch.cumsum(ok.long(), -1) == 1)
            grabbed = first.any(-1)
            self.holding = torch.where(first, torch.zeros_like(self.holding), self.holding)
            self.pl_held_by = torch.where(grabbed, first.long().argmax(-1), self.pl_held_by)
            events["grasp"] |= first
        else:
            corners = self.net_handles()                                    # [N, 4, 3]
            d = (gp[:, :, None] - corners[:, None]).norm(dim=-1)           # [N, A, 4]
            free = (self.corner_held_by < 0)[:, None].expand(n, a, 4)
            d = torch.where(free, d, torch.full_like(d, 1e3))
            best = d.argmin(-1)
            dbest = d.min(-1).values
            ok = closing & (dbest < NET_GRASP_RADIUS) & (self.holding < 0)
            for ag in range(a):   # sequential so two drones cannot take one corner
                c = best[:, ag]
                still_free = self.corner_held_by[ar, c] < 0
                take = ok[:, ag] & still_free
                self.corner_held_by[ar, c] = torch.where(take, torch.full_like(c, ag), self.corner_held_by[ar, c])
                self.holding[:, ag] = torch.where(take, c, self.holding[:, ag])
                events["grasp"][:, ag] |= take

    def _shield(self, v_cmd: torch.Tensor):
        cyl_c = self.zone_c[:, 1:2]
        cyl_r = self.zone_r[:, 1:2]
        cyl_a = self.zone_active[:, 1:2]
        return shield_velocity(self.pos, v_cmd, cyl_c, cyl_r, cyl_a,
                               self.box_c, self.box_h, self.box_active, margin=0.6)

    def _collisions(self) -> torch.Tensor:
        """Mark collided drones and return crashed worlds [N]."""
        r = spec.DRONE_RADIUS
        alive = self.alive
        # obstacles
        rel = self.pos[:, :, None] - self.box_c[:, None]
        closest = torch.maximum(torch.minimum(rel, self.box_h[:, None]), -self.box_h[:, None])
        dist = (rel - closest).norm(dim=-1)
        hit_box = ((dist < r) & self.box_active[:, None]).any(-1)
        # other drones
        dd = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        both = alive[:, :, None] & alive[:, None]
        hit_mate = ((dd < 2 * r) & ~eye & both).any(-1)
        # ground or water
        surf = self.surface_height(self.pos[..., :2])
        if self.scene == "trash":
            low = self.pos[..., 2] - r < surf
            hard = low & (self.vel[..., 2] < -2.0)
            # gentle landing: rest on the ground
            self.pos[..., 2] = torch.where(low & ~hard & alive, surf + r, self.pos[..., 2])
            self.vel[..., 2] = torch.where(low & ~hard & alive, self.vel[..., 2].clamp_min(0.0), self.vel[..., 2])
            hit_ground = hard
        else:
            hit_ground = self.pos[..., 2] < surf + 0.05
        hit = (hit_box | hit_mate | hit_ground) & alive
        self.collided |= hit
        return hit.any(-1)

    def _trash_delivered(self) -> torch.Tensor:
        free = self.pl_held_by < 0
        in_bin = (self.pl_pos[:, :2] - self.zone_c[:, 0, :2]).norm(dim=-1) < self.zone_r[:, 0]
        low = self.pl_pos[:, 2] < self.pl_size[:, 2] / 2 + 0.05
        return free & in_bin & low

    # ------------------------------------------------------------------ sensing
    def _geometry(self) -> Dict[str, torch.Tensor]:
        geom = {"box_c": self.box_c, "box_h": self.box_h, "box_active": self.box_active}
        sph_c, sph_r, sph_a = [], [], []
        far = torch.full_like(self.pos, 1e4)
        sph_c.append(torch.where(self.alive.unsqueeze(-1), self.pos, far))
        sph_r.append(torch.full(self.pos.shape[:2], spec.DRONE_RADIUS * 0.8, device=self.device))
        if self.scene == "trash":
            geom["ground_z"] = torch.zeros(self.num_envs, device=self.device)
            sph_c.append(self.pl_pos[:, None])
            sph_r.append(self.pl_size[:, None, :].max(-1).values / 2)
        else:
            geom["ground_z"] = torch.full((self.num_envs,), -12.0, device=self.device)   # seabed
            sph_c.append(self.fish.pos)
            sph_r.append(torch.full(self.fish.pos.shape[:2], self.fish.radius, device=self.device))
            sph_c.append(self.net.pos)
            sph_r.append(torch.full(self.net.pos.shape[:2], 0.12, device=self.device))
        geom["sph_c"] = torch.cat(sph_c, 1)
        geom["sph_r"] = torch.cat(sph_r, 1)
        return geom

    def _scan(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        geom = {k: v[env_ids] for k, v in self._geometry().items()}
        ext, ext_d = self._solid_ranges()
        imgs, downs, pts = [], [], None
        for ag in range(self.num_agents):
            g = dict(geom)
            if self._drone_spheres_first:      # a drone must not see its own sphere
                sc = g["sph_c"].clone()
                sc[:, ag] = 1e4
                g["sph_c"] = sc
            waves = None
            if self.waves is not None:
                waves = _SubWaves(self.waves, env_ids)
            img, down, p = self.lidar_model.scan(
                self.pos[env_ids, ag:ag + 1], self.quat[env_ids, ag:ag + 1], g, waves=waves,
                t=self.t[env_ids], turbidity=self.turbidity[env_ids],
                solid_range=None if ext is None else ext[env_ids, ag:ag + 1].reshape(len(env_ids), 1, -1),
                solid_range_down=None if ext_d is None else ext_d[env_ids, ag:ag + 1].reshape(len(env_ids), 1, -1))
            imgs.append(img)
            downs.append(down)
            if ag == 0:
                pts = p
        self._lidar[env_ids] = torch.cat(imgs, 1)
        self._lidar_down[env_ids] = torch.cat(downs, 1)
        for k, v in pts.items():
            if k not in self._lidar_pts:
                self._lidar_pts[k] = torch.full((self.num_envs,) + v.shape[1:], float("nan"), device=self.device)
            self._lidar_pts[k][env_ids] = v

    # ------------------------------------------------------------------ observations
    def observe(self) -> Dict[str, torch.Tensor]:
        """Current observation for every world, without stepping."""
        return self._obs()

    def _obs(self) -> Dict[str, torch.Tensor]:
        n, a = self.num_envs, self.num_agents
        dev = self.device
        obs = spec.empty_obs(n, a, dev)
        grav_b = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand_as(self.pos))
        surf = self.surface_height(self.pos[..., :2])
        height = self.pos[..., 2] - surf
        holding = (self.holding >= 0).float()
        obs["proprio"] = torch.cat([self.vel, grav_b, self.ang, self.grip.float().unsqueeze(-1),
                                    holding.unsqueeze(-1), height.unsqueeze(-1), self.last_action], -1)
        apos, arad, akind = self.anchor_positions()
        rel = apos - self.pos.unsqueeze(2)
        inside = self._anchor_inside(apos, arad, akind)
        onehot = torch.nn.functional.one_hot(akind, len(spec.ANCHOR_TYPES)).float()
        obs["anchors"] = torch.cat([rel, onehot, arad.unsqueeze(-1), inside.float().unsqueeze(-1)], -1)
        if a > 1:
            for i in range(a):
                others = [j for j in range(a) if j != i][:spec.MAX_MATES]
                o = torch.tensor(others, device=dev)
                obs["mates"][:, i, :len(others)] = torch.cat([
                    self.pos[:, o] - self.pos[:, i:i + 1], self.vel[:, o] - self.vel[:, i:i + 1],
                    (self.holding[:, o] >= 0).float().unsqueeze(-1)], -1)
                obs["mates_mask"][:, i, :len(others)] = self.alive[:, o]
        obs["lidar"] = self._lidar.clone()
        obs["lidar_down"] = self._lidar_down.clone()
        priv, mask = self._priv_tokens()
        obs["priv"], obs["priv_mask"] = priv, mask
        obs["g"] = self._goal_space(apos, inside, height)
        return obs

    def _anchor_inside(self, apos, arad, akind) -> torch.Tensor:
        d_xy = (self.pos.unsqueeze(2)[..., :2] - apos[..., :2]).norm(dim=-1)
        d3 = (self.pos.unsqueeze(2) - apos).norm(dim=-1)
        zone = arad > 0
        inside = torch.where(zone, d_xy < arad, d3 < 0.5)
        return inside & (akind != ANCHOR_KIND["none"])

    def _priv_tokens(self):
        n, a = self.num_envs, self.num_agents
        dev = self.device
        rows = []   # each: (pos [N,3], vel [N,3], size [N,3], type, active [N])
        oh = lambda t: torch.nn.functional.one_hot(torch.tensor(PRIV_TYPE[t], device=dev), len(spec.PRIV_TYPES)).float()
        for b in range(self.nb):
            t = "boat" if (self.scene == "ocean" and b == 0) else "obstacle"
            rows.append((self.box_c[:, b], self.box_v[:, b], self.box_h[:, b] * 2, t, self.box_active[:, b]))
        if self.scene == "trash":
            rows.append((self.pl_pos, self.pl_vel, self.pl_size, "payload", torch.ones(n, dtype=torch.bool, device=dev)))
            r = self.zone_r[:, 0]
            rows.append((self.zone_c[:, 0], torch.zeros(n, 3, device=dev), torch.stack([r, r, torch.full_like(r, 1.0)], -1),
                         "zone", self.zone_active[:, 0]))
        else:
            ext = self.net.pos.amax(1) - self.net.pos.amin(1)
            rows.append((self.net.centroid(), self.net.vel.mean(1), ext, "net", torch.ones(n, dtype=torch.bool, device=dev)))
            for s in range(spec.N_FISH_SCHOOLS):
                c = self.fish.school_centers()[:, s]
                rows.append((c, torch.zeros(n, 3, device=dev), torch.tensor([2.0, 2.0, 1.0], device=dev).expand(n, 3),
                             "fish_school", torch.ones(n, dtype=torch.bool, device=dev)))
        r = self.zone_r[:, 1]
        rows.append((self.zone_c[:, 1], torch.zeros(n, 3, device=dev), torch.stack([r, r, torch.full_like(r, 10.0)], -1),
                     "zone", self.zone_active[:, 1]))
        rows = rows[:spec.MAX_PRIV]
        priv = torch.zeros(n, a, spec.MAX_PRIV, spec.PRIV_DIM, device=dev)
        mask = torch.zeros(n, a, spec.MAX_PRIV, dtype=torch.bool, device=dev)
        for i, (p, v, s, t, act) in enumerate(rows):
            relp = p.unsqueeze(1) - self.pos
            priv[:, :, i] = torch.cat([relp, v.unsqueeze(1).expand(n, a, 3), s.unsqueeze(1).expand(n, a, 3),
                                       oh(t).expand(n, a, -1)], -1)
            mask[:, :, i] = act.unsqueeze(-1).expand(n, a)
        return priv, mask

    def _goal_space(self, apos, inside, height) -> torch.Tensor:
        n, a = self.num_envs, self.num_agents
        dev = self.device
        g = torch.zeros(n, a, spec.G_DIM, device=dev)
        S = spec.G_SLICES
        g[..., S["drone_pos"]] = self.pos
        g[..., S["drone_vel"]] = self.vel
        g[..., S["speed"]] = self.vel.norm(dim=-1, keepdim=True)
        g[..., S["tilt"]] = tilt_of(self.quat).unsqueeze(-1)
        g[..., S["height"]] = height.unsqueeze(-1)
        g[..., S["thrust"]] = self.hover.unsqueeze(-1)
        g[..., S["grip_closed"]] = self.grip.float().unsqueeze(-1)
        holding = self.holding >= 0
        g[..., S["holding"]] = holding.float().unsqueeze(-1)
        if self.scene == "trash":
            ppos = self.pl_pos.unsqueeze(1).expand(n, a, 3)
            pvel = self.pl_vel.unsqueeze(1).expand(n, a, 3)
            g[..., S["held_mass"]] = torch.where(holding, self.pl_mass.unsqueeze(-1), torch.zeros_like(self.mass)).unsqueeze(-1)
            g[..., S["swing"]] = torch.where(holding, self.swing.unsqueeze(-1), torch.zeros_like(self.mass)).unsqueeze(-1)
        else:
            corners = self.net_handles()
            cvel = self.net.vel[:, self.net.corners]
            d = (self.gripper_pos()[:, :, None] - corners[:, None]).norm(dim=-1)
            idx = torch.where(holding, self.holding, d.argmin(-1))
            ppos = torch.gather(corners, 1, idx.unsqueeze(-1).expand(n, a, 3))
            pvel = torch.gather(cvel, 1, idx.unsqueeze(-1).expand(n, a, 3))
            g[..., S["net_centroid_rel"]] = self.net.centroid().unsqueeze(1) - self.pos
            g[..., S["net_stretch"]] = self.net_stretch.view(n, 1, 1).expand(n, a, 1)
            g[..., S["fish_in_net"]] = self.fish_frac.view(n, 1, 1).expand(n, a, 1)
        g[..., S["payload_rel"]] = ppos - self.pos
        g[..., S["payload_vel"]] = pvel
        g[..., S["jerk"]] = self.jerk.unsqueeze(-1).clamp(max=500.0)
        g[..., S["anchor_rel_drone"]] = (apos - self.pos.unsqueeze(2)).reshape(n, a, -1)
        g[..., S["anchor_rel_payload"]] = (apos - ppos.unsqueeze(2)).reshape(n, a, -1)
        g[..., S["anchor_inside"]] = inside.float()
        # nearest obstacle
        rel = self.pos[:, :, None] - self.box_c[:, None]
        closest = torch.maximum(torch.minimum(rel, self.box_h[:, None]), -self.box_h[:, None])
        vec = rel - closest
        dist = vec.norm(dim=-1) - spec.DRONE_RADIUS
        dist = torch.where(self.box_active[:, None], dist, torch.full_like(dist, 2 * spec.ARENA_HALF))
        dmin, imin = dist.min(-1)
        nvec = torch.gather(vec, 2, imin[..., None, None].expand(n, a, 1, 3)).squeeze(2)
        nrm = nvec / nvec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bvel = torch.gather(self.box_v[:, None].expand(n, a, -1, 3), 2, imin[..., None, None].expand(n, a, 1, 3)).squeeze(2)
        g[..., S["obstacle_dist"]] = dmin.clamp(max=2 * spec.ARENA_HALF).unsqueeze(-1)
        g[..., S["obstacle_closing"]] = (-((self.vel - bvel) * nrm).sum(-1)).unsqueeze(-1)
        if a > 1:
            dd = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
            eye = torch.eye(a, dtype=torch.bool, device=dev)
            ok = self.alive[:, None] & ~eye
            dd = torch.where(ok, dd, torch.full_like(dd, 2 * spec.ARENA_HALF))
            g[..., S["mate_dist"]] = dd.min(-1).values.unsqueeze(-1)
        else:
            g[..., S["mate_dist"]] = 2 * spec.ARENA_HALF
        g[..., S["collision"]] = self.collided.float().unsqueeze(-1)
        g[..., S["shield"]] = self.shield_hit.float().unsqueeze(-1)
        return g

    # ------------------------------------------------------------------ state copy (CEM)
    _STATE = ("t", "steps", "pos", "vel", "ang", "quat", "mass", "drag", "alive", "grip", "holding", "last_action",
              "hover", "acc_prev", "vel_prev", "jerk", "ext_force", "v_int", "collided", "shield_hit", "shield_count",
              "wind", "wind_mean", "gust", "gust_time", "box_c", "box_h", "box_v", "box_active", "box_moving",
              "orbit_c", "orbit_r", "orbit_w", "orbit_ph", "zone_c", "zone_r", "zone_active", "pl_pos", "pl_vel",
              "pl_size", "pl_mass", "pl_kind", "pl_held_by", "swing", "turbidity", "net_stretch", "fish_frac",
              "corner_held_by", "anc_kind", "anc_src", "anc_point", "anc_radius", "exp_phase", "_lidar", "_lidar_down")

    def get_state(self) -> Dict[str, torch.Tensor]:
        st = {k: getattr(self, k).clone() for k in self._STATE}
        for name in ("waves", "net", "fish"):
            sub = getattr(self, name)
            if sub is not None:
                for k, v in sub.state().items():
                    st[f"{name}.{k}"] = v.clone()
        return st

    def set_state(self, state: Dict[str, torch.Tensor], env_ids: Optional[torch.Tensor] = None,
                  src_ids: Optional[torch.Tensor] = None) -> None:
        """Write rows src_ids of state into worlds env_ids (default: all, same rows)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        src = env_ids if src_ids is None else torch.as_tensor(src_ids, device=self.device).reshape(-1)
        for k in self._STATE:
            getattr(self, k)[env_ids] = state[k][src]
        for name in ("waves", "net", "fish"):
            sub = getattr(self, name)
            if sub is not None:
                sub.load({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(name + ".")}, env_ids, src)
        self._reset_backend(env_ids)

    def copy_state(self, src: int, dst_ids: torch.Tensor) -> None:
        dst_ids = torch.as_tensor(dst_ids, device=self.device).reshape(-1)
        st = self.get_state()
        self.set_state(st, dst_ids, torch.full_like(dst_ids, int(src)))

    # ------------------------------------------------------------------ expert
    def expert_action(self) -> torch.Tensor:
        from .expert import expert_action
        return expert_action(self)

    # ------------------------------------------------------------------ demo
    def perturb(self, env_id: int, kind: str, **kw) -> None:
        i = int(env_id)
        dev = self.device
        if kind == "drop_obstacle":
            b = self.nb - 1
            p = torch.tensor(kw.get("pos", [0.0, 0.0]), dtype=torch.float32, device=dev)[:2]
            half = float(kw.get("half", 0.6))
            self.box_h[i, b] = half
            base = self.surface_height(p.view(1, 1, 2).expand(self.num_envs, 1, 2))[i, 0] if self.scene == "ocean" else 0.0
            self.box_c[i, b] = torch.cat([p, torch.tensor([float(base) + half], device=dev)])
            self.box_active[i, b] = True
        elif kind == "gust":
            v = torch.tensor(kw.get("vec", [4.0, 0.0, 0.0]), dtype=torch.float32, device=dev)
            self.gust[i] = v
            self.gust_time[i] = float(kw.get("duration", 2.0))
        elif kind == "waves":
            if self.waves is not None:
                self.waves.scale[i] = float(kw.get("scale", 1.5))
        elif kind == "move_payload":
            if self.scene == "trash":
                p = torch.tensor(kw.get("pos", [0.0, 0.0]), dtype=torch.float32, device=dev)[:2]
                self.pl_held_by[i] = -1
                self.holding[i] = -1
                self.pl_pos[i] = torch.cat([p, self.pl_size[i, 2:3] / 2])
                self.pl_vel[i] = 0.0
        elif kind == "disable_drone":
            ag = int(kw.get("agent", 0))
            self.alive[i, ag] = False
            self.holding[i, ag] = -1
            if self.corner_held_by is not None:
                self.corner_held_by[i] = torch.where(self.corner_held_by[i] == ag, torch.full_like(self.corner_held_by[i], -1),
                                                     self.corner_held_by[i])
            if self.scene == "trash" and int(self.pl_held_by[i]) == ag:
                self.pl_held_by[i] = -1
            self.pos[i, ag] = torch.tensor([0.0, 0.0, -50.0], device=dev)
            self.vel[i, ag] = 0.0
        else:
            raise ValueError(f"unknown perturbation {kind!r}")
        self._reset_backend(torch.tensor([i], device=dev))

    def render_state(self, env_id: int) -> dict:
        i = int(env_id)
        L = lambda x: [round(float(v), 4) for v in x.reshape(-1).tolist()]
        st = {"t": float(self.t[i]), "scene": self.scene, "shield_hits": int(self.shield_count[i]),
              "wind": L(self.wind[i] + self.gust[i])}
        st["drones"] = [{"pos": L(self.pos[i, j]), "quat": L(self.quat[i, j]), "grip": bool(self.grip[i, j]),
                         "holding": bool(self.holding[i, j] >= 0), "alive": bool(self.alive[i, j])}
                        for j in range(self.num_agents)]
        st["obstacles"] = [{"name": f"obstacle_{b}" if not (self.scene == "ocean" and b == 0) else "boat_hull",
                            "pos": L(self.box_c[i, b]), "size": L(self.box_h[i, b] * 2), "moving": bool(self.box_moving[i, b]),
                            "dropped": b >= self.nb - EXTRA_OBSTACLE_SLOTS}
                           for b in range(self.nb) if bool(self.box_active[i, b]) and not (self.scene == "ocean" and b == 0)]
        zones = []
        if self.scene == "trash":
            zones.append({"name": "bin", "pos": L(self.zone_c[i, 0]), "radius": float(self.zone_r[i, 0]), "height": 1.0, "kind": "bin"})
            st["payloads"] = [{"name": "payload", "pos": L(self.pl_pos[i]), "quat": [1, 0, 0, 0], "size": L(self.pl_size[i]),
                               "kind": "cooler" if int(self.pl_kind[i]) == 1 else "bag"}]
            st.update(waves=None, boat=None, net=None, fish=[])
        else:
            zones.append({"name": "boat", "pos": L(self.zone_c[i, 0]), "radius": float(self.zone_r[i, 0]), "height": 1.0, "kind": "boat"})
            st["payloads"] = []
            st["waves"] = self.waves.params(i)
            st["boat"] = {"pos": L(self.zone_c[i, 0]), "quat": [1, 0, 0, 0], "size": [2 * x for x in BOAT_HALF]}
            st["net"] = {"rows": self.net.rows, "cols": self.net.cols,
                         "nodes": [L(p) for p in self.net.pos[i]],
                         "held": [int((self.corner_held_by[i] == j).nonzero()[0]) if bool((self.corner_held_by[i] == j).any()) else -1
                                  for j in range(self.num_agents)]}
            st["fish"] = [L(p) for p in self.fish.pos[i]]
        if bool(self.zone_active[i, 1]):
            zones.append({"name": "nofly_0", "pos": L(self.zone_c[i, 1]), "radius": float(self.zone_r[i, 1]), "height": 20.0, "kind": "nofly"})
        st["zones"] = zones
        apos, arad, akind = self.anchor_positions()
        st["anchors"] = [{"pos": L(apos[i, 0, k]), "kind": spec.ANCHOR_TYPES[int(akind[i, 0, k])], "radius": float(arad[i, 0, k])}
                         for k in range(spec.N_ANCHORS)]
        lid = {}
        for k in ("nir", "green"):
            if k in self._lidar_pts:
                p = self._lidar_pts[k][i]
                p = p[torch.isfinite(p).all(-1)]
                lid[k] = [L(x) for x in p]
            else:
                lid[k] = []
        st["lidar"] = lid
        return st


class _SubWaves:
    """View of a WaveField restricted to a subset of worlds (for partial LiDAR scans)."""

    def __init__(self, w: WaveField, ids: torch.Tensor):
        self.w, self.ids = w, ids

    def _swap(self):
        w, ids = self.w, self.ids
        return {k: getattr(w, k) for k in ("amp", "k", "dir", "omega", "phase", "scale")}, \
            {k: getattr(w, k)[ids] for k in ("amp", "k", "dir", "omega", "phase", "scale")}

    def _call(self, fn, *args):
        full, sub = self._swap()
        for k, v in sub.items():
            setattr(self.w, k, v)
        try:
            return fn(*args)
        finally:
            for k, v in full.items():
                setattr(self.w, k, v)

    def height(self, xy, t):
        return self._call(self.w.height, xy, t)

    def ray_hit(self, o, d, t):
        return self._call(self.w.ray_hit, o, d, t)


def make_env(scene: str = "trash", num_envs: int = 64, device: str = "cpu", backend: str = "torch", **kw):
    """Factory. backend="torch" runs anywhere PyTorch runs; backend="genesis" needs genesis-world."""
    cfg = EnvConfig(scene=scene, num_envs=num_envs, device=device)
    for k, v in kw.items():
        setattr(cfg, k, v)
    if backend == "torch":
        return SkyEnv(cfg)
    if backend == "genesis":
        from .backends.genesis_backend import GenesisSkyEnv
        return GenesisSkyEnv(cfg)
    raise ValueError(f"unknown backend {backend!r}")
