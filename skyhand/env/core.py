"""SkyEnv: batched environment with a pure-torch physics backend.

Runs anywhere PyTorch runs, with no Genesis install. The Genesis backend
subclasses it and overrides only the drone rigid-body hooks.

Scenes (see skyhand/spec.py)
----------------------------
trash:    each world is a city (street grid, cars, trash piles at the curb, a
          garbage truck) or open terrain with scattered litter (rolling hills, lakes
          with a tide, trees, rocks, birds). Trash and non-trash items can be grasped.
delivery: organ cooler from pad_pickup to pad_dropoff across rugged terrain
          (mountains, valleys, lakes), birds, no-fly zones.
ocean:    team net fishing with fish schools, whales, dolphins, turtles (bycatch).

Contract: skyhand/spec.py. step() does not auto-reset; call reset(env_ids) for
worlds that are done or truncated after storing their transitions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .. import spec
from ..spec import AnchorBinding, StepOut
from . import scenes as S
from .animals import MarineAnimals
from .controller import GRAVITY, VelocityController
from .fish import FishSchools
from .items import HANDLE_HEIGHT, ItemCatalog
from .lidar import LidarModel
from .math3d import quat_from_yaw, quat_integrate, quat_rotate, quat_rotate_inv, tilt_of
from .net import NetSim
from .shield import shield_velocity
from .terrain import NO_WATER, Terrain
from .waves import WaveField

NET_GRASP_RADIUS = 0.3
NET_HANDLE_HEIGHT = 0.6          # buoy line above each net corner, grabbed by the drone
ITEM_GRASP_RADIUS = 0.15
VEL_KI = 1.0                     # integral gain on velocity error
BOAT_HALF = (2.0, 1.0, 0.4)
BOAT_RADIUS = 2.5
SEABED_Z = -12.0
FAR = 1.0e4
ANCHOR_KIND = {k: i for i, k in enumerate(spec.ANCHOR_TYPES)}
PRIV_TYPE = {k: i for i, k in enumerate(spec.PRIV_TYPES)}
BOX_KINDS = ("building", "car", "truck", "bench", "hydrant", "trunk", "boat", "dropped", "bin")
SPHERE_KINDS = ("rock", "canopy", "bird")
ZONE_KINDS = ("bin", "truck", "pad_pickup", "pad_dropoff", "nofly", "boat")
VARIANTS = ("city", "terrain", "delivery", "ocean", "world")
BK = {k: i for i, k in enumerate(BOX_KINDS)}
SK = {k: i for i, k in enumerate(SPHERE_KINDS)}
ZK = {k: i for i, k in enumerate(ZONE_KINDS)}
VK = {k: i for i, k in enumerate(VARIANTS)}
DROPPED_SLOT = spec.MAX_BOXES - 1        # reserved for the demo's "drop obstacle"
BOAT_SLOT = spec.MAX_BOXES - 2           # boat hull box (ocean and world)
ZS = spec.ZONE_SLOT


@dataclass
class EnvConfig:
    scene: str = "trash"
    num_envs: int = 64
    device: str = "cpu"
    seed: int = 0
    lidar: bool = True              # False: skip scans, LiDAR images stay at "no return"
    shield: bool = True
    domain_rand: bool = False       # master switch for all domain randomisation (off: no sim2real needed)
    rand_drone_dynamics: bool = True  # drone mass and drag randomisation (only with domain_rand)
    enable_wind: bool = True        # ambient/mean wind (only with domain_rand)
    dynamic_wind: bool = True       # stochastic turbulence (only with domain_rand)
    enable_gusts: bool = True       # gusts (only with domain_rand)
    random_anchors: bool = True     # pretraining: randomise anchors on reset
    city_prob: float = 0.5          # trash scene: fraction of worlds that are cities (rest: terrain litter)
    nofly_prob: float = 0.5
    max_wind: float = 3.0
    expert_mode: str = "task"
    genesis_lidar: bool = False     # Genesis backend only (see backend docstring)


class SkyEnv:
    def __init__(self, cfg: Optional[EnvConfig] = None, **kw):
        cfg = cfg or EnvConfig()
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown EnvConfig field {k!r}")
            setattr(cfg, k, v)
        assert cfg.scene in spec.SCENES, cfg.scene
        self.cfg = cfg
        self.scene = cfg.scene
        self.num_envs = n = cfg.num_envs
        self.num_agents = a = spec.SCENE_AGENTS[cfg.scene]
        self.arena_half = spec.SCENE_ARENA_HALF[cfg.scene]
        self.max_speed = spec.SCENE_MAX_SPEED[cfg.scene]
        self.max_vz = spec.SCENE_MAX_VZ[cfg.scene]
        self.base_drag = spec.SCENE_DRAG[cfg.scene]
        self.episode_steps = spec.SCENE_EPISODE_STEPS[cfg.scene]
        self.device = dev = torch.device(cfg.device)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.pyrng = random.Random(cfg.seed)
        self.ctrl = VelocityController()
        self.lidar_model = LidarModel(dev)
        self.catalog = ItemCatalog(dev)
        zf = lambda *s: torch.zeros(*s, device=dev)
        zb = lambda *s: torch.zeros(*s, dtype=torch.bool, device=dev)
        zl = lambda *s, v=0: torch.full(s, v, dtype=torch.long, device=dev)
        # time
        self.t = zf(n)
        self.steps = zl(n)
        self.variant = zl(n)
        # drones
        self.pos, self.vel, self.ang = zf(n, a, 3), zf(n, a, 3), zf(n, a, 3)
        self.quat = zf(n, a, 4)
        self.quat[..., 0] = 1.0
        self.mass = torch.full((n, a), spec.DRONE_MASS, device=dev)
        self.drag = torch.full((n, a), self.base_drag, device=dev)
        self.alive = torch.ones(n, a, dtype=torch.bool, device=dev)
        self.grip = zb(n, a)
        self.holding = zl(n, a, v=-1)                 # item index (trash, delivery) or net corner (ocean)
        self.last_held = zl(n, a, v=-1)               # last item released, kept as the payload until the next grasp
        self.last_action = zf(n, a, spec.ACTION_DIM)
        self.hover = torch.ones(n, a, device=dev)
        self.acc_prev, self.vel_prev, self.jerk = zf(n, a, 3), zf(n, a, 3), zf(n, a)
        self.item_force = zf(n, a, 3)                 # pull of carried items on each drone
        self.net_force = zf(n, a, 3)                  # pull of the held net corner on each drone
        self.v_int = zf(n, a, 3)
        self.collided = zb(n, a)
        self.shield_hit = zb(n, a)
        self.shield_count = zl(n)
        self.swing = zf(n, a)
        # wind
        self.wind, self.wind_mean = zf(n, 3), zf(n, 3)
        self.gust, self.gust_time = zf(n, 3), zf(n)
        # terrain and water
        self.terrain = Terrain(n, dev, res=spec.SCENE_TERRAIN_RES[cfg.scene], half=self.arena_half)
        self.has_items = cfg.scene in ("trash", "delivery", "world")
        self.has_net = cfg.scene in ("ocean", "world")
        self.corner_offset = spec.MAX_ITEMS if self.has_items else 0   # net corner c has handle index offset + c
        # boxes
        B = spec.MAX_BOXES
        self.box_c, self.box_h, self.box_v = zf(n, B, 3), zf(n, B, 3), zf(n, B, 3)
        self.box_active, self.box_moving = zb(n, B), zb(n, B)
        self.box_kind = zl(n, B)
        self.lane_a, self.lane_b = zf(n, B, 2), zf(n, B, 2)
        self.lane_speed, self.lane_phase = zf(n, B), zf(n, B)
        # spheres (rocks, canopies, birds)
        Sn = spec.MAX_SPHERES
        self.sph_c, self.sph_v, self.sph_r = zf(n, Sn, 3), zf(n, Sn, 3), zf(n, Sn)
        self.sph_active, self.sph_moving = zb(n, Sn), zb(n, Sn)
        self.sph_kind = zl(n, Sn)
        self.orb_c, self.orb_r, self.orb_w, self.orb_ph = zf(n, Sn, 3), zf(n, Sn), zf(n, Sn), zf(n, Sn)
        # zones: 0 drop-off, 1 nofly_0, 2 nofly_1, 3 pad_pickup
        Z = spec.N_ZONES
        self.zone_c, self.zone_r, self.zone_drop_h = zf(n, Z, 3), zf(n, Z), zf(n, Z)
        self.zone_active = zb(n, Z)
        self.zone_kind = zl(n, Z)
        # items
        I = spec.MAX_ITEMS
        self.it_type = zl(n, I)
        self.it_active, self.it_delivered = zb(n, I), zb(n, I)
        self.it_pos, self.it_vel, self.it_half = zf(n, I, 3), zf(n, I, 3), zf(n, I, 3)
        self.it_sphere = zb(n, I)
        self.it_mass = zf(n, I)
        self.it_held_by = zl(n, I, v=-1)
        self.trash_init = zf(n)
        # ocean
        sea = self.has_net
        self.waves = WaveField(n, dev) if sea else None
        self.net = NetSim(n, dev) if sea else None
        self.fish = FishSchools(n, dev) if sea else None
        self.animals = MarineAnimals(n, dev) if sea else None
        self.turbidity = torch.full((n,), 0.3, device=dev)
        self.net_stretch, self.fish_frac = zf(n), zf(n)
        self.bycatch, self.animal_gap = zb(n), torch.full((n,), 2 * self.arena_half, device=dev)
        self.corner_held_by = zl(n, 4, v=-1)
        # anchors
        K = spec.N_ANCHORS
        self.anc_kind = zl(n, a, K)
        self.anc_src = zl(n, a, K, v=-1)              # fixed entity index, or -1
        self.anc_sel = zl(n, a, K)                    # selector id (1 + index into spec.ANCHOR_SELECTORS), 0 = none
        self.anc_point, self.anc_radius = zf(n, a, K, 3), zf(n, a, K)
        # expert
        self.exp_phase = zl(n, a)
        self.exp_cruise = zf(n, a)
        # LiDAR cache
        self._lidar = torch.full((n, a, spec.LIDAR_CH, spec.LIDAR_V, spec.LIDAR_H), 255, dtype=torch.uint8, device=dev)
        self._lidar[:, :, 2] = 0
        self._lidar_down = torch.full((n, a, spec.LIDAR_CH, spec.DOWN_V, spec.DOWN_H), 255, dtype=torch.uint8, device=dev)
        self._lidar_down[:, :, 2] = 0
        self._lidar_pts: Dict[str, torch.Tensor] = {}
        self._drone_spheres_first = True
        self._build_entities()
        self._post_init()
        self.reset()

    # ================================================================ backend hooks
    def _post_init(self) -> None:
        """Backend hook called before the first reset."""

    def _integrate_drones(self, force: torch.Tensor, torque: torch.Tensor, dt: float) -> None:
        alive = self.alive.unsqueeze(-1)
        inertia = self.ctrl.inertia.to(self.device)
        self.vel = torch.where(alive, self.vel + dt * force / self.mass.unsqueeze(-1), self.vel)
        self.pos = torch.where(alive, self.pos + dt * self.vel, self.pos)
        gyro = torch.cross(self.ang, self.ang * inertia, dim=-1)
        self.ang = torch.where(alive, self.ang + dt * (torque - gyro) / inertia, self.ang).clamp(-20.0, 20.0)
        self.quat = torch.where(alive, quat_integrate(self.quat, self.ang, dt), self.quat)

    def _solid_ranges(self):
        """External raycaster distances (Genesis). None = analytic geometry."""
        return None, None

    def _reset_backend(self, env_ids: torch.Tensor) -> None:
        """Backend hook: push freshly reset or edited state to the simulator."""

    # ================================================================ helpers
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def gripper_pos(self) -> torch.Tensor:
        off = torch.tensor(spec.GRIPPER_OFFSET, device=self.device).expand_as(self.pos)
        return self.pos + quat_rotate(self.quat, off)

    def item_handles(self) -> torch.Tensor:
        """Grab points of all items [N, I, 3]; a held item's handle is the holder's gripper."""
        h = self.it_pos.clone()
        h[..., 2] += self.it_half[..., 2] + HANDLE_HEIGHT
        held = self.it_held_by >= 0
        if held.any():
            gp = self.gripper_pos()
            idx = self.it_held_by.clamp_min(0).unsqueeze(-1).expand(-1, -1, 3)
            h = torch.where(held.unsqueeze(-1), torch.gather(gp, 1, idx), h)
        return h

    def net_handles(self) -> torch.Tensor:
        h = self.net.corner_pos().clone()
        h[..., 2] += NET_HANDLE_HEIGHT
        return h

    def ground_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height at xy [N, M, 2] -> [N, M]."""
        return self.terrain.height(xy)

    def water_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Water surface at xy [N, M, 2] -> [N, M]: tide level plus waves (NO_WATER where none)."""
        lvl = self.terrain.water_level(self.t)
        h = lvl.view(-1, *([1] * (xy.dim() - 2))).expand(xy.shape[:-1])
        if self.waves is not None:          # waves take [N, M, 2]: flatten any middle dimensions
            h = h + self.waves.height(xy.reshape(xy.shape[0], -1, 2), self.t).reshape(xy.shape[:-1])
        return torch.where(lvl.view(-1, *([1] * (xy.dim() - 2))) > NO_WATER / 2, h, torch.full_like(h, NO_WATER))

    def surface_height(self, xy: torch.Tensor) -> torch.Tensor:
        """What a drone would hit going straight down: terrain or water [N, M]."""
        return torch.maximum(self.ground_height(xy), self.water_height(xy))

    # ================================================================ entities
    def _build_entities(self) -> None:
        I = spec.MAX_ITEMS
        items = [f"item_{i}" for i in range(I)] + [f"item_{i}.handle" for i in range(I)]
        obstacles = [f"obstacle_{k}" for k in range(spec.N_NAMED_OBSTACLES)]
        drones = [f"drone_{i}" for i in range(self.num_agents)]
        net = [f"net.corner_{i}" for i in range(4)] + ["net.center"]
        if self.scene == "trash":
            names = items + ["bin", "nofly_0"] + obstacles + drones
        elif self.scene == "delivery":
            names = ["item_0", "item_0.handle", "pad_pickup", "pad_dropoff", "nofly_0", "nofly_1"] + drones
        elif self.scene == "ocean":
            names = net + ["boat"] + [f"fish_school_{i}" for i in range(spec.N_FISH_SCHOOLS)] + ["nofly_0"] + \
                list(self.animals.names) + drones
        else:
            names = items + ["bin", "pad_pickup", "pad_dropoff", "boat", "nofly_0", "nofly_1"] + net + \
                [f"fish_school_{i}" for i in range(spec.N_FISH_SCHOOLS)] + list(self.animals.names) + obstacles + drones
        kinds = []
        for nm in names:
            if nm.endswith(".handle") or nm.startswith("net.corner"):
                kinds.append("handle")
            elif nm in ZS:
                kinds.append("zone")
            else:
                kinds.append("point")
        self._ent_names: List[str] = names
        self._ent_kind = torch.tensor([ANCHOR_KIND[k] for k in kinds], device=self.device)
        self._ent_item = torch.tensor([int(nm.split("_")[1].split(".")[0]) if nm.startswith("item_") else -1
                                       for nm in names], device=self.device)
        self._ent_corner = torch.tensor([int(nm.split("_")[1]) if nm.startswith("net.corner_") else -1
                                         for nm in names], device=self.device)
        self._ent_is_handle = torch.tensor([nm.endswith(".handle") for nm in names], device=self.device)
        self._ent_zone = [ZS.get(nm) for nm in names]
        # selector candidates: entity indices each selector chooses among (empty if the scene has none)
        idx = lambda pred: torch.tensor([e for e, nm in enumerate(names) if pred(nm)], dtype=torch.long, device=self.device)
        handles = lambda nm: nm.startswith("item_") and nm.endswith(".handle")
        animals = set(self.animals.names) if self.animals is not None else set()
        cands = {"nearest:trash": idx(handles), "nearest:item": idx(handles),
                 "nearest:net_corner": idx(lambda nm: nm.startswith("net.corner_")),
                 "nearest:fish_school": idx(lambda nm: nm.startswith("fish_school_")),
                 "nearest:animal": idx(lambda nm: nm in animals)}
        for t in spec.ITEM_TYPES:
            cands[f"nearest:{t}"] = idx(handles)
        self._selectors = [(sel, cands[sel]) for sel, _ in spec.ANCHOR_SELECTORS]
        self._sel_id = {sel: i + 1 for i, (sel, _) in enumerate(spec.ANCHOR_SELECTORS)}
        self._item_handle_ent = torch.tensor([names.index(f"item_{k}.handle") if f"item_{k}.handle" in names else -1
                                              for k in range(spec.MAX_ITEMS)], device=self.device)
        self._corner_ent = torch.tensor([names.index(f"net.corner_{c}") if f"net.corner_{c}" in names else -1
                                         for c in range(4)], device=self.device)

    def entity_names(self) -> List[str]:
        return list(self._ent_names)

    def _entity_positions(self) -> torch.Tensor:
        n = self.num_envs
        cache = {}
        if self.has_items:
            cache["items"], cache["handles"] = self.it_pos, self.item_handles()
        if self.has_net:
            cache["corners"], cache["school"] = self.net_handles(), self.fish.school_centers()
        cols = []
        for nm in self._ent_names:
            if nm.startswith("item_"):
                k = int(nm.split("_")[1].split(".")[0])
                cols.append((cache["handles"] if nm.endswith(".handle") else cache["items"])[:, k])
            elif nm in ZS:
                cols.append(self.zone_c[:, ZS[nm]])
            elif nm.startswith("net.corner_"):
                cols.append(cache["corners"][:, int(nm.split("_")[1])])
            elif nm == "net.center":
                cols.append(self.net.centroid())
            elif nm.startswith("fish_school_"):
                cols.append(cache["school"][:, int(nm.split("_")[2])])
            elif self.animals is not None and nm in self.animals.names:
                cols.append(self.animals.pos[:, self.animals.names.index(nm)])
            elif nm.startswith("obstacle_"):
                cols.append(self.box_c[:, int(nm.split("_")[1])])
            elif nm.startswith("drone_"):
                cols.append(self.pos[:, int(nm.split("_")[1])])
            else:
                raise KeyError(nm)
        out = torch.stack(cols, 1)
        assert out.shape[:2] == (n, len(self._ent_names))
        return out

    def _entity_radius(self) -> torch.Tensor:
        r = torch.zeros(self.num_envs, len(self._ent_names), device=self.device)
        for i, z in enumerate(self._ent_zone):
            if z is not None:
                r[:, i] = self.zone_r[:, z]
        return r

    def _entity_active(self) -> torch.Tensor:
        """[N, E] bool: entity exists in this world (items and zones may be unused)."""
        act = torch.ones(self.num_envs, len(self._ent_names), dtype=torch.bool, device=self.device)
        for i, nm in enumerate(self._ent_names):
            it = int(self._ent_item[i])
            if it >= 0:
                act[:, i] = self.it_active[:, it] & ~self.it_delivered[:, it]
            z = self._ent_zone[i]
            if z is not None:
                act[:, i] = self.zone_active[:, z]
            if nm.startswith("obstacle_"):
                act[:, i] = self.box_active[:, int(nm.split("_")[1])]
        return act

    def entity_descriptions(self, env_id: int = 0) -> Dict[str, str]:
        """Human-readable descriptions for the language layer."""
        i = int(env_id)
        pos = self._entity_positions()[i]
        act = self._entity_active()[i]
        d = {}
        for e, name in enumerate(self._ent_names):
            p = [round(x, 1) for x in pos[e].tolist()]
            off = "" if bool(act[e]) else " (inactive)"
            it = int(self._ent_item[e])
            if it >= 0:
                typ = spec.ITEM_TYPES[int(self.it_type[i, it])]
                tag = "organ transport" if typ == "cooler" else ("trash" if spec.ITEM_IS_TRASH[int(self.it_type[i, it])] else "not trash")
                d[name] = f"{typ} ({tag}){' handle (grab here)' if name.endswith('.handle') else ''} at {p}{off}"
            elif name == "bin":
                kind = ZONE_KINDS[int(self.zone_kind[i, ZS["bin"]])]
                what = "garbage truck" if kind == "truck" else "trash bin"
                d[name] = f"{what} drop-off, radius {float(self.zone_r[i, ZS['bin']]):.1f} m at {p}{off}"
            elif name.startswith("pad_"):
                z = ZS[name]
                d[name] = f"hospital helipad ({'pickup' if name == 'pad_pickup' else 'drop-off'}), radius {float(self.zone_r[i, z]):.1f} m at {p}{off}"
            elif name.startswith("nofly"):
                z = ZS[name]
                d[name] = f"no-fly cylinder radius {float(self.zone_r[i, z]):.1f} m at {p}{off}"
            elif name.startswith("obstacle"):
                b = int(name.split("_")[1])
                d[name] = f"{BOX_KINDS[int(self.box_kind[i, b])]} at {p}" + ("" if bool(self.box_active[i, b]) else " (inactive)")
            elif name.startswith("net.corner"):
                d[name] = f"net corner (grab here) at {p}"
            elif name == "net.center":
                d[name] = f"centre of the fishing net at {p}"
            elif name == "boat":
                d[name] = f"boat deck, delivery zone radius {BOAT_RADIUS} m at {p}"
            elif name.startswith("fish_school"):
                d[name] = f"school of fish under water near {p}"
            elif self.animals is not None and name in self.animals.names:
                d[name] = f"{name.split('_')[0]} (keep it out of the net) at {p}"
            elif name.startswith("drone"):
                d[name] = f"drone at {p}"
            else:
                d[name] = f"{name} at {p}"
        return d

    # ================================================================ anchors
    def randomize_anchors(self, env_ids: torch.Tensor) -> None:
        m, a, K = len(env_ids), self.num_agents, spec.N_ANCHORS
        if m == 0:
            return
        kinds = self._ent_kind
        act = self._entity_active()[env_ids]                                  # [m, E]
        E = len(self._ent_names)

        def pick(pool_mask):                                                  # pool_mask [E] bool
            w = (act & pool_mask[None]).float()
            w = torch.where(w.sum(-1, keepdim=True) > 0, w, act.float() + 1e-6)
            idx = torch.multinomial(w.repeat_interleave(a, 0).cpu(), 1, generator=self.gen).view(m, a).to(self.device)
            return idx

        handles = kinds == ANCHOR_KIND["handle"]
        zones = kinds == ANCHOR_KIND["zone"]
        points = kinds == ANCHOR_KIND["point"]
        use = lambda p: self._rand(m, a) < p
        src = torch.full((m, a, K), -1, dtype=torch.long, device=self.device)
        src[..., 0] = torch.where(use(0.75), pick(handles), src[..., 0])
        if self.has_net:   # team net hauls: each drone gets its own corner
            first = self._ent_names.index("net.corner_0")
            corner = first + torch.arange(a, device=self.device).expand(m, a) % 4
            team = use(0.6) if self.scene == "ocean" else (self._rand(m, 1) < 0.3).expand(m, a)
            src[..., 0] = torch.where(team, corner, src[..., 0])
        drop = torch.tensor([nm in ("bin", "boat", "pad_dropoff") for nm in self._ent_names], device=self.device)
        src[..., 1] = torch.where(use(0.6), pick(drop), torch.where(use(0.5), pick(zones), src[..., 1]))
        src[..., 2] = torch.where(use(0.5), pick(zones), src[..., 2])
        src[..., 3] = torch.where(use(0.4), pick(points), src[..., 3])
        xy = (self._rand(m, a, K, 2) - 0.5) * 2 * (self.arena_half - 3)
        ground = torch.maximum(self.terrain.height(xy.reshape(m, -1, 2), env_ids),
                               self.terrain.water_level(self.t[env_ids], env_ids)[:, None]).reshape(m, a, K)
        if self.waves is not None:
            ground = torch.zeros_like(ground)
        pt = torch.cat([xy, (ground + 1.5 + 3.0 * self._rand(m, a, K)).unsqueeze(-1)], -1)
        # selector anchors, so the policy learns "the closest one of a kind"
        sel = torch.zeros((m, a, K), dtype=torch.long, device=self.device)
        if self.has_items:
            sel[..., 0] = torch.where(use(0.25), self._sel_id["nearest:trash"], sel[..., 0])
        if self.has_net:
            sel[..., 2] = torch.where(use(0.3), self._sel_id["nearest:animal"], sel[..., 2])
            sel[..., 3] = torch.where(use(0.2), self._sel_id["nearest:fish_school"], sel[..., 3])
        src = torch.where(sel > 0, torch.full_like(src, -1), src)
        self.anc_src[env_ids] = src
        self.anc_sel[env_ids] = sel
        self.anc_point[env_ids] = pt
        sel_kind = torch.tensor([ANCHOR_KIND["none"]] + [ANCHOR_KIND[k] for _, k in spec.ANCHOR_SELECTORS], device=self.device)
        self.anc_kind[env_ids] = torch.where(sel > 0, sel_kind[sel],
                                             torch.where(src >= 0, kinds[src.clamp_min(0)], torch.full_like(src, ANCHOR_KIND["point"])))
        self.anc_radius[env_ids] = 0.0

    def bind_anchors(self, env_ids, bindings: List[AnchorBinding], agent: Optional[int] = None) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        agents = range(self.num_agents) if agent is None else [agent]
        for k in range(spec.N_ANCHORS):
            b = bindings[k] if k < len(bindings) else None
            for ag in agents:
                self.anc_sel[env_ids, ag, k] = 0
                if b is None or b.kind == "none":
                    self.anc_src[env_ids, ag, k] = -1
                    self.anc_kind[env_ids, ag, k] = ANCHOR_KIND["none"]
                    self.anc_point[env_ids, ag, k] = 0.0
                    self.anc_radius[env_ids, ag, k] = 0.0
                    continue
                if b.entity is not None and b.entity in self._sel_id:
                    self.anc_src[env_ids, ag, k] = -1
                    self.anc_sel[env_ids, ag, k] = self._sel_id[b.entity]
                    kind = b.kind if b.kind in ("handle", "point", "zone") else spec.SELECTOR_KIND[b.entity]
                    if kind == "point" and spec.SELECTOR_KIND[b.entity] == "handle":
                        kind = "handle"
                elif b.entity is not None:
                    if b.entity not in self._ent_names:
                        raise KeyError(f"unknown entity {b.entity!r}; known: {self._ent_names} "
                                       f"and selectors {list(self._sel_id)}")
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

    def anchor_src(self, ent: Optional[torch.Tensor] = None, act: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[N, A, K] entity index each anchor points at this step: the fixed binding, or the
        closest valid match for a selector (-1 for a point anchor or an unmatched selector)."""
        src = self.anc_src
        if not bool((self.anc_sel > 0).any()):
            return src
        n, a, K = src.shape
        ent = self._entity_positions() if ent is None else ent
        act = self._entity_active() if act is None else act
        out = src.clone()
        off = self.corner_offset
        held_item = (self.holding >= 0) & (self.holding < off) if self.has_items else torch.zeros_like(self.holding, dtype=torch.bool)
        held_corner = self.holding >= off if self.has_net else torch.zeros_like(held_item)
        for sid, (sel, cand) in enumerate(self._selectors, start=1):
            m = self.anc_sel == sid
            if not bool(m.any()):
                continue
            if cand.numel() == 0:
                out = torch.where(m, torch.full_like(out, -1), out)
                continue
            valid = act[:, cand].unsqueeze(1).expand(n, a, -1).clone()               # [N, A, C]
            sticky = torch.full((n, a), -1, dtype=torch.long, device=self.device)
            if sel.startswith("nearest:") and sel not in ("nearest:net_corner", "nearest:fish_school", "nearest:animal"):
                it = self._ent_item[cand].clamp_min(0)                                # item index of each candidate
                typ = self.it_type[:, it]                                             # [N, C]
                if sel == "nearest:trash":
                    ok = self.catalog.is_trash[typ]
                elif sel == "nearest:item":
                    ok = torch.ones_like(typ, dtype=torch.bool)
                else:
                    ok = typ == self.catalog.id(sel.split(":", 1)[1])
                valid &= ok.unsqueeze(1)
                hk = self.holding.clamp(0, spec.MAX_ITEMS - 1)
                h_ok = held_item & torch.gather(ok, 1, hk)
                sticky = torch.where(h_ok, self._item_handle_ent[hk], sticky)
            elif sel == "nearest:net_corner":
                free = (self.corner_held_by < 0).unsqueeze(1) | \
                    (self.corner_held_by.unsqueeze(1) == torch.arange(a, device=self.device).view(1, a, 1))
                valid &= free
                sticky = torch.where(held_corner, self._corner_ent[(self.holding - off).clamp(0, 3)], sticky)
            d = (self.pos[:, :, None] - ent[:, cand][:, None]).norm(dim=-1)
            d = torch.where(valid, d, torch.full_like(d, FAR))
            best = torch.where(valid.any(-1), cand[d.argmin(-1)], torch.full((n, a), -1, dtype=torch.long, device=self.device))
            choose = torch.where(sticky >= 0, sticky, best)
            out = torch.where(m, choose.unsqueeze(-1).expand(n, a, K), out)
        return out

    def anchor_positions(self):
        """(pos [N, A, K, 3], radius [N, A, K], kind [N, A, K]). Unused slots and selectors
        with nothing to match report kind 'none' and the drone's own position."""
        ent = self._entity_positions()
        er = self._entity_radius()
        src_eff = self.anchor_src(ent)
        n, a, K = src_eff.shape
        src = src_eff.clamp_min(0)
        gpos = torch.gather(ent[:, None].expand(n, a, -1, 3), 2, src.unsqueeze(-1).expand(n, a, K, 3))
        grad = torch.gather(er[:, None].expand(n, a, -1), 2, src)
        bound = src_eff >= 0
        pos = torch.where(bound.unsqueeze(-1), gpos, self.anc_point)
        rad = torch.where(bound & (self.anc_radius == 0), grad, self.anc_radius)
        kind = torch.where((self.anc_sel > 0) & ~bound, torch.full_like(self.anc_kind, ANCHOR_KIND["none"]), self.anc_kind)
        none = kind == ANCHOR_KIND["none"]
        pos = torch.where(none.unsqueeze(-1), self.pos.unsqueeze(2).expand_as(pos), pos)
        return pos, rad, kind

    def anchor_item(self) -> torch.Tensor:
        """[N, A] item index anchor 0 points at this step (its item or its handle), else -1."""
        src = self.anchor_src()[..., 0]
        it = self._ent_item[src.clamp_min(0)]
        return torch.where(src >= 0, it, torch.full_like(src, -1))

    def anchor_handle(self) -> torch.Tensor:
        """[N, A] handle-table index anchor 0 points at this step: an item (its index) or a net
        corner (corner_offset + c); -1 otherwise."""
        srcf = self.anchor_src()[..., 0]
        src = srcf.clamp_min(0)
        it, co = self._ent_item[src], self._ent_corner[src]
        h = torch.where(it >= 0, it, torch.where(co >= 0, co + self.corner_offset, torch.full_like(it, -1)))
        return torch.where(srcf >= 0, h, torch.full_like(h, -1))

    # ================================================================ reset
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if len(env_ids) > 0:
            self._reset_worlds(env_ids, len(env_ids), self.num_agents)
            self._reset_backend(env_ids)
            if self.cfg.lidar:
                self._scan(env_ids)
        return self._obs()

    def _reset_worlds(self, ids: torch.Tensor, m: int, a: int) -> None:
        cfg = self.cfg
        dr = cfg.domain_rand
        rand_drone = dr and cfg.rand_drone_dynamics
        self.t[ids] = 0.0
        self.steps[ids] = 0
        self.vel[ids] = 0.0
        self.ang[ids] = 0.0
        self.quat[ids] = quat_from_yaw((self._rand(m, a) - 0.5) * 2 * math.pi)
        self.mass[ids] = spec.DRONE_MASS * (1.0 + (0.2 * (self._rand(m, a) - 0.5) if rand_drone else 0.0))
        self.drag[ids] = self.base_drag * (1.0 + (0.4 * (self._rand(m, a) - 0.5) if rand_drone else 0.0))
        for name in ("grip", "collided", "shield_hit"):
            getattr(self, name)[ids] = False
        self.alive[ids] = True
        self.holding[ids] = -1
        self.last_held[ids] = -1
        for name in ("last_action", "acc_prev", "vel_prev", "jerk", "item_force", "net_force", "v_int", "swing", "exp_cruise"):
            getattr(self, name)[ids] = 0.0
        self.hover[ids] = 1.0
        self.shield_count[ids] = 0
        self.exp_phase[ids] = 0
        ang = self._rand(m) * 2 * math.pi
        wind_on = dr and cfg.enable_wind
        self.wind_mean[ids] = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * \
            (self._rand(m, 1) * cfg.max_wind if wind_on else 0.0)
        self.wind[ids] = self.wind_mean[ids]
        self.gust[ids] = 0.0
        self.gust_time[ids] = 0.0
        # clear world contents
        self.box_active[ids] = False
        self.box_moving[ids] = False
        self.box_v[ids] = 0.0
        self.sph_active[ids] = False
        self.sph_moving[ids] = False
        self.sph_v[ids] = 0.0
        self.zone_active[ids] = False
        self.it_active[ids] = False
        self.it_delivered[ids] = False
        self.it_held_by[ids] = -1
        self.it_vel[ids] = 0.0
        self.corner_held_by[ids] = -1
        self.net_stretch[ids] = 0.0
        self.fish_frac[ids] = 0.0
        self.bycatch[ids] = False
        if self.scene == "ocean":
            self._reset_ocean(ids, m, a)
        elif self.scene == "world":
            self._reset_world(ids, m, a)
        else:
            self._reset_land(ids, m, a)
        if cfg.random_anchors:
            self.randomize_anchors(ids)

    def _reset_land(self, ids: torch.Tensor, m: int, a: int) -> None:
        cfg = self.cfg
        rng = self.pyrng
        if self.scene == "trash":
            city = self._rand(m) < cfg.city_prob
        else:
            city = torch.zeros(m, dtype=torch.bool, device=self.device)
        flat_ids, rough_ids = ids[city], ids[~city]
        if len(flat_ids):
            self.terrain.set_flat(flat_ids, 0.0)
        if len(rough_ids):
            self.terrain.generate(rough_ids, "rolling" if self.scene == "trash" else "remote", self.gen)
        hs = self.terrain.h[ids].detach().cpu().numpy()
        wb = self.terrain.water_base[ids].tolist()
        ta = self.terrain.tide_amp[ids].tolist()
        layouts = []
        for j in range(m):
            if bool(city[j]):
                layouts.append(S.city(rng, cfg.nofly_prob))
                continue
            g = S.Ground(hs[j], 0.0, wb[j], ta[j], half=self.arena_half)
            layouts.append(S.litter(rng, g, cfg.nofly_prob) if self.scene == "trash" else S.delivery(rng, g, cfg.nofly_prob))
        self._load_layouts(ids, layouts)

    def _load_layouts(self, ids: torch.Tensor, layouts: List[S.Layout]) -> None:
        dev = self.device
        m = len(ids)
        B, Sn, I, Z = spec.MAX_BOXES, spec.MAX_SPHERES, spec.MAX_ITEMS, spec.N_ZONES
        bc = torch.zeros(m, B, 3); bh = torch.zeros(m, B, 3); bact = torch.zeros(m, B, dtype=torch.bool)
        bkind = torch.zeros(m, B, dtype=torch.long); bmov = torch.zeros(m, B, dtype=torch.bool)
        la = torch.zeros(m, B, 2); lb = torch.zeros(m, B, 2); ls = torch.zeros(m, B); lp = torch.zeros(m, B)
        sc = torch.zeros(m, Sn, 3); sr = torch.zeros(m, Sn); sact = torch.zeros(m, Sn, dtype=torch.bool)
        skind = torch.zeros(m, Sn, dtype=torch.long); smov = torch.zeros(m, Sn, dtype=torch.bool)
        oc = torch.zeros(m, Sn, 3); orr = torch.zeros(m, Sn); ow = torch.zeros(m, Sn); oph = torch.zeros(m, Sn)
        ityp = torch.zeros(m, I, dtype=torch.long); ixy = torch.zeros(m, I, 2); iact = torch.zeros(m, I, dtype=torch.bool)
        zc = torch.zeros(m, Z, 3); zr = torch.zeros(m, Z); zact = torch.zeros(m, Z, dtype=torch.bool)
        zkind = torch.zeros(m, Z, dtype=torch.long); zdh = torch.zeros(m, Z)
        dxy = torch.zeros(m, self.num_agents, 2); dh = torch.zeros(m, self.num_agents)
        var = torch.zeros(m, dtype=torch.long)
        for j, L in enumerate(layouts):
            var[j] = VK[L.variant]
            for k, b in enumerate(L.boxes[:B - 1]):
                bc[j, k] = torch.tensor(b.c); bh[j, k] = torch.tensor(b.h); bact[j, k] = True; bkind[j, k] = BK[b.kind]
                if b.lane is not None:
                    bmov[j, k] = True
                    la[j, k] = torch.tensor(b.lane[0:2]); lb[j, k] = torch.tensor(b.lane[2:4])
                    ls[j, k] = b.lane[4]; lp[j, k] = b.lane[5]
            for k, s in enumerate(L.spheres[:Sn]):
                sc[j, k] = torch.tensor(s.c); sr[j, k] = s.r; sact[j, k] = True; skind[j, k] = SK[s.kind]
                if s.orbit is not None:
                    smov[j, k] = True
                    oc[j, k] = torch.tensor(s.orbit[0:3]); orr[j, k] = s.orbit[3]; ow[j, k] = s.orbit[4]; oph[j, k] = s.orbit[5]
            for k, (typ, x, y) in enumerate(L.items[:I]):
                ityp[j, k] = self.catalog.id(typ); ixy[j, k] = torch.tensor([x, y]); iact[j, k] = True
            for slot, (kind, x, y, r, drop_h) in L.zones.items():
                zc[j, slot] = torch.tensor([x, y, 0.0]); zr[j, slot] = r; zact[j, slot] = True
                zkind[j, slot] = ZK[kind]; zdh[j, slot] = drop_h
            for k in range(self.num_agents):
                x, y, h = L.drones[min(k, len(L.drones) - 1)]
                dxy[j, k] = torch.tensor([x, y]); dh[j, k] = h
        to = lambda x: x.to(dev)
        self.variant[ids] = to(var)
        self.box_c[ids], self.box_h[ids], self.box_active[ids] = to(bc), to(bh), to(bact)
        self.box_kind[ids], self.box_moving[ids] = to(bkind), to(bmov)
        self.lane_a[ids], self.lane_b[ids], self.lane_speed[ids], self.lane_phase[ids] = to(la), to(lb), to(ls), to(lp)
        self.sph_c[ids], self.sph_r[ids], self.sph_active[ids] = to(sc), to(sr), to(sact)
        self.sph_kind[ids], self.sph_moving[ids] = to(skind), to(smov)
        self.orb_c[ids], self.orb_r[ids], self.orb_w[ids], self.orb_ph[ids] = to(oc), to(orr), to(ow), to(oph)
        # zones sit on the ground
        zc = to(zc)
        zc[..., 2] = self.terrain.height(zc[..., :2], ids)
        self.zone_c[ids], self.zone_r[ids], self.zone_active[ids] = zc, to(zr), to(zact)
        self.zone_kind[ids], self.zone_drop_h[ids] = to(zkind), to(zdh)
        # items rest on the ground
        ityp = to(ityp)
        self.it_type[ids], self.it_active[ids] = ityp, to(iact)
        half = self.catalog.half[ityp]
        self.it_half[ids] = half
        self.it_sphere[ids] = self.catalog.sphere[ityp]
        self.it_mass[ids] = self.catalog.mass[ityp]
        ixy = to(ixy)
        gz = self.terrain.height(ixy, ids)
        self.it_pos[ids] = torch.cat([ixy, (gz + half[..., 2]).unsqueeze(-1)], -1)
        trash = self.catalog.is_trash[ityp] & to(iact)
        self.trash_init[ids] = trash.float().sum(-1)
        # drones above the ground at their start spots
        dxy = to(dxy)
        gz = torch.maximum(self.terrain.height(dxy, ids), self.terrain.water_level(self.t[ids], ids)[:, None])
        self.pos[ids] = torch.cat([dxy, (gz + to(dh)).unsqueeze(-1)], -1)
        self._move_dynamic(0.0, ids)

    def _place_boat(self, ids: torch.Tensor, xy: torch.Tensor) -> None:
        b, z = BOAT_SLOT, ZS["boat"]
        self.zone_c[ids, z] = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
        self.zone_r[ids, z] = BOAT_RADIUS
        self.zone_kind[ids, z] = ZK["boat"]
        self.zone_drop_h[ids, z] = 6.0
        self.zone_active[ids, z] = True
        self.box_h[ids, b] = torch.tensor(BOAT_HALF, device=self.device)
        self.box_c[ids, b] = self.zone_c[ids, z]
        self.box_active[ids, b] = True
        self.box_moving[ids, b] = False
        self.box_kind[ids, b] = BK["boat"]

    def _reset_sea_life(self, ids: torch.Tensor, net_xy: torch.Tensor, home_c: torch.Tensor, home_r: torch.Tensor,
                        animal_r: Optional[torch.Tensor] = None) -> None:
        """Fish schools stay in the disc (home_c, home_r); animals roam animal_r (default home_r), which
        must cover the net so that bycatch can happen where the net is."""
        m = len(ids)
        self.net.reset(ids, torch.cat([net_xy, torch.zeros(m, 1, device=self.device)], -1), self._rand(m) * 2 * math.pi)
        self.fish.reset(ids, self.gen, home_c, home_r)
        self.animals.reset(ids, self.gen, home_c, home_r if animal_r is None else animal_r)
        self.turbidity[ids] = 0.1 + 0.7 * self._rand(m)

    def _reset_ocean(self, ids: torch.Tensor, m: int, a: int) -> None:
        H = self.arena_half
        self.variant[ids] = VK["ocean"]
        self.terrain.set_flat(ids, SEABED_Z, water=0.0)
        self.waves.randomize(ids, self.gen)
        boat_xy = (self._rand(m, 2) - 0.5) * 2 * (H - 6)
        self._place_boat(ids, boat_xy)
        nf = self._rand(m) < self.cfg.nofly_prob
        nf_xy = (self._rand(m, 2) - 0.5) * 2 * (H - 6)
        z1 = ZS["nofly_0"]
        self.zone_c[ids, z1] = torch.cat([nf_xy, torch.zeros(m, 1, device=self.device)], -1)
        self.zone_r[ids, z1] = 2.5 + 2.0 * self._rand(m)
        self.zone_kind[ids, z1] = ZK["nofly"]
        self.zone_active[ids, z1] = nf & ((nf_xy - boat_xy).norm(dim=-1) > 8.0)
        net_xy = boat_xy + torch.nn.functional.normalize(self._randn(m, 2), dim=-1) * (12.0 + 6.0 * self._rand(m, 1))
        # net centre inside the fish disc (schools are reachable), animals roam a wider disc around it
        net_xy = net_xy * ((H - 9.0) / net_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        self._reset_sea_life(ids, net_xy, torch.zeros(m, 2, device=self.device), torch.full((m,), H - 9.0, device=self.device),
                             torch.full((m,), H - 3.0, device=self.device))
        k = torch.arange(a, device=self.device) % 4
        cxy = self.net.pos[ids][:, self.net.corners[k], :2]
        out = torch.nn.functional.normalize(cxy - net_xy[:, None], dim=-1)
        d_xy = cxy + out * (1.5 + 1.5 * self._rand(m, a, 1))
        self.pos[ids] = torch.cat([d_xy, 2.0 + 2.0 * self._rand(m, a, 1)], -1)
        self.t[ids] = self._rand(m) * 20.0

    def _reset_world(self, ids: torch.Tensor, m: int, a: int) -> None:
        """Generic pretraining world: countryside, lakes, coast and sea, a town, two clinics."""
        import numpy as np
        layouts = [S.world(self.pyrng, self.terrain.res, self.arena_half, self.cfg.nofly_prob) for _ in range(m)]
        h = torch.tensor(np.stack([L.heights for L in layouts]), dtype=torch.float32)
        self.terrain.set_heights(ids, h, water=0.0, tide_amp=0.3, gen=self.gen)
        self.waves.randomize(ids, self.gen, max_total_amp=0.25)
        self._load_layouts(ids, layouts)
        dev = self.device
        self._place_boat(ids, torch.tensor([L.boat for L in layouts], dtype=torch.float32, device=dev))
        sea = torch.tensor([L.sea for L in layouts], dtype=torch.float32, device=dev)
        self._reset_sea_life(ids, torch.tensor([L.net for L in layouts], dtype=torch.float32, device=dev), sea[:, :2], sea[:, 2])
        self.t[ids] = self._rand(m) * 20.0

    # ================================================================ step
    def step(self, actions: torch.Tensor) -> StepOut:
        n, a = self.num_envs, self.num_agents
        actions = actions.to(self.device).float().clamp(-1.0, 1.0)
        assert actions.shape == (n, a, spec.ACTION_DIM), actions.shape
        actions = torch.where(self.alive.unsqueeze(-1), actions, torch.zeros_like(actions))
        events = {k: torch.zeros(n, a, dtype=torch.bool, device=self.device) for k in spec.EVENT_KEYS}
        self._update_grip(actions, events)
        v_cmd, _ = VelocityController.command(actions, self.max_speed, self.max_vz)
        if self.cfg.shield:
            v_cmd, hit = self._shield(v_cmd)
        else:
            hit = torch.zeros(n, a, dtype=torch.bool, device=self.device)
        self.shield_hit = hit & self.alive
        self.shield_count += self.shield_hit.sum(-1)
        self.collided = torch.zeros(n, a, dtype=torch.bool, device=self.device)
        crash = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(spec.SUBSTEPS):
            self._move_dynamic(spec.PHYS_DT)
            self._update_wind(spec.PHYS_DT)
            mass = self.mass + self._held_mass()
            self.v_int = (self.v_int + spec.PHYS_DT * (v_cmd - self.vel)).clamp(-2.0, 2.0)
            thrust, torque, hover = self.ctrl(actions, self.quat, self.vel, self.ang, mass=mass,
                                              v_des=v_cmd + VEL_KI * self.v_int)
            self.hover = hover
            bz = quat_rotate(self.quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(self.pos))
            rel_air = self.vel - (self.wind + self.gust).unsqueeze(1)
            force = thrust.unsqueeze(-1) * bz - self.drag.unsqueeze(-1) * rel_air * rel_air.norm(dim=-1, keepdim=True).clamp_min(1.0)
            force[..., 2] -= self.mass * GRAVITY
            force = torch.where(self.alive.unsqueeze(-1), force + self.item_force + self.net_force, torch.zeros_like(force))
            self._integrate_drones(force, torque, spec.PHYS_DT)
            if self.has_items:
                self._items_substep(spec.PHYS_DT, events)
            crash |= self._collisions()
        if self.has_net:
            self._ocean_policy_step(events)
        self.t += spec.POLICY_DT
        self.steps += 1
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
        if self.has_items:
            events["delivered"] |= self._check_delivered().unsqueeze(-1).expand(n, a)
        ceiling = self.terrain.max_height().clamp_min(0.0)[:, None] + 60.0
        oob = ((self.pos[..., :2].abs() > self.arena_half + 5).any(-1) | (self.pos[..., 2] > ceiling)) & self.alive
        done = crash | oob.any(-1) | ~self.alive.any(-1)
        truncated = (self.steps >= self.episode_steps) & ~done
        # which drones ended the episode (crashed or left the arena); their teammates' last
        # transitions are cut off, not terminal. A world that ends otherwise marks every drone.
        agent_done = (self.collided | oob) & done.unsqueeze(-1)
        agent_done |= (done & ~agent_done.any(-1)).unsqueeze(-1)
        return StepOut(obs=self._obs(), done=done, truncated=truncated, events=events,
                       info={"shield_count": self.shield_count.clone(), "agent_done": agent_done})

    # ================================================================ dynamics
    def _held_mass(self) -> torch.Tensor:
        m = torch.zeros_like(self.mass)
        if not self.has_items:
            return m
        held = self.it_held_by >= 0
        return m.scatter_add(1, self.it_held_by.clamp_min(0), torch.where(held, self.it_mass, torch.zeros_like(self.it_mass)))

    def _move_dynamic(self, dt: float, ids: Optional[torch.Tensor] = None) -> None:
        """Cars along their lanes, birds on their orbits."""
        rows = slice(None) if ids is None else ids
        mv = self.box_moving[rows]
        if mv.any():
            a, b = self.lane_a[rows], self.lane_b[rows]
            L = (b - a).norm(dim=-1).clamp_min(1e-3)
            self.lane_phase[rows] = self.lane_phase[rows] + dt * self.lane_speed[rows] / (2 * L)
            ph = self.lane_phase[rows] % 1.0
            tri = 1.0 - (2.0 * ph - 1.0).abs()                                   # 0 -> 1 -> 0
            tgt = a + (b - a) * tri.unsqueeze(-1)
            old = self.box_c[rows][..., :2]
            vel = torch.cat([(tgt - old) / max(dt, 1e-6), torch.zeros_like(tgt[..., :1])], -1)
            m3 = mv.unsqueeze(-1)
            self.box_v[rows] = torch.where(m3, vel if dt > 0 else torch.zeros_like(vel), torch.zeros_like(vel))
            bc = self.box_c[rows].clone()
            bc[..., :2] = torch.where(mv.unsqueeze(-1), tgt, bc[..., :2])
            self.box_c[rows] = bc
        sm = self.sph_moving[rows]
        if sm.any():
            self.orb_ph[rows] = self.orb_ph[rows] + dt * self.orb_w[rows]
            ph = self.orb_ph[rows]
            c = self.orb_c[rows]
            tgt = c + torch.stack([torch.cos(ph), torch.sin(ph), torch.zeros_like(ph)], -1) * self.orb_r[rows].unsqueeze(-1)
            old = self.sph_c[rows]
            vel = (tgt - old) / max(dt, 1e-6) if dt > 0 else torch.zeros_like(tgt)
            self.sph_v[rows] = torch.where(sm.unsqueeze(-1), vel, torch.zeros_like(vel))
            self.sph_c[rows] = torch.where(sm.unsqueeze(-1), tgt, old)

    def _update_wind(self, dt: float) -> None:
        dr = self.cfg.domain_rand
        wind_on = dr and self.cfg.enable_wind
        if not wind_on:
            self.wind.zero_()
            self.wind_mean.zero_()
        else:
            sigma = 0.8 if self.cfg.dynamic_wind else 0.0
            self.wind = self.wind + 0.5 * (self.wind_mean - self.wind) * dt + \
                sigma * math.sqrt(dt) * self._randn(self.num_envs, 3) * torch.tensor([1.0, 1.0, 0.2], device=self.device)
        self.gust_time = (self.gust_time - dt).clamp_min(0.0)
        if dr and self.cfg.enable_gusts:
            self.gust = torch.where((self.gust_time > 0).unsqueeze(-1), self.gust, torch.zeros_like(self.gust))
        else:
            self.gust.zero_()
            self.gust_time.zero_()

    def _items_substep(self, dt: float, events) -> None:
        held = self.it_held_by >= 0                                            # [N, I]
        active = self.it_active & ~self.it_delivered
        free = active & ~held
        # free items: fall, rest on the terrain, sink in lakes
        v = self.it_vel.clone()
        v[..., 2] -= GRAVITY * dt
        p = self.it_pos + dt * v
        gz = self.terrain.height(p[..., :2])
        floor = gz + self.it_half[..., 2]
        on_ground = p[..., 2] <= floor
        p[..., 2] = torch.maximum(p[..., 2], floor)
        v[..., 2] = torch.where(on_ground, torch.zeros_like(v[..., 2]), v[..., 2])
        v[..., :2] = torch.where(on_ground.unsqueeze(-1), v[..., :2] * (1.0 - 8.0 * dt), v[..., :2])
        f = free.unsqueeze(-1)
        self.it_pos = torch.where(f, p, self.it_pos)
        self.it_vel = torch.where(f, v, self.it_vel)
        lvl = self.terrain.water_level(self.t)[:, None]
        sunk = free & (self.it_pos[..., 2] < lvl - 0.3) & (gz < lvl)
        if sunk.any():
            self.it_active = self.it_active & ~sunk
            events["lost"] |= sunk.any(-1, keepdim=True).expand(-1, self.num_agents)
        # held items hang on a rope below the gripper (position-based pendulum)
        self.item_force = torch.zeros_like(self.item_force)
        if not held.any():
            self.swing = torch.zeros_like(self.swing)
            return
        gp = self.gripper_pos()
        holder = self.it_held_by.clamp_min(0)
        hidx = holder.unsqueeze(-1).expand(-1, -1, 3)
        hp = torch.gather(gp, 1, hidx)                                         # [N, I, 3]
        hv = torch.gather(self.vel, 1, hidx)
        L = (self.it_half[..., 2] + HANDLE_HEIGHT).unsqueeze(-1)
        v = self.it_vel.clone()
        v[..., 2] -= GRAVITY * dt
        v = v - dt * 0.4 * (v - (self.wind + self.gust)[:, None]) / self.it_mass.clamp_min(0.05).unsqueeze(-1)
        p = self.it_pos + dt * v
        d = p - hp
        p_new = hp + d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6) * L
        p_new[..., 2] = torch.maximum(p_new[..., 2], self.terrain.height(p_new[..., :2]) + self.it_half[..., 2])
        v_new = (p_new - self.it_pos) / dt
        acc = (v_new - self.it_vel) / dt
        tension = self.it_mass.unsqueeze(-1) * (acc - torch.tensor([0.0, 0.0, -GRAVITY], device=self.device))
        lim = 3 * GRAVITY * spec.DRONE_MASS
        tension = tension.clamp(-lim, lim)
        h3 = held.unsqueeze(-1)
        self.it_pos = torch.where(h3, p_new, self.it_pos)
        self.it_vel = torch.where(h3, v_new, self.it_vel)
        rel, relv = self.it_pos - hp, self.it_vel - hv
        sw = torch.cross(rel, relv, dim=-1).norm(dim=-1) / (L.squeeze(-1) ** 2)
        sw = torch.where(held, sw, torch.zeros_like(sw))
        self.item_force = torch.zeros_like(self.item_force).scatter_add(
            1, hidx, torch.where(h3, -tension, torch.zeros_like(tension)))
        self.swing = torch.zeros_like(self.swing).scatter_add(1, holder, sw)

    def _check_delivered(self) -> torch.Tensor:
        """Items released or resting inside a drop-off: the bin / garbage truck (slot 'bin';
        trash is removed, it went in) or the clinic pad (slot 'pad_dropoff'; its small drop
        height means the cooler must be set down). Returns [N] bool for this step."""
        any_new = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        trash = self.catalog.is_trash[self.it_type]
        for slot in (ZS["bin"], ZS["pad_dropoff"]):
            free = self.it_active & ~self.it_delivered & (self.it_held_by < 0)
            zc, zr, zh = self.zone_c[:, slot:slot + 1], self.zone_r[:, slot:slot + 1], self.zone_drop_h[:, slot:slot + 1]
            inside = ((self.it_pos[..., :2] - zc[..., :2]).norm(dim=-1) < zr) & \
                ((self.it_pos[..., 2] - self.it_half[..., 2] - zc[..., 2]) < zh.clamp_min(0.3)) & self.zone_active[:, slot:slot + 1]
            new = free & inside
            if new.any():
                self.it_delivered |= new
                if slot == ZS["bin"]:
                    self.it_active = self.it_active & ~(new & trash)
                any_new |= new.any(-1)
        return any_new

    def _ocean_policy_step(self, events) -> None:
        n, a = self.num_envs, self.num_agents
        ar = torch.arange(n, device=self.device)
        pinned = torch.zeros(n, self.net.p, dtype=torch.bool, device=self.device)
        pin_pos = self.net.pos.clone()
        gp = self.gripper_pos()
        up = torch.tensor([0.0, 0.0, NET_HANDLE_HEIGHT], device=self.device)
        for c in range(4):
            holder = self.corner_held_by[:, c]
            held = holder >= 0
            node = int(self.net.corners[c])
            pinned[:, node] = held
            pin_pos[:, node] = torch.where(held.unsqueeze(-1), gp[ar, holder.clamp_min(0)] - up, pin_pos[:, node])
        wh = self.water_height(self.net.pos[..., :2])
        wv = self.waves.water_velocity(self.net.pos - torch.nn.functional.pad(
            self.terrain.water_level(self.t).view(-1, 1, 1).expand(-1, self.net.p, 1), (2, 0)), self.t)
        floor = self.ground_height(self.net.pos[..., :2])
        hull = (self.box_c[:, BOAT_SLOT:BOAT_SLOT + 1], self.box_h[:, BOAT_SLOT:BOAT_SLOT + 1],
                self.box_active[:, BOAT_SLOT:BOAT_SLOT + 1])
        pin_force, stretch = self.net.step(spec.POLICY_DT, pinned, pin_pos, wh, wv, floor=floor, boxes=hull)
        self.net_stretch = stretch.clamp_min(0.0)
        # the net never tears: a corner pulled too far slips out of its gripper (a release event)
        for c in range(4):
            slip = self.net.slip[:, int(self.net.corners[c])] & (self.corner_held_by[:, c] >= 0)
            if slip.any():
                who = self.corner_held_by[:, c].clamp_min(0)
                lost_grip = slip.unsqueeze(-1) & (torch.arange(a, device=self.device)[None] == who[:, None])
                events["slip"] |= lost_grip
                events["release"] |= lost_grip
                self.holding = torch.where(lost_grip, torch.full_like(self.holding, -1), self.holding)
                self.corner_held_by[:, c] = torch.where(slip, torch.full_like(who, -1), self.corner_held_by[:, c])
        f = torch.zeros(n, a, 3, device=self.device)
        for c in range(4):
            holder = self.corner_held_by[:, c]
            held = holder >= 0
            node = int(self.net.corners[c])
            f[ar, holder.clamp_min(0)] += torch.where(held.unsqueeze(-1), pin_force[:, node], torch.zeros(n, 3, device=self.device))
        lim = 2.0 * GRAVITY * spec.DRONE_MASS
        self.net_force = f.clamp(-lim, lim)
        drones = torch.where(self.alive.unsqueeze(-1), self.pos, torch.full_like(self.pos, FAR))
        self.fish.step(spec.POLICY_DT, drones, self.water_height(self.fish.pos[..., :2]), self.net.pos, gen=self.gen)
        net_held = (self.corner_held_by >= 0).any(-1)
        self.fish.update_capture(self.net.contains(self.fish.pos), self.net.pos, self.net.vel, net_held)
        self.fish_frac = self.fish.caught.float().mean(-1)
        self.animals.step(spec.POLICY_DT, self.t, self.water_height(self.animals.pos[..., :2]), self.gen)
        new_tangle = self.animals.update_tangle(self.net.pos, self.net.vel, net_held)
        self.bycatch = self.animals.tangled.any(-1)
        self.animal_gap = self.animals.net_distance(self.net.pos)
        events["bycatch"] |= new_tangle.any(-1, keepdim=True).expand(n, a)
        zb = ZS["boat"]
        bz = self.water_height(self.zone_c[:, zb:zb + 1, :2])[:, 0]
        self.zone_c[:, zb, 2] = bz
        self.box_c[:, BOAT_SLOT, 2] = bz
        cen = self.net.centroid()
        delivered = ((cen[:, :2] - self.zone_c[:, zb, :2]).norm(dim=-1) < BOAT_RADIUS) & (cen[:, 2] > bz + 0.2) & \
            (self.fish_frac > 0) & self.zone_active[:, zb]
        events["delivered"] |= delivered.unsqueeze(-1).expand(n, a)

    def _handle_table(self):
        """Every grabbable handle: items first (indices 0..MAX_ITEMS-1), then net corners
        (corner_offset + c). Returns (handle pos, body pos, body vel [N, H, 3], free [N, H],
        grasp radius [H], is_item [H])."""
        hp, bp, bv, free, rad, isit = [], [], [], [], [], []
        if self.has_items:
            hp.append(self.item_handles())
            bp.append(self.it_pos)
            bv.append(self.it_vel)
            free.append(self.it_active & ~self.it_delivered & (self.it_held_by < 0))
            rad.append(torch.full((spec.MAX_ITEMS,), ITEM_GRASP_RADIUS, device=self.device))
            isit.append(torch.ones(spec.MAX_ITEMS, dtype=torch.bool, device=self.device))
        if self.has_net:
            nh = self.net_handles()
            hp.append(nh)
            bp.append(nh)
            bv.append(self.net.vel[:, self.net.corners])
            free.append(self.corner_held_by < 0)
            rad.append(torch.full((4,), NET_GRASP_RADIUS, device=self.device))
            isit.append(torch.zeros(4, dtype=torch.bool, device=self.device))
        return (torch.cat(hp, 1), torch.cat(bp, 1), torch.cat(bv, 1), torch.cat(free, 1), torch.cat(rad), torch.cat(isit))

    def _update_grip(self, actions: torch.Tensor, events) -> None:
        closed = (actions[..., 4] > spec.GRIP_CLOSED_THRESHOLD) & self.alive
        closing = closed & ~self.grip
        opening = ~closed & self.grip
        self.grip = closed
        n, a = self.num_envs, self.num_agents
        ar = torch.arange(n, device=self.device)
        off = self.corner_offset
        if opening.any():
            rel = opening & (self.holding >= 0)
            events["release"] |= rel
            if self.has_net:
                for c in range(4):
                    holder = self.corner_held_by[:, c]
                    rel_c = (holder >= 0) & rel[ar, holder.clamp_min(0)]
                    self.corner_held_by[:, c] = torch.where(rel_c, torch.full_like(holder, -1), holder)
            if self.has_items:
                held_rel = (self.it_held_by >= 0) & torch.gather(rel, 1, self.it_held_by.clamp_min(0))
                self.it_held_by = torch.where(held_rel, torch.full_like(self.it_held_by, -1), self.it_held_by)
                is_item = self.holding < off
                self.last_held = torch.where(rel & is_item, self.holding, self.last_held)
            self.holding = torch.where(rel, torch.full_like(self.holding, -1), self.holding)
        if not closing.any():
            return
        gp = self.gripper_pos()
        hpos, _, hvel, free, radius, isit = self._handle_table()
        d = (gp[:, :, None] - hpos[:, None]).norm(dim=-1)                       # [N, A, H]
        d = torch.where(free[:, None], d, torch.full_like(d, FAR))
        best, dbest = d.argmin(-1), d.min(-1).values
        ok = closing & (dbest < radius[best]) & (self.holding < 0)
        vrel = torch.gather((self.vel[:, :, None] - hvel[:, None]).norm(dim=-1), 2, best.unsqueeze(-1)).squeeze(-1)
        ok &= ~isit[best] | (vrel < spec.GRASP_MAX_REL_SPEED)
        for ag in range(a):          # sequential so two drones never take the same handle
            c = best[:, ag]
            item = isit[c]
            if self.has_items:
                ci = c.clamp(max=spec.MAX_ITEMS - 1)
                still_i = self.it_held_by[ar, ci] < 0
            else:
                ci, still_i = c, torch.zeros_like(item)
            if self.has_net:
                cc = (c - off).clamp(0, 3)
                still_c = self.corner_held_by[ar, cc] < 0
            else:
                cc, still_c = c, torch.zeros_like(item)
            take = ok[:, ag] & torch.where(item, still_i, still_c)
            if self.has_items:
                self.it_held_by[ar, ci] = torch.where(take & item, torch.full_like(ci, ag), self.it_held_by[ar, ci])
            if self.has_net:
                self.corner_held_by[ar, cc] = torch.where(take & ~item, torch.full_like(cc, ag), self.corner_held_by[ar, cc])
            self.holding[:, ag] = torch.where(take, c, self.holding[:, ag])
            self.last_held[:, ag] = torch.where(take, torch.full_like(c, -1), self.last_held[:, ag])
            events["grasp"][:, ag] |= take

    def _floor(self, pos: torch.Tensor) -> torch.Tensor:
        """Lowest allowed drone height [N, A] at pos: terrain or water plus clearance."""
        return self.surface_height(pos[..., :2]) + spec.DRONE_RADIUS + 0.05

    def _shield(self, v_cmd: torch.Tensor):
        bc = torch.cat([self.box_c, self.sph_c], 1)
        bh = torch.cat([self.box_h, self.sph_r.unsqueeze(-1).expand(-1, -1, 3)], 1)
        ba = torch.cat([self.box_active, self.sph_active], 1)
        floor = self._floor(self.pos + v_cmd * 0.5)
        return shield_velocity(self.pos, v_cmd, self.zone_c[:, 1:3], self.zone_r[:, 1:3],
                               self.zone_active[:, 1:3], bc, bh, ba, margin=0.6, floor=floor, vel=self.vel)

    def _collisions(self) -> torch.Tensor:
        r = spec.DRONE_RADIUS
        alive = self.alive
        rel = self.pos[:, :, None] - self.box_c[:, None]
        closest = torch.maximum(torch.minimum(rel, self.box_h[:, None]), -self.box_h[:, None])
        hit_box = (((rel - closest).norm(dim=-1) < r) & self.box_active[:, None]).any(-1)
        ds = (self.pos[:, :, None] - self.sph_c[:, None]).norm(dim=-1)
        hit_sph = ((ds < r + self.sph_r[:, None]) & self.sph_active[:, None]).any(-1)
        dd = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        hit_mate = ((dd < 2 * r) & ~eye & alive[:, :, None] & alive[:, None]).any(-1)
        ground = self.ground_height(self.pos[..., :2])
        water = self.water_height(self.pos[..., :2])
        low = self.pos[..., 2] - r < ground
        hard = low & (self.vel[..., 2] < -2.0)
        soft = low & ~hard & alive
        self.pos[..., 2] = torch.where(soft, ground + r, self.pos[..., 2])
        self.vel[..., 2] = torch.where(soft, self.vel[..., 2].clamp_min(0.0), self.vel[..., 2])
        wet = self.pos[..., 2] < water + 0.05
        hit_net = torch.zeros_like(wet)
        if self.has_net:                    # rotors and net ropes do not mix
            hit_net = self.net.segment_distance(self.pos) < r + 0.05
        hit = (hit_box | hit_sph | hit_mate | hard | wet | hit_net) & alive
        self.collided |= hit
        return hit.any(-1)

    # ================================================================ sensing
    def _geometry(self) -> Dict[str, torch.Tensor]:
        n = self.num_envs
        box_c, box_h, box_a = [self.box_c], [self.box_h], [self.box_active]
        sph_c = [torch.where(self.alive.unsqueeze(-1), self.pos, torch.full_like(self.pos, FAR)), self.sph_c]
        sph_r = [torch.full(self.pos.shape[:2], spec.DRONE_RADIUS * 0.8, device=self.device), self.sph_r]
        sph_a = [self.alive, self.sph_active]
        if self.has_items:
            box_c.append(self.it_pos); box_h.append(self.it_half); box_a.append(self.it_active & ~self.it_sphere)
            sph_c.append(self.it_pos); sph_r.append(self.it_half[..., 0]); sph_a.append(self.it_active & self.it_sphere)
        if self.has_net:
            ones = lambda k: torch.ones(n, k, dtype=torch.bool, device=self.device)
            sph_c += [self.fish.pos, self.net.pos, self.animals.pos]
            sph_r += [torch.full(self.fish.pos.shape[:2], self.fish.radius, device=self.device),
                      torch.full(self.net.pos.shape[:2], 0.12, device=self.device), self.animals.radius.expand(n, -1)]
            sph_a += [ones(self.fish.f), ones(self.net.p), ones(self.animals.k)]
        return {"box_c": torch.cat(box_c, 1), "box_h": torch.cat(box_h, 1), "box_active": torch.cat(box_a, 1),
                "sph_c": torch.cat(sph_c, 1), "sph_r": torch.cat(sph_r, 1), "sph_active": torch.cat(sph_a, 1)}

    def _scan(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        geom = {k: v[env_ids] for k, v in self._geometry().items()}
        ids = env_ids
        geom["terrain"] = lambda o, d: self.terrain.ray_hit(o, d, spec.LIDAR_MAX_RANGE, ids)
        water = None if bool((self.terrain.water_base[ids] <= NO_WATER / 2).all()) else SurfaceWater(self, ids)
        ext, ext_d = self._solid_ranges()
        imgs, downs, pts = [], [], None
        for ag in range(self.num_agents):
            g = dict(geom)
            if self._drone_spheres_first:
                sc = g["sph_c"].clone()
                sc[:, ag] = FAR
                g["sph_c"] = sc
            img, down, p = self.lidar_model.scan(
                self.pos[ids, ag:ag + 1], self.quat[ids, ag:ag + 1], g, waves=water, t=self.t[ids],
                turbidity=self.turbidity[ids],
                solid_range=None if ext is None else ext[ids, ag:ag + 1].reshape(len(ids), 1, -1),
                solid_range_down=None if ext_d is None else ext_d[ids, ag:ag + 1].reshape(len(ids), 1, -1))
            imgs.append(img)
            downs.append(down)
            if ag == 0:
                pts = p
        self._lidar[ids] = torch.cat(imgs, 1)
        self._lidar_down[ids] = torch.cat(downs, 1)
        for k, v in pts.items():
            if k not in self._lidar_pts:
                self._lidar_pts[k] = torch.full((self.num_envs,) + v.shape[1:], float("nan"), device=self.device)
            self._lidar_pts[k][ids] = v

    # ================================================================ observations
    def observe(self) -> Dict[str, torch.Tensor]:
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
        self._anchor_kind_now = akind
        inside = self._anchor_inside(apos, arad, akind)
        onehot = torch.nn.functional.one_hot(akind, len(spec.ANCHOR_TYPES)).float()
        obs["anchors"] = torch.cat([apos - self.pos.unsqueeze(2), onehot, arad.unsqueeze(-1), inside.float().unsqueeze(-1)], -1)
        if a > 1:
            for i in range(a):
                others = torch.tensor([j for j in range(a) if j != i][:spec.MAX_MATES], device=dev)
                k = len(others)
                obs["mates"][:, i, :k] = torch.cat([self.pos[:, others] - self.pos[:, i:i + 1],
                                                    self.vel[:, others] - self.vel[:, i:i + 1],
                                                    (self.holding[:, others] >= 0).float().unsqueeze(-1)], -1)
                obs["mates_mask"][:, i, :k] = self.alive[:, others]
        obs["lidar"] = self._lidar.clone()
        obs["lidar_down"] = self._lidar_down.clone()
        obs["priv"], obs["priv_mask"] = self._priv_tokens()
        obs["g"] = self._goal_space(apos, inside, height)
        return obs

    def _anchor_inside(self, apos, arad, akind) -> torch.Tensor:
        d_xy = (self.pos.unsqueeze(2)[..., :2] - apos[..., :2]).norm(dim=-1)
        d3 = (self.pos.unsqueeze(2) - apos).norm(dim=-1)
        inside = torch.where(arad > 0, d_xy < arad, d3 < 0.5)
        return inside & (akind != ANCHOR_KIND["none"])

    def _entity_table(self):
        """All physical entities as rows for privileged tokens: (pos, vel, size, type, active)."""
        n, dev = self.num_envs, self.device
        P = len(spec.PRIV_TYPES)
        rows = []

        def add(pos, vel, size, typ, act):
            if typ.dim() == 1:
                typ = typ.unsqueeze(0).expand(n, -1)
            rows.append((pos, vel, size, typ, act))

        bt = torch.where(self.box_kind == BK["boat"], PRIV_TYPE["boat"], PRIV_TYPE["box"])
        add(self.box_c, self.box_v, 2 * self.box_h, bt, self.box_active)
        add(self.sph_c, self.sph_v, 2 * self.sph_r.unsqueeze(-1).expand(-1, -1, 3),
            torch.full_like(self.sph_kind, PRIV_TYPE["sphere"]), self.sph_active)
        if self.has_items:
            is_trash = self.catalog.is_trash[self.it_type]
            cooler = self.it_type == self.catalog.id("cooler")
            it = torch.where(cooler, PRIV_TYPE["cooler"], torch.where(is_trash, PRIV_TYPE["item_trash"], PRIV_TYPE["item_other"]))
            add(self.it_pos, self.it_vel, 2 * self.it_half, it, self.it_active & ~(self.it_delivered & is_trash))
        zsize = torch.stack([self.zone_r, self.zone_r, torch.where(self.zone_kind == ZK["nofly"], 20.0, 1.0)], -1)
        add(self.zone_c, torch.zeros_like(self.zone_c), zsize, torch.full_like(self.zone_kind, PRIV_TYPE["zone"]), self.zone_active)
        if self.has_net:
            ext = self.net.pos.amax(1) - self.net.pos.amin(1)
            add(self.net.centroid()[:, None], self.net.vel.mean(1, keepdim=True), ext[:, None],
                torch.full((n, 1), PRIV_TYPE["net"], device=dev), torch.ones(n, 1, dtype=torch.bool, device=dev))
            sc = self.fish.school_centers()
            add(sc, torch.zeros_like(sc), torch.tensor([2.0, 2.0, 1.0], device=dev).expand_as(sc),
                torch.full(sc.shape[:2], PRIV_TYPE["fish_school"], device=dev), torch.ones(sc.shape[:2], dtype=torch.bool, device=dev))
            ar = self.animals.radius.view(1, -1, 1).expand(n, -1, 3) * 2
            add(self.animals.pos, self.animals.vel, ar, torch.full((n, self.animals.k), PRIV_TYPE["animal"], device=dev),
                torch.ones(n, self.animals.k, dtype=torch.bool, device=dev))
        pos = torch.cat([r[0] for r in rows], 1)
        vel = torch.cat([r[1] for r in rows], 1)
        size = torch.cat([r[2] for r in rows], 1)
        typ = torch.cat([r[3] for r in rows], 1)
        act = torch.cat([r[4] for r in rows], 1)
        return pos, vel, size, torch.nn.functional.one_hot(typ.long(), P).float(), act

    def _priv_tokens(self):
        n, a = self.num_envs, self.num_agents
        pos, vel, size, typ, act = self._entity_table()
        E = pos.shape[1]
        k = min(spec.MAX_PRIV, E)
        rel = pos[:, None] - self.pos[:, :, None]                               # [N, A, E, 3]
        dist = torch.where(act[:, None], rel.norm(dim=-1), torch.full(rel.shape[:3], FAR, device=self.device))
        dk, idx = torch.topk(dist, k, dim=-1, largest=False)
        g3 = lambda x: torch.gather(x[:, None].expand(n, a, E, x.shape[-1]), 2, idx.unsqueeze(-1).expand(n, a, k, x.shape[-1]))
        tok = torch.cat([torch.gather(rel, 2, idx.unsqueeze(-1).expand(n, a, k, 3)), g3(vel), g3(size), g3(typ)], -1)
        priv = torch.zeros(n, a, spec.MAX_PRIV, spec.PRIV_DIM, device=self.device)
        mask = torch.zeros(n, a, spec.MAX_PRIV, dtype=torch.bool, device=self.device)
        priv[:, :, :k] = tok
        mask[:, :, :k] = dk < FAR / 2
        return torch.where(mask.unsqueeze(-1), priv, torch.zeros_like(priv)), mask

    def payload_index(self) -> torch.Tensor:
        """[N, A] handle-table index treated as the payload: held; else the last-held item (so
        'released at the bin' stays measurable) unless anchor 0 now points at something else;
        else the item or net corner bound to anchor 0; else the nearest available handle."""
        hpos, bpos, _, free, _, isit = self._handle_table()
        if self.has_items:
            it_ok = self.it_active & ~self.it_delivered
            avail = torch.cat([it_ok, torch.ones(self.num_envs, hpos.shape[1] - spec.MAX_ITEMS, dtype=torch.bool,
                                                 device=self.device)], 1)
        else:
            avail = torch.ones_like(free)
        held = self.holding
        a0 = self.anchor_handle()
        a0_ok = (a0 >= 0) & torch.gather(avail, 1, a0.clamp_min(0))
        last = self.last_held
        last_ok = (last >= 0) & ((a0 < 0) | (a0 == last))
        d = (self.gripper_pos()[:, :, None] - hpos[:, None]).norm(dim=-1)
        d = torch.where(avail[:, None], d, torch.full_like(d, FAR))
        return torch.where(held >= 0, held, torch.where(last_ok, last, torch.where(a0_ok, a0, d.argmin(-1))))

    def _goal_space(self, apos, inside, height) -> torch.Tensor:
        n, a = self.num_envs, self.num_agents
        dev = self.device
        g = torch.zeros(n, a, spec.G_DIM, device=dev)
        Sl = spec.G_SLICES
        g[..., Sl["drone_vel"]] = self.vel
        g[..., Sl["speed"]] = self.vel.norm(dim=-1, keepdim=True)
        g[..., Sl["tilt"]] = tilt_of(self.quat).unsqueeze(-1)
        g[..., Sl["height"]] = height.unsqueeze(-1)
        over = self.water_height(self.pos[..., :2]) > self.ground_height(self.pos[..., :2])
        g[..., Sl["over_water"]] = over.float().unsqueeze(-1)
        g[..., Sl["thrust"]] = self.hover.unsqueeze(-1)
        g[..., Sl["grip_closed"]] = self.grip.float().unsqueeze(-1)
        holding = self.holding >= 0
        g[..., Sl["holding"]] = holding.float().unsqueeze(-1)
        pidx = self.payload_index()
        _, bpos, bvel, _, _, isit = self._handle_table()
        idx3 = pidx.unsqueeze(-1).expand(n, a, 3)
        ppos = torch.gather(bpos, 1, idx3)                     # item centre, or the corner's grab point
        pvel = torch.gather(bvel, 1, idx3)
        if self.has_items:
            is_item = isit[pidx]
            ii = pidx.clamp(max=spec.MAX_ITEMS - 1)
            ptype = torch.gather(self.it_type, 1, ii)
            any_item = (self.it_active | self.it_delivered).any(-1, keepdim=True) & is_item
            g[..., Sl["payload_type"]] = torch.nn.functional.one_hot(ptype, len(spec.ITEM_TYPES)).float() * any_item.unsqueeze(-1).float()
            g[..., Sl["payload_is_trash"]] = (self.catalog.is_trash[ptype] & any_item).float().unsqueeze(-1)
            mass = torch.gather(self.it_mass, 1, ii)
            g[..., Sl["held_mass"]] = torch.where(holding & is_item, mass, torch.zeros_like(mass)).unsqueeze(-1)
            g[..., Sl["swing"]] = self.swing.unsqueeze(-1)
            left = (self.it_active & ~self.it_delivered & self.catalog.is_trash[self.it_type]).float().sum(-1)
            g[..., Sl["trash_left"]] = (left / self.trash_init.clamp_min(1.0)).view(n, 1, 1).expand(n, a, 1)
        if self.has_net:
            g[..., Sl["net_centroid_rel"]] = self.net.centroid().unsqueeze(1) - self.pos
            g[..., Sl["net_stretch"]] = self.net_stretch.view(n, 1, 1).expand(n, a, 1)
            g[..., Sl["fish_in_net"]] = self.fish_frac.view(n, 1, 1).expand(n, a, 1)
            g[..., Sl["animal_dist"]] = self.animal_gap.clamp(max=spec.FAR_DIST).view(n, 1, 1).expand(n, a, 1)
            g[..., Sl["bycatch"]] = self.bycatch.float().view(n, 1, 1).expand(n, a, 1)
        else:
            g[..., Sl["animal_dist"]] = spec.FAR_DIST
        g[..., Sl["payload_rel"]] = ppos - self.pos
        g[..., Sl["payload_vel"]] = pvel
        g[..., Sl["jerk"]] = self.jerk.unsqueeze(-1).clamp(max=500.0)
        g[..., Sl["anchor_rel_drone"]] = (apos - self.pos.unsqueeze(2)).reshape(n, a, -1)
        g[..., Sl["anchor_rel_payload"]] = (apos - ppos.unsqueeze(2)).reshape(n, a, -1)
        g[..., Sl["anchor_inside"]] = inside.float()
        g[..., Sl["anchor_active"]] = (self._anchor_kind_now != ANCHOR_KIND["none"]).float()
        # nearest obstacle (boxes and spheres)
        rel = self.pos[:, :, None] - self.box_c[:, None]
        closest = torch.maximum(torch.minimum(rel, self.box_h[:, None]), -self.box_h[:, None])
        vec_b = rel - closest
        d_b = torch.where(self.box_active[:, None], vec_b.norm(dim=-1), torch.full(vec_b.shape[:3], FAR, device=dev))
        vec_s = self.pos[:, :, None] - self.sph_c[:, None]
        d_s = vec_s.norm(dim=-1) - self.sph_r[:, None]
        d_s = torch.where(self.sph_active[:, None], d_s, torch.full_like(d_s, FAR))
        dist = torch.cat([d_b, d_s], -1) - spec.DRONE_RADIUS
        vec = torch.cat([vec_b, vec_s], 2)
        ovel = torch.cat([self.box_v, self.sph_v], 1)
        dmin, imin = dist.min(-1)
        nvec = torch.gather(vec, 2, imin[..., None, None].expand(n, a, 1, 3)).squeeze(2)
        nrm = nvec / nvec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bvel = torch.gather(ovel[:, None].expand(n, a, -1, 3), 2, imin[..., None, None].expand(n, a, 1, 3)).squeeze(2)
        g[..., Sl["obstacle_dist"]] = dmin.clamp(max=spec.FAR_DIST).unsqueeze(-1)
        g[..., Sl["obstacle_closing"]] = (-((self.vel - bvel) * nrm).sum(-1)).unsqueeze(-1)
        if a > 1:
            dd = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
            ok = self.alive[:, None] & ~torch.eye(a, dtype=torch.bool, device=dev)
            g[..., Sl["mate_dist"]] = torch.where(ok, dd, torch.full_like(dd, spec.FAR_DIST)).min(-1).values.clamp(
                max=spec.FAR_DIST).unsqueeze(-1)
        else:
            g[..., Sl["mate_dist"]] = spec.FAR_DIST
        g[..., Sl["collision"]] = self.collided.float().unsqueeze(-1)
        g[..., Sl["shield"]] = self.shield_hit.float().unsqueeze(-1)
        return g

    # ================================================================ state copy (CEM)
    _STATE = ("t", "steps", "variant", "pos", "vel", "ang", "quat", "mass", "drag", "alive", "grip", "holding",
              "last_held", "last_action", "hover", "acc_prev", "vel_prev", "jerk", "item_force", "net_force", "v_int", "collided", "shield_hit",
              "shield_count", "swing", "wind", "wind_mean", "gust", "gust_time",
              "box_c", "box_h", "box_v", "box_active", "box_moving", "box_kind", "lane_a", "lane_b", "lane_speed",
              "lane_phase", "sph_c", "sph_v", "sph_r", "sph_active", "sph_moving", "sph_kind", "orb_c", "orb_r",
              "orb_w", "orb_ph", "zone_c", "zone_r", "zone_drop_h", "zone_active", "zone_kind",
              "it_type", "it_active", "it_delivered", "it_pos", "it_vel", "it_half", "it_sphere", "it_mass",
              "it_held_by", "trash_init", "turbidity", "net_stretch", "fish_frac", "bycatch", "animal_gap",
              "corner_held_by", "anc_kind", "anc_src", "anc_sel", "anc_point", "anc_radius", "exp_phase", "exp_cruise",
              "_lidar", "_lidar_down")
    _SUBS = ("terrain", "waves", "net", "fish", "animals")

    def get_state(self) -> Dict[str, torch.Tensor]:
        st = {k: getattr(self, k).clone() for k in self._STATE}
        for name in self._SUBS:
            sub = getattr(self, name)
            if sub is not None:
                for k, v in sub.state().items():
                    st[f"{name}.{k}"] = v.clone()
        return st

    def set_state(self, state: Dict[str, torch.Tensor], env_ids: Optional[torch.Tensor] = None,
                  src_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        src = env_ids if src_ids is None else torch.as_tensor(src_ids, device=self.device).reshape(-1)
        for k in self._STATE:
            getattr(self, k)[env_ids] = state[k][src]
        for name in self._SUBS:
            sub = getattr(self, name)
            if sub is not None:
                sub.load({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(name + ".")}, env_ids, src)
        self._reset_backend(env_ids)

    def copy_state(self, src: int, dst_ids: torch.Tensor) -> None:
        dst_ids = torch.as_tensor(dst_ids, device=self.device).reshape(-1)
        self.set_state(self.get_state(), dst_ids, torch.full_like(dst_ids, int(src)))

    # ================================================================ expert
    def expert_action(self) -> torch.Tensor:
        from .expert import expert_action
        return expert_action(self)

    # ================================================================ demo
    def perturb(self, env_id: int, kind: str, **kw) -> None:
        i = int(env_id)
        dev = self.device
        if kind == "drop_obstacle":
            b = DROPPED_SLOT
            p = torch.tensor(kw.get("pos", [0.0, 0.0]), dtype=torch.float32, device=dev)[:2]
            half = float(kw.get("half", 0.6))
            base = float(self.surface_height(p.view(1, 1, 2).expand(self.num_envs, 1, 2))[i, 0])
            self.box_h[i, b] = half
            self.box_c[i, b] = torch.cat([p, torch.tensor([base + half], device=dev)])
            self.box_active[i, b] = True
            self.box_moving[i, b] = False
            self.box_kind[i, b] = BK["dropped"]
        elif kind == "gust":
            if not (self.cfg.domain_rand and self.cfg.enable_gusts):
                return
            self.gust[i] = torch.tensor(kw.get("vec", [4.0, 0.0, 0.0]), dtype=torch.float32, device=dev)
            self.gust_time[i] = float(kw.get("duration", 2.0))
        elif kind == "waves":
            if self.waves is not None:
                self.waves.scale[i] = float(kw.get("scale", 1.5))
        elif kind == "tide":
            self.terrain.tide_offset[i] = float(kw.get("offset", 0.5))
        elif kind == "move_payload":
            if self.has_items:
                k = int(kw.get("item", max(int(self.anchor_item()[i, 0]), 0)))
                p = torch.tensor(kw.get("pos", [0.0, 0.0]), dtype=torch.float32, device=dev)[:2]
                if int(self.it_held_by[i, k]) >= 0:
                    self.holding[i, int(self.it_held_by[i, k])] = -1
                self.it_held_by[i, k] = -1
                gz = float(self.terrain.height(p.view(1, 1, 2).expand(self.num_envs, 1, 2))[i, 0])
                self.it_pos[i, k] = torch.cat([p, torch.tensor([gz], device=dev) + self.it_half[i, k, 2:3]])
                self.it_vel[i, k] = 0.0
        elif kind == "disable_drone":
            ag = int(kw.get("agent", 0))
            self.alive[i, ag] = False
            self.holding[i, ag] = -1
            if self.has_net:
                ch = self.corner_held_by[i]
                self.corner_held_by[i] = torch.where(ch == ag, torch.full_like(ch, -1), ch)
            if self.has_items:
                hb = self.it_held_by[i]
                self.it_held_by[i] = torch.where(hb == ag, torch.full_like(hb, -1), hb)
            self.pos[i, ag] = torch.tensor([0.0, 0.0, -50.0], device=dev)
            self.vel[i, ag] = 0.0
        else:
            raise ValueError(f"unknown perturbation {kind!r}")
        self._reset_backend(torch.tensor([i], device=dev))

    def render_state(self, env_id: int, include_terrain: bool = True) -> dict:
        """JSON-serialisable frame (schema in skyhand/spec.py). Terrain heights are included
        when include_terrain is True; terrain.version tells the client when they change."""
        i = int(env_id)
        L = lambda x: [round(float(v), 3) for v in x.reshape(-1).tolist()]
        st = {"t": float(self.t[i]), "scene": self.scene, "variant": VARIANTS[int(self.variant[i])],
              "arena_half": self.arena_half, "max_speed": self.max_speed, "shield_hits": int(self.shield_count[i]),
              "wind": L(self.wind[i] + self.gust[i])}
        tr = self.terrain
        st["terrain"] = None if (self.scene == "ocean") else {
            "version": int(tr.version[i]), "res": tr.res, "half": tr.half,
            "heights": [L(row) for row in tr.h[i]] if include_terrain else None}
        lvl = float(tr.water_level(self.t[i:i + 1], torch.tensor([i], device=self.device))[0])
        st["water"] = None if lvl <= NO_WATER / 2 else {
            "level": round(lvl, 3), "waves": self.waves.params(i) if self.waves is not None else None}
        st["drones"] = [{"pos": L(self.pos[i, j]), "quat": L(self.quat[i, j]), "grip": bool(self.grip[i, j]),
                         "holding": bool(self.holding[i, j] >= 0), "alive": bool(self.alive[i, j])}
                        for j in range(self.num_agents)]
        listed = (self.it_active[i] | self.it_delivered[i]) if self.has_items else None
        st["items"] = [{"name": f"item_{k}", "type": spec.ITEM_TYPES[int(self.it_type[i, k])],
                        "is_trash": bool(spec.ITEM_IS_TRASH[int(self.it_type[i, k])]),
                        "shape": "sphere" if bool(self.it_sphere[i, k]) else "box",
                        "pos": L(self.it_pos[i, k]), "size": L(self.it_half[i, k] * 2),
                        "held": bool(self.it_held_by[i, k] >= 0), "delivered": bool(self.it_delivered[i, k])}
                       for k in range(spec.MAX_ITEMS) if bool(listed[k])] if self.has_items else []
        st["boxes"] = [{"name": f"box_{b}", "kind": BOX_KINDS[int(self.box_kind[i, b])], "pos": L(self.box_c[i, b]),
                        "size": L(self.box_h[i, b] * 2), "moving": bool(self.box_moving[i, b])}
                       for b in range(spec.MAX_BOXES) if bool(self.box_active[i, b]) and int(self.box_kind[i, b]) != BK["boat"]]
        st["spheres"] = [{"name": f"sphere_{s}", "kind": SPHERE_KINDS[int(self.sph_kind[i, s])], "pos": L(self.sph_c[i, s]),
                          "radius": round(float(self.sph_r[i, s]), 3), "moving": bool(self.sph_moving[i, s])}
                         for s in range(spec.MAX_SPHERES) if bool(self.sph_active[i, s])]
        znames = {v: k for k, v in ZS.items()}
        zones = []
        for z in range(spec.N_ZONES):
            if not bool(self.zone_active[i, z]):
                continue
            kind = ZONE_KINDS[int(self.zone_kind[i, z])]
            name = znames[z]
            height = {"nofly": 20.0, "truck": 3.0}.get(kind, 1.0)          # the truck zone covers its 3 m bed
            zones.append({"name": name, "pos": L(self.zone_c[i, z]), "radius": float(self.zone_r[i, z]),
                          "height": height, "kind": kind})
        st["zones"] = zones
        apos, arad, akind = self.anchor_positions()
        st["anchors"] = [{"pos": L(apos[i, 0, k]), "kind": spec.ANCHOR_TYPES[int(akind[i, 0, k])], "radius": float(arad[i, 0, k])}
                         for k in range(spec.N_ANCHORS)]
        if self.has_net:
            st["boat"] = {"pos": L(self.zone_c[i, ZS["boat"]]), "quat": [1, 0, 0, 0], "size": [2 * x for x in BOAT_HALF]}
            st["net"] = {"rows": self.net.rows, "cols": self.net.cols, "nodes": [L(p) for p in self.net.pos[i]],
                         "held": [int((self.corner_held_by[i] == j).nonzero()[0]) if bool((self.corner_held_by[i] == j).any()) else -1
                                  for j in range(self.num_agents)]}
            st["fish"] = [L(p) for p in self.fish.pos[i]]
            st["animals"] = [{"name": nm, "type": nm.split("_")[0], "pos": L(self.animals.pos[i, k]),
                              "vel": L(self.animals.vel[i, k]), "radius": float(self.animals.radius[k]),
                              "tangled": bool(self.animals.tangled[i, k])} for k, nm in enumerate(self.animals.names)]
        else:
            st.update(boat=None, net=None, fish=[], animals=[])
        lid = {}
        for k in ("nir", "green"):
            p = self._lidar_pts.get(k)
            if p is None:
                lid[k] = []
            else:
                p = p[i]
                lid[k] = [L(x) for x in p[torch.isfinite(p).all(-1)]]
        st["lidar"] = lid
        return st


class SurfaceWater:
    """Water surface for the LiDAR over a subset of worlds: tide level plus waves (if any),
    present only where the terrain is lower than the surface (sea, lakes, rivers)."""

    def __init__(self, env: "SkyEnv", ids: torch.Tensor):
        self.env, self.ids = env, ids
        self.waves = _SubWaves(env.waves, ids) if env.waves is not None else None

    def height(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lvl = self.env.terrain.water_level(t, self.ids).view(-1, *([1] * (xy.dim() - 2))).expand(xy.shape[:-1])
        return lvl + self.waves.height(xy, t) if self.waves is not None else lvl

    def ray_hit(self, o: torch.Tensor, d: torch.Tensor, t: torch.Tensor, iters: int = 3) -> torch.Tensor:
        inf = torch.full(o.shape[:2], float("inf"), device=o.device)
        eta = self.height(o[..., :2], t)
        dz = d[..., 2]
        above = o[..., 2] > eta
        going = torch.where(above, dz < -1e-6, dz > 1e-6)
        safe = torch.where(going, dz, torch.ones_like(dz))
        s = ((eta - o[..., 2]) / safe).clamp_min(0.0)
        for _ in range(iters):                      # fixed point on the moving surface
            p = o + d * s.unsqueeze(-1)
            s = torch.where(going, ((self.height(p[..., :2], t) - o[..., 2]) / safe).clamp_min(0.0), s)
        p = o + d * s.unsqueeze(-1)
        wet = self.env.terrain.height(p[..., :2], self.ids) < self.height(p[..., :2], t)
        lvl = self.env.terrain.water_level(t, self.ids)[:, None]
        return torch.where(going & wet & (lvl > NO_WATER / 2), s, inf)


class _SubWaves:
    """View of a WaveField restricted to a subset of worlds (for partial LiDAR scans)."""

    _KEYS = ("amp", "k", "dir", "omega", "phase", "scale")

    def __init__(self, w: WaveField, ids: torch.Tensor):
        self.w, self.ids = w, ids

    def _call(self, fn, *args):
        full = {k: getattr(self.w, k) for k in self._KEYS}
        for k in self._KEYS:
            setattr(self.w, k, full[k][self.ids])
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
    """Factory. backend="torch" runs anywhere; backend="genesis" needs genesis-world."""
    cfg = EnvConfig(scene=scene, num_envs=num_envs, device=device)
    for k, v in kw.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown EnvConfig field {k!r}")
        setattr(cfg, k, v)
    if backend == "torch":
        return SkyEnv(cfg)
    if backend == "genesis":
        from .backends.genesis_backend import GenesisSkyEnv
        return GenesisSkyEnv(cfg)
    raise ValueError(f"unknown backend {backend!r}")
