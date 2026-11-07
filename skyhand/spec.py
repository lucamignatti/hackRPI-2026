"""Shared contract between env backends, training, mission, and demo code.

Depends only on torch and the standard library so it imports wherever PyTorch does.
Python >= 3.9: keep `from __future__ import annotations`, no match statements.

Conventions
-----------
- World frame: z up, metres, seconds. Water surface rest height is z = 0 in the
  ocean scene; ground is z = 0 in the trash scene.
- Quaternions are (w, x, y, z), same as Genesis.
- Batched tensors are [N, A, ...]: N parallel worlds, A agents (drones) per world.
  A is fixed per env instance (trash: 1, ocean: OCEAN_AGENTS).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import torch

# ---------------------------------------------------------------- timing
PHYS_DT = 0.01            # physics step (s)
SUBSTEPS = 5              # physics steps per policy step -> 20 Hz policy
POLICY_DT = PHYS_DT * SUBSTEPS
LIDAR_EVERY = 2           # policy steps between LiDAR scans -> 10 Hz
EPISODE_STEPS = 900       # 45 s episodes (default; see SCENE_EPISODE_STEPS)
Z_RESAMPLE_STEPS = 100    # policy steps a world keeps its z during pretraining

# ---------------------------------------------------------------- action
ACTION_DIM = 5            # [vx, vy, vz, yaw_rate, grip] each in [-1, 1]
MAX_SPEED = 6.0           # m/s, scale for vx, vy (default; see SCENE_MAX_SPEED)
MAX_VZ = 3.0              # m/s, scale for vz (default; see SCENE_MAX_VZ)
MAX_YAW_RATE = 2.0        # rad/s
GRIP_CLOSED_THRESHOLD = 0.0   # grip > 0 means closed

# ---------------------------------------------------------------- drone
DRONE_MASS = 2.0          # kg
DRONE_RADIUS = 0.35       # m, collision sphere
GRIPPER_OFFSET = (0.0, 0.0, -0.18)   # gripper point in body frame (m)
GRASP_RADIUS = 0.10       # m, handle must be this close to the gripper
GRASP_MAX_REL_SPEED = 0.6 # m/s, max relative speed to grasp
MAX_TILT = 0.6            # rad, controller tilt limit

# ---------------------------------------------------------------- scenes
# trash:    each world is either a city (street grid of buildings, trash piled at the
#           curb, a garbage truck as drop-off) or open terrain with scattered litter.
# delivery: long-range, fast organ delivery between two remote clinics (Zipline-style,
#           rural Africa): a 600 m arena of hills, mesas, a river valley and lakes,
#           a clinic building at each pad, acacia trees, vultures, large no-fly
#           areas such as a wildlife reserve.
# ocean:    team net fishing among fish schools and marine animals (bycatch).
# world:    the generic pretraining world that contains all of the above at once: a
#           400 m map with countryside, lakes, a coastline into open sea, a town with
#           trash piles and a garbage truck, two clinics, a boat with the net, fish,
#           animals, birds, and no-fly zones. Each task is a subset of this world.
SCENES = ("trash", "delivery", "ocean", "world")
TRASH_AGENTS = 1
DELIVERY_AGENTS = 1
OCEAN_AGENTS = 4          # one drone per net corner
WORLD_AGENTS = 3
SCENE_AGENTS = {"trash": TRASH_AGENTS, "delivery": DELIVERY_AGENTS, "ocean": OCEAN_AGENTS, "world": WORLD_AGENTS}
ARENA_HALF = 25.0         # trash and ocean are 50 x 50 m
# Per-scene physical scale. Actions are normalised: vx, vy in [-1, 1] times the
# scene's max speed. Delivery is a streamlined fast drone with far less drag.
SCENE_ARENA_HALF = {"trash": 25.0, "delivery": 300.0, "ocean": 25.0, "world": 200.0}
SCENE_MAX_SPEED = {"trash": 6.0, "delivery": 18.0, "ocean": 6.0, "world": 12.0}
SCENE_MAX_VZ = {"trash": 4.0, "delivery": 6.0, "ocean": 3.0, "world": 5.0}
SCENE_DRAG = {"trash": 0.35, "delivery": 0.035, "ocean": 0.35, "world": 0.08}
SCENE_EPISODE_STEPS = {"trash": 900, "delivery": 1800, "ocean": 900, "world": 1800}
SCENE_TERRAIN_RES = {"trash": 64, "delivery": 64, "ocean": 64, "world": 128}
TERRAIN_RES = 64          # heightfield samples per side (default; see SCENE_TERRAIN_RES)
MAX_BOXES = 64            # buildings (packed city lots), cars, benches, hydrants, truck, boat hull, tree trunks ...
MAX_SPHERES = 40          # rocks, tree canopies, birds
MAX_ITEMS = 16            # graspable objects per world
NET_ROWS = 8
NET_COLS = 8
N_FISH_SCHOOLS = 2
FISH_PER_SCHOOL = 16
N_WAVES = 4
ANIMAL_TYPES = ("whale", "dolphin", "turtle")
ANIMAL_COUNTS = (1, 2, 2)  # per ocean world
N_ANIMALS = sum(ANIMAL_COUNTS)

# ---------------------------------------------------------------- graspable items
# name, shape ("box" or "sphere"), size (box: full x, y, z; sphere: radius, -, -),
# mass (kg), is_trash. Shapes are chosen to be easy to tell apart on LiDAR:
# a round bag, a flat pizza box, a cube, a thin tall bottle, a tall narrow backpack ...
ITEM_CATALOG = (
    ("trash_bag",     "sphere", (0.28, 0.0, 0.0),   0.8, True),
    ("pizza_box",     "box",    (0.45, 0.45, 0.06), 0.4, True),
    ("cardboard_box", "box",    (0.40, 0.40, 0.40), 0.6, True),
    ("bottle",        "box",    (0.09, 0.09, 0.30), 0.2, True),
    ("backpack",      "box",    (0.32, 0.20, 0.45), 1.0, False),
    ("ball",          "sphere", (0.12, 0.0, 0.0),   0.4, False),
    ("flower_pot",    "box",    (0.30, 0.30, 0.35), 1.2, False),
    ("cooler",        "box",    (0.45, 0.30, 0.30), 1.3, False),   # organ transport, delivery scene
)
ITEM_TYPES = tuple(c[0] for c in ITEM_CATALOG)
ITEM_IS_TRASH = tuple(c[4] for c in ITEM_CATALOG)

# ---------------------------------------------------------------- LiDAR
LIDAR_CH = 3              # 0: NIR range, 1: green range, 2: green intensity
LIDAR_V, LIDAR_H = 16, 64 # vertical channels x azimuth bins, spherical pattern
LIDAR_V_FOV = (-45.0, 15.0)   # degrees, elevation range of the spherical pattern
DOWN_V, DOWN_H = 8, 8     # downward grid under the gripper
DOWN_HALF_WIDTH = 0.6     # m, half width of the downward grid at 1 m depth
LIDAR_MAX_RANGE = 30.0    # m
GREEN_MAX_DEPTH = 6.0     # m, green returns vanish below this depth
# Stored as uint8: range -> round(255 * range / LIDAR_MAX_RANGE); no return -> 255.
# Intensity -> round(255 * intensity), intensity in [0, 1].

# ---------------------------------------------------------------- anchors
N_ANCHORS = 4
ANCHOR_TYPES = ("none", "point", "handle", "zone")
ANCHOR_DIM = 3 + len(ANCHOR_TYPES) + 2   # rel pos (drone frame, world-aligned), type one-hot, radius, inside

# ---------------------------------------------------------------- tokens
# proprio: vel (3, world), projected gravity in body (3), ang vel body (3),
#          grip_closed, holding, height, last action (5)
PROPRIO_DIM = 3 + 3 + 3 + 1 + 1 + 1 + ACTION_DIM
MAX_MATES = OCEAN_AGENTS - 1
MATE_DIM = 3 + 3 + 1                     # rel pos, rel vel, holding
MAX_PRIV = 24             # the nearest MAX_PRIV entities to each drone
PRIV_TYPES = ("box", "sphere", "item_trash", "item_other", "cooler", "net", "fish_school", "boat",
              "zone", "animal")
PRIV_DIM = 3 + 3 + 3 + len(PRIV_TYPES)   # rel pos, vel, size, type one-hot

# ---------------------------------------------------------------- goal space g
# Ordered (name, size, scale). B, the discriminator, and reward programs read g.
# scale is a fixed normaliser: networks see g / scale. Reward programs see raw g.
_G_FIELDS = [
    ("drone_pos", 3, 25.0),        # world position
    ("drone_vel", 3, 6.0),
    ("speed", 1, 6.0),
    ("tilt", 1, 1.0),              # rad
    ("height", 1, 10.0),           # above the terrain or water surface below the drone
    ("over_water", 1, 1.0),        # 1 if the surface below the drone is water
    ("thrust", 1, 1.0),            # collective thrust / hover thrust
    ("grip_closed", 1, 1.0),
    ("holding", 1, 1.0),
    ("held_mass", 1, 2.0),
    ("payload_rel", 3, 25.0),      # payload position minus drone position. Payload = the held item, else
                                   # the last-held item (until another is grasped or a0 moves on), else
                                   # the item bound to a0, else the nearest available item
    ("payload_type", len(ITEM_TYPES), 1.0),   # one-hot item type of the payload (zeros: net corner / none)
    ("payload_is_trash", 1, 1.0),
    ("payload_vel", 3, 6.0),
    ("swing", 1, 3.0),             # payload angular rate about the gripper (rad/s)
    ("jerk", 1, 50.0),             # drone jerk magnitude (m/s^3)
    ("anchor_rel_drone", 3 * N_ANCHORS, 25.0),    # anchor k position minus drone position
    ("anchor_rel_payload", 3 * N_ANCHORS, 25.0),  # anchor k position minus payload position
    ("anchor_inside", N_ANCHORS, 1.0),            # drone inside anchor k zone
    ("anchor_active", N_ANCHORS, 1.0),            # 1 if anchor k is in use (an unused or unmatched slot reads
                                                  # as zero offsets, indistinguishable from an anchor at the drone)
    ("obstacle_dist", 1, 10.0),    # distance from drone surface to nearest obstacle
    ("obstacle_closing", 1, 6.0),  # closing speed toward it (positive = approaching)
    ("trash_left", 1, 1.0),        # fraction of this world's trash not yet delivered (trash scene)
    ("net_centroid_rel", 3, 25.0), # net centroid minus drone position (zeros outside ocean)
    ("net_stretch", 1, 1.0),       # max edge stretch ratio minus 1
    ("fish_in_net", 1, 1.0),       # fraction of fish inside the net volume
    ("animal_dist", 1, 10.0),      # nearest marine animal to any net node (ocean; large elsewhere)
    ("bycatch", 1, 1.0),           # 1 while an animal is tangled in the net
    ("mate_dist", 1, 10.0),        # nearest teammate distance (ARENA_HALF*2 if none)
    ("collision", 1, 1.0),         # 1 on a step with a collision
    ("shield", 1, 1.0),            # 1 on a step where the shield intervened
]
G_SLICES: "OrderedDict[str, slice]" = OrderedDict()
_i = 0
for _name, _size, _ in _G_FIELDS:
    G_SLICES[_name] = slice(_i, _i + _size)
    _i += _size
G_DIM = _i
G_FIELD_SIZES: Dict[str, int] = {n: s for n, s, _ in _G_FIELDS}


def g_scale(device=None) -> torch.Tensor:
    """Per-dimension normaliser for g, shape [G_DIM]."""
    parts = [torch.full((s,), float(c)) for _, s, c in _G_FIELDS]
    return torch.cat(parts).to(device)


def g_field(g: torch.Tensor, name: str) -> torch.Tensor:
    """View of one field of g, keeps the trailing dimension."""
    return g[..., G_SLICES[name]]


def anchor_rel(g: torch.Tensor, name: str, k: int) -> torch.Tensor:
    """anchor_rel_drone / anchor_rel_payload for anchor k, shape [..., 3]."""
    return g_field(g, name)[..., 3 * k: 3 * k + 3]


# ---------------------------------------------------------------- observation dict
# Every backend's reset() and step() return obs with exactly these keys/shapes.
OBS_SPEC = {
    "proprio": ("float32", (PROPRIO_DIM,)),
    "anchors": ("float32", (N_ANCHORS, ANCHOR_DIM)),
    "mates": ("float32", (MAX_MATES, MATE_DIM)),
    "mates_mask": ("bool", (MAX_MATES,)),          # True = slot used
    "lidar": ("uint8", (LIDAR_CH, LIDAR_V, LIDAR_H)),
    "lidar_down": ("uint8", (LIDAR_CH, DOWN_V, DOWN_H)),
    "priv": ("float32", (MAX_PRIV, PRIV_DIM)),     # critics only
    "priv_mask": ("bool", (MAX_PRIV,)),
    "g": ("float32", (G_DIM,)),
}
ACTOR_KEYS = ("proprio", "anchors", "mates", "mates_mask", "lidar", "lidar_down")
CRITIC_KEYS = ACTOR_KEYS + ("priv", "priv_mask")

_DTYPES = {"float32": torch.float32, "bool": torch.bool, "uint8": torch.uint8}


def empty_obs(n: int, a: int, device=None) -> Dict[str, torch.Tensor]:
    """Zero-filled obs dict with the right shapes and dtypes."""
    return {k: torch.zeros((n, a) + shape, dtype=_DTYPES[dt], device=device)
            for k, (dt, shape) in OBS_SPEC.items()}


def check_obs(obs: Dict[str, torch.Tensor], n: int, a: int) -> None:
    """Raise AssertionError if obs does not match OBS_SPEC."""
    assert set(obs) == set(OBS_SPEC), f"keys {sorted(obs)} != {sorted(OBS_SPEC)}"
    for k, (dt, shape) in OBS_SPEC.items():
        t = obs[k]
        assert t.dtype == _DTYPES[dt], f"{k}: dtype {t.dtype} != {dt}"
        assert tuple(t.shape) == (n, a) + shape, f"{k}: shape {tuple(t.shape)} != {(n, a) + shape}"
        if t.dtype == torch.float32:
            assert torch.isfinite(t).all(), f"{k}: non-finite values"


EVENT_KEYS = ("collision", "shield", "grasp", "release", "delivered", "slip", "bycatch", "lost")


@dataclass
class StepOut:
    obs: Dict[str, torch.Tensor]
    done: torch.Tensor                    # [N] bool, terminal (crash, out of bounds)
    truncated: torch.Tensor               # [N] bool, time limit
    events: Dict[str, torch.Tensor]       # EVENT_KEYS -> [N, A] bool
    info: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------- anchor binding
# Selector anchors: instead of one fixed entity, an anchor can name a kind of thing and
# resolve every step to the closest one to its drone. Item selectors stick with the item
# the drone is holding; the net-corner selector skips corners other drones hold. A
# selector with nothing left to match (e.g. no trash remains) reads as an unused slot.
ANCHOR_SELECTORS = (("nearest:trash", "handle"), ("nearest:item", "handle")) + \
    tuple((f"nearest:{t}", "handle") for t in ITEM_TYPES) + \
    (("nearest:net_corner", "handle"), ("nearest:fish_school", "point"), ("nearest:animal", "point"))
SELECTOR_KIND = dict(ANCHOR_SELECTORS)


@dataclass
class AnchorBinding:
    """What anchor k points at. Exactly one of entity / point is set.

    entity: a name from env.entity_names(), e.g. "item_3.handle", "bin", "boat",
            "nofly_0", "net.corner_2", "fish_school_0", or a selector from
            ANCHOR_SELECTORS such as "nearest:trash".
    point:  fixed world point (x, y, z).
    radius: zone radius in metres (0 = plain point).
    kind:   one of ANCHOR_TYPES.
    """
    kind: str = "point"
    entity: Optional[str] = None
    point: Optional[Sequence[float]] = None
    radius: float = 0.0


# ---------------------------------------------------------------- env API
class SkyEnvAPI(Protocol):
    scene: str
    num_envs: int
    num_agents: int
    device: torch.device

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]: ...
    def step(self, actions: torch.Tensor) -> StepOut: ...           # actions [N, A, ACTION_DIM]
    def observe(self) -> Dict[str, torch.Tensor]: ...               # current obs, no step
    def bind_anchors(self, env_ids: torch.Tensor, bindings: List[AnchorBinding],
                     agent: Optional[int] = None) -> None: ...
    def randomize_anchors(self, env_ids: torch.Tensor) -> None: ...
    def entity_names(self) -> List[str]: ...
    def copy_state(self, src: int, dst_ids: torch.Tensor) -> None: ...
    def get_state(self) -> Dict[str, torch.Tensor]: ...
    def set_state(self, state: Dict[str, torch.Tensor], env_ids: Optional[torch.Tensor] = None) -> None: ...
    def expert_action(self) -> torch.Tensor: ...                    # [N, A, ACTION_DIM]
    def render_state(self, env_id: int) -> dict: ...
    def perturb(self, env_id: int, kind: str, **kwargs) -> None: ...


# ---------------------------------------------------------------- render state (demo)
# env.render_state(i) returns a JSON-serialisable dict:
# {
#   "t": float, "scene": "trash" | "delivery" | "ocean", "variant": "city" | "terrain" | "delivery" | "ocean",
#   "max_speed": float,                                  # the scene's horizontal speed limit (m/s)
#   "arena_half": float,
#   "terrain":   None | {"version": int, "res": int, "half": float, "heights": [[res] * res]},
#                # heights[i][j] at x = -half + 2*half*j/(res-1), y = -half + 2*half*i/(res-1).
#                # Sent only when "version" changed since the last call with send_terrain=False
#   "water":     None | {"level": float, "waves": None | {"amp","k","dir","omega","phase": [N_WAVES]}},
#                # terrain scenes: flat water at "level" wherever the terrain is lower (lakes);
#                # ocean: waves everywhere around level 0.
#   "drones":    [{"pos": [3], "quat": [4], "grip": bool, "holding": bool, "alive": bool}],
#   "items":     [{"name": "item_3", "type": ITEM_TYPES entry, "is_trash": bool, "shape": "box"|"sphere",
#                  "pos": [3], "size": [3], "held": bool, "delivered": bool}],
#                # delivered items stay listed with delivered=True (trash rests where it was dropped)
#   "boxes":     [{"name": str, "kind": "building"|"car"|"truck"|"bench"|"hydrant"|"trunk"|"boat"|"dropped"|"bin",
#                  "pos": [3], "size": [3], "moving": bool}],
#   "spheres":   [{"name": str, "kind": "rock"|"canopy"|"bird", "pos": [3], "radius": float, "moving": bool}],
#   "zones":     [{"name": str, "pos": [3], "radius": float, "height": float,
#                  "kind": "bin"|"truck"|"pad_pickup"|"pad_dropoff"|"nofly"|"boat"}],
#   "anchors":   [{"pos": [3], "kind": str, "radius": float}],
#   "boat":      None | {"pos": [3], "quat": [4], "size": [3]},
#   "net":       None | {"rows": int, "cols": int, "nodes": [[3] * rows*cols], "held": [int corner or -1 per drone]},
#   "fish":      [[3]],
#   "animals":   [{"name": str, "type": "whale"|"dolphin"|"turtle", "pos": [3], "vel": [3], "radius": float,
#                  "tangled": bool}],
#   "lidar":     {"nir": [[3]], "green": [[3]]},   # world-frame hit points of drone 0
#   "shield_hits": int, "wind": [3]
# }
# perturb kinds: "drop_obstacle" (pos), "gust" (vec, duration), "waves" (scale),
#                "move_payload" (pos, item), "disable_drone" (agent), "tide" (offset).

# ---------------------------------------------------------------- entity names (anchors, language layer)
# env.entity_names() returns a fixed list per scene; env.entity_descriptions() describes each
# for world 0 (type, position, "(inactive)" for unused slots).
#   trash:    item_0 .. item_{MAX_ITEMS-1} and item_i.handle (graspable objects, trash and not),
#             "bin" (drop-off: garbage truck in a city world, trash bin on terrain), "nofly_0",
#             obstacle_0 .. obstacle_7 (nearest boxes: buildings, cars ...), "drone_0"
#   delivery: item_0 and item_0.handle (the organ cooler), "pad_pickup", "pad_dropoff",
#             "nofly_0", "nofly_1", "drone_0"
#   ocean:    net.corner_0 .. net.corner_3, net.center, "boat", fish_school_0 .., "nofly_0",
#             whale_0, dolphin_0, dolphin_1, turtle_0, turtle_1, drone_0 .. drone_{OCEAN_AGENTS-1}
#   world:    everything: item_i and item_i.handle, "bin" (garbage truck in town), "pad_pickup",
#             "pad_dropoff", "boat", "nofly_0", "nofly_1", net.corner_0 .. 3, net.center,
#             fish_school_i, the animals, obstacle_0 .. obstacle_7, drone_0 .. drone_2
# Zone slots, the same in every scene (unused slots are inactive):
ZONE_SLOT = {"bin": 0, "nofly_0": 1, "nofly_1": 2, "pad_pickup": 3, "pad_dropoff": 4, "boat": 5}
N_ZONES = 6
N_NAMED_OBSTACLES = 8
