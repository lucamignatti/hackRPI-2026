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
EPISODE_STEPS = 600       # 30 s episodes
Z_RESAMPLE_STEPS = 100    # policy steps a world keeps its z during pretraining

# ---------------------------------------------------------------- action
ACTION_DIM = 5            # [vx, vy, vz, yaw_rate, grip] each in [-1, 1]
MAX_SPEED = 6.0           # m/s, scale for vx, vy
MAX_VZ = 3.0              # m/s, scale for vz
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
SCENES = ("trash", "ocean")
TRASH_AGENTS = 1
OCEAN_AGENTS = 3
ARENA_HALF = 15.0         # both scenes are 30 x 30 m
TRASH_N_OBSTACLES = 6     # boxes, of which TRASH_N_MOVING move
TRASH_N_MOVING = 2
NET_ROWS = 8
NET_COLS = 8
N_FISH_SCHOOLS = 2
FISH_PER_SCHOOL = 16
N_WAVES = 4

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
MAX_PRIV = 16
PRIV_TYPES = ("obstacle", "payload", "net", "fish_school", "boat", "zone")
PRIV_DIM = 3 + 3 + 3 + len(PRIV_TYPES)   # rel pos, vel, size, type one-hot

# ---------------------------------------------------------------- goal space g
# Ordered (name, size, scale). B, the discriminator, and reward programs read g.
# scale is a fixed normaliser: networks see g / scale. Reward programs see raw g.
_G_FIELDS = [
    ("drone_pos", 3, 15.0),        # world position
    ("drone_vel", 3, 6.0),
    ("speed", 1, 6.0),
    ("tilt", 1, 1.0),              # rad
    ("height", 1, 10.0),           # above ground or wave surface
    ("thrust", 1, 1.0),            # collective thrust / hover thrust
    ("grip_closed", 1, 1.0),
    ("holding", 1, 1.0),
    ("held_mass", 1, 2.0),
    ("payload_rel", 3, 15.0),      # payload position minus drone position
    ("payload_vel", 3, 6.0),
    ("swing", 1, 3.0),             # payload angular rate about the gripper (rad/s)
    ("jerk", 1, 50.0),             # drone jerk magnitude (m/s^3)
    ("anchor_rel_drone", 3 * N_ANCHORS, 15.0),    # anchor k position minus drone position
    ("anchor_rel_payload", 3 * N_ANCHORS, 15.0),  # anchor k position minus payload position
    ("anchor_inside", N_ANCHORS, 1.0),            # drone inside anchor k zone
    ("obstacle_dist", 1, 10.0),    # distance from drone surface to nearest obstacle
    ("obstacle_closing", 1, 6.0),  # closing speed toward it (positive = approaching)
    ("net_centroid_rel", 3, 15.0), # net centroid minus drone position (zeros in trash)
    ("net_stretch", 1, 1.0),       # max edge stretch ratio minus 1
    ("fish_in_net", 1, 1.0),       # fraction of fish inside the net volume
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


EVENT_KEYS = ("collision", "shield", "grasp", "release", "delivered", "tear")


@dataclass
class StepOut:
    obs: Dict[str, torch.Tensor]
    done: torch.Tensor                    # [N] bool, terminal (crash, tear)
    truncated: torch.Tensor               # [N] bool, time limit
    events: Dict[str, torch.Tensor]       # EVENT_KEYS -> [N, A] bool
    info: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------- anchor binding
@dataclass
class AnchorBinding:
    """What anchor k points at. Exactly one of entity / point is set.

    entity: a name from env.entity_names(), e.g. "bag.handle", "bin", "boat",
            "nofly_0", "net.corner_2", "fish_school_0".
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
#   "t": float, "scene": "trash" | "ocean",
#   "drones":    [{"pos": [3], "quat": [4], "grip": bool, "holding": bool}],
#   "payloads":  [{"name": str, "pos": [3], "quat": [4], "size": [3], "kind": "bag"|"cooler"}],
#   "obstacles": [{"name": str, "pos": [3], "size": [3], "moving": bool, "dropped": bool}],  # dropped = placed by perturb
#   "zones":     [{"name": str, "pos": [3], "radius": float, "height": float, "kind": "bin"|"nofly"|"boat"}],
#   "anchors":   [{"pos": [3], "kind": str, "radius": float}],
#   "waves":     None | {"amp": [N_WAVES], "k": [N_WAVES], "dir": [N_WAVES], "omega": [N_WAVES], "phase": [N_WAVES]},
#   "boat":      None | {"pos": [3], "quat": [4], "size": [3]},
#   "net":       None | {"rows": int, "cols": int, "nodes": [[3] * rows*cols], "held": [int corner idx or -1 per drone]},
#   "fish":      [[3]],             # every fish position
#   "lidar":     {"nir": [[3]], "green": [[3]]},   # world-frame hit points of drone 0
#   "shield_hits": int, "wind": [3]
# }
# perturb kinds: "drop_obstacle" (pos), "gust" (vec, duration), "waves" (scale),
#                "move_payload" (pos), "disable_drone" (agent).
