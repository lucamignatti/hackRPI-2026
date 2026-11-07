"""Hand-built scene entity descriptions and fake envs for the mission tests (no skyhand.env).

The description strings follow the format env.entity_descriptions() is expected to produce:
  "item_3": "pizza_box (trash) at [x, y, z]", "item_3.handle": "pizza_box handle (grab here) at [x, y, z]",
  non-trash items "backpack at [...]" or "flower_pot (not trash) at [...]", unused slots contain "(inactive)".
"""
from __future__ import annotations

from typing import Dict, List

import torch

from skyhand import spec


def _item(ents: Dict[str, str], i: int, typ: str, pos, trash=None, inactive=False):
    p = "[%.1f, %.1f, %.1f]" % tuple(pos)
    tag = "" if trash is None else (" (trash)" if trash else " (not trash)")
    if inactive:
        ents[f"item_{i}"] = f"{typ}{tag} (inactive)"
        ents[f"item_{i}.handle"] = f"{typ} handle (inactive)"
    else:
        ents[f"item_{i}"] = f"{typ}{tag} at {p}"
        ents[f"item_{i}.handle"] = f"{typ} handle (grab here) at [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2] + 0.1:.1f}]"


def trash_city() -> Dict[str, str]:
    """Drone at the origin, garbage truck at (12, 0). Active trash: items 0, 2, 3, 4."""
    e: Dict[str, str] = {}
    _item(e, 0, "trash_bag", (6.0, 2.0, 0.3), trash=True)
    _item(e, 1, "backpack", (1.0, 1.0, 0.2))                        # nearest item, not trash
    _item(e, 2, "pizza_box", (3.0, 0.0, 0.05), trash=True)
    _item(e, 3, "pizza_box", (9.0, 5.0, 0.05), trash=True)
    _item(e, 4, "bottle", (4.0, -1.0, 0.15), trash=True)
    _item(e, 5, "flower_pot", (2.0, -1.0, 0.2), trash=False)
    _item(e, 6, "cardboard_box", (0.5, 0.0, 0.2), trash=True, inactive=True)   # inactive: must be skipped
    for i in range(7, spec.MAX_ITEMS):
        e[f"item_{i}"] = "(inactive)"
        e[f"item_{i}.handle"] = "(inactive)"
    e["bin"] = "garbage truck drop-off at [12.0, 0.0, 0.0], radius 1.5"
    e["nofly_0"] = "no-fly zone at [-10.0, 10.0, 0.0], radius 4.0"
    for i in range(spec.N_NAMED_OBSTACLES):
        e[f"obstacle_{i}"] = f"building at [{-20.0 + 5 * i:.1f}, 15.0, 4.0], size [4.0, 4.0, 8.0]"
    e["drone_0"] = "drone at [0.0, 0.0, 2.0]"
    return e


def trash_no_positions() -> Dict[str, str]:
    """Descriptions without coordinates: order falls back to the listing order."""
    return {"item_0": "bottle (trash)", "item_0.handle": "bottle handle",
            "item_1": "ball", "item_1.handle": "ball handle",
            "item_2": "trash_bag (trash)", "item_2.handle": "trash_bag handle",
            "item_3": "(inactive)", "item_3.handle": "(inactive)",
            "item_4": "pizza_box (trash)", "item_4.handle": "pizza_box handle",
            "bin": "trash bin drop-off", "nofly_0": "no-fly zone", "drone_0": "drone"}


def delivery() -> Dict[str, str]:
    return {"item_0": "cooler at [-15.0, -12.0, 3.1]", "item_0.handle": "cooler handle (grab here) at [-15.0, -12.0, 3.4]",
            "pad_pickup": "pickup helipad at [-15.0, -12.0, 2.8], radius 2.0",
            "pad_dropoff": "hospital helipad at [16.0, 14.0, 6.5], radius 2.0",
            "nofly_0": "no-fly zone at [0.0, 2.0, 0.0], radius 5.0",
            "nofly_1": "no-fly zone at [8.0, -6.0, 0.0], radius 4.0",
            "drone_0": "drone at [-14.0, -12.0, 5.0]"}


def ocean() -> Dict[str, str]:
    e = {f"net.corner_{i}": f"net corner {i} (grab here) at [{(-2.0, 2.0, 2.0, -2.0)[i]:.1f}, "
                            f"{(-2.0, -2.0, 2.0, 2.0)[i]:.1f}, 0.0]" for i in range(4)}
    e.update({"net.center": "net centre at [0.0, 0.0, -0.5]", "boat": "boat at [15.0, 0.0, 0.0], radius 3.0",
              "fish_school_0": "fish school at [-8.0, 6.0, -2.0]", "fish_school_1": "fish school at [12.0, 2.0, -2.0]",
              "nofly_0": "no-fly zone at [-15.0, -15.0, 0.0], radius 3.0",
              "whale_0": "whale at [-6.0, 9.0, -3.0]", "dolphin_0": "dolphin at [4.0, -5.0, -1.0]",
              "dolphin_1": "dolphin at [6.0, 8.0, -1.0]", "turtle_0": "turtle at [-3.0, -9.0, -0.5]",
              "turtle_1": "turtle at [9.0, -9.0, -0.5]"})
    for i in range(spec.OCEAN_AGENTS):
        e[f"drone_{i}"] = f"drone at [{-2.0 + 2 * i:.1f}, -2.0, 2.0]"
    return e


def world() -> Dict[str, str]:
    """The generic 400 m world, described like env.entity_descriptions() does it.

    Town around (-45, 15) with the garbage truck at (-55, 20); clinics at (60, -80) (pickup, the cooler waits
    there as item_5, not item_0) and (-90, 110) (drop-off); sea to the east with the boat at (120, 0).
    drone_0 is in town, drone_1 at the pickup clinic, drone_2 near the boat.
    Active trash: items 0, 1, 2, 4. Not trash: item_3 (backpack, nearest to drone_0), item_6 (ball), the cooler.
    Inactive: item_7 (a trash bag) and items 8..15.
    """
    e: Dict[str, str] = {}

    def item(i, typ, pos, note, inactive=False):
        p = "[%.1f, %.1f, %.1f]" % tuple(pos)
        if inactive:
            e[f"item_{i}"] = f"{typ} ({note}) (inactive)"
            e[f"item_{i}.handle"] = f"{typ} handle (inactive)"
        else:
            e[f"item_{i}"] = f"{typ} ({note}) at {p}"
            e[f"item_{i}.handle"] = f"{typ} handle (grab here) at [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2] + 0.3:.1f}]"

    item(0, "trash_bag", (-45.0, 12.0, 0.3), "trash")
    item(1, "pizza_box", (-38.0, 14.0, 0.05), "trash")
    item(2, "bottle", (20.0, 60.0, 0.15), "trash")                 # countryside litter
    item(3, "backpack", (-39.0, 9.0, 0.2), "not trash")            # nearest item to drone_0
    item(4, "cardboard_box", (-50.0, 25.0, 0.2), "trash")
    item(5, "cooler", (60.0, -80.0, 3.1), "organ transport")       # the organ cooler at pad_pickup
    item(6, "ball", (10.0, 50.0, 0.12), "not trash")
    item(7, "trash_bag", (-40.0, 10.0, 0.3), "trash", inactive=True)
    for i in range(8, spec.MAX_ITEMS):
        e[f"item_{i}"] = "(inactive)"
        e[f"item_{i}.handle"] = "(inactive)"
    e["bin"] = "garbage truck drop-off, radius 3.2 m at [-55.0, 20.0, 0.0]"
    e["pad_pickup"] = "clinic helipad (pickup), radius 3.0 m at [60.0, -80.0, 2.8]"
    e["pad_dropoff"] = "hospital helipad (drop-off), radius 3.0 m at [-90.0, 110.0, 6.5]"
    e["boat"] = "boat deck, delivery zone radius 2.5 m at [120.0, 0.0, 0.5]"
    e["nofly_0"] = "no-fly zone (wildlife reserve), radius 15.0 m at [0.0, 40.0, 0.0]"
    e["nofly_1"] = "no-fly zone, radius 12.0 m at [-60.0, 70.0, 0.0]"
    for i, (x, y) in enumerate(((97.0, -3.0), (103.0, -3.0), (103.0, 3.0), (97.0, 3.0))):
        e[f"net.corner_{i}"] = f"net corner {i} (grab here) at [{x:.1f}, {y:.1f}, 0.0]"
    e["net.center"] = "net centre at [100.0, 0.0, -0.5]"
    e["fish_school_0"] = "fish school at [140.0, 30.0, -2.0]"
    e["fish_school_1"] = "fish school at [105.0, -10.0, -2.0]"
    e["whale_0"] = "whale (keep it out of the net) at [150.0, -20.0, -3.0]"
    e["dolphin_0"] = "dolphin (keep it out of the net) at [110.0, 15.0, -1.0]"
    e["dolphin_1"] = "dolphin (keep it out of the net) at [130.0, -5.0, -1.0]"
    e["turtle_0"] = "turtle (keep it out of the net) at [95.0, -15.0, -0.5]"
    e["turtle_1"] = "turtle (keep it out of the net) at [125.0, 20.0, -0.5]"
    for i in range(spec.N_NAMED_OBSTACLES):
        e[f"obstacle_{i}"] = f"building at [{-60.0 + 6 * i:.1f}, 30.0, 5.0], size [5.0, 5.0, 10.0]"
    for i, pos in enumerate(((-40.0, 10.0, 3.0), (55.0, -75.0, 4.0), (110.0, 5.0, 3.0))):
        e[f"drone_{i}"] = "drone at [%.1f, %.1f, %.1f]" % pos
    return e


SCENE_ENTS = {"trash": trash_city, "delivery": delivery, "ocean": ocean, "world": world}


class FakeEnv:
    """Records bind_anchors calls; step returns random g. Enough for bind_mission / to_bindings."""

    def __init__(self, scene: str, entities: Dict[str, str], num_envs: int = 2):
        self.scene = scene
        self._ents = dict(entities)
        self.num_envs = num_envs
        self.num_agents = spec.SCENE_AGENTS[scene]
        self.device = torch.device("cpu")
        self.bound: Dict[int, List[spec.AnchorBinding]] = {}

    def entity_names(self) -> List[str]:
        return list(self._ents)

    def entity_descriptions(self) -> Dict[str, str]:
        return dict(self._ents)

    def bind_anchors(self, env_ids, bindings, agent=None):
        assert len(bindings) == spec.N_ANCHORS
        for b in bindings:
            assert b.kind in spec.ANCHOR_TYPES
            assert b.entity is None or b.entity in self._ents, b.entity
        self.bound[0 if agent is None else int(agent)] = list(bindings)

    def step(self, actions):
        n, a = self.num_envs, self.num_agents
        obs = spec.empty_obs(n, a)
        obs["g"] = torch.randn(n, a, spec.G_DIM) * 3
        z = torch.zeros(n, a, dtype=torch.bool)
        return spec.StepOut(obs=obs, done=torch.zeros(n, dtype=torch.bool),
                            truncated=torch.zeros(n, dtype=torch.bool), events={k: z for k in spec.EVENT_KEYS})


class PointMassEnv:
    """Batched single-drone point mass: velocity = action[:3] * MAX_SPEED. g has drone_pos, drone_vel, speed
    and anchor_rel_drone[0] for a bound fixed point. Implements the env API cem_refine needs."""

    def __init__(self, num_envs: int, start=(0.0, 0.0, 2.0)):
        self.num_envs, self.num_agents, self.device = num_envs, 1, torch.device("cpu")
        self.pos = torch.tensor(start).repeat(num_envs, 1, 1)
        self.vel = torch.zeros(num_envs, 1, 3)
        self.anchor = torch.zeros(num_envs, 1, 3)

    def bind_anchors(self, env_ids, bindings, agent=None):
        self.anchor[env_ids, 0] = torch.tensor(bindings[0].point, dtype=torch.float32)

    def get_state(self):
        return {"pos": self.pos.clone(), "vel": self.vel.clone(), "anchor": self.anchor.clone()}

    def set_state(self, state, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self.pos[ids], self.vel[ids], self.anchor[ids] = state["pos"][ids], state["vel"][ids], state["anchor"][ids]

    def copy_state(self, src, dst_ids):
        self.pos[dst_ids], self.vel[dst_ids], self.anchor[dst_ids] = (self.pos[src].clone(), self.vel[src].clone(),
                                                                  self.anchor[src].clone())

    def observe(self):
        obs = spec.empty_obs(self.num_envs, 1)
        g = obs["g"]
        g[..., spec.G_SLICES["drone_pos"]] = self.pos
        g[..., spec.G_SLICES["drone_vel"]] = self.vel
        g[..., spec.G_SLICES["speed"]] = self.vel.norm(dim=-1, keepdim=True)
        ard = spec.G_SLICES["anchor_rel_drone"].start
        g[..., ard:ard + 3] = self.anchor - self.pos
        return obs

    def step(self, actions):
        self.vel = actions[..., :3].clamp(-1, 1) * spec.MAX_SPEED
        self.pos = self.pos + self.vel * spec.POLICY_DT
        n = self.num_envs
        z = torch.zeros(n, 1, dtype=torch.bool)
        return spec.StepOut(obs=self.observe(), done=torch.zeros(n, dtype=torch.bool),
                            truncated=torch.zeros(n, dtype=torch.bool), events={k: z for k in spec.EVENT_KEYS})
