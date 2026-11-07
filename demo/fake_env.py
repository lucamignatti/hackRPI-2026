"""Kinematic preview env for the demo (`python -m demo.server --fake-env`) and for tests.

It follows the SkyEnvAPI surface and the render-state schema in skyhand/spec.py, but the
physics is a toy: drones follow the velocity command with a first-order lag, items snap
to the gripper when grasped and drop straight down when released, nothing collides.
Use it to work on the viewer and the server while skyhand/env/core.py is unavailable.
The UI labels it ("sim: kinematic preview") so it is never mistaken for the simulator.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from skyhand import spec
from skyhand.spec import AnchorBinding, StepOut

TYPE_INFO = {c[0]: c for c in spec.ITEM_CATALOG}
WATER_LEVEL = 0.5
WORLD_SEA_LEVEL = 0.0      # world: tide level of the sea and lakes (waves on top)


def _hz(item_type: str) -> float:
    _, shape, size, _, _ = TYPE_INFO[item_type]
    return size[0] if shape == "sphere" else size[2] / 2


def _size3(item_type: str) -> List[float]:
    _, shape, size, _, _ = TYPE_INFO[item_type]
    return [2 * size[0]] * 3 if shape == "sphere" else list(size)


class FakeEnv:
    """Toy env. Layout is the same in every world; drone / item state is per world."""

    is_fake = True

    def __init__(self, scene: str = "trash", num_envs: int = 1, device: str = "cpu", seed: int = 0,
                 variant: Optional[str] = None, enable_gusts: bool = False, domain_rand: bool = False,
                 lidar: bool = True, **_ignored):
        assert scene in spec.SCENES, scene
        self.scene = scene
        self.num_envs = int(num_envs)
        self.num_agents = spec.SCENE_AGENTS[scene]
        self.arena_half = spec.SCENE_ARENA_HALF.get(scene, spec.ARENA_HALF)
        self.max_speed = spec.SCENE_MAX_SPEED.get(scene, spec.MAX_SPEED)
        self.max_vz = spec.SCENE_MAX_VZ.get(scene, spec.MAX_VZ)
        self.episode_steps = spec.SCENE_EPISODE_STEPS.get(scene, spec.EPISODE_STEPS)
        self.k = self.arena_half / 25.0          # layout scale (delivery: 600 m arena)
        self.res = spec.SCENE_TERRAIN_RES.get(scene, spec.TERRAIN_RES)
        self.has_net = scene in ("ocean", "world")
        self.device = torch.device(device)
        self.seed = int(seed)
        self.lidar_on = lidar
        self.gusts_enabled = bool(enable_gusts)
        self.cfg = type("Cfg", (), {"enable_gusts": self.gusts_enabled, "domain_rand": bool(domain_rand)})()
        if variant is None:
            variant = "ocean" if scene == "ocean" else "world" if scene == "world" else \
                "terrain" if scene == "delivery" else ("city" if self.seed % 2 == 0 else "terrain")
        self.variant = variant
        self.terrain_version = 0
        self._sent_version = -1
        self.tide = 0.0
        self.wave_scale = 1.0
        self._build_layout()
        self.reset()

    # ------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        r = lambda *s: torch.rand(*s, generator=g)
        k = self.k
        self.boxes: List[dict] = []
        self.spheres: List[dict] = []
        self.zones: List[dict] = []
        self.animals: List[dict] = []
        self.item_types: List[str] = []
        item_xy: List[List[float]] = []
        self.heights: Optional[torch.Tensor] = None
        self.waves = None
        if self.variant in ("terrain",):
            self._make_terrain(g)
        if self.scene == "world":
            self._build_world(g)
            return
        if self.variant == "city":
            for bx in (-15.0, 0.0, 15.0):
                for by in (-15.0, 0.0, 15.0):
                    if (bx, by) == (0.0, 0.0):
                        continue
                    h = 6.0 + 12.0 * float(r(1))
                    self.boxes.append({"name": f"building_{len(self.boxes)}", "kind": "building",
                                       "pos": [bx, by, h / 2], "size": [9.0, 9.0, h], "moving": False})
            self.cars = [(-7.5, -20.0, 1.0, 4.0), (7.5, 18.0, -1.0, 3.0), (-20.0, 7.5, 1.0, 5.0)]
            for k, _ in enumerate(self.cars):
                self.boxes.append({"name": f"car_{k}", "kind": "car", "pos": [0, 0, 0.75], "size": [1.8, 4.2, 1.5],
                                   "moving": True})
            self.boxes.append({"name": "truck", "kind": "truck", "pos": [9.5, -4.0, 1.6], "size": [2.5, 7.0, 3.2],
                               "moving": False})
            self.boxes.append({"name": "hydrant_0", "kind": "hydrant", "pos": [-9.8, 3.0, 0.4], "size": [0.3, 0.3, 0.8],
                               "moving": False})
            self.boxes.append({"name": "bench_0", "kind": "bench", "pos": [5.2, 9.8, 0.25], "size": [1.6, 0.5, 0.5],
                               "moving": False})
            self.zones.append({"name": "bin", "kind": "truck", "pos": [9.5, -5.5, 3.2], "radius": 1.3, "height": 1.0})
            types = ["trash_bag", "pizza_box", "bottle", "cardboard_box", "backpack", "ball"]
            curbs = [[-9.8, -6.0], [-6.0, 9.8], [5.5, -9.8], [9.8, 6.0], [-9.8, 8.0], [3.0, 9.8]]
            self.item_types, item_xy = types, curbs
        elif self.scene == "trash":   # terrain litter
            for k in range(6):
                x, y = float(r(1) * 40 - 20), float(r(1) * 40 - 20)
                self.boxes.append({"name": f"trunk_{k}", "kind": "trunk", "pos": [x, y, self.ground(x, y) + 1.5],
                                   "size": [0.4, 0.4, 3.0], "moving": False})
                self.spheres.append({"name": f"canopy_{k}", "kind": "canopy", "pos": [x, y, self.ground(x, y) + 3.8],
                                     "radius": 1.6, "moving": False})
            for k in range(3):
                x, y = float(r(1) * 40 - 20), float(r(1) * 40 - 20)
                self.spheres.append({"name": f"rock_{k}", "kind": "rock", "pos": [x, y, self.ground(x, y) + 0.3],
                                     "radius": 0.7, "moving": False})
            self.zones.append({"name": "bin", "kind": "bin", "pos": [2.0, -2.0, self.ground(2.0, -2.0)], "radius": 0.8,
                               "height": 1.0})
            self.item_types = ["trash_bag", "bottle", "pizza_box", "flower_pot", "ball"]
            item_xy = [[-6.0, 4.0], [7.0, 6.0], [-3.0, -8.0], [10.0, -9.0], [-12.0, -3.0]]
        elif self.scene == "delivery":
            pk, dp = [-17.0 * k, -16.0 * k], [16.0 * k, 15.0 * k]
            self.zones.append({"name": "pad_dropoff", "kind": "pad_dropoff", "pos": dp + [self.ground(*dp)],
                               "radius": 1.5, "height": 0.1})
            self.zones.append({"name": "nofly_0", "kind": "nofly", "pos": [0.0, 2.0 * k, 0.0], "radius": 4.0 * k, "height": 80.0})
            self.zones.append({"name": "pad_pickup", "kind": "pad_pickup", "pos": pk + [self.ground(*pk)],
                               "radius": 1.5, "height": 0.1})
            self.item_types = ["cooler"]
            item_xy = [[pk[0] + 0.6, pk[1]]]
            self.birds = [(float(r(1) * 30 - 15) * k, float(r(1) * 30 - 15) * k, 3.0 * k + 3 * float(r(1)), 15.0 + 40 * float(r(1)))
                          for _ in range(4)]
            for j in range(4):
                self.spheres.append({"name": f"bird_{j}", "kind": "bird", "pos": [0, 0, 0], "radius": 0.6, "moving": True})
            for j in range(12):
                x, y = float(r(1) * 40 - 20) * k, float(r(1) * 40 - 20) * k
                self.boxes.append({"name": f"trunk_{j}", "kind": "trunk", "pos": [x, y, self.ground(x, y) + 1.5],
                                   "size": [0.4, 0.4, 3.0], "moving": False})
                self.spheres.append({"name": f"canopy_{j}", "kind": "canopy", "pos": [x, y, self.ground(x, y) + 3.8],
                                     "radius": 1.6, "moving": False})
            for j, (cx, cy) in enumerate((pk, dp)):           # a clinic and two huts at each pad
                self.boxes.append({"name": f"clinic_{j}", "kind": "building", "pos": [cx + 10, cy, self.ground(cx + 10, cy) + 2],
                                   "size": [8.0, 6.0, 4.0], "moving": False})
                for h in range(2):
                    x, y = cx - 8 + 5 * h, cy + 9
                    self.boxes.append({"name": f"hut_{2 * j + h}", "kind": "hut", "pos": [x, y, self.ground(x, y) + 1.3],
                                       "size": [3.0, 3.0, 2.6], "moving": False})
        else:   # ocean
            self.waves = {"amp": [0.12, 0.08, 0.05, 0.03], "k": [0.35, 0.6, 0.9, 1.4], "dir": [0.3, 0.8, -0.4, 1.9],
                          "phase": [0.0, 1.0, 2.0, 3.0]}
            self.waves["omega"] = [math.sqrt(9.81 * k) for k in self.waves["k"]]
            self.boat = {"pos": [0.0, -16.0, 0.4], "quat": [1.0, 0.0, 0.0, 0.0], "size": [5.0, 2.2, 1.2]}
            self.boxes.append({"name": "boat", "kind": "boat", "pos": self.boat["pos"], "size": self.boat["size"],
                               "moving": False})
            self.zones.append({"name": "boat", "kind": "boat", "pos": [0.0, -16.0, 1.0], "radius": 3.0, "height": 2.0})
            self.school_c = [[6.0, 6.0], [-8.0, 4.0]]
            k = 0
            for t, cnt in zip(spec.ANIMAL_TYPES, spec.ANIMAL_COUNTS):
                for j in range(cnt):
                    rad = {"whale": 2.5, "dolphin": 0.9, "turtle": 0.5}[t]
                    self.animals.append({"name": f"{t}_{j}", "type": t, "radius": rad,
                                         "orbit": (float(r(1) * 20 - 10), float(r(1) * 20 - 5), 4.0 + 6 * float(r(1)),
                                                   0.1 + 0.2 * float(r(1)), float(r(1) * 6.28))})
                    k += 1
        if self.scene != "ocean":
            self.zones.append({"name": "nofly_1" if self.scene == "delivery" else "nofly_0", "kind": "nofly",
                               "pos": [-12.0 * k, 12.0 * k, 0.0], "radius": 3.0 * k, "height": 30.0 * min(k, 3)})
        self.item_xy = item_xy

    # ------------------------------------------------------------ world: everything in one 400 m map
    def _ocean_waves(self, amp_scale: float = 1.0) -> dict:
        w = {"amp": [a * amp_scale for a in (0.12, 0.08, 0.05, 0.03)], "k": [0.35, 0.6, 0.9, 1.4],
             "dir": [0.3, 0.8, -0.4, 1.9], "phase": [0.0, 1.0, 2.0, 3.0]}
        w["omega"] = [math.sqrt(9.81 * k) for k in w["k"]]
        return w

    def _build_world(self, g: torch.Generator) -> None:
        """Hills, a lake and a coast (sea on the west), a town with a garbage truck, two clinics with
        pads and huts, countryside litter, the boat / net / fish / animals at sea, trees, rocks, birds
        and two no-fly zones. Sea and lake share the tide level; waves everywhere on top."""
        r = lambda: float(torch.rand(1, generator=g))
        R, H = self.res, self.arena_half
        lin = torch.linspace(-H, H, R)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        bump = lambda cx, cy, s, a: a * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))
        h = 3.0 + 1.5 * torch.sin(x / 35.0) * torch.cos(y / 41.0)
        h = h + bump(110, 110, 40, 28) + bump(140, -120, 30, 18) + bump(-20, 150, 35, 14)
        h = h - bump(70, 60, 22, 6.0)                                    # lake
        coast = torch.clamp((-60.0 - x) / 40.0, 0.0, 1.0)               # beach at x ~ -75, open sea west of -100
        h = h - coast * (h + 6.0)
        for cx, cy, s in ((40, -60, 45), (-40, -140, 14), (-50, 40, 14)):  # flat town and clinic sites
            w = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
            h = h * (1 - w) + 3.0 * w
        self.heights = h.clamp_min(-8.0)
        self.waves = self._ocean_waves(0.8)
        gr = self.ground
        # town: block grid, a garbage truck ("bin" zone on its roof), parked cars, a hydrant
        for bx in (10.0, 30.0, 50.0, 70.0):
            for by in (-80.0, -60.0, -40.0):
                if (bx, by) == (50.0, -60.0):
                    continue
                hb = 6.0 + 10.0 * r()
                self.boxes.append({"name": f"building_{len(self.boxes)}", "kind": "building",
                                   "pos": [bx, by, gr(bx, by) + hb / 2], "size": [12.0, 12.0, hb], "moving": False})
        tz = gr(50.0, -60.0)
        self.boxes.append({"name": "truck", "kind": "truck", "pos": [50.0, -60.0, tz + 1.6], "size": [2.5, 7.0, 3.2],
                           "moving": False})
        self.zones.append({"name": "bin", "kind": "truck", "pos": [50.0, -61.5, tz + 3.2], "radius": 1.3, "height": 1.0})
        self.cars = []
        for j, (cx, cy) in enumerate(((20.0, -50.0), (60.0, -70.0))):
            self.boxes.append({"name": f"car_{j}", "kind": "car", "pos": [cx, cy, gr(cx, cy) + 0.75],
                               "size": [1.8, 4.2, 1.5], "moving": False})
        self.boxes.append({"name": "hydrant_0", "kind": "hydrant", "pos": [40.0, -51.0, gr(40, -51) + 0.4],
                           "size": [0.3, 0.3, 0.8], "moving": False})
        # clinics: pads, a clinic building and two huts each
        pk, dp = [-40.0, -140.0], [-50.0, 40.0]
        self.zones.append({"name": "pad_pickup", "kind": "pad_pickup", "pos": pk + [gr(*pk)], "radius": 1.5, "height": 0.1})
        self.zones.append({"name": "pad_dropoff", "kind": "pad_dropoff", "pos": dp + [gr(*dp)], "radius": 1.5,
                           "height": 0.1})
        for j, (cx, cy) in enumerate((pk, dp)):
            self.boxes.append({"name": f"clinic_{j}", "kind": "building", "pos": [cx + 10, cy, gr(cx + 10, cy) + 2],
                               "size": [8.0, 6.0, 4.0], "moving": False})
            for hh in range(2):
                hx, hy = cx - 8 + 5 * hh, cy + 9
                self.boxes.append({"name": f"hut_{2 * j + hh}", "kind": "hut", "pos": [hx, hy, gr(hx, hy) + 1.3],
                                   "size": [3.0, 3.0, 2.6], "moving": False})
        # no-fly zones
        self.zones.append({"name": "nofly_0", "kind": "nofly", "pos": [0.0, 100.0, 0.0], "radius": 15.0, "height": 60.0})
        self.zones.append({"name": "nofly_1", "kind": "nofly", "pos": [120.0, -20.0, 0.0], "radius": 12.0, "height": 60.0})
        # trees and rocks on land (away from the town and clinics), birds
        for j in range(14):
            tx, ty = 40.0 + 150.0 * r() - 60.0, 360.0 * r() - 180.0
            if (abs(tx - 40) < 50 and abs(ty + 60) < 40) or gr(tx, ty) < 1.0:
                continue
            self.boxes.append({"name": f"trunk_{j}", "kind": "trunk", "pos": [tx, ty, gr(tx, ty) + 1.5],
                               "size": [0.4, 0.4, 3.0], "moving": False})
            self.spheres.append({"name": f"canopy_{j}", "kind": "canopy", "pos": [tx, ty, gr(tx, ty) + 3.8],
                                 "radius": 1.6, "moving": False})
        for j in range(4):
            rx, ry = 100.0 * r(), 300.0 * r() - 150.0
            self.spheres.append({"name": f"rock_{j}", "kind": "rock", "pos": [rx, ry, gr(rx, ry) + 0.3],
                                 "radius": 0.8, "moving": False})
        self.birds = [(150.0 * r() - 50.0, 300.0 * r() - 150.0, 25.0 + 20 * r(), 18.0 + 15 * r()) for _ in range(4)]
        for j in range(4):
            self.spheres.append({"name": f"bird_{j}", "kind": "bird", "pos": [0, 0, 0], "radius": 0.6, "moving": True})
        # items: curbside trash in town, countryside litter, things that are not trash, the organ cooler
        self.item_types = ["trash_bag", "pizza_box", "bottle", "cardboard_box", "bottle", "backpack", "ball",
                           "flower_pot", "cooler"]
        self.item_xy = [[24.0, -50.0], [44.0, -70.0], [64.0, -50.0], [90.0, 40.0], [-20.0, 60.0], [64.0, -90.0],
                        [20.0, 20.0], [100.0, -60.0], [pk[0] + 0.6, pk[1]]]
        # sea: boat, net, fish schools, animals
        self.sea_c = [-150.0, 70.0]
        self.boat = {"pos": [-140.0, 55.0, 0.4], "quat": [1.0, 0.0, 0.0, 0.0], "size": [5.0, 2.2, 1.2]}
        self.boxes.append({"name": "boat", "kind": "boat", "pos": self.boat["pos"], "size": self.boat["size"],
                           "moving": False})
        self.zones.append({"name": "boat", "kind": "boat", "pos": [-140.0, 55.0, 1.0], "radius": 3.0, "height": 2.0})
        self.school_c = [[-165.0, 90.0], [-175.0, 50.0]]
        for t, cnt in zip(spec.ANIMAL_TYPES, spec.ANIMAL_COUNTS):
            for j in range(cnt):
                rad = {"whale": 2.5, "dolphin": 0.9, "turtle": 0.5}[t]
                self.animals.append({"name": f"{t}_{j}", "type": t, "radius": rad,
                                     "orbit": (self.sea_c[0] + 30 * r() - 15, self.sea_c[1] + 30 * r() - 15,
                                               4.0 + 6 * r(), 0.1 + 0.2 * r(), r() * 6.28)})

    @property
    def n_items(self) -> int:
        return len(self.item_types) if self.scene != "ocean" else 0

    @property
    def corner_base(self) -> int:
        """Index of net corner 0 in self.items (ocean: 0, world: after the items)."""
        return self.n_items

    def _is_corner(self, m: int) -> bool:
        return self.has_net and m >= self.corner_base

    def _corner_xy(self) -> List[List[float]]:
        if self.scene == "ocean":
            return [[-2.5, -9.5], [2.5, -9.5], [2.5, -4.5], [-2.5, -4.5]]
        cx, cy = self.sea_c
        return [[cx - 3, cy - 3], [cx + 3, cy - 3], [cx + 3, cy + 3], [cx - 3, cy + 3]]

    def _water_level(self) -> float:
        return (WORLD_SEA_LEVEL if self.scene == "world" else WATER_LEVEL) + self.tide

    def _make_terrain(self, g: torch.Generator) -> None:
        R, H, k = self.res, self.arena_half, self.k
        lin = torch.linspace(-H, H, R)
        y, x = torch.meshgrid(lin, lin, indexing="ij")      # heights[i][j] at (x_j, y_i)
        a_k = min(k, 3.0)
        bump = lambda cx, cy, s, a: a * a_k * torch.exp(-((x - cx * k) ** 2 + (y - cy * k) ** 2) / (2 * (s * k) ** 2))
        h = 1.2 + 0.6 * a_k * torch.sin(x / (6.0 * k)) * torch.cos(y / (7.0 * k))
        h = h + bump(14, 12, 5.0, 12.0) + bump(20, -18, 4.0, 7.0) + bump(-20, 18, 6.0, 5.0)
        h = h - bump(-6, 8, 6.0, 3.2) - bump(4, -12, 3.5, 2.5)
        if self.scene == "delivery":
            for cx, cy in ((-17 * k, -16 * k), (16 * k, 15 * k)):
                w = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (8.0 * k * k))
                h = h * (1 - w) + 1.5 * w
        self.heights = h.clamp_min(-2.0)

    def ground(self, x: float, y: float) -> float:
        if self.heights is None:
            return 0.0
        R, H = self.res, self.arena_half
        fx = (min(max(x, -H), H) + H) / (2 * H) * (R - 1)
        fy = (min(max(y, -H), H) + H) / (2 * H) * (R - 1)
        j0, i0 = min(int(fx), R - 2), min(int(fy), R - 2)
        tx, ty = fx - j0, fy - i0
        hh = self.heights
        return float((1 - ty) * ((1 - tx) * hh[i0, j0] + tx * hh[i0, j0 + 1]) + ty * ((1 - tx) * hh[i0 + 1, j0] + tx * hh[i0 + 1, j0 + 1]))

    def _ground_t(self, p: torch.Tensor) -> torch.Tensor:
        """Surface height (terrain or water) under points p [..., 3]."""
        flat = p.reshape(-1, 3)
        if self.variant == "ocean":
            return torch.zeros(p.shape[:-1])
        out = torch.tensor([self.ground(float(a), float(b)) for a, b in flat[:, :2].tolist()]).reshape(p.shape[:-1])
        if self.heights is not None:
            out = out.clamp_min(self._water_level())
        return out

    # ------------------------------------------------------------ API
    def entity_names(self) -> List[str]:
        if self.scene == "trash":
            items = [f"item_{i}" for i in range(spec.MAX_ITEMS)] + [f"item_{i}.handle" for i in range(spec.MAX_ITEMS)]
            return items + ["bin", "nofly_0"] + [f"obstacle_{k}" for k in range(spec.N_NAMED_OBSTACLES)] + ["drone_0"]
        if self.scene == "delivery":
            return ["item_0", "item_0.handle", "pad_pickup", "pad_dropoff", "nofly_0", "nofly_1", "drone_0"]
        if self.scene == "world":
            items = [f"item_{i}" for i in range(spec.MAX_ITEMS)] + [f"item_{i}.handle" for i in range(spec.MAX_ITEMS)]
            names = items + ["bin", "pad_pickup", "pad_dropoff", "boat", "nofly_0", "nofly_1"]
            names += [f"net.corner_{c}" for c in range(4)] + ["net.center"]
            names += [f"fish_school_{s}" for s in range(spec.N_FISH_SCHOOLS)] + [a["name"] for a in self.animals]
            return names + [f"obstacle_{k}" for k in range(spec.N_NAMED_OBSTACLES)] + \
                [f"drone_{a}" for a in range(self.num_agents)]
        names = [f"net.corner_{c}" for c in range(4)] + ["net.center", "boat"]
        names += [f"fish_school_{s}" for s in range(spec.N_FISH_SCHOOLS)] + ["nofly_0"]
        names += [a["name"] for a in self.animals] + [f"drone_{a}" for a in range(self.num_agents)]
        return names

    def entity_descriptions(self) -> Dict[str, str]:
        out = {}
        pos = self._entity_positions()[0]
        for k, n in enumerate(self.entity_names()):
            p = pos[k].tolist()
            where = f"at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
            if n.startswith("item_"):
                i = int(n.split(".")[0].split("_")[1])
                if i >= len(self.item_types):
                    out[n] = "(inactive)"
                    continue
                t = self.item_types[i]
                kind = "trash" if TYPE_INFO[t][4] else "not trash"
                out[n] = f"{'handle of ' if n.endswith('.handle') else ''}{t.replace('_', ' ')} ({kind}) {where}"
            elif n == "nofly_0" and self.scene == "ocean":
                out[n] = "no-fly zone (inactive)"
            else:
                out[n] = f"{n.replace('_', ' ')} {where}"
        return out

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        N, A = self.num_envs, self.num_agents
        if env_ids is None or not hasattr(self, "drone"):
            env_ids = torch.arange(N)
            self.drone = torch.zeros(N, A, 3)
            self.vel = torch.zeros(N, A, 3)
            self.alive = torch.ones(N, A, dtype=torch.bool)
            self.grip = torch.zeros(N, A, dtype=torch.bool)
            self.held = torch.full((N, A), -1, dtype=torch.long)
            self.last_held = torch.full((N, A), -1, dtype=torch.long)
            self.last_act = torch.zeros(N, A, spec.ACTION_DIM)
            self.t = torch.zeros(N)
            self.steps = torch.zeros(N, dtype=torch.long)
            self.shield_count = torch.zeros(N, dtype=torch.long)
            self.gust = torch.zeros(N, 3)
            M = max(self.n_items, 1 if not self.has_net else 0) + (4 if self.has_net else 0)
            self.items = torch.zeros(N, M, 3)
            self.delivered = torch.zeros(N, M, dtype=torch.bool)
            self.extra_boxes: List[dict] = []
            self.bindings = [[AnchorBinding(kind="none")] * spec.N_ANCHORS for _ in range(A)]
        ids = env_ids.tolist()
        for i in ids:
            for a in range(A):
                if self.scene == "ocean":
                    sx, sy = [(-2.5, -9.5), (2.5, -9.5), (2.5, -4.5), (-2.5, -4.5)][a]
                    self.drone[i, a] = torch.tensor([sx * 1.4, sy - 1.5, 2.0])
                elif self.scene == "world":
                    x, y = -60.0 + 4.0 * a, 20.0
                    self.drone[i, a] = torch.tensor([x, y, self.ground(x, y) + 0.3])
                elif self.scene == "delivery":
                    x, y = -17.0 * self.k - 1.5, -16.0 * self.k
                    self.drone[i, a] = torch.tensor([x, y, self.ground(x, y) + 0.3])
                else:
                    self.drone[i, a] = torch.tensor([0.0, 0.0, self.ground(0, 0) + 0.3])
            if self.scene != "ocean":
                for m, (x, y) in enumerate(self.item_xy):
                    self.items[i, m] = torch.tensor([x, y, self.ground(x, y) + _hz(self.item_types[m])])
            if self.has_net:
                lvl = self._water_level() if self.scene == "world" else 0.0
                for c, (cx, cy) in enumerate(self._corner_xy()):
                    self.items[i, self.corner_base + c] = torch.tensor([cx, cy, lvl])
        self.vel[env_ids] = 0
        self.alive[env_ids] = True
        self.grip[env_ids] = False
        self.held[env_ids] = -1
        self.last_held[env_ids] = -1
        self.delivered[env_ids] = False
        self.t[env_ids] = 0
        self.steps[env_ids] = 0
        self.gust[env_ids] = 0
        if 0 in ids:
            self.extra_boxes = []
        return self._obs()

    def bind_anchors(self, env_ids, bindings: List[AnchorBinding], agent: Optional[int] = None) -> None:
        bl = list(bindings) + [AnchorBinding(kind="none")] * (spec.N_ANCHORS - len(bindings))
        for a in ([agent] if agent is not None else range(self.num_agents)):
            self.bindings[a] = bl[:spec.N_ANCHORS]

    def randomize_anchors(self, env_ids) -> None:
        pass

    def expert_action(self) -> torch.Tensor:
        raise NotImplementedError("fake env has no expert")

    def observe(self) -> Dict[str, torch.Tensor]:
        return self._obs()

    # ------------------------------------------------------------ geometry
    def _handles(self) -> torch.Tensor:
        """[N, M, 3] grasp points: item tops (corners themselves in the ocean)."""
        if self.scene == "ocean":
            return self.items.clone()
        off = torch.tensor([_hz(t) + 0.03 for t in self.item_types] + [0.0] * (self.items.shape[1] - self.n_items))
        h = self.items.clone()
        h[..., 2] += off
        return h

    def _zone(self, name: str) -> Optional[dict]:
        return next((z for z in self.zones if z["name"] == name), None)

    def _boxes_now(self) -> List[dict]:
        out = []
        t = float(self.t[0])
        for b in self.boxes:
            b = dict(b)
            if b["kind"] == "car" and b["moving"]:
                k = int(b["name"].split("_")[1])
                x0, y0, d, v = self.cars[k]
                s = ((y0 if abs(x0) < 10 else x0) + d * v * t + 25.0) % 50.0 - 25.0
                b["pos"] = [x0, s, 0.75] if abs(x0) < 10 else [s, y0, 0.75]
                b["size"] = [1.8, 4.2, 1.5] if abs(x0) < 10 else [4.2, 1.8, 1.5]
            out.append(b)
        return out + self.extra_boxes

    def _spheres_now(self) -> List[dict]:
        out = []
        t = float(self.t[0])
        for s in self.spheres:
            s = dict(s)
            if s["kind"] == "bird":
                cx, cy, rad, z = self.birds[int(s["name"].split("_")[1])]
                w = 0.6 / rad * 3
                s["pos"] = [cx + rad * math.cos(w * t), cy + rad * math.sin(w * t), z + 0.5 * math.sin(2 * t)]
            out.append(s)
        return out

    def _animals_now(self) -> List[dict]:
        out = []
        t = float(self.t[0])
        for a in self.animals:
            cx, cy, rad, w, ph = a["orbit"]
            ang = ph + w * t
            z = {"whale": -1.5, "dolphin": -0.4 + 0.6 * math.sin(3 * t + ph), "turtle": -0.6}[a["type"]]
            pos = [cx + rad * math.cos(ang), cy + rad * math.sin(ang), z]
            vel = [-rad * w * math.sin(ang), rad * w * math.cos(ang), 0.0]
            out.append({"name": a["name"], "type": a["type"], "pos": pos, "vel": vel, "radius": a["radius"],
                        "tangled": False})
        return out

    def _net_nodes(self, i: int) -> torch.Tensor:
        c = self.items[i, self.corner_base:]   # corners 0..3: (r0,c0), (r0,cN), (rN,cN), (rN,c0)
        R, C = spec.NET_ROWS, spec.NET_COLS
        u = torch.linspace(0, 1, C).view(1, C, 1)
        v = torch.linspace(0, 1, R).view(R, 1, 1)
        top = c[0] * (1 - u) + c[1] * u
        bot = c[3] * (1 - u) + c[2] * u
        p = top * (1 - v) + bot * v
        sag = (torch.sin(math.pi * u) * torch.sin(math.pi * v)).squeeze(-1) * 1.2
        p = p.clone()
        p[..., 2] = p[..., 2] - sag
        return p.reshape(-1, 3)

    def _entity_positions(self) -> torch.Tensor:
        """[N, E, 3] world positions in entity_names() order (inactive slots at the origin)."""
        N = self.num_envs
        names = self.entity_names()
        out = torch.zeros(N, len(names), 3)
        hand = self._handles()
        boxes = self._boxes_now()
        animals = {a["name"]: a for a in self._animals_now()}
        for k, n in enumerate(names):
            if n.startswith("item_"):
                i = int(n.split(".")[0].split("_")[1])
                if i < self.n_items:
                    out[:, k] = hand[:, i] if n.endswith(".handle") else self.items[:, i]
            elif n.startswith("net.corner_"):
                out[:, k] = self.items[:, self.corner_base + int(n.rsplit("_", 1)[1])]
            elif n == "net.center":
                out[:, k] = self.items[:, self.corner_base:self.corner_base + 4].mean(1)
            elif n.startswith("fish_school_"):
                s = int(n.rsplit("_", 1)[1])
                out[:, k] = torch.tensor(self._school_center(s) + [-0.8])
            elif n.startswith("obstacle_"):
                j = int(n.rsplit("_", 1)[1])
                if j < len(boxes):
                    out[:, k] = torch.tensor(boxes[j]["pos"])
            elif n.startswith("drone_"):
                out[:, k] = self.drone[:, int(n.rsplit("_", 1)[1])]
            elif n in animals:
                out[:, k] = torch.tensor(animals[n]["pos"])
            elif self._zone(n) is not None:
                out[:, k] = torch.tensor(self._zone(n)["pos"])
        return out

    def _school_center(self, s: int) -> List[float]:
        cx, cy = self.school_c[s]
        t = float(self.t[0])
        return [cx + 2.0 * math.cos(0.1 * t + s), cy + 2.0 * math.sin(0.1 * t + s)]

    def _anchor_state(self):
        """pos [N, A, K, 3], kind index [A, K], radius [A, K]."""
        N, A, K = self.num_envs, self.num_agents, spec.N_ANCHORS
        names = self.entity_names()
        ent = self._entity_positions()
        pos = torch.zeros(N, A, K, 3)
        kind = torch.zeros(A, K, dtype=torch.long)
        rad = torch.zeros(A, K)
        for a in range(A):
            for k, b in enumerate(self.bindings[a]):
                if b.kind == "none":
                    continue
                kind[a, k] = spec.ANCHOR_TYPES.index(b.kind) if b.kind in spec.ANCHOR_TYPES else 1
                if b.entity is not None and b.entity in names:
                    pos[:, a, k] = ent[:, names.index(b.entity)]
                    z = self._zone(b.entity)
                    rad[a, k] = b.radius or (z["radius"] if z else 0.0)
                elif b.point is not None:
                    pos[:, a, k] = torch.tensor([float(v) for v in b.point])
                    rad[a, k] = b.radius
                else:
                    kind[a, k] = 0
        return pos, kind, rad

    # ------------------------------------------------------------ obs
    def _obs(self) -> Dict[str, torch.Tensor]:
        N, A = self.num_envs, self.num_agents
        obs = spec.empty_obs(N, A)
        apos, akind, arad = self._anchor_state()
        surf = self._ground_t(self.drone)
        height = self.drone[..., 2] - surf
        holding = self.held >= 0
        rel = apos - self.drone.unsqueeze(2)
        inside = (rel[..., :2].norm(dim=-1) < arad.unsqueeze(0)) & (arad.unsqueeze(0) > 0)
        p = obs["proprio"]
        p[..., 0:3] = self.vel
        p[..., 5] = -1.0
        p[..., 9] = self.grip.float()
        p[..., 10] = holding.float()
        p[..., 11] = height
        p[..., 12:17] = self.last_act
        an = obs["anchors"]
        an[..., 0:3] = rel
        an[..., 3:7] = torch.nn.functional.one_hot(akind, len(spec.ANCHOR_TYPES)).float().unsqueeze(0)
        an[..., 7] = arad.unsqueeze(0)
        an[..., 8] = inside.float()
        obs["lidar"].fill_(255)
        obs["lidar_down"].fill_(255)
        if A > 1:
            for a in range(A):
                others = [b for b in range(A) if b != a]
                for j, b in enumerate(others[:spec.MAX_MATES]):
                    obs["mates"][:, a, j, 0:3] = self.drone[:, b] - self.drone[:, a]
                    obs["mates"][:, a, j, 3:6] = self.vel[:, b] - self.vel[:, a]
                    obs["mates"][:, a, j, 6] = holding[:, b].float()
                    obs["mates_mask"][:, a, j] = True
        g = obs["g"]
        S = spec.G_SLICES
        hand = self._handles()
        lh = self.last_held
        payload = torch.where((lh >= 0).unsqueeze(-1),
                              torch.gather(self.items, 1, lh.clamp_min(0).unsqueeze(-1).expand(N, A, 3)),
                              self.drone)
        g[..., S["drone_pos"]] = self.drone
        g[..., S["drone_vel"]] = self.vel
        g[..., S["speed"]] = self.vel.norm(dim=-1, keepdim=True)
        g[..., S["height"]] = height.unsqueeze(-1)
        g[..., S["grip_closed"]] = self.grip.float().unsqueeze(-1)
        g[..., S["holding"]] = holding.float().unsqueeze(-1)
        g[..., S["payload_rel"]] = payload - self.drone
        g[..., S["anchor_rel_drone"]] = rel.reshape(N, A, -1)
        g[..., S["anchor_rel_payload"]] = (apos - payload.unsqueeze(2)).reshape(N, A, -1)
        g[..., S["anchor_inside"]] = inside.float()
        g[..., S["obstacle_dist"]] = 10.0
        g[..., S["animal_dist"]] = 10.0
        g[..., S["mate_dist"]] = 2 * self.arena_half
        if self.scene in ("trash", "world"):
            trash = torch.tensor([TYPE_INFO[t][4] for t in self.item_types])
            left = (trash & ~self.delivered[:, :self.n_items]).sum(-1).float() / trash.sum().clamp_min(1)
            g[..., S["trash_left"]] = left.view(N, 1, 1).expand(N, A, 1)
        if self.scene != "ocean":
            for a in range(A):
                for i in range(N):
                    h = int(self.held[i, a])
                    if h >= 0 and not self._is_corner(h):
                        t = self.item_types[h]
                        g[i, a, S["payload_type"]] = torch.nn.functional.one_hot(
                            torch.tensor(spec.ITEM_TYPES.index(t)), len(spec.ITEM_TYPES)).float()
                        g[i, a, S["payload_is_trash"]] = float(TYPE_INFO[t][4])
                        g[i, a, S["held_mass"]] = TYPE_INFO[t][3]
        if self.has_net:
            cb = self.corner_base
            cen = self.items[:, cb:cb + 4].mean(1, keepdim=True)
            g[..., S["net_centroid_rel"]] = cen - self.drone
        return obs

    # ------------------------------------------------------------ step
    def step(self, actions: torch.Tensor) -> StepOut:
        N, A = self.num_envs, self.num_agents
        act = actions.detach().float().cpu().clamp(-1, 1)
        cmd = torch.stack([act[..., 0] * self.max_speed, act[..., 1] * self.max_speed, act[..., 2] * self.max_vz], -1)
        alive = self.alive.unsqueeze(-1)
        self.vel = torch.where(alive, self.vel + 0.6 * (cmd - self.vel), torch.zeros_like(self.vel))
        self.vel = self.vel + self.gust.unsqueeze(1) * 0.02 * alive
        self.drone = self.drone + self.vel * spec.POLICY_DT
        self.drone[..., :2] = self.drone[..., :2].clamp(-self.arena_half, self.arena_half)
        floor = self._ground_t(self.drone) + 0.1
        self.drone[..., 2] = torch.maximum(self.drone[..., 2], floor)
        self.drone[..., 2] = torch.where(self.alive, self.drone[..., 2], floor)
        self.last_act = act
        ev = {k: torch.zeros(N, A, dtype=torch.bool) for k in spec.EVENT_KEYS}
        closed = (act[..., 4] > spec.GRIP_CLOSED_THRESHOLD) & self.alive
        grip_pt = self.drone + torch.tensor(spec.GRIPPER_OFFSET)
        hand = self._handles()
        reach = torch.tensor([0.45 if (self.scene == "ocean" or self._is_corner(m)) else 0.15
                              for m in range(self.items.shape[1])])
        for i in range(N):
            for a in range(A):
                h = int(self.held[i, a])
                if closed[i, a] and h < 0 and not self.grip[i, a]:
                    d = (hand[i] - grip_pt[i, a]).norm(dim=-1)
                    taken = set(int(x) for x in self.held[i] if int(x) >= 0)
                    for m in torch.argsort(d).tolist():
                        if d[m] < reach[m] and m not in taken and not bool(self.delivered[i, m]):
                            self.held[i, a] = m
                            self.last_held[i, a] = m
                            ev["grasp"][i, a] = True
                        break
                elif not closed[i, a] and h >= 0:
                    self.held[i, a] = -1
                    ev["release"][i, a] = True
                    if self._is_corner(h):
                        self.items[i, h, 2] = self._water_level()
                    elif self.scene != "ocean":
                        x, y = float(self.items[i, h, 0]), float(self.items[i, h, 1])
                        self.items[i, h, 2] = max(self.ground(x, y), self._water_level() if self.heights is not None
                                                  else -1e9) + _hz(self.item_types[h])
                        drops = [self.zones[0]] if self.scene != "world" else \
                            [z for z in self.zones if z["name"] in ("bin", "pad_dropoff", "boat")]
                        for drop in drops:
                            if math.hypot(x - drop["pos"][0], y - drop["pos"][1]) < drop["radius"]:
                                self.delivered[i, h] = True
                                ev["delivered"][i, a] = True
                                self.items[i, h, 2] = drop["pos"][2] - 0.3
                                break
        self.grip = closed
        for i in range(N):
            for a in range(A):
                h = int(self.held[i, a])
                if h >= 0:
                    off = 0.0 if (self.scene == "ocean" or self._is_corner(h)) else _hz(self.item_types[h]) + 0.03
                    self.items[i, h] = grip_pt[i, a] - torch.tensor([0.0, 0.0, off])
        self.t += spec.POLICY_DT
        self.steps += 1
        trunc = self.steps >= self.episode_steps
        done = torch.zeros(N, dtype=torch.bool)
        return StepOut(obs=self._obs(), done=done, truncated=trunc, events=ev)

    # ------------------------------------------------------------ perturb
    def perturb(self, env_id: int, kind: str, **kw) -> None:
        i = int(env_id)
        if kind == "drop_obstacle":
            x, y = [float(v) for v in kw["pos"][:2]]
            hf = float(kw.get("half", 0.6))
            self.extra_boxes.append({"name": f"dropped_{len(self.extra_boxes)}", "kind": "dropped",
                                     "pos": [x, y, self.ground(x, y) + hf], "size": [2 * hf] * 3, "moving": False})
        elif kind == "gust":
            if self.gusts_enabled:
                self.gust[i] = torch.tensor([float(v) for v in kw["vec"][:3]])
        elif kind == "waves":
            self.wave_scale = float(kw.get("scale", 1.0))
        elif kind == "tide":
            if self.heights is not None:
                self.tide = float(kw.get("offset", 0.0))
        elif kind == "move_payload":
            if self.scene == "ocean":
                return
            item = kw.get("item", 0)
            m = int(str(item).split("_")[-1]) if item is not None else 0
            m = max(0, min(m, self.n_items - 1))
            x, y = [float(v) for v in kw["pos"][:2]]
            for a in range(self.num_agents):
                if int(self.held[i, a]) == m:
                    self.held[i, a] = -1
            self.items[i, m] = torch.tensor([x, y, self.ground(x, y) + _hz(self.item_types[m])])
            self.delivered[i, m] = False
        elif kind == "disable_drone":
            a = int(kw["agent"])
            self.alive[i, a] = False
            self.held[i, a] = -1

    # ------------------------------------------------------------ render
    def render_state(self, env_id: int, include_terrain: bool = True) -> dict:
        i = int(env_id)
        terrain = None
        if self.heights is not None:
            terrain = {"version": self.terrain_version, "res": self.res, "half": self.arena_half}
            if include_terrain:
                terrain["heights"] = [[round(v, 3) for v in row] for row in self.heights.tolist()]
                self._sent_version = self.terrain_version
        water = None
        if self.variant in ("ocean", "world"):
            w = dict(self.waves)
            w["amp"] = [a * self.wave_scale for a in w["amp"]]
            water = {"level": 0.0 if self.variant == "ocean" else self._water_level(), "waves": w}
        elif self.heights is not None:
            water = {"level": WATER_LEVEL + self.tide, "waves": None}
        apos, akind, arad = self._anchor_state()
        anchors = [{"pos": apos[i, 0, k].tolist(), "kind": spec.ANCHOR_TYPES[int(akind[0, k])],
                    "radius": float(arad[0, k])} for k in range(spec.N_ANCHORS)]
        items = []
        if self.scene != "ocean":
            for m, t in enumerate(self.item_types):
                items.append({"name": f"item_{m}", "type": t, "is_trash": bool(TYPE_INFO[t][4]),
                              "shape": TYPE_INFO[t][1], "pos": self.items[i, m].tolist(), "size": _size3(t),
                              "held": bool((self.held[i] == m).any()), "delivered": bool(self.delivered[i, m])})
        net = fish = None
        animals = []
        if self.has_net:
            net = {"rows": spec.NET_ROWS, "cols": spec.NET_COLS, "nodes": self._net_nodes(i).tolist(),
                   "held": [int(h) for h in self.held[i].tolist()]}
            fish = []
            t = float(self.t[0])
            for s in range(spec.N_FISH_SCHOOLS):
                cx, cy = self._school_center(s)
                for f in range(spec.FISH_PER_SCHOOL):
                    a = f * 2.4 + t * 0.8
                    fish.append([cx + 1.2 * math.cos(a), cy + 1.2 * math.sin(a), -0.8 - 0.3 * math.sin(f)])
            animals = self._animals_now()
            nodes = self._net_nodes(i)
            for an in animals:
                d = (nodes - torch.tensor(an["pos"])).norm(dim=-1).min()
                an["tangled"] = bool(d < an["radius"] * 0.6)
        d0 = self.drone[i, 0].tolist()
        nir = []
        if self.lidar_on:
            for k in range(24):
                ang = k * 2 * math.pi / 24
                x, y = d0[0] + 3 * math.cos(ang), d0[1] + 3 * math.sin(ang)
                nir.append([x, y, float(self._ground_t(torch.tensor([x, y, 0.0])))])
        return {
            "t": float(self.t[i]), "scene": self.scene, "variant": self.variant, "arena_half": self.arena_half, "max_speed": self.max_speed,
            "terrain": terrain, "water": water,
            "drones": [{"pos": self.drone[i, a].tolist(), "quat": [1.0, 0.0, 0.0, 0.0], "grip": bool(self.grip[i, a]),
                        "holding": bool(self.held[i, a] >= 0), "alive": bool(self.alive[i, a])}
                       for a in range(self.num_agents)],
            "items": items, "boxes": self._boxes_now(), "spheres": self._spheres_now(),
            "zones": [z for z in self.zones if not (self.scene == "ocean" and z["kind"] == "nofly")],
            "anchors": anchors, "boat": dict(self.boat) if self.has_net else None,
            "net": net, "fish": fish or [], "animals": animals,
            "lidar": {"nir": nir, "green": []}, "shield_hits": int(self.shield_count[i]),
            "wind": self.gust[i].tolist(),
        }


def make_fake_env(scene: str = "trash", num_envs: int = 1, device: str = "cpu", **kw) -> FakeEnv:
    return FakeEnv(scene, num_envs=num_envs, device=device, **kw)
