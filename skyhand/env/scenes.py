"""Per-world layout generation for the trash and delivery scenes (plain Python).

Each generator returns a Layout of Python lists for one world; SkyEnv copies a
batch of layouts into its tensors. Placement uses the world's heightfield (numpy)
so objects sit on dry land with a modest slope. Layouts are reproducible from the
env's seeded random.Random.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import spec

H = spec.SCENE_ARENA_HALF["trash"]
STREET = 7.0


@dataclass
class Box:
    kind: str
    c: Tuple[float, float, float]
    h: Tuple[float, float, float]                     # half extents
    lane: Optional[Tuple[float, float, float, float, float, float]] = None   # ax, ay, bx, by, speed, phase


@dataclass
class Sphere:
    kind: str
    c: Tuple[float, float, float]
    r: float
    orbit: Optional[Tuple[float, float, float, float, float, float]] = None  # cx, cy, cz, radius, omega, phase


@dataclass
class Layout:
    variant: str
    boxes: List[Box] = field(default_factory=list)
    spheres: List[Sphere] = field(default_factory=list)
    items: List[Tuple[str, float, float]] = field(default_factory=list)      # type, x, y
    zones: Dict[int, Tuple[str, float, float, float, float]] = field(default_factory=dict)  # slot -> kind, x, y, r, drop_h
    drones: List[Tuple[float, float, float]] = field(default_factory=list)   # x, y, height above ground


class Ground:
    """Bilinear sampling of one world's heightfield, plus water level for dry-land tests."""

    def __init__(self, h: Optional[np.ndarray], flat_z: float, water: float, tide_amp: float, half: float = H):
        self.h, self.flat_z, self.water, self.tide, self.half = h, flat_z, water, tide_amp, half
        self.res = None if h is None else h.shape[0]

    def __call__(self, x: float, y: float) -> float:
        if self.h is None:
            return self.flat_z
        r = self.res - 1
        hf = self.half
        u = min(max((x + hf) / (2 * hf) * r, 0.0), r - 1e-4)
        v = min(max((y + hf) / (2 * hf) * r, 0.0), r - 1e-4)
        j, i = int(u), int(v)
        fu, fv = u - j, v - i
        h = self.h
        return float(h[i, j] * (1 - fu) * (1 - fv) + h[i, j + 1] * fu * (1 - fv)
                     + h[i + 1, j] * (1 - fu) * fv + h[i + 1, j + 1] * fu * fv)

    def dry(self, x: float, y: float, margin: float = 0.3) -> bool:
        return self(x, y) > self.water + self.tide + margin

    def slope(self, x: float, y: float, d: float = 1.0) -> float:
        return max(abs(self(x + d, y) - self(x - d, y)), abs(self(x, y + d) - self(x, y - d))) / (2 * d)

    def spot(self, rng: random.Random, half: Optional[float] = None, max_slope: float = 0.35,
             avoid: List[Tuple[float, float, float]] = (), tries: int = 200,
             near: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float]:
        """A dry, gentle spot at least avoid-radius from each (x, y, r) in avoid; within
        near = (x, y, r) when given."""
        half = self.half - 3 if half is None else half
        best = None
        for _ in range(tries):
            if near is not None:
                a, rr = rng.uniform(0, 2 * math.pi), near[2] * math.sqrt(rng.random())
                x = min(max(near[0] + rr * math.cos(a), -half), half)
                y = min(max(near[1] + rr * math.sin(a), -half), half)
            else:
                x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            if not self.dry(x, y) or self.slope(x, y) > max_slope:
                continue
            if all((x - ax) ** 2 + (y - ay) ** 2 > ar * ar for ax, ay, ar in avoid):
                return x, y
            best = best or (x, y)
        return best or (rng.uniform(-half, half), rng.uniform(-half, half))


# ---------------------------------------------------------------- shared decorations
def _trees_rocks(L: Layout, g: Ground, rng: random.Random, n_trees: int, n_rocks: int, avoid) -> None:
    for _ in range(n_trees):
        x, y = g.spot(rng, max_slope=0.6, avoid=avoid)
        z = g(x, y)
        th = rng.uniform(2.0, 4.0)
        L.boxes.append(Box("trunk", (x, y, z + th / 2), (0.2, 0.2, th / 2)))
        r = rng.uniform(1.0, 1.8)
        L.spheres.append(Sphere("canopy", (x, y, z + th + 0.6 * r), r))
        avoid = list(avoid) + [(x, y, r + 1.0)]
    for _ in range(n_rocks):
        x, y = g.spot(rng, max_slope=0.8, avoid=avoid)
        r = rng.uniform(0.3, 0.9)
        L.spheres.append(Sphere("rock", (x, y, g(x, y) + 0.3 * r), r))


def _nofly(L: Layout, rng: random.Random, slot: int, keep_clear: List[Tuple[float, float, float]]) -> None:
    for _ in range(50):
        x, y, r = rng.uniform(-H + 6, H - 6), rng.uniform(-H + 6, H - 6), rng.uniform(2.5, 5.0)
        if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 2.0) ** 2 for a, b, c in keep_clear):
            L.zones[slot] = ("nofly", x, y, r, 0.0)
            return


def _clear_start(L: Layout, g: Ground, rng: random.Random, near: Tuple[float, float],
                 rmin: float, rmax: float, height: Tuple[float, float]) -> Tuple[float, float, float]:
    """A drone start rmin..rmax from `near`, clear of every sphere and box footprint."""
    hlo, hhi = height
    for k in range(200):
        r = rng.uniform(rmin, rmax + 0.1 * k)
        a = rng.uniform(0, 2 * math.pi)
        x = min(max(near[0] + r * math.cos(a), -g.half + 2), g.half - 2)
        y = min(max(near[1] + r * math.sin(a), -g.half + 2), g.half - 2)
        if not g.dry(x, y, 0.0):
            continue
        h = rng.uniform(hlo, hhi)
        z = g(x, y) + h
        ok = all(math.hypot(x - s.c[0], y - s.c[1]) > s.r + 1.5 or abs(z - s.c[2]) > s.r + 1.5 for s in L.spheres)
        ok = ok and all(abs(x - b.c[0]) > b.h[0] + 1.2 or abs(y - b.c[1]) > b.h[1] + 1.2 or z > b.c[2] + b.h[2] + 1.2
                        for b in L.boxes)
        if ok:
            return x, y, h
    return near[0], near[1], hhi + 4.0


def _packed_block(L: Layout, bx: float, by: float, block: float, sidewalk: float, base: float,
                  lots: int, rng: random.Random, hmin: float = 8.0, hmax: float = 30.0) -> None:
    """Fill one city block with buildings built lot to lot (thin gaps between them), like a
    Manhattan block: the block's sidewalk ring stays free for trash piles and hydrants."""
    span = block - 2 * sidewalk
    lot = span / lots
    gap = 0.3
    block_h = rng.uniform(hmin, hmax)
    for i in range(lots):
        for j in range(lots):
            x0, x1 = bx + sidewalk + i * lot + gap / 2, bx + sidewalk + (i + 1) * lot - gap / 2
            y0, y1 = by + sidewalk + j * lot + gap / 2, by + sidewalk + (j + 1) * lot - gap / 2
            ht = min(max(block_h * rng.uniform(0.6, 1.4), hmin * 0.75), hmax * 1.2)
            L.boxes.append(Box("building", ((x0 + x1) / 2, (y0 + y1) / 2, base + ht / 2),
                               ((x1 - x0) / 2, (y1 - y0) / 2, ht / 2)))


# ---------------------------------------------------------------- trash: city
def city(rng: random.Random, nofly_prob: float) -> Layout:
    L = Layout("city")
    g = Ground(None, 0.0, -1e4, 0.0)
    # a tight grid: big blocks packed lot to lot with buildings, narrow streets between
    nb = 2
    street = 4.5
    block = (2 * H - (nb + 1) * street) / nb
    starts = [-H + street + k * (block + street) for k in range(nb)]           # block min corner
    streets = [-H + street / 2 + k * (block + street) for k in range(nb + 1)]  # street centre lines
    sidewalk = 1.8                 # bags sit at the curb, ~1.3 m clear of the walls
    blocks = [(bx, by) for bx in starts for by in starts]
    for bx, by in blocks:
        _packed_block(L, bx, by, block, sidewalk, 0.0, 3, rng, hmin=6.0, hmax=18.0)
    # garbage truck parked mid-block on a street; its zone is the drop-off
    truck_x = rng.random() < 0.5
    ts = rng.choice(streets)
    p = rng.choice(starts) + block / 2
    tx, ty = (p, ts) if truck_x else (ts, p)
    L.boxes.append(Box("truck", (tx, ty, 1.5), (3.0, 1.2, 1.5) if truck_x else (1.2, 3.0, 1.5)))
    L.zones[0] = ("truck", tx, ty, 3.2, 6.0)
    # cars drive back and forth down the middle of the other streets
    lanes = [(ax, s_) for ax in (True, False) for s_ in streets if not (ax == truck_x and s_ == ts)]
    rng.shuffle(lanes)
    for along_x, s_ in lanes[:3]:
        span = H - 2.5
        h = (2.1, 0.9, 0.75) if along_x else (0.9, 2.1, 0.75)
        lane = (-span, s_, span, s_, rng.uniform(3.0, 6.0), rng.uniform(0, 1)) if along_x else \
            (s_, -span, s_, span, rng.uniform(3.0, 6.0), rng.uniform(0, 1))
        L.boxes.append(Box("car", (lane[0], lane[1], 0.75), h, lane))
    # trash piles on the sidewalk against a building, NYC style
    piles = []
    for _ in range(rng.choice([2, 3])):
        bx, by = rng.choice(blocks)
        side = rng.randrange(4)
        t = rng.uniform(1.5, block - 1.5)
        e = 0.45                   # at the curb (street side of the sidewalk)
        cx, cy = [(bx + t, by + e), (bx + t, by + block - e), (bx + e, by + t), (bx + block - e, by + t)][side]
        piles.append((cx, cy, side))
    for cx, cy, side in piles:
        for _ in range(rng.randint(3, 4)):
            if len(L.items) >= spec.MAX_ITEMS - 2:
                break
            typ = rng.choices(["trash_bag", "cardboard_box", "pizza_box", "bottle"], [0.65, 0.15, 0.12, 0.08])[0]
            off = rng.uniform(-1.2, 1.2)
            jit = rng.uniform(-0.2, 0.2)
            x, y = (cx + off, cy + jit) if side < 2 else (cx + jit, cy + off)
            L.items.append((typ, x, y))
    # things on the sidewalk that are not trash
    for _ in range(rng.randint(1, 2)):
        if len(L.items) >= spec.MAX_ITEMS:
            break
        bx, by = rng.choice(blocks)
        L.items.append((rng.choice(["backpack", "ball", "flower_pot"]), bx + rng.uniform(1.0, block - 1.0), by + 0.6))
    for _ in range(rng.randint(2, 4)):
        bx, by = rng.choice(blocks)
        L.boxes.append(Box("hydrant", (bx + rng.uniform(1.0, block - 1.0), by + block - 0.4, 0.4), (0.15, 0.15, 0.4)))
    for _ in range(rng.randint(1, 2)):
        bx, by = rng.choice(blocks)
        L.boxes.append(Box("bench", (bx + 0.6, by + rng.uniform(1.5, block - 1.5), 0.25), (0.25, 0.8, 0.25)))
    if rng.random() < nofly_prob:
        _nofly(L, rng, 1, [(tx, ty, 3.5)] + [(x, y, 1.0) for _, x, y in L.items])
    ix, iy = rng.choice(streets), rng.choice(streets)
    L.drones = [(ix, iy, rng.uniform(3.0, 6.0))]
    return L


# ---------------------------------------------------------------- trash: litter on terrain
def litter(rng: random.Random, g: Ground, nofly_prob: float) -> Layout:
    L = Layout("terrain")
    bx, by = g.spot(rng, half=H - 6, max_slope=0.2)
    L.boxes.append(Box("bin", (bx, by, g(bx, by) + 0.5), (0.45, 0.45, 0.5)))
    L.zones[0] = ("bin", bx, by, 1.0, 4.0)
    avoid = [(bx, by, 3.0)]
    for _ in range(rng.randint(4, 7)):
        x, y = g.spot(rng, avoid=avoid)
        L.items.append((rng.choice(["trash_bag", "pizza_box", "cardboard_box", "bottle"]), x, y))
        avoid.append((x, y, 1.0))
    for _ in range(rng.randint(1, 3)):
        x, y = g.spot(rng, avoid=avoid)
        L.items.append((rng.choice(["backpack", "ball", "flower_pot"]), x, y))
        avoid.append((x, y, 1.0))
    for _ in range(rng.randint(0, 2)):
        x, y = g.spot(rng, max_slope=0.15, avoid=avoid)
        L.boxes.append(Box("bench", (x, y, g(x, y) + 0.25), (0.8, 0.25, 0.25)))
    _trees_rocks(L, g, rng, rng.randint(3, 6), rng.randint(3, 6), avoid)
    if rng.random() < nofly_prob:
        _nofly(L, rng, 1, [(bx, by, 2.0)] + [(x, y, 1.0) for _, x, y in L.items])
    L.drones = [_clear_start(L, g, rng, (bx, by), 4.0, 12.0, (2.5, 4.5))]
    return L


# ---------------------------------------------------------------- delivery (long range, remote)
def delivery(rng: random.Random, g: Ground, nofly_prob: float) -> Layout:
    """Two remote clinics 300+ m apart, each with a clinic building; acacias,
    and a wildlife-reserve no-fly zone that often sits across the route."""
    L = Layout("delivery")
    half = g.half - 15
    best = None
    for _ in range(80):
        a = g.spot(rng, half=half, max_slope=0.12)
        b = g.spot(rng, half=half, max_slope=0.12, avoid=[(a[0], a[1], 300.0)])
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if best is None or d > best[2]:
            best = (a, b, d)
        if d > 380.0:
            break
    (ax, ay), (bx, by), _ = best
    L.zones[3] = ("pad_pickup", ax, ay, 3.0, 0.0)
    L.zones[4] = ("pad_dropoff", bx, by, 3.0, 0.6)
    L.items.append(("cooler", ax, ay))
    keep = [(ax, ay, 8.0), (bx, by, 8.0)]
    for px, py in ((ax, ay), (bx, by)):                  # a clinic building beside each pad
        cx, cy = g.spot(rng, max_slope=0.25, near=(px, py, 18.0), avoid=keep)
        L.boxes.append(Box("building", (cx, cy, g(cx, cy) + 2.0), (4.0, 3.0, 2.0)))
        keep.append((cx, cy, 6.0))
    # wildlife reserve no-fly zones near the middle of the route
    mx, my = (ax + bx) / 2, (ay + by) / 2
    for slot in (1, 2):
        if rng.random() < nofly_prob:
            for _ in range(50):
                r = rng.uniform(35.0, 70.0)
                x, y = mx + rng.uniform(-90, 90), my + rng.uniform(-90, 90)
                if math.hypot(x - ax, y - ay) > r + 30 and math.hypot(x - bx, y - by) > r + 30:
                    L.zones[slot] = ("nofly", x, y, r, 0.0)
                    break
    for _ in range(rng.randint(8, 12)):                  # acacias: short trunk, wide flat crown
        x, y = g.spot(rng, max_slope=0.5, avoid=keep)
        z = g(x, y)
        L.boxes.append(Box("trunk", (x, y, z + 1.5), (0.25, 0.25, 1.5)))
        L.spheres.append(Sphere("canopy", (x, y, z + 4.0), rng.uniform(2.0, 3.0)))
    for _ in range(rng.randint(2, 4)):
        x, y = g.spot(rng, max_slope=0.8, avoid=keep)
        r = rng.uniform(0.8, 2.0)
        L.spheres.append(Sphere("rock", (x, y, g(x, y) + 0.3 * r), r))
    L.drones = [_clear_start(L, g, rng, (ax, ay), 6.0, 12.0, (3.0, 5.0))]
    return L


# ---------------------------------------------------------------- generic world
@dataclass
class WorldLayout(Layout):
    heights: Optional[np.ndarray] = None
    sea: Tuple[float, float, float] = (0.0, 0.0, 10.0)          # deep-water disc for fish and animals
    boat: Tuple[float, float] = (0.0, 0.0)
    net: Tuple[float, float] = (0.0, 0.0)


def world_terrain(rng: random.Random, res: int, half: float):
    """Heightfield [res, res] with hills, lakes, a coastline into deep sea, and a flat town
    plateau. Returns (heights, sea direction unit vector, coast offset, town rect)."""
    lin = np.linspace(-half, half, res)
    X, Y = np.meshgrid(lin, lin)                                # rows = y, cols = x
    h = np.full_like(X, 3.0)                                   # countryside sits a few metres above sea level

    def bump(cx, cy, amp, sig):
        return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig * sig))

    for _ in range(9):
        h += bump(rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(3, 18), rng.uniform(15, 40))
    for _ in range(2):
        h += bump(rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(20, 35), rng.uniform(20, 35))
    for _ in range(2):                                           # inland lake basins
        h -= bump(rng.uniform(-half * 0.6, half * 0.6), rng.uniform(-half * 0.6, half * 0.6),
                  rng.uniform(8, 14), rng.uniform(12, 22))
    for _ in range(3):
        k = 2 * math.pi / rng.uniform(40, 120)
        th = rng.uniform(0, 2 * math.pi)
        h += rng.uniform(0.5, 1.5) * np.sin(k * (math.cos(th) * X + math.sin(th) * Y) + rng.uniform(0, 6.3))
    # coastline: the sea side of the map slopes into deep water
    th = rng.uniform(0, 2 * math.pi)
    sea_dir = (math.cos(th), math.sin(th))
    c0 = rng.uniform(0.45, 0.6) * half
    s = X * sea_dir[0] + Y * sea_dir[1] - c0                     # > 0 offshore
    w = 1.0 / (1.0 + np.exp(-s / 8.0))
    h = h * (1 - w) + (-4.0 - 10.0 * np.clip(s / 60.0, 0, 1)) * w
    # town plateau on dry land, away from the coast
    for _ in range(100):
        tx, ty = rng.uniform(-half * 0.6, half * 0.6), rng.uniform(-half * 0.6, half * 0.6)
        if tx * sea_dir[0] + ty * sea_dir[1] - c0 < -70:
            break
    tw = rng.uniform(70, 90) / 2
    town = (tx - tw, ty - tw, tx + tw, ty + tw)
    inside = np.clip(np.minimum.reduce([X - town[0], town[2] - X, Y - town[1], town[3] - Y]) / 12.0 + 1.0, 0, 1)
    sel = inside >= 0.999
    plateau = max(float(np.median(h[sel])) if sel.any() else 2.0, 2.0)
    h = h * (1 - inside) + plateau * inside
    return np.clip(h, -16, 60), sea_dir, c0, town, plateau


def world(rng: random.Random, res: int, half: float, nofly_prob: float) -> WorldLayout:
    h, sea_dir, c0, town, plateau = world_terrain(rng, res, half)
    L = WorldLayout("world", heights=h)
    g = Ground(h, 0.0, 0.0, 0.3, half=half)
    # ---- town: street grid, cars, trash piles at the curb, garbage truck
    x0, y0, x1, y1 = town
    nb = 3
    street, sidewalk = 5.0, 1.8
    block = ((x1 - x0) - (nb + 1) * street) / nb
    xs = [x0 + street + k * (block + street) for k in range(nb)]
    ys = [y0 + street + k * (block + street) for k in range(nb)]
    sx = [x0 + street / 2 + k * (block + street) for k in range(nb + 1)]
    sy = [y0 + street / 2 + k * (block + street) for k in range(nb + 1)]
    blocks = [(bx, by) for bx in xs for by in ys]
    for bx, by in blocks:
        _packed_block(L, bx, by, block, sidewalk, plateau, 2, rng, hmin=6.0, hmax=18.0)
    tx = rng.choice(sx)
    ty = rng.choice(ys) + block / 2                              # parked mid-block
    L.boxes.append(Box("truck", (tx, ty, plateau + 1.5), (1.2, 3.0, 1.5)))
    lanes = [("x", s_) for s_ in sy] + [("y", s_) for s_ in sx if s_ != tx]
    rng.shuffle(lanes)
    for ax, s_ in lanes[:3]:
        if ax == "x":
            lane = (x0 + 2, s_, x1 - 2, s_, rng.uniform(3, 6), rng.random())
            hh = (2.1, 0.9, 0.75)
        else:
            lane = (s_, y0 + 2, s_, y1 - 2, rng.uniform(3, 6), rng.random())
            hh = (0.9, 2.1, 0.75)
        L.boxes.append(Box("car", (lane[0], lane[1], plateau + 0.75), hh, lane))
    L.zones[0] = ("truck", tx, ty, 3.2, 6.0)
    for _ in range(2):
        bx, by = rng.choice(blocks)
        cx, cy = bx + rng.uniform(1.5, block - 1.5), by + 0.45
        for _ in range(rng.randint(3, 4)):
            typ = rng.choices(["trash_bag", "cardboard_box", "pizza_box", "bottle"], [0.65, 0.15, 0.12, 0.08])[0]
            L.items.append((typ, cx + rng.uniform(-1.2, 1.2), cy + rng.uniform(-0.2, 0.2)))
    for _ in range(2):
        bx, by = rng.choice(blocks)
        L.boxes.append(Box("hydrant", (bx + rng.uniform(1.0, block - 1.0), by + block - 0.4, plateau + 0.4), (0.15, 0.15, 0.4)))
    keep = [((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * 1.45)]
    # ---- clinics with helipads on dry land outside town, the cooler at the pickup pad
    pa = g.spot(rng, max_slope=0.12, avoid=keep)
    pb = g.spot(rng, max_slope=0.12, avoid=keep + [(pa[0], pa[1], 120.0)])
    L.zones[3] = ("pad_pickup", pa[0], pa[1], 3.0, 0.0)
    L.zones[4] = ("pad_dropoff", pb[0], pb[1], 3.0, 0.6)
    L.items.append(("cooler", pa[0], pa[1]))
    keep += [(pa[0], pa[1], 10.0), (pb[0], pb[1], 10.0)]
    for px, py in (pa, pb):
        cx, cy = g.spot(rng, max_slope=0.3, near=(px, py, 16.0), avoid=keep)
        L.boxes.append(Box("building", (cx, cy, g(cx, cy) + 2.0), (4.0, 3.0, 2.0)))
        keep.append((cx, cy, 6.0))
    # ---- countryside litter and things that are not trash, near the town edge
    for _ in range(4):
        x, y = g.spot(rng, near=((x0 + x1) / 2, (y0 + y1) / 2, 90.0), avoid=keep)
        L.items.append((rng.choice(["trash_bag", "pizza_box", "cardboard_box", "bottle"]), x, y))
        keep.append((x, y, 1.0))
    for _ in range(2):
        x, y = g.spot(rng, near=((x0 + x1) / 2, (y0 + y1) / 2, 90.0), avoid=keep)
        L.items.append((rng.choice(["backpack", "ball", "flower_pot"]), x, y))
        keep.append((x, y, 1.0))
    L.items = L.items[:spec.MAX_ITEMS]
    _trees_rocks(L, g, rng, rng.randint(8, 12), rng.randint(4, 6), keep)
    # ---- the sea: boat, net, and a deep-water disc for fish and animals
    sd = sea_dir
    perp = (-sd[1], sd[0])
    along = rng.uniform(-half * 0.4, half * 0.4)
    base = (sd[0] * (c0 + 45.0) + perp[0] * along, sd[1] * (c0 + 45.0) + perp[1] * along)
    L.sea = (base[0] + sd[0] * 25.0, base[1] + sd[1] * 25.0, 40.0)
    L.boat = (base[0] - sd[0] * 10.0, base[1] - sd[1] * 10.0)
    L.net = (base[0] + sd[0] * 12.0 + perp[0] * rng.uniform(-8, 8), base[1] + sd[1] * 12.0 + perp[1] * rng.uniform(-8, 8))
    for slot in (1, 2):
        if rng.random() < nofly_prob:
            for _ in range(50):
                x, y, r = rng.uniform(-half + 30, half - 30), rng.uniform(-half + 30, half - 30), rng.uniform(15.0, 35.0)
                if all(math.hypot(x - a, y - b) > r + c + 5 for a, b, c in keep + [(L.boat[0], L.boat[1], 15.0)]):
                    L.zones[slot] = ("nofly", x, y, r, 0.0)
                    break
    # ---- three drones near the coast between town and sea
    start = (base[0] - sd[0] * 75.0, base[1] - sd[1] * 75.0)        # on land, ~30 m inland
    L.drones = [_clear_start(L, g, rng, start, 4.0 + 6.0 * k, 10.0 + 6.0 * k, (3.0, 6.0)) for k in range(spec.WORLD_AGENTS)]
    return L
