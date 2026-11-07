"""Reward expression language over the goal-space vector g.

Model text is parsed with ``ast.parse(mode="eval")`` and walked with a strict
whitelist; it is never passed to eval/exec. The tree is compiled into nested
closures of torch ops, so a reward program is a batched function
``f(g: Tensor[..., G_DIM]) -> Tensor[...]`` (float32).

Semantics
---------
- Reward programs see raw (unnormalised) g, see spec._G_FIELDS.
- Points are positions relative to the drone (the drone is the origin):
  drone = 0, payload = payload_rel, net = net_centroid_rel,
  a<k> = anchor_rel_drone[k]. dist(p, q) = |q - p|, dz(p, q) = q.z - p.z.
- Truthiness is ``x > 0.5`` everywhere (comparisons return 0/1 floats,
  ``and``/``or``/``not``/``where`` and done_when use the same threshold).
- payload_is("pizza_box") is 1.0 when the payload's item type (payload_type one-hot) is that
  type. Its argument is the ONLY string literal the language accepts and must be one of
  spec.ITEM_TYPES.
- Division is guarded (|denominator| >= 1e-6) and the final result is passed
  through nan_to_num, so a program never returns NaN/inf.
"""
from __future__ import annotations

import ast
import math
from typing import Callable, Dict, List, Optional, Set

import torch

from .. import spec

__all__ = ["DSLError", "CompiledReward", "compile_reward", "describe_vocabulary",
           "SCALARS", "POINTS", "FUNCTIONS", "CONSTANTS", "ITEM_TYPES"]

ITEM_TYPES = tuple(spec.ITEM_TYPES)

MAX_SOURCE_LEN = 2000
MAX_NODES = 400
MAX_POW = 8
MAX_LITERAL = 1e6


class DSLError(ValueError):
    """Raised for any expression outside the reward language."""


Fn = Callable[[torch.Tensor], torch.Tensor]   # g [..., G_DIM] -> [...]


def _field1(name: str) -> Fn:
    s = spec.G_SLICES[name]
    assert s.stop - s.start == 1, name
    return lambda g: g[..., s.start]


def _comp(name: str, i: int) -> Fn:
    s = spec.G_SLICES[name]
    return lambda g: g[..., s.start + i]


# name -> (getter, description)
SCALARS: "Dict[str, tuple]" = {
    "speed": (_field1("speed"), "drone speed (m/s)"),
    "tilt": (_field1("tilt"), "drone tilt (rad)"),
    "height": (_field1("height"), "drone height above the terrain or water surface below it (m)"),
    "over_water": (_field1("over_water"), "1 if the surface below the drone is water (lake / ocean)"),
    "thrust": (_field1("thrust"), "collective thrust / hover thrust (~1 at hover)"),
    "grip_closed": (_field1("grip_closed"), "1 if the gripper is closed"),
    "holding": (_field1("holding"), "1 if holding a handle (an item handle or a net corner)"),
    "held_mass": (_field1("held_mass"), "mass carried (kg)"),
    "payload_is_trash": (_field1("payload_is_trash"), "1 if the payload is a trash item"),
    "swing": (_field1("swing"), "payload swing rate (rad/s), 0 when not holding"),
    "jerk": (_field1("jerk"), "drone jerk (m/s^3, typically 0-100)"),
    "obstacle_dist": (_field1("obstacle_dist"), "distance from drone surface to nearest obstacle (m)"),
    "obstacle_closing": (_field1("obstacle_closing"), "closing speed toward nearest obstacle (m/s, >0 approaching)"),
    "trash_left": (_field1("trash_left"), "fraction of this world's trash not yet delivered (trash scene, 0-1)"),
    "net_stretch": (_field1("net_stretch"), "max net edge stretch ratio minus 1 (ocean)"),
    "fish_in_net": (_field1("fish_in_net"), "fraction of fish inside the net (ocean, 0-1)"),
    "animal_dist": (_field1("animal_dist"), "distance from the nearest marine animal to the net (m, ocean; "
                                            "large elsewhere)"),
    "bycatch": (_field1("bycatch"), "1 while a marine animal is tangled in the net (ocean)"),
    "mate_dist": (_field1("mate_dist"), "distance to nearest teammate drone (m)"),
    "collision": (_field1("collision"), "1 on a step with a collision"),
    "shield": (_field1("shield"), "1 on a step where the safety shield intervened"),
    "vx": (_comp("drone_vel", 0), "drone velocity x (m/s)"),
    "vy": (_comp("drone_vel", 1), "drone velocity y (m/s)"),
    "vz": (_comp("drone_vel", 2), "drone velocity z (m/s, up positive)"),
    "payload_speed": (lambda g: g[..., spec.G_SLICES["payload_vel"]].norm(dim=-1), "payload speed (m/s)"),
}


def _point_getter(name: str) -> Callable[[torch.Tensor], Optional[torch.Tensor]]:
    """Point position relative to the drone, [..., 3]; None for the drone (origin)."""
    if name == "drone":
        return lambda g: None
    if name == "payload":
        s = spec.G_SLICES["payload_rel"]
        return lambda g: g[..., s]
    if name == "net":
        s = spec.G_SLICES["net_centroid_rel"]
        return lambda g: g[..., s]
    k = int(name[1:])
    return lambda g: spec.anchor_rel(g, "anchor_rel_drone", k)


POINTS: Dict[str, str] = {
    "drone": "the drone itself",
    "payload": ("the payload: the item the drone holds, else the item bound to anchor a0, else the nearest item "
                "(ocean: the net corner held or nearest)"),
    "net": "the fishing net centroid (ocean)",
}
for _k in range(spec.N_ANCHORS):
    POINTS[f"a{_k}"] = f"anchor {_k} (bound to a scene entity or a fixed point)"
_POINT_GET = {n: _point_getter(n) for n in POINTS}

CONSTANTS = {"pi": math.pi, "True": 1.0, "False": 0.0}

FUNCTIONS: Dict[str, str] = {
    "dist": "dist(p, q): 3D distance between points",
    "distxy": "distxy(p, q): horizontal distance between points",
    "dx": "dx(p, q): q.x - p.x",
    "dy": "dy(p, q): q.y - p.y",
    "dz": "dz(p, q): q.z - p.z (signed, up positive)",
    "inside": "inside(ak): 1 if the drone is inside anchor k's zone (zones) or within 0.5 m (points)",
    "active": "active(ak): 1 if anchor k is in use (an unused slot, or a selector like nearest:trash with nothing left, is 0)",
    "abs": "abs(x)",
    "min": "min(x, y, ...): elementwise min of 2+ args",
    "max": "max(x, y, ...): elementwise max of 2+ args",
    "clamp": "clamp(x, lo, hi)",
    "exp": "exp(x)",
    "sqrt": "sqrt(x) (x clamped at 0)",
    "log1p": "log1p(x) (x clamped above -1)",
    "tanh": "tanh(x)",
    "where": "where(cond, a, b): a where cond > 0.5 else b",
    "payload_is": ("payload_is('type'): 1 if the payload's item type is 'type' (a quoted item type: "
                   + ", ".join(ITEM_TYPES) + ")"),
}


def describe_vocabulary() -> str:
    """Compact description of the language for the LLM prompt."""
    lines = ["Reward language (a single Python-like expression; evaluated per drone per step):",
             "Scalars:"]
    lines += [f"  {n}: {d}" for n, (_, d) in SCALARS.items()]
    lines.append("Points (only usable as arguments of dist/distxy/dx/dy/dz/inside):")
    lines += [f"  {n}: {d}" for n, d in POINTS.items()]
    lines.append("Functions:")
    lines += [f"  {d}" for d in FUNCTIONS.values()]
    lines.append("Operators: + - * / ** (constant integer exponent, |n| <= %d), unary -, "
                 "comparisons < <= > >= == != (give 1.0 / 0.0), and, or, not (truthy means > 0.5), "
                 "numeric literals, pi." % MAX_POW)
    lines.append("Item types (only as the quoted argument of payload_is): " + ", ".join(ITEM_TYPES) + ".")
    lines.append("Not allowed: attribute access, indexing, strings (except the item type in payload_is), lambdas, "
                 "comprehensions, other names.")
    return "\n".join(lines)


def _allowed_names() -> str:
    return ("scalars: " + ", ".join(SCALARS) + "; points: " + ", ".join(POINTS) +
            "; functions: " + ", ".join(FUNCTIONS) + "; constants: pi")


# ------------------------------------------------------------------ compiler
_BINOPS = {
    ast.Add: torch.add,
    ast.Sub: torch.sub,
    ast.Mult: torch.mul,
}
_CMPOPS = {
    ast.Lt: torch.lt, ast.LtE: torch.le, ast.Gt: torch.gt,
    ast.GtE: torch.ge, ast.Eq: torch.eq, ast.NotEq: torch.ne,
}
_UNARY_FN = {
    "abs": torch.abs,
    "exp": lambda x: torch.exp(x.clamp(max=60.0)),
    "sqrt": lambda x: torch.sqrt(x.clamp_min(0.0)),
    "log1p": lambda x: torch.log1p(x.clamp_min(-1.0 + 1e-6)),
    "tanh": torch.tanh,
}
_POINT_FN = ("dist", "distxy", "dx", "dy", "dz")


def _truthy(x: torch.Tensor) -> torch.Tensor:
    return x > 0.5


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    b = torch.where(b.abs() < eps, torch.where(b < 0, torch.full_like(b, -eps), torch.full_like(b, eps)), b)
    return a / b


class _Compiler:
    def __init__(self, source: str):
        self.source = source
        self.names: Set[str] = set()

    def err(self, node: ast.AST, msg: str) -> DSLError:
        col = getattr(node, "col_offset", None)
        where = f" (at column {col})" if col is not None else ""
        return DSLError(f"{msg}{where} in {self.source!r}. Allowed {_allowed_names()}.")

    def scalar(self, node: ast.AST) -> Fn:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (str, bytes)):
                raise self.err(node, f"strings are not allowed (got {v!r})")
            if not isinstance(v, (int, float)):
                raise self.err(node, f"literal {v!r} is not a number")
            if not math.isfinite(float(v)) or abs(float(v)) > MAX_LITERAL:
                raise self.err(node, f"literal {v!r} is too large (|x| <= {MAX_LITERAL:g})")
            c = float(v)
            return lambda g: torch.full(g.shape[:-1], c, dtype=g.dtype, device=g.device)
        if isinstance(node, ast.Name):
            n = node.id
            if n in SCALARS:
                self.names.add(n)
                get = SCALARS[n][0]
                return get
            if n in CONSTANTS:
                c = CONSTANTS[n]
                return lambda g: torch.full(g.shape[:-1], c, dtype=g.dtype, device=g.device)
            if n in POINTS:
                raise self.err(node, f"{n!r} is a point; use it inside dist/distxy/dx/dy/dz/inside")
            if n in FUNCTIONS:
                raise self.err(node, f"{n!r} is a function and must be called")
            raise self.err(node, f"unknown name {n!r}")
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op is ast.Pow:
                return self.power(node)
            a, b = self.scalar(node.left), self.scalar(node.right)
            if op in _BINOPS:
                f = _BINOPS[op]
                return lambda g: f(a(g), b(g))
            if op is ast.Div:
                return lambda g: _safe_div(a(g), b(g))
            raise self.err(node, f"operator {op.__name__} is not allowed")
        if isinstance(node, ast.UnaryOp):
            x = self.scalar(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda g: -x(g)
            if isinstance(node.op, ast.UAdd):
                return x
            if isinstance(node.op, ast.Not):
                return lambda g: (~_truthy(x(g))).to(g.dtype)
            raise self.err(node, f"unary operator {type(node.op).__name__} is not allowed")
        if isinstance(node, ast.BoolOp):
            parts = [self.scalar(v) for v in node.values]
            is_and = isinstance(node.op, ast.And)

            def boolop(g, parts=parts, is_and=is_and):
                acc = _truthy(parts[0](g))
                for p in parts[1:]:
                    acc = (acc & _truthy(p(g))) if is_and else (acc | _truthy(p(g)))
                return acc.to(g.dtype)
            return boolop
        if isinstance(node, ast.Compare):
            terms = [self.scalar(node.left)] + [self.scalar(c) for c in node.comparators]
            ops = []
            for o in node.ops:
                if type(o) not in _CMPOPS:
                    raise self.err(node, f"comparison {type(o).__name__} is not allowed")
                ops.append(_CMPOPS[type(o)])

            def compare(g, terms=terms, ops=ops):
                vals = [t(g) for t in terms]
                acc = ops[0](vals[0], vals[1])
                for i in range(1, len(ops)):
                    acc = acc & ops[i](vals[i], vals[i + 1])
                return acc.to(g.dtype)
            return compare
        if isinstance(node, ast.IfExp):
            c, a, b = self.scalar(node.test), self.scalar(node.body), self.scalar(node.orelse)
            return lambda g: torch.where(_truthy(c(g)), a(g), b(g))
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Attribute):
            raise self.err(node, "attribute access is not allowed")
        if isinstance(node, ast.Subscript):
            raise self.err(node, "indexing / subscripts are not allowed")
        if isinstance(node, ast.Lambda):
            raise self.err(node, "lambdas are not allowed")
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise self.err(node, "comprehensions are not allowed")
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            raise self.err(node, "strings are not allowed")
        raise self.err(node, f"{type(node).__name__} is not allowed")

    def power(self, node: ast.BinOp) -> Fn:
        e = node.right
        neg = False
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            neg = isinstance(e.op, ast.USub)
            e = e.operand
        if not (isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) and not isinstance(e.value, bool)
                and float(e.value).is_integer()):
            raise self.err(node, "the exponent of ** must be a constant integer")
        n = int(e.value) * (-1 if neg else 1)
        if abs(n) > MAX_POW:
            raise self.err(node, f"exponent {n} is too large (|n| <= {MAX_POW})")
        base = self.scalar(node.left)
        if n >= 0:
            return lambda g: base(g) ** n
        return lambda g: _safe_div(torch.ones_like(g[..., 0]), base(g) ** (-n))

    def point(self, node: ast.AST, fname: str) -> Callable[[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(node, ast.Name) or node.id not in POINTS:
            raise self.err(node, f"{fname}() takes points ({', '.join(POINTS)}) as arguments")
        self.names.add(node.id)
        return _POINT_GET[node.id]

    def call(self, node: ast.Call) -> Fn:
        if not isinstance(node.func, ast.Name):
            raise self.err(node, "only plain function calls are allowed")
        fname = node.func.id
        if fname not in FUNCTIONS:
            raise self.err(node, f"unknown function {fname!r}")
        if node.keywords:
            raise self.err(node, f"{fname}() does not take keyword arguments")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise self.err(node, "star arguments are not allowed")
        args = node.args
        nargs = len(args)

        def need(k: int):
            if nargs != k:
                raise self.err(node, f"{fname}() takes {k} argument(s), got {nargs}")

        if fname in _POINT_FN:
            need(2)
            p, q = self.point(args[0], fname), self.point(args[1], fname)

            def rel(g):
                pp, qq = p(g), q(g)
                if pp is None and qq is None:
                    return torch.zeros(g.shape[:-1] + (3,), dtype=g.dtype, device=g.device)
                if pp is None:
                    return qq
                if qq is None:
                    return -pp
                return qq - pp
            if fname == "dist":
                return lambda g: rel(g).norm(dim=-1)
            if fname == "distxy":
                return lambda g: rel(g)[..., :2].norm(dim=-1)
            idx = {"dx": 0, "dy": 1, "dz": 2}[fname]
            return lambda g: rel(g)[..., idx]
        if fname in ("inside", "active"):
            need(1)
            a = args[0]
            if not (isinstance(a, ast.Name) and a.id in POINTS and a.id.startswith("a") and a.id[1:].isdigit()):
                raise self.err(node, "%s() takes an anchor a0..a%d" % (fname, spec.N_ANCHORS - 1))
            self.names.add(a.id)
            col = spec.G_SLICES["anchor_" + fname].start + int(a.id[1:])
            return lambda g: g[..., col]
        if fname == "payload_is":
            need(1)
            a = args[0]
            if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                raise self.err(node, "payload_is() takes one quoted item type: " + ", ".join(ITEM_TYPES))
            if a.value not in ITEM_TYPES:
                raise self.err(node, f"unknown item type {a.value!r}; item types: " + ", ".join(ITEM_TYPES))
            self.names.add("payload_is")
            col = spec.G_SLICES["payload_type"].start + ITEM_TYPES.index(a.value)
            return lambda g: (g[..., col] > 0.5).to(g.dtype)
        if fname in _UNARY_FN:
            need(1)
            x, f = self.scalar(args[0]), _UNARY_FN[fname]
            return lambda g: f(x(g))
        if fname in ("min", "max"):
            if nargs < 2:
                raise self.err(node, f"{fname}() takes 2 or more arguments")
            xs = [self.scalar(a) for a in args]
            f = torch.minimum if fname == "min" else torch.maximum

            def mm(g, xs=xs, f=f):
                acc = xs[0](g)
                for x in xs[1:]:
                    acc = f(acc, x(g))
                return acc
            return mm
        if fname == "clamp":
            need(3)
            x, lo, hi = (self.scalar(a) for a in args)
            return lambda g: torch.minimum(torch.maximum(x(g), lo(g)), hi(g))
        if fname == "where":
            need(3)
            c, a, b = (self.scalar(x) for x in args)
            return lambda g: torch.where(_truthy(c(g)), a(g), b(g))
        raise self.err(node, f"unknown function {fname!r}")  # pragma: no cover


class CompiledReward:
    """Callable reward program. ``fn(g [..., G_DIM]) -> [...]`` float32."""

    def __init__(self, source: str, fn: Fn, names: Set[str]):
        self.source = source
        self._fn = fn
        self.names = frozenset(names)

    @property
    def anchors_used(self) -> List[str]:
        return sorted(n for n in self.names if n.startswith("a") and n[1:].isdigit())

    def __call__(self, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-1] != spec.G_DIM:
            raise ValueError(f"g has trailing dim {g.shape[-1]}, expected G_DIM={spec.G_DIM}")
        if not torch.is_floating_point(g):
            g = g.float()
        out = self._fn(g).to(torch.float32)
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def truthy(self, g: torch.Tensor) -> torch.Tensor:
        """Boolean version (for done_when): value > 0.5."""
        return self(g) > 0.5

    def __repr__(self) -> str:
        return f"CompiledReward({self.source!r})"


def compile_reward(source: str) -> CompiledReward:
    """Parse and compile a reward expression. Raises DSLError on anything not whitelisted."""
    if not isinstance(source, str):
        raise DSLError(f"expression must be a string, got {type(source).__name__}")
    src = source.strip()
    if not src:
        raise DSLError("empty expression")
    if len(src) > MAX_SOURCE_LEN:
        raise DSLError(f"expression too long ({len(src)} > {MAX_SOURCE_LEN} chars)")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error in {src!r}: {e.msg}") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        raise DSLError(f"expression too complex ({n_nodes} nodes > {MAX_NODES})")
    c = _Compiler(src)
    fn = c.scalar(tree.body)
    return CompiledReward(src, fn, c.names)
