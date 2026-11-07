"""Fallback reward-program evaluator and plain-language explainer.

Used only when skyhand.mission is not importable. Same surface language as
PLAN.md section 6.2: g field names, dist(), inside(), abs/min/max/clamp/exp,
comparisons, and/or/not, arithmetic. Parsed with `ast` against a whitelist and
evaluated as batched torch ops over g [..., G_DIM]. No eval().
"""
from __future__ import annotations

import ast
from typing import Callable, Dict, List, Optional

import torch

from skyhand import spec

ANCHOR_NAMES = tuple(f"a{k}" for k in range(spec.N_ANCHORS))
SCALAR_FIELDS = tuple(n for n, s in spec.G_FIELD_SIZES.items() if s == 1)
BODIES = ("drone", "payload")
FUNCS = ("dist", "distxy", "inside", "abs", "min", "max", "clamp", "exp")


class RewardSyntaxError(ValueError):
    pass


def _check(node: ast.AST) -> None:
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name,
          ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
          ast.Not, ast.And, ast.Or, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.IfExp)
    for n in ast.walk(node):
        if not isinstance(n, ok):
            raise RewardSyntaxError(f"not allowed: {type(n).__name__}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise RewardSyntaxError("only numeric constants")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in FUNCS or n.keywords:
                raise RewardSyntaxError("unknown function")
        if isinstance(n, ast.Name) and n.id not in FUNCS + SCALAR_FIELDS + ANCHOR_NAMES + BODIES:
            raise RewardSyntaxError(f"unknown name {n.id!r}")


def parse(expr: str) -> ast.Expression:
    expr = expr.replace("&&", " and ").replace("||", " or ")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise RewardSyntaxError(str(e)) from e
    _check(tree)
    return tree


def _anchor_idx(node) -> int:
    if not (isinstance(node, ast.Name) and node.id in ANCHOR_NAMES):
        raise RewardSyntaxError("expected an anchor a0..a3")
    return int(node.id[1:])


def _rel(g, body: str, k: int):
    return spec.anchor_rel(g, "anchor_rel_drone" if body == "drone" else "anchor_rel_payload", k)


def _eval(node, g):
    f = lambda x: x if torch.is_tensor(x) else torch.full(g.shape[:-1], float(x), device=g.device)
    if isinstance(node, ast.Expression):
        return _eval(node.body, g)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in SCALAR_FIELDS:
            return spec.g_field(g, node.id)[..., 0]
        raise RewardSyntaxError(f"{node.id} can only be used inside dist()/inside()")
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, g)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        return 1.0 - (f(v) > 0.5).float()
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, g), _eval(node.right, g)
        op = type(node.op)
        if op is ast.Add:
            return a + b
        if op is ast.Sub:
            return a - b
        if op is ast.Mult:
            return a * b
        if op is ast.Div:
            if torch.is_tensor(b):
                return a / torch.where(b.abs() < 1e-6, torch.full_like(b, 1e-6), b)
            return a / b if b != 0 else a * 0.0
        if op is ast.Pow:
            return f(a) ** b
    if isinstance(node, ast.BoolOp):
        vals = [(f(_eval(v, g)) > 0.5) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out.float()
    if isinstance(node, ast.Compare):
        left = f(_eval(node.left, g))
        out = torch.ones_like(left, dtype=torch.bool)
        for op, rnode in zip(node.ops, node.comparators):
            right = f(_eval(rnode, g))
            out &= {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right,
                    ast.GtE: left >= right, ast.Eq: left == right, ast.NotEq: left != right}[type(op)]
            left = right
        return out.float()
    if isinstance(node, ast.IfExp):
        c = f(_eval(node.test, g)) > 0.5
        return torch.where(c, f(_eval(node.body, g)), f(_eval(node.orelse, g)))
    if isinstance(node, ast.Call):
        name, args = node.func.id, node.args
        if name in ("dist", "distxy"):
            if len(args) != 2:
                raise RewardSyntaxError(f"{name} takes 2 arguments")
            x, y = args
            dims = 2 if name == "distxy" else 3                  # distxy: horizontal distance
            names = [getattr(x, "id", None), getattr(y, "id", None)]
            if set(names) == {"drone", "payload"}:
                return spec.g_field(g, "payload_rel")[..., :dims].norm(dim=-1)
            if names[0] in BODIES and names[1] in ANCHOR_NAMES:
                return _rel(g, names[0], _anchor_idx(y))[..., :dims].norm(dim=-1)
            if names[1] in BODIES and names[0] in ANCHOR_NAMES:
                return _rel(g, names[1], _anchor_idx(x))[..., :dims].norm(dim=-1)
            if names[0] in ANCHOR_NAMES and names[1] in ANCHOR_NAMES:
                return (_rel(g, "drone", _anchor_idx(x)) - _rel(g, "drone", _anchor_idx(y)))[..., :dims].norm(dim=-1)
            raise RewardSyntaxError(f"{name}(drone|payload|aK, aK)")
        if name == "inside":
            if len(args) != 1:
                raise RewardSyntaxError("inside takes 1 argument")
            return spec.g_field(g, "anchor_inside")[..., _anchor_idx(args[0])]
        vals = [f(_eval(a, g)) for a in args]
        if name == "abs" and len(vals) == 1:
            return vals[0].abs()
        if name == "exp" and len(vals) == 1:
            return vals[0].clamp(max=30.0).exp()
        if name == "min" and len(vals) >= 2:
            out = vals[0]
            for v in vals[1:]:
                out = torch.minimum(out, v)
            return out
        if name == "max" and len(vals) >= 2:
            out = vals[0]
            for v in vals[1:]:
                out = torch.maximum(out, v)
            return out
        if name == "clamp" and len(vals) == 3:
            return torch.maximum(torch.minimum(vals[0], vals[2]), vals[1])
        raise RewardSyntaxError(f"bad call to {name}")
    raise RewardSyntaxError(f"unsupported {type(node).__name__}")


def compile_expr(expr: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """expr -> fn(g [..., G_DIM]) -> r [...] (float tensor)."""
    tree = parse(expr)

    def fn(g: torch.Tensor) -> torch.Tensor:
        v = _eval(tree, g)
        if not torch.is_tensor(v):
            v = torch.full(g.shape[:-1], float(v), device=g.device)
        return v.float()

    fn.expr = expr  # type: ignore[attr-defined]
    return fn


# ------------------------------------------------------------------ plain language
_FIELD_WORDS = {
    "holding": "holding the payload", "swing": "payload swing", "jerk": "jerky motion", "tilt": "tilting",
    "thrust": "thrust use", "speed": "speed", "height": "height", "collision": "collisions",
    "shield": "safety-shield interventions", "net_stretch": "net stretch", "fish_in_net": "fish inside the net",
    "obstacle_dist": "distance to obstacles", "mate_dist": "distance to teammates", "grip_closed": "closed gripper",
    "payload_is_trash": "carrying trash", "trash_left": "trash left on the ground", "bycatch": "animals tangled in the net",
    "animal_dist": "distance from the net to marine animals", "over_water": "flying over water",
    "held_mass": "carried mass",
}


def _src(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return "?"


def _terms(node, sign=1.0, out=None):
    out = [] if out is None else out
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        _terms(node.left, sign, out)
        _terms(node.right, sign * (1 if isinstance(node.op, ast.Add) else -1), out)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        _terms(node.operand, -sign, out)
    else:
        out.append((sign, node))
    return out


def _describe(node, anchors: Dict[str, str]) -> str:
    an = lambda n: f"{n.id} ({anchors.get(n.id, 'unbound')})" if isinstance(n, ast.Name) and n.id in ANCHOR_NAMES else _src(n)
    if isinstance(node, ast.Call) and node.func.id in ("dist", "distxy") and len(node.args) == 2:
        what = "horizontal distance" if node.func.id == "distxy" else "distance"
        return f"{what} from the {_src(node.args[0])} to {an(node.args[1])}"
    if isinstance(node, ast.Call) and node.func.id == "inside" and node.args:
        return f"being inside {an(node.args[0])}"
    if isinstance(node, ast.Name):
        return _FIELD_WORDS.get(node.id, node.id.replace("_", " "))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        word = {ast.Lt: "below", ast.LtE: "at most", ast.Gt: "above", ast.GtE: "at least"}.get(type(node.ops[0]), "equal to")
        return f"{_describe(node.left, anchors)} {word} {_src(node.comparators[0])}"
    return _src(node)


def explain(reward: str, done_when: Optional[str], anchors: Dict[str, str], strict: bool = False) -> List[str]:
    """Bullet list describing a reward program in plain language.

    strict=True returns [] when the expression uses vocabulary this module does not know
    (the mission package's DSL is a superset); the caller then shows its own explanation."""
    lines: List[str] = []
    try:
        tree = parse(reward).body
        if done_when and strict:
            parse(done_when)
    except RewardSyntaxError as e:
        return [] if strict else [f"(could not parse reward: {e})"]
    for sign, node in _terms(tree):
        w = 1.0
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            neg = lambda n: isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant)
            if neg(node.left):
                w, node = -float(node.left.operand.value), node.right
            elif isinstance(node.left, ast.Constant):
                w, node = float(node.left.value), node.right
            elif isinstance(node.right, ast.Constant):
                w, node = float(node.right.value), node.left
        s = sign * w
        what = _describe(node, anchors)
        if isinstance(node, ast.Call) and node.func.id in ("dist", "distxy"):
            lines.append(("Minimise " if s < 0 else "Maximise ") + what + f" (weight {abs(s):g})")
        else:
            lines.append(("Reward " if s > 0 else "Penalise ") + what + f" (weight {abs(s):g})")
    if done_when:
        try:
            dt = parse(done_when).body
            parts = dt.values if isinstance(dt, ast.BoolOp) else [dt]
            joiner = " and " if not isinstance(dt, ast.BoolOp) or isinstance(dt.op, ast.And) else " or "
            lines.append("Mission complete when " + joiner.join(_describe(p, anchors) if not isinstance(p, ast.UnaryOp)
                                                              else "not " + _describe(p.operand, anchors) for p in parts))
        except RewardSyntaxError as e:
            lines.append(f"(could not parse done condition: {e})")
    return lines
