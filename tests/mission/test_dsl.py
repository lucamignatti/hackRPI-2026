from __future__ import annotations

import itertools

import pytest
import torch

from skyhand import spec
from skyhand.mission.dsl import POINTS, SCALARS, DSLError, compile_reward, describe_vocabulary

S = spec.G_SLICES


def make_g(m=5, seed=0):
    torch.manual_seed(seed)
    return torch.randn(m, spec.G_DIM) * 3


def point_vec(g, name):
    if name == "drone":
        return torch.zeros(g.shape[0], 3)
    if name == "payload":
        return g[:, S["payload_rel"]]
    if name == "net":
        return g[:, S["net_centroid_rel"]]
    k = int(name[1:])
    return g[:, S["anchor_rel_drone"]][:, 3 * k:3 * k + 3]


FIELD_SCALARS = ["speed", "tilt", "height", "thrust", "grip_closed", "holding", "held_mass", "swing", "jerk",
                 "obstacle_dist", "obstacle_closing", "net_stretch", "fish_in_net", "mate_dist", "collision", "shield"]


@pytest.mark.parametrize("name", FIELD_SCALARS)
def test_scalars(name):
    g = make_g()
    out = compile_reward(name)(g)
    assert out.shape == (5,) and out.dtype == torch.float32
    assert torch.allclose(out, g[:, S[name]].squeeze(-1))


def test_scalar_values_placed_by_slice():
    g = torch.zeros(2, spec.G_DIM)
    g[:, S["holding"]] = 1.0
    g[0, S["swing"]] = 0.7
    g[:, S["drone_vel"]] = torch.tensor([1.0, -2.0, 3.0])
    assert compile_reward("holding")(g).tolist() == [1.0, 1.0]
    assert torch.allclose(compile_reward("swing")(g), torch.tensor([0.7, 0.0]))
    assert compile_reward("vz")(g).tolist() == [3.0, 3.0]
    assert compile_reward("vx + vy")(g).tolist() == [-1.0, -1.0]


def test_all_point_pairs():
    g = make_g(7)
    names = list(POINTS)
    for p, q in itertools.product(names, names):
        rel = point_vec(g, q) - point_vec(g, p)
        assert torch.allclose(compile_reward(f"dist({p}, {q})")(g), rel.norm(dim=-1), atol=1e-5), (p, q)
        assert torch.allclose(compile_reward(f"distxy({p}, {q})")(g), rel[:, :2].norm(dim=-1), atol=1e-5)
        assert torch.allclose(compile_reward(f"dz({p}, {q})")(g), rel[:, 2], atol=1e-5)
        assert torch.allclose(compile_reward(f"dx({p}, {q})")(g), rel[:, 0], atol=1e-5)


def test_specific_pairs_hand_built():
    g = torch.zeros(1, spec.G_DIM)
    ard = S["anchor_rel_drone"].start
    g[0, ard + 3:ard + 6] = torch.tensor([1.0, 0.0, 0.0])    # a1
    g[0, ard + 6:ard + 9] = torch.tensor([4.0, 4.0, 2.0])    # a2
    g[0, S["payload_rel"]] = torch.tensor([0.0, 0.0, -1.0])
    assert compile_reward("dist(a1, a2)")(g).item() == pytest.approx(29 ** 0.5)
    assert compile_reward("distxy(a1, a2)")(g).item() == pytest.approx(5.0)
    assert compile_reward("dz(a1, a2)")(g).item() == pytest.approx(2.0)
    assert compile_reward("dz(payload, drone)")(g).item() == pytest.approx(1.0)
    assert compile_reward("dist(drone, a1)")(g).item() == pytest.approx(1.0)
    assert compile_reward("dist(payload, a1)")(g).item() == pytest.approx(2 ** 0.5)


def test_inside():
    g = torch.zeros(3, spec.G_DIM)
    g[1, S["anchor_inside"].start + 2] = 1.0
    assert compile_reward("inside(a2)")(g).tolist() == [0.0, 1.0, 0.0]
    assert compile_reward("inside(a0)")(g).tolist() == [0.0, 0.0, 0.0]


def test_operators_logic_functions():
    g = torch.zeros(4, spec.G_DIM)
    g[:, S["speed"]] = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    g[:, S["holding"]] = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    f = lambda e: compile_reward(e)(g).tolist()
    assert f("speed > 1.5") == [0, 0, 1, 1]
    assert f("speed >= 1") == [0, 1, 1, 1]
    assert f("speed == 2") == [0, 0, 1, 0]
    assert f("0 < speed < 3") == [0, 1, 1, 0]
    assert f("holding and speed > 1.5") == [0, 0, 0, 1]
    assert f("holding or speed > 1.5") == [0, 1, 1, 1]
    assert f("not holding") == [1, 0, 1, 0]
    assert f("where(holding, speed, -1)") == [-1, 1, -1, 3]
    assert f("min(speed, 1.5, 2)") == [0, 1, 1.5, 1.5]
    assert f("max(speed, 1.5)") == [1.5, 1.5, 2, 3]
    assert f("clamp(speed, 0.5, 2.5)") == [0.5, 1, 2, 2.5]
    assert f("speed ** 2") == [0, 1, 4, 9]
    assert f("-speed + 2*3 - 1/2") == [5.5, 4.5, 3.5, 2.5]
    assert f("abs(1 - speed)") == [1, 0, 1, 2]
    out = compile_reward("exp(-speed) + sqrt(speed) + log1p(speed) + tanh(speed)")(g)
    s = g[:, S["speed"]].squeeze(-1)
    assert torch.allclose(out, torch.exp(-s) + s.sqrt() + torch.log1p(s) + torch.tanh(s))
    assert torch.isfinite(compile_reward("1 / speed")(g)).all()
    assert f("3") == [3, 3, 3, 3]


def test_batched_shapes():
    r = compile_reward("-dist(payload, a1) + 3*holding")
    assert r(torch.zeros(6, 2, spec.G_DIM)).shape == (6, 2)
    assert r(torch.zeros(spec.G_DIM)).shape == ()
    assert r.anchors_used == ["a1"]
    with pytest.raises(ValueError):
        r(torch.zeros(3, 10))


@pytest.mark.parametrize("bad", [
    "g.__class__", "speed.real", "__import__('os')", "g[0]", "speed[0]", "(lambda x: x)(1)",
    "[x for x in range(3)]", "'hello'", "foo + 1", "speed ** holding", "speed ** 100", "speed ** 0.5",
    "dist(drone)", "dist(speed, a1)", "inside(drone)", "payload + 1", "exp", "min(speed)", "open('f')",
    "speed if", "", "abs(x=1)", "eval('1')", "f'{speed}'", "1e300", "speed @ speed", "speed // 2",
])
def test_rejections(bad):
    with pytest.raises(DSLError):
        compile_reward(bad)


def test_error_lists_allowed_names():
    with pytest.raises(DSLError) as e:
        compile_reward("velocity")
    assert "unknown name 'velocity'" in str(e.value) and "obstacle_dist" in str(e.value)


def test_vocabulary_doc():
    doc = describe_vocabulary()
    for n in list(SCALARS) + list(POINTS) + ["dist", "inside", "where"]:
        assert n in doc
