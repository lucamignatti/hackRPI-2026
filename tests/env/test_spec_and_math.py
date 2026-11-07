import ast
import math
import pathlib

import torch

from skyhand import spec
from skyhand.env import math3d as m

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_g_layout_consistent():
    total = sum(sl.stop - sl.start for sl in spec.G_SLICES.values())
    assert total == spec.G_DIM
    assert spec.g_scale().shape == (spec.G_DIM,)
    assert (spec.g_scale() > 0).all()
    g = torch.arange(spec.G_DIM).float()
    assert spec.anchor_rel(g, "anchor_rel_drone", 2).tolist() == \
        g[spec.G_SLICES["anchor_rel_drone"]][6:9].tolist()


def test_empty_obs_passes_check():
    obs = spec.empty_obs(3, 2)
    spec.check_obs(obs, 3, 2)


def test_quat_rotate_roundtrip():
    q = m.quat_normalize(torch.randn(10, 4))
    v = torch.randn(10, 3)
    assert torch.allclose(m.quat_rotate_inv(q, m.quat_rotate(q, v)), v, atol=1e-5)
    assert torch.allclose(m.quat_rotate(q, v).norm(dim=-1), v.norm(dim=-1), atol=1e-5)


def test_yaw_and_tilt():
    q = m.quat_from_yaw(torch.tensor([0.3, -1.2]))
    assert torch.allclose(m.yaw_of(q), torch.tensor([0.3, -1.2]), atol=1e-5)
    assert torch.allclose(m.tilt_of(q), torch.zeros(2), atol=1e-3)
    q2 = m.quat_from_axis_angle(torch.tensor([1.0, 0.0, 0.0]), torch.tensor(0.4))
    assert abs(float(m.tilt_of(q2)) - 0.4) < 1e-4


def test_quat_integrate_constant_rate():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        q = m.quat_integrate(q, torch.tensor([0.0, 0.0, 1.0]), 0.01)
    assert abs(float(m.yaw_of(q)) - 1.0) < 1e-2


def _py_files(sub):
    return [p for p in (ROOT / "skyhand" / sub).rglob("*.py")]


def test_no_genesis_import_outside_backend():
    """Training and inference must import where Genesis is not installed."""
    for p in (ROOT / "skyhand").rglob("*.py"):
        if p.name == "genesis_backend.py":
            continue
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "genesis" for n in names), f"{p} imports genesis"


def test_py39_compatible_env_and_spec():
    files = _py_files("env") + [ROOT / "skyhand" / "spec.py"]
    for p in files:
        src = p.read_text()
        tree = ast.parse(src)
        assert not any(type(n).__name__ == "Match" for n in ast.walk(tree)), f"match statement in {p}"
        if src.strip() and any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body):
            assert "from __future__ import annotations" in src, f"{p} lacks future annotations"
