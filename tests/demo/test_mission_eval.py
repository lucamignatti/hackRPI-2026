"""Mission success eval (demo.mission_eval): scoring rules, per-world compile, CLI."""
from __future__ import annotations

import json

import pytest
import torch

from demo import mission_eval as E
from skyhand import spec
from skyhand.env.core import make_env


class ZeroAgent:
    """Linear B, zero actions: never completes anything, never crashes."""
    device = "cpu"

    def __init__(self, d=8):
        self.d = d
        self.W = torch.randn(spec.G_DIM, d, generator=torch.Generator().manual_seed(0))
        self.seen_z = None

    def encode_g(self, g):
        return g.float() @ self.W

    def act(self, obs, z, noise=0.0):
        self.seen_z = z
        n, a = obs["proprio"].shape[:2]
        out = torch.zeros(n, a, spec.ACTION_DIM)
        out[..., 4] = -1.0
        return out


def mk(sc, n, **kw):
    kw.setdefault("lidar", False)
    kw.setdefault("random_anchors", False)
    return make_env(sc, num_envs=n, device="cpu", **kw)


def test_fractions_add_up_and_policy_gets_one_z_per_world_and_drone():
    ag = ZeroAgent()
    r = E.run_mission("world", E.MISSIONS["world"][0], ag, envs=2, max_steps=5, make_env=mk,
                      z_samples_envs=2, z_samples_steps=3)
    assert r.controller == "policy" and r.worlds == 2
    assert abs(r.success + r.failure + r.timeout - 1.0) < 1e-6
    assert r.timeout == 1.0                                              # 5 steps finish nothing
    assert ag.seen_z.shape == (2, spec.SCENE_AGENTS["world"], ag.d)      # per-drone jobs -> [N, A, d]


def test_each_world_compiles_against_its_own_layout():
    env = mk("trash", 4, seed=0)
    descs = [E._WorldView(env, i).entity_descriptions() for i in range(4)]
    assert descs[1] == env.entity_descriptions(1) and descs[0] == env.entity_descriptions(0)


@pytest.mark.slow
def test_scripted_controller_flies_to_the_bin():
    lidar_mk = lambda sc, n, **kw: mk(sc, n, **{"lidar": True, **kw})   # noqa: E731  it climbs over what LiDAR sees
    r = E.run_mission("trash", "fly to the bin", None, envs=2, make_env=lidar_mk, seed=0)
    assert r.controller == "scripted" and r.success == 1.0 and r.mean_success_s is not None


def test_done_rule_matches_the_server():
    class M:   # per-drone jobs: every drone with a done_when must meet it once (latched)
        agent_done_fns = {0: lambda g: g[..., 0], 1: lambda g: g[..., 0], 2: None}
    g = torch.zeros(3, spec.G_DIM)
    latched = set()
    g[0, 0] = 1.0
    assert not E._done_now(M, g, latched) and latched == {0}
    g[0, 0], g[1, 0] = 0.0, 1.0
    assert E._done_now(M, g, latched)                                    # drone 0 stays done once met


def test_cli_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "MISSIONS", {"trash": ["fly to the bin"]})
    out = tmp_path / "eval.json"
    res = E.main(["--baseline", "scripted", "--scenes", "trash", "--envs", "1", "--max-steps", "3",
                  "--device", "cpu", "--z-samples-envs", "1", "--z-samples-steps", "2", "--out", str(out)])
    rows = json.loads(out.read_text())
    assert len(res) == len(rows) == 1 and rows[0]["controller"] == "scripted" and rows[0]["worlds"] == 1


def test_warns_when_the_mission_compiler_is_missing():
    assert E.check_api(E.offline_api()) is None
    assert "keyword parser" in E.check_api({})
