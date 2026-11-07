"""Judge demo server tests: FastAPI TestClient, torch backend, CPU, LiDAR on."""
from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from demo import pipeline as P
from demo.server import DemoSession, create_app, default_make_env
from skyhand import spec
from skyhand.env.core import make_env

WEB = Path(__file__).resolve().parents[2] / "demo" / "web"
RENDER_KEYS = {"t", "scene", "drones", "payloads", "obstacles", "zones", "anchors", "waves", "boat", "net",
               "fish", "lidar", "shield_hits", "wind"}


class FakeAgent:
    """Stands in for the trained FB agent: linear B, zero actions."""

    def __init__(self, d=8):
        self.d = d
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(spec.G_DIM, d, generator=g)

    def encode_g(self, g):
        return g.float() @ self.W

    def act(self, obs, z, noise=0.0):
        n, a = obs["proprio"].shape[:2]
        return torch.zeros(n, a, spec.ACTION_DIM)


def make_session(scene="trash", pipeline=None, **kw):
    pipe = pipeline or P.Pipeline(mission_api={})   # keyword parser, scripted fallback
    return DemoSession(scene, num_envs=1, pipeline=pipe, make_env=default_make_env("torch", "cpu", lidar=True),
                       seed=3, **kw)


@pytest.fixture
def client():
    s = make_session()
    with TestClient(create_app(s)) as c:
        c.session = s
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and j["scene"] == "trash"
    assert j["policy_mode"] == P.POLICY_SCRIPTED
    assert j["mission_mode"] == P.MISSION_KEYWORD


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "importmap" in r.text
    assert client.get("/static/main.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_scene_switch(client):
    r = client.post("/api/scene", json={"scene": "ocean"})
    assert r.status_code == 200 and r.json()["scene"] == "ocean"
    assert client.get("/api/state").json()["state"]["scene"] == "ocean"
    assert client.post("/api/scene", json={"scene": "mars"}).status_code == 400
    assert client.post("/api/scene", json={"scene": "trash"}).json()["num_agents"] == spec.TRASH_AGENTS


def test_mission_compile_fallback_parser(client):
    r = client.post("/api/mission", json={"text": "grab the bag and drop it in the bin without getting near the moving carts"})
    assert r.status_code == 200
    m = r.json()
    assert m["source"] == P.MISSION_KEYWORD
    assert m["anchors"]["a0"]["target"] == "payload.handle" and m["anchors"]["a1"]["target"] == "bin"
    assert "dist(payload, a1)" in m["reward"] and "obstacle_dist" in m["reward"]
    assert m["explanation"] and all(isinstance(x, str) for x in m["explanation"])
    # compiling does not start flying
    assert client.get("/api/state").json()["demo"]["flying"] is False
    assert client.post("/api/mission", json={"text": "  "}).status_code == 400


def test_mission_compile_injected_pipeline():
    calls = {}

    def fake_compile(text, scene, entities):
        calls["entities"] = entities
        return SimpleNamespace(anchors={"a0": "bin", "a1": None, "a2": None, "a3": None}, reward="-dist(drone, a0)",
                               done_when="dist(drone, a0) < 1.0", explanation="Fly to the bin.", offline=False)

    pipe = P.Pipeline(mission_api={"compile_mission": fake_compile})
    s = make_session(pipeline=pipe)
    with TestClient(create_app(s)) as c:
        m = c.post("/api/mission", json={"text": "go to the bin"}).json()
        assert m["source"] == P.MISSION_GEMINI
        assert m["explanation"][0] == "Fly to the bin."
        assert m["anchors"]["a0"]["target"] == "bin"
        assert "bin" in calls["entities"]
        assert c.post("/api/approve").status_code == 200
        st = c.get("/api/state").json()
        assert st["demo"]["mission_mode"] == P.MISSION_GEMINI
        assert st["state"]["anchors"][0]["kind"] == "zone"


def _ws_frames(c, n=3, timeout=10.0):
    frames = []
    with c.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        t0 = time.time()
        while len(frames) < n and time.time() - t0 < timeout:
            msg = ws.receive_json()
            if msg["type"] == "frame":
                frames.append(msg)
    return frames


@pytest.mark.parametrize("scene,text", [("trash", "grab the bag and drop it in the bin"),
                                        ("ocean", "net the school closest to the boat")])
def test_approve_and_ws_frames(scene, text):
    s = make_session(scene)
    with TestClient(create_app(s)) as c:
        assert c.post("/api/approve").status_code == 400          # nothing compiled yet
        assert c.post("/api/mission", json={"text": text}).status_code == 200
        r = c.post("/api/approve")
        assert r.status_code == 200 and r.json()["flying"] is True
        frames = _ws_frames(c, 3)
        assert len(frames) == 3
        for f in frames:
            st = f["state"]
            assert RENDER_KEYS <= set(st), RENDER_KEYS - set(st)
            assert st["scene"] == scene
            assert len(st["drones"]) == s.env.num_agents
            for d in st["drones"]:
                assert len(d["pos"]) == 3 and len(d["quat"]) == 4 and {"grip", "holding"} <= set(d)
            assert len(st["anchors"]) == spec.N_ANCHORS
            assert {"nir", "green"} <= set(st["lidar"])
            if scene == "ocean":
                assert set(st["waves"]) == {"amp", "k", "dir", "omega", "phase"}
                assert len(st["net"]["nodes"]) == st["net"]["rows"] * st["net"]["cols"]
                assert len(st["fish"]) > 0 and st["boat"] is not None
            else:
                assert st["waves"] is None and st["payloads"][0]["kind"] in ("bag", "cooler")
            demo = f["demo"]
            for k in ("status", "policy_mode", "mission_mode", "reward", "shield_hits", "mission_text", "step", "zmap"):
                assert k in demo
            assert demo["mission_text"] == text
            assert demo["zmap"]["available"] is False
        assert frames[-1]["demo"]["step"] > 0
        assert frames[-1]["demo"]["reward"] is not None
        assert len(frames[-1]["state"]["lidar"]["nir"]) > 0


def test_perturb_trash(client):
    st0 = client.get("/api/state").json()["state"]
    n_obs = len(st0["obstacles"])
    r = client.post("/api/perturb", json={"kind": "gust", "vec": [5.0, 0.0, 0.0], "duration": 3.0})
    assert r.status_code == 200
    assert float(client.session.env.gust[0, 0]) == pytest.approx(5.0)
    r = client.post("/api/perturb", json={"kind": "drop_obstacle", "pos": [1.0, 2.0]})
    assert r.status_code == 200
    obs = client.get("/api/state").json()["state"]["obstacles"]
    assert len(obs) == n_obs + 1
    assert any(abs(o["pos"][0] - 1.0) < 1e-3 and abs(o["pos"][1] - 2.0) < 1e-3 for o in obs)
    r = client.post("/api/perturb", json={"kind": "move_payload", "pos": [-4.0, 3.0]})
    assert r.status_code == 200
    p = client.get("/api/state").json()["state"]["payloads"][0]["pos"]
    assert abs(p[0] + 4.0) < 0.2 and abs(p[1] - 3.0) < 0.2
    assert client.post("/api/perturb", json={"kind": "waves", "scale": 2.0}).status_code == 200   # no-op in trash
    assert client.post("/api/perturb", json={"kind": "disable_drone", "agent": 0}).status_code == 200
    assert client.get("/api/state").json()["state"]["drones"][0]["alive"] is False
    assert client.post("/api/perturb", json={"kind": "earthquake"}).status_code == 400
    assert client.post("/api/perturb", json={"kind": "drop_obstacle"}).status_code == 400


def test_perturb_ocean():
    s = make_session("ocean")
    with TestClient(create_app(s)) as c:
        st0 = c.get("/api/state").json()["state"]
        amp0 = np.array(st0["waves"]["amp"])
        assert c.post("/api/perturb", json={"kind": "waves", "scale": 2.0}).status_code == 200
        amp1 = np.array(c.get("/api/state").json()["state"]["waves"]["amp"])
        np.testing.assert_allclose(amp1, 2.0 * amp0, rtol=1e-4)
        n_obs = len(st0["obstacles"])
        assert c.post("/api/perturb", json={"kind": "drop_obstacle", "pos": [0.0, 0.0]}).status_code == 200
        assert len(c.get("/api/state").json()["state"]["obstacles"]) == n_obs + 1
        assert c.post("/api/perturb", json={"kind": "disable_drone", "agent": 1}).status_code == 200
        drones = c.get("/api/state").json()["state"]["drones"]
        assert drones[1]["alive"] is False and drones[0]["alive"] is True
        assert c.post("/api/perturb", json={"kind": "disable_drone", "agent": 9}).status_code == 400
        assert c.post("/api/perturb", json={"kind": "gust"}).status_code == 200
        assert float(s.env.gust[0].norm()) > 0
        assert c.post("/api/perturb", json={"kind": "move_payload", "pos": [0, 0]}).json()["note"]


def test_stop_reset(client):
    client.post("/api/mission", json={"text": "fly to the bin"})
    assert client.post("/api/approve").json()["flying"] is True
    r = client.post("/api/stop")
    assert r.status_code == 200 and r.json()["flying"] is False and r.json()["status"] == "stopped"
    r = client.post("/api/reset")
    assert r.status_code == 200 and r.json()["status"] in ("idle", "awaiting approval")
    assert client.get("/api/state").json()["demo"]["step"] == 0
    # can re-approve the same mission after a reset
    assert client.post("/api/approve").json()["flying"] is True


def test_episode_success_reported():
    s = make_session()
    s.compile("fly to the bin")
    s.approve()
    for _ in range(600):
        s.tick()
        if not s.flying:
            break
    assert s.status == "success", s.message


def test_fallback_controller_moves_toward_a0():
    env = make_env("trash", num_envs=1, seed=5, random_anchors=False, lidar=True)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile("grab the bag and drop it in the bin", "trash", env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = env._obs()
    d0 = float(obs["anchors"][0, 0, 0, :3].norm())
    best = d0
    for _ in range(300):
        obs = env.step(act(obs, None)).obs
        best = min(best, float(obs["anchors"][0, 0, 0, :3].norm()))
        if bool(obs["proprio"][0, 0, 10] > 0.5):
            break
    assert best < 0.5 * d0 and best < 1.0, (d0, best)


def test_fallback_controller_ocean_grabs_corners():
    env = make_env("ocean", num_envs=1, seed=0, random_anchors=False, lidar=False)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile("net the school closest to the boat", "ocean", env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = env._obs()
    for _ in range(300):
        out = env.step(act(obs, None))
        obs = out.obs
        assert not bool(out.done[0])
    assert bool((obs["proprio"][0, :, 10] > 0.5).any())


def test_pca_zmap_with_fake_agent():
    pipe = P.Pipeline(agent=FakeAgent(8), mission_api={}, z_samples_envs=4, z_samples_steps=5)
    mk = default_make_env("torch", "cpu", lidar=False)
    zm = pipe.zmap(lambda sc, n: mk(sc, n), torch.randn(8))
    assert zm["available"] is True
    names = [p["name"] for p in zm["points"]]
    assert "Trash pickup" in names and "Organ delivery" in names and "Net fishing" in names
    assert sum(p["current"] for p in zm["points"]) == 1
    for p in zm["points"]:
        assert np.isfinite(p["x"]) and np.isfinite(p["y"]) and abs(p["x"]) <= 1 + 1e-6 and abs(p["y"]) <= 1 + 1e-6
    out = P.pca_2d(np.random.randn(5, 8))
    assert out.shape == (5, 2)


def test_trained_mode_labels_and_z(tmp_path):
    pipe = P.Pipeline(agent=FakeAgent(8), mission_api={}, z_samples_envs=4, z_samples_steps=5)
    s = DemoSession("trash", 1, pipeline=pipe, make_env=default_make_env("torch", "cpu", lidar=False), seed=1)
    s.compile("grab the bag and drop it in the bin")
    info = s.approve()
    assert info["policy_mode"] == P.POLICY_TRAINED
    assert s.z is not None and s.z.shape == (8,)
    assert abs(float(s.z.norm()) - np.sqrt(8)) < 1e-3
    assert s.zmap["available"] and any(p["current"] for p in s.zmap["points"])
    s.tick()
    assert s.extras()["z_norm"] is not None


def test_reward_language_safe():
    from demo.rewardlang import RewardSyntaxError, compile_expr
    g = torch.zeros(3, spec.G_DIM)
    assert compile_expr("-dist(payload, a1) + 3*holding - 5*inside(a2)")(g).shape == (3,)
    for bad in ("__import__('os')", "g.sum()", "[1, 2]", "foo + 1", "dist(bar, a1)"):
        with pytest.raises(RewardSyntaxError):
            compile_expr(bad)(g)


def test_index_pins_cdn_only():
    html = (WEB / "index.html").read_text()
    urls = re.findall(r"https?://[^\s\"'<>]+", html)
    assert urls, "expected the three.js importmap"
    allowed = ("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
               "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/")
    for u in urls:
        assert u in allowed, u
    js = (WEB / "main.js").read_text()
    assert not re.search(r"https?://", js), "main.js must load three only through the importmap"


def test_mainjs_wave_formula_matches_waves_params():
    from skyhand.env.waves import WaveField
    keys = set(WaveField(1).params(0))
    assert keys == {"amp", "k", "dir", "omega", "phase"}
    js = (WEB / "main.js").read_text()
    body = js[js.index("export function waveHeight"):]
    body = body[:body.index("\n}\n")]
    for k in keys:
        assert f"w.{k}[i]" in body, k
    assert re.search(r"w\.amp\[i\]\s*\*\s*Math\.cos\(\s*w\.k\[i\]\s*\*\s*\(\s*Math\.cos\(w\.dir\[i\]\)\s*\*\s*x\s*\+\s*"
                     r"Math\.sin\(w\.dir\[i\]\)\s*\*\s*y\s*\)\s*-\s*w\.omega\[i\]\s*\*\s*t\s*\+\s*w\.phase\[i\]\s*\)", body)


def test_wave_formula_numeric_match():
    """Python port of the JS formula against WaveField.height (with a live scale)."""
    import math
    from skyhand.env.waves import WaveField
    w = WaveField(1)
    w.scale[0] = 1.7
    p = w.params(0)
    xy = torch.tensor([[[1.3, -2.2], [7.0, 4.0]]])
    t = torch.tensor([3.21])
    ref = w.height(xy, t)[0]
    for j, (x, y) in enumerate(xy[0].tolist()):
        h = sum(p["amp"][i] * math.cos(p["k"][i] * (math.cos(p["dir"][i]) * x + math.sin(p["dir"][i]) * y)
                                       - p["omega"][i] * 3.21 + p["phase"][i]) for i in range(len(p["amp"])))
        assert abs(h - float(ref[j])) < 1e-4
