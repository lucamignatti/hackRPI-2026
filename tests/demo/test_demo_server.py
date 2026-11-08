"""Judge demo server tests.

Most tests run against demo.fake_env.FakeEnv, a kinematic env that follows the
SkyEnvAPI surface and the render-state schema in skyhand/spec.py. The test_real_env_*
tests run the same flows on skyhand.env.core.make_env for all three scenes.
"""
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
from demo.fake_env import FakeEnv
from demo.server import DemoSession, create_app, default_make_env, gusts_enabled
from skyhand import spec

WEB = Path(__file__).resolve().parents[2] / "demo" / "web"
RENDER_KEYS = {"t", "scene", "variant", "arena_half", "terrain", "water", "drones", "items", "boxes", "spheres",
               "zones", "anchors", "boat", "net", "fish", "animals", "lidar", "shield_hits", "wind"}
BOX_KINDS = {"building", "car", "truck", "bench", "hydrant", "trunk", "boat", "dropped", "bin"}
SPHERE_KINDS = {"rock", "canopy"}
ZONE_KINDS = {"bin", "truck", "pad_pickup", "pad_dropoff", "nofly", "boat"}
EXAMPLES = {"trash": "pick up the trash bag and put it in the bin without hitting the cars",
            "delivery": "deliver the organ cooler to the drop-off pad smoothly and stay out of the no-fly zones",
            "ocean": "net the school closest to the boat without catching any whales",
            "world": "collect the trash and put it in the garbage truck without hitting the cars"}
SPLIT = ("drone 0 pick up the pizza box and drop it in the garbage truck, drone 1 deliver the organ cooler "
         "to the clinic, drone 2 fly to the boat")


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


def fake_mk(**kw):
    return default_make_env("torch", "cpu", lidar=True, fake=True, **kw)


def make_session(scene="trash", pipeline=None, make_env=None, seed=2, **kw):
    pipe = pipeline or P.Pipeline(mission_api={})   # keyword parser, scripted fallback
    return DemoSession(scene, num_envs=1, pipeline=pipe, make_env=make_env or fake_mk(), seed=seed, **kw)


@pytest.fixture
def client():
    s = make_session()
    with TestClient(create_app(s)) as c:
        c.session = s
        yield c


def _ws_frames(c, n=3, timeout=10.0):
    """(first non-frame messages, frames)."""
    frames, other = [], []
    with c.websocket_connect("/ws") as ws:
        t0 = time.time()
        while len(frames) < n and time.time() - t0 < timeout:
            msg = ws.receive_json()
            (frames if msg["type"] == "frame" else other).append(msg)
    return other, frames


def check_render_state(st, scene, num_agents):
    """The render-state schema of skyhand/spec.py, as the viewer consumes it."""
    assert RENDER_KEYS <= set(st), RENDER_KEYS - set(st)
    assert st["scene"] == scene and st["arena_half"] == pytest.approx(spec.SCENE_ARENA_HALF[scene])
    if "max_speed" in st:
        assert st["max_speed"] == pytest.approx(spec.SCENE_MAX_SPEED[scene])
    assert st["variant"] in ("city", "terrain", "ocean", scene)   # spec: city|terrain|ocean; core.py sends "delivery"
    assert len(st["drones"]) == num_agents
    for d in st["drones"]:
        assert len(d["pos"]) == 3 and len(d["quat"]) == 4 and {"grip", "holding", "alive"} <= set(d)
    assert len(st["anchors"]) == spec.N_ANCHORS
    for a in st["anchors"]:
        assert len(a["pos"]) == 3 and a["kind"] in spec.ANCHOR_TYPES and "radius" in a
    for it in st["items"]:
        assert re.fullmatch(r"item_\d+", it["name"]) and it["type"] in spec.ITEM_TYPES
        assert it["shape"] in ("box", "sphere") and len(it["pos"]) == 3 and len(it["size"]) == 3
        assert {"is_trash", "held", "delivered"} <= set(it)
    for b in st["boxes"]:
        assert b["kind"] in BOX_KINDS and len(b["pos"]) == 3 and len(b["size"]) == 3 and "moving" in b
    for s in st["spheres"]:
        assert s["kind"] in SPHERE_KINDS and len(s["pos"]) == 3 and "radius" in s
    for z in st["zones"]:
        assert z["kind"] in ZONE_KINDS and {"name", "pos", "radius", "height"} <= set(z)
    for a in st["animals"]:
        assert a["type"] in spec.ANIMAL_TYPES and {"name", "pos", "vel", "radius", "tangled"} <= set(a)
    assert {"nir", "green"} <= set(st["lidar"])
    ter, water = st["terrain"], st["water"]
    if ter is not None:
        assert {"version", "res", "half"} <= set(ter)
    if scene == "ocean":
        assert ter is None
        assert water["level"] == 0.0 and set(water["waves"]) == {"amp", "k", "dir", "omega", "phase"}
        assert len(st["net"]["nodes"]) == st["net"]["rows"] * st["net"]["cols"]
        assert len(st["fish"]) > 0 and st["boat"] is not None and len(st["animals"]) == spec.N_ANIMALS
    elif scene == "world":
        # everything at once: terrain (128 x 128), tide level AND waves, town, clinics, sea life
        assert st["variant"] == "world"
        assert ter is not None and ter["res"] == spec.SCENE_TERRAIN_RES["world"] and ter["half"] == pytest.approx(200.0)
        assert isinstance(water["level"], float) and set(water["waves"]) == {"amp", "k", "dir", "omega", "phase"}
        assert len(water["waves"]["amp"]) == len(water["waves"]["k"]) > 0
        assert len(st["net"]["nodes"]) == st["net"]["rows"] * st["net"]["cols"]
        assert st["boat"] is not None and len(st["fish"]) > 0 and st["animals"] and st["items"]
        zk = {z["kind"] for z in st["zones"]}
        assert {"pad_pickup", "pad_dropoff"} <= zk and zk & {"truck", "bin"}     # no-fly zones vary per world
        kinds = {b["kind"] for b in st["boxes"]}
        assert "building" in kinds
    else:
        assert water is None or water["waves"] is None
        assert st["items"], "terrain scenes have items"
    if scene == "delivery":
        assert ter is not None and water is not None
        assert {"pad_pickup", "pad_dropoff"} <= {z["kind"] for z in st["zones"]}
        assert st["items"][0]["name"] == "item_0" and st["items"][0]["type"] == "cooler"


# ------------------------------------------------------------------ basics
def test_health(client):
    j = client.get("/api/health").json()
    assert j["ok"] and j["scene"] == "trash"
    assert j["policy_mode"] == P.POLICY_SCRIPTED
    assert j["mission_mode"] == P.MISSION_KEYWORD


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "importmap" in r.text
    assert 'id="btn-voice"' in r.text and 'id="voice-status"' in r.text
    for sc in spec.SCENES:
        assert f'data-scene="{sc}"' in r.text
    js = client.get("/static/main.js")
    assert js.status_code == 200 and "MediaRecorder" in js.text and 'fetch("/api/transcribe"' in js.text
    assert client.get("/static/style.css").status_code == 200


def test_voice_transcription_requires_server_key(client, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    r = client.post("/api/transcribe", content=b"recording", headers={"content-type": "audio/webm"})
    assert r.status_code == 503 and "ELEVENLABS_API_KEY" in r.json()["detail"]


def test_voice_transcription_returns_reviewable_text(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    async def fake_transcribe(audio, content_type):
        assert audio == b"recording" and content_type == "audio/webm"
        return {"text": "fly to the clinic", "language": "en", "provider": "ElevenLabs", "model": "scribe_v2"}

    monkeypatch.setattr("demo.server.transcribe_elevenlabs", fake_transcribe)
    r = client.post("/api/transcribe", content=b"recording", headers={"content-type": "audio/webm"})
    assert r.status_code == 200
    assert r.json() == {"text": "fly to the clinic", "language": "en", "provider": "ElevenLabs", "model": "scribe_v2"}


def test_voice_transcription_rejects_non_audio(client, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    r = client.post("/api/transcribe", content=b"not audio", headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_scene_switch(client):
    for sc in spec.SCENES:
        r = client.post("/api/scene", json={"scene": sc})
        assert r.status_code == 200 and r.json()["scene"] == sc
        assert r.json()["num_agents"] == spec.SCENE_AGENTS[sc]
        assert client.get("/api/state").json()["state"]["scene"] == sc
    assert client.post("/api/scene", json={"scene": "mars"}).status_code == 400


def test_info_reports_sim_and_gusts(client):
    info = client.get("/api/info").json()
    assert info["sim"] == "kinematic preview" and info["gusts"] is False
    assert info["scenes"] == list(spec.SCENES)


# ------------------------------------------------------------------ keyword parser
def _item_of(env, anchor):
    name = anchor.split(".")[0]
    return P.item_types(env)[name]


def test_keyword_parser_trash_picks_item_by_type():
    for seed in (2, 3):                      # city and terrain worlds
        env = FakeEnv("trash", seed=seed)
        m = P.keyword_compile("pick up the trash bag and put it in the bin without hitting the cars", "trash", env)
        assert m.source == P.MISSION_KEYWORD
        assert m.anchors["a0"].endswith(".handle") and _item_of(env, m.anchors["a0"]) == "trash_bag"
        assert m.anchors["a1"] == "bin"
        assert "dist(payload, a1)" in m.reward and "obstacle_dist" in m.reward
        assert "payload_is_trash" in m.reward             # "trash": penalise holding non-trash
        assert m.bindings[0].kind == "handle" and m.bindings[1].kind == "zone"
        m = P.keyword_compile("grab the pizza box and drop it in the garbage truck", "trash", env)
        assert _item_of(env, m.anchors["a0"]) == "pizza_box"
        m = P.keyword_compile("move the backpack to the bin", "trash", env)
        assert _item_of(env, m.anchors["a0"]) == "backpack" or "backpack" not in P.item_types(env).values()
        m = P.keyword_compile("clean up the litter", "trash", env)
        t = _item_of(env, m.anchors["a0"])
        assert spec.ITEM_IS_TRASH[spec.ITEM_TYPES.index(t)]
        m = P.keyword_compile("fly to the bin and hover", "trash", env)
        assert m.anchors["a0"] == "bin" and m.anchors["a1"] is None and m.done_when


def test_keyword_parser_delivery():
    env = FakeEnv("delivery", seed=1)
    m = P.keyword_compile(EXAMPLES["delivery"], "delivery", env)
    assert m.anchors["a0"] == "item_0.handle" and m.anchors["a1"] == "pad_dropoff"
    assert {m.anchors["a2"], m.anchors["a3"]} == {"nofly_0", "nofly_1"}
    assert "swing" in m.reward and "inside(a2)" in m.reward and "inside(a3)" in m.reward
    m = P.keyword_compile("fly to the pickup pad", "delivery", env)
    assert m.anchors["a0"] == "pad_pickup"
    m = P.keyword_compile("fly to the drop-off pad", "delivery", env)
    assert m.anchors["a0"] == "pad_dropoff"


def test_keyword_parser_ocean():
    env = FakeEnv("ocean", seed=0)
    m = P.keyword_compile(EXAMPLES["ocean"], "ocean", env)
    assert m.anchors["a0"] == "net.corner_0" and m.anchors["a1"] == "boat"
    assert m.anchors["a2"].startswith("fish_school_")
    assert "bycatch" in m.reward and "animal_dist" in m.reward
    m = P.keyword_compile("fly to the whale", "ocean", env)
    assert m.anchors["a0"] == "whale_0"
    m = P.keyword_compile("fly to the boat", "ocean", env)
    assert m.anchors["a0"] == "boat"


def test_entity_kind_inference():
    assert P.infer_kind("item_3.handle") == "handle" and P.infer_kind("net.corner_1") == "handle"
    for z in ("bin", "pad_pickup", "pad_dropoff", "nofly_1", "boat"):
        assert P.infer_kind(z) == "zone"
    assert P.infer_kind("whale_0") == "point" and P.infer_kind("item_2") == "point"


def test_mission_compile_endpoint(client):
    r = client.post("/api/mission", json={"text": EXAMPLES["trash"]})
    assert r.status_code == 200
    m = r.json()
    assert m["source"] == P.MISSION_KEYWORD
    assert re.fullmatch(r"item_\d+\.handle", m["anchors"]["a0"]["target"]) and m["anchors"]["a1"]["target"] == "bin"
    assert "trash bag" in m["anchors"]["a0"]["description"]
    assert m["explanation"] and all(isinstance(x, str) for x in m["explanation"])
    assert client.get("/api/state").json()["demo"]["flying"] is False
    assert client.post("/api/mission", json={"text": "  "}).status_code == 400


def test_mission_compile_injected_pipeline():
    calls = {}

    def fake_compile(text, scene, entities):
        calls["entities"] = entities
        return SimpleNamespace(anchors={"a0": "pad_dropoff", "a1": None, "a2": None, "a3": None},
                               reward="-dist(drone, a0)", done_when="dist(drone, a0) < 1.0",
                               explanation="Fly to the drop-off pad.", offline=False)

    pipe = P.Pipeline(mission_api={"compile_mission": fake_compile})
    s = make_session("delivery", pipeline=pipe, seed=1)
    with TestClient(create_app(s)) as c:
        m = c.post("/api/mission", json={"text": "go to the drop-off pad"}).json()
        assert m["source"] == P.MISSION_GEMINI
        assert m["explanation"][0] == "Fly to the drop-off pad."
        assert m["anchors"]["a0"]["target"] == "pad_dropoff"
        assert {"pad_pickup", "pad_dropoff", "item_0.handle"} <= set(calls["entities"])
        assert c.post("/api/approve").status_code == 200
        st = c.get("/api/state").json()
        assert st["demo"]["mission_mode"] == P.MISSION_GEMINI
        assert st["state"]["anchors"][0]["kind"] == "zone"


# ------------------------------------------------------------------ frames
@pytest.mark.parametrize("scene", spec.SCENES)
def test_approve_and_ws_frames(scene):
    s = make_session(scene, seed=1 if scene == "delivery" else 2)
    text = EXAMPLES[scene]
    with TestClient(create_app(s)) as c:
        assert c.post("/api/approve").status_code == 400          # nothing compiled yet
        assert c.post("/api/mission", json={"text": text}).status_code == 200
        r = c.post("/api/approve")
        assert r.status_code == 200 and r.json()["flying"] is True
        other, frames = _ws_frames(c, 3)
        assert other[0]["type"] == "hello" and other[0]["scene"] == scene and "gusts" in other[0]
        assert other[1]["type"] == "terrain"
        assert len(frames) == 3
        for f in frames:
            check_render_state(f["state"], scene, s.env.num_agents)
            demo = f["demo"]
            for k in ("status", "policy_mode", "mission_mode", "reward", "shield_hits", "mission_text", "step",
                      "zmap", "gusts", "sim"):
                assert k in demo
            assert demo["mission_text"] == text
            assert demo["zmap"]["available"] is False
        assert frames[-1]["demo"]["step"] > 0
        assert frames[-1]["demo"]["reward"] is not None


def test_terrain_sent_only_on_change():
    s = make_session("delivery", seed=1)
    with TestClient(create_app(s)) as c:
        other, frames = _ws_frames(c, 4)
        ter = other[1]["terrain"]                                   # full terrain on connect
        assert ter is not None and len(ter["heights"]) == spec.TERRAIN_RES == ter["res"]
        assert all(len(row) == ter["res"] for row in ter["heights"])
        keys = {f["state"]["terrain"]["key"] for f in frames}
        assert keys == {ter["key"]}
        with_heights = [f for f in frames if "heights" in f["state"]["terrain"]]
        assert len(with_heights) <= 1                               # at most the first broadcast
        assert all("heights" not in f["state"]["terrain"] for f in frames[1:])
        assert "heights" in c.get("/api/state").json()["state"]["terrain"]   # one-off GET always has them
        assert c.get("/api/terrain").json()["terrain"]["key"] == ter["key"]
        # a new terrain version is broadcast once with heights
        s.env.terrain_version += 1
        s.env.heights = s.env.heights + 0.5
        _, frames = _ws_frames(c, 3)
        hs = [f for f in frames if "heights" in f["state"]["terrain"]]
        assert len(hs) == 1 and hs[0]["state"]["terrain"]["key"] != ter["key"]
        # scene switch: new key even if the new env restarts its version at 0
        c.post("/api/scene", json={"scene": "trash"})
        c.post("/api/scene", json={"scene": "delivery"})
        assert c.get("/api/terrain").json()["terrain"]["key"] != ter["key"]


def test_terrain_without_send_terrain_kwarg():
    """Envs whose render_state(i) only takes env_id still work (heights cached server side)."""

    class OldSig(FakeEnv):
        def render_state(self, env_id):
            return super().render_state(env_id)

    s = make_session("delivery", seed=1, make_env=lambda sc, n, **kw: OldSig(sc, n, **kw))
    f1 = s.frame(include_terrain=False)
    assert "heights" in f1["state"]["terrain"]
    f2 = s.frame(include_terrain=False)
    assert "heights" not in f2["state"]["terrain"]
    assert s.terrain()["heights"] and s.frame()["state"]["terrain"]["heights"]


def test_city_has_no_terrain_and_building_boxes():
    s = make_session("trash", seed=2)
    st = s.frame()["state"]
    assert st["variant"] == "city" and st["terrain"] is None and st["water"] is None
    kinds = {b["kind"] for b in st["boxes"]}
    assert {"building", "car", "truck"} <= kinds
    assert any(z["name"] == "bin" and z["kind"] == "truck" for z in st["zones"])
    assert {i["is_trash"] for i in st["items"]} == {True, False}


# ------------------------------------------------------------------ perturbations
def test_perturb_trash(client):
    st0 = client.get("/api/state").json()["state"]
    n_box = len(st0["boxes"])
    r = client.post("/api/perturb", json={"kind": "gust", "vec": [5.0, 0.0, 0.0], "duration": 3.0})
    assert r.status_code == 200 and "gusts are off" in r.json()["note"]
    assert float(client.session.env.gust[0].norm()) == 0.0          # env ignores gusts when disabled
    assert client.post("/api/perturb", json={"kind": "drop_obstacle", "pos": [1.0, 2.0]}).status_code == 200
    boxes = client.get("/api/state").json()["state"]["boxes"]
    assert len(boxes) == n_box + 1
    assert any(b["kind"] == "dropped" and abs(b["pos"][0] - 1.0) < 1e-3 and abs(b["pos"][1] - 2.0) < 1e-3 for b in boxes)
    # move_payload defaults to the item a0 is bound to
    client.post("/api/mission", json={"text": "grab the pizza box and drop it in the bin"})
    a0 = client.session.pending.anchors["a0"].split(".")[0]
    r = client.post("/api/perturb", json={"kind": "move_payload", "pos": [-4.0, 3.0]})
    assert r.status_code == 200 and r.json()["params"]["item"] == int(a0.split("_")[1])
    it = next(i for i in client.get("/api/state").json()["state"]["items"] if i["name"] == a0)
    assert abs(it["pos"][0] + 4.0) < 0.2 and abs(it["pos"][1] - 3.0) < 0.2
    r = client.post("/api/perturb", json={"kind": "move_payload", "pos": [2.0, 2.0], "item": "item_1"})
    assert r.json()["params"]["item"] == 1
    assert client.post("/api/perturb", json={"kind": "move_payload", "pos": [2, 2], "item": "boat"}).status_code == 400
    assert client.post("/api/perturb", json={"kind": "waves", "scale": 2.0}).json()["note"]   # no-op off the ocean
    assert client.post("/api/perturb", json={"kind": "disable_drone", "agent": 0}).status_code == 200
    assert client.get("/api/state").json()["state"]["drones"][0]["alive"] is False
    assert client.post("/api/perturb", json={"kind": "earthquake"}).status_code == 400
    assert client.post("/api/perturb", json={"kind": "drop_obstacle"}).status_code == 400


def test_perturb_tide_and_gusts_enabled():
    s = make_session("delivery", seed=1, make_env=fake_mk(enable_gusts=True))
    with TestClient(create_app(s)) as c:
        assert c.get("/api/info").json()["gusts"] is True
        lvl0 = c.get("/api/state").json()["state"]["water"]["level"]
        r = c.post("/api/perturb", json={"kind": "tide", "offset": 0.8})
        assert r.status_code == 200 and r.json()["params"] == {"offset": 0.8} and not r.json()["note"]
        assert c.get("/api/state").json()["state"]["water"]["level"] == pytest.approx(lvl0 + 0.8)
        assert c.post("/api/perturb", json={"kind": "tide", "offset": 9}).json()["params"]["offset"] == 2.0
        r = c.post("/api/perturb", json={"kind": "gust", "vec": [5.0, 0.0, 0.0]})
        assert not r.json()["note"] and float(s.env.gust[0, 0]) == pytest.approx(5.0)


def test_gusts_enabled_helper():
    cfg = lambda **kw: SimpleNamespace(cfg=SimpleNamespace(**kw))
    assert gusts_enabled(cfg(domain_rand=True, enable_gusts=True))
    assert not gusts_enabled(cfg(domain_rand=False, enable_gusts=True))
    assert not gusts_enabled(cfg(domain_rand=True, enable_gusts=False))
    assert not gusts_enabled(SimpleNamespace())


def test_perturb_ocean():
    s = make_session("ocean")
    with TestClient(create_app(s)) as c:
        st0 = c.get("/api/state").json()["state"]
        amp0 = np.array(st0["water"]["waves"]["amp"])
        assert c.post("/api/perturb", json={"kind": "waves", "scale": 2.0}).status_code == 200
        amp1 = np.array(c.get("/api/state").json()["state"]["water"]["waves"]["amp"])
        np.testing.assert_allclose(amp1, 2.0 * amp0, rtol=1e-4)
        assert c.post("/api/perturb", json={"kind": "disable_drone", "agent": 1}).status_code == 200
        drones = c.get("/api/state").json()["state"]["drones"]
        assert drones[1]["alive"] is False and drones[0]["alive"] is True
        assert c.post("/api/perturb", json={"kind": "disable_drone", "agent": 9}).status_code == 400
        assert c.post("/api/perturb", json={"kind": "move_payload", "pos": [0, 0]}).json()["note"]
        assert c.post("/api/perturb", json={"kind": "tide", "offset": 1.0}).json()["note"]


# ------------------------------------------------------------------ flying
def test_stop_reset(client):
    client.post("/api/mission", json={"text": "fly to the bin"})
    assert client.post("/api/approve").json()["flying"] is True
    r = client.post("/api/stop")
    assert r.status_code == 200 and r.json()["flying"] is False and r.json()["status"] == "stopped"
    r = client.post("/api/reset")
    assert r.status_code == 200 and r.json()["status"] in ("idle", "awaiting approval")
    assert client.get("/api/state").json()["demo"]["step"] == 0
    assert client.post("/api/approve").json()["flying"] is True


@pytest.mark.parametrize("scene,seed,text", [
    ("trash", 2, "fly to the bin"),
    ("trash", 2, "pick up the trash bag and put it in the garbage truck"),     # city: truck zone on top of the truck
    ("trash", 3, "grab the bottle and drop it in the bin"),                     # terrain litter world
    ("delivery", 1, EXAMPLES["delivery"]),
])
def test_episode_success_fake_env(scene, seed, text):
    s = make_session(scene, seed=seed)
    s.compile(text)
    s.approve()
    for _ in range(spec.EPISODE_STEPS * 2):
        s.tick()
        if not s.flying:
            break
    assert s.status == "success", s.message
    if text != "fly to the bin":
        assert any(i["delivered"] for i in s.frame()["state"]["items"])


def test_fallback_controller_ocean_grabs_corners_fake():
    env = FakeEnv("ocean", 1, seed=0)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(EXAMPLES["ocean"], "ocean", env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = env.observe()
    for _ in range(300):
        obs = env.step(act(obs, None)).obs
    assert bool((obs["proprio"][0, :, 10] > 0.5).all())


def test_scripted_controller_climbs_over_obstacles():
    """A near LiDAR return in a near-horizontal row makes the drone climb and slow down in transit."""
    obs = spec.empty_obs(1, 1)
    obs["anchors"][0, 0, 0, 0:3] = torch.tensor([20.0, 0.0, -4.0])     # a0 far ahead
    obs["anchors"][0, 0, 0, 3 + spec.ANCHOR_TYPES.index("handle")] = 1.0
    obs["proprio"][0, 0, 11] = 4.5                                       # at cruise height
    obs["lidar"].fill_(255)
    free = P.ScriptedController("trash", 1, 1)(obs)
    assert abs(float(free[0, 0, 2])) < 0.05 and float(free[0, 0, 0]) > 0.4
    obs["lidar"][0, 0, 0, 12, 0] = int(255 * 2.0 / spec.LIDAR_MAX_RANGE)  # wall 2 m away
    ctrl = P.ScriptedController("trash", 1, 1)
    for _ in range(5):
        act = ctrl(obs)
    assert float(act[0, 0, 2]) > 0.2 and float(act[0, 0, 0]) < float(free[0, 0, 0])
    obs["lidar"].zero_()                                                 # all-zero lidar = sensor off: ignored
    act = P.ScriptedController("trash", 1, 1)(obs)
    assert abs(float(act[0, 0, 2])) < 0.05


# ------------------------------------------------------------------ world (fake env)
def test_split_by_drone():
    shared, per = P.split_by_drone(SPLIT, 3)
    assert shared == "" and set(per) == {0, 1, 2}
    assert "pizza box" in per[0] and "cooler" in per[1] and "boat" in per[2] and "drone" not in per[1]
    shared, per = P.split_by_drone("drone 0 fly to the drop-off pad and the other drones collect trash", 3)
    assert per == {0: "fly to the drop-off pad"} and shared == "collect trash"
    assert P.split_by_drone("fly to the boat", 3) == ("fly to the boat", {})
    assert P.split_by_drone("drone 7 fly to the boat", 3)[1] == {}          # no such drone


def test_world_keyword_parser_per_drone_fake():
    env = FakeEnv("world", 1, seed=2)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(SPLIT, "world", env)
    assert set(m.agents) == {0, 1, 2}
    types = P.item_types(env)
    a = {ag: j["anchors"] for ag, j in m.agents.items()}
    assert types[a[0]["a0"].split(".")[0]] == "pizza_box" and a[0]["a1"] == "bin"
    assert types[a[1]["a0"].split(".")[0]] == "cooler" and a[1]["a1"] == "pad_dropoff"
    assert a[2]["a0"] == "boat" and m.agents[2]["done_when"] == "inside(a0)"
    pub = m.public(env.entity_descriptions())
    assert set(pub["agents"]) == {"0", "1", "2"} and pub["agents"]["2"]["anchors"]["a0"]["target"] == "boat"
    assert m.tasks(3)[2].startswith("fly to boat")
    # one shared carry text: every drone its own item of that kind
    m = pipe.compile("collect the trash and put it in the bin", "world", env)
    items = [j["anchors"]["a0"] for j in m.agents.values()]
    assert len(set(items)) == 3 and all(P._is_trash(env, i.split(".")[0]) for i in items)
    # net fishing: one shared mission, every drone its own corner, all haul
    m = pipe.compile(EXAMPLES["ocean"], "world", env)
    assert m.agents is None and m.anchors["a0"] == "net.corner_{agent}"
    pipe.bind(env, m, torch.tensor([0]))
    assert [b[0].entity for b in env.bindings] == ["net.corner_0", "net.corner_1", "net.corner_2"]
    assert pipe.haul_mask(env) == [True, True, True]


def test_world_episode_per_drone_success_fake():
    """Each drone does its own job; success needs every drone's own done_when."""
    s = make_session("world", seed=2)
    s.compile(SPLIT)
    s.approve()
    assert s.extras()["agent_tasks"][1].startswith("carry")
    for _ in range(s.max_steps):
        s.tick()
        if not s.flying:
            break
    assert s.status == "success", s.message
    assert s.done_agents == {0, 1, 2}
    st = s.frame()["state"]
    delivered = {i["type"] for i in st["items"] if i["delivered"]}
    assert {"pizza_box", "cooler"} <= delivered


def test_world_net_haul_fake():
    env = FakeEnv("world", 1, seed=2)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(EXAMPLES["ocean"], "world", env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = env.observe()
    for _ in range(900):
        obs = env.step(act(obs, None)).obs
        if bool((obs["proprio"][0, :, 10] > 0.5).all()):
            break
    assert bool((obs["proprio"][0, :, 10] > 0.5).all())


def test_world_trained_per_drone_z():
    """Per-drone jobs get one z per drone ([A, d]); the policy is called with [N, A, d]."""
    seen = {}

    class Agent(FakeAgent):
        def act(self, obs, z, noise=0.0):
            seen["z"] = tuple(z.shape)
            return super().act(obs, z, noise)

    pipe = P.Pipeline(agent=Agent(8), mission_api={}, z_samples_envs=2, z_samples_steps=4)
    s = DemoSession("world", 1, pipeline=pipe, make_env=default_make_env(lidar=False, fake=True), seed=2)
    s.compile(SPLIT)
    s.approve()
    assert s.z is not None and s.z.shape == (3, 8)
    assert torch.allclose(s.z.norm(dim=-1), torch.full((3,), np.sqrt(8)), atol=1e-3)
    assert not torch.allclose(s.z[0], s.z[2])
    s.tick()
    assert seen["z"] == (1, 3, 8)
    assert sum(p["current"] for p in s.zmap["points"]) == 3
    assert s.extras()["z_per_drone"] is True


# ------------------------------------------------------------------ z map / trained mode
def test_reference_missions_use_new_names():
    by_scene = {m["scene"] for m in P.REFERENCE_MISSIONS}
    assert by_scene == set(spec.SCENES)
    for m in P.REFERENCE_MISSIONS:
        names = FakeEnv(m["scene"], seed=1 if m["scene"] == "delivery" else 2).entity_names()
        for v in m["anchors"].values():
            assert v in names, (m["name"], v)
        P.rewardlang.compile_expr(m["reward"])(torch.zeros(2, spec.G_DIM))


def test_pca_zmap_with_fake_agent():
    pipe = P.Pipeline(agent=FakeAgent(8), mission_api={}, z_samples_envs=4, z_samples_steps=5)
    mk = default_make_env("torch", "cpu", lidar=False, fake=True)
    zm = pipe.zmap(lambda sc, n: mk(sc, n), torch.randn(8))
    assert zm["available"] is True
    names = [p["name"] for p in zm["points"]]
    assert "Trash pickup" in names and "Organ delivery" in names and "Net fishing" in names
    assert sum(p["current"] for p in zm["points"]) == 1
    for p in zm["points"]:
        assert np.isfinite(p["x"]) and np.isfinite(p["y"]) and abs(p["x"]) <= 1 + 1e-6 and abs(p["y"]) <= 1 + 1e-6
    assert P.pca_2d(np.random.randn(5, 8)).shape == (5, 2)


def test_trained_mode_labels_and_z():
    pipe = P.Pipeline(agent=FakeAgent(8), mission_api={}, z_samples_envs=4, z_samples_steps=5)
    s = DemoSession("delivery", 1, pipeline=pipe, make_env=default_make_env(lidar=False, fake=True), seed=1)
    s.compile(EXAMPLES["delivery"])
    info = s.approve()
    assert info["policy_mode"] == P.POLICY_TRAINED
    assert s.z is not None and s.z.shape == (8,)
    assert abs(float(s.z.norm()) - np.sqrt(8)) < 1e-3
    assert s.zmap["available"] and any(p["current"] for p in s.zmap["points"])
    s.tick()
    assert s.extras()["z_norm"] is not None


def test_reward_language_safe():
    from demo.rewardlang import RewardSyntaxError, compile_expr, explain
    g = torch.zeros(3, spec.G_DIM)
    assert compile_expr("-dist(payload, a1) + 3*holding - 5*inside(a2)")(g).shape == (3,)
    assert compile_expr("-3*holding*(1 - payload_is_trash) - 5*bycatch - trash_left")(g).shape == (3,)
    assert any("animals tangled" in line for line in explain("-5*bycatch", None, {}))
    for bad in ("__import__('os')", "g.sum()", "[1, 2]", "foo + 1", "dist(bar, a1)", "payload_type"):
        with pytest.raises(RewardSyntaxError):
            compile_expr(bad)(g)


# ------------------------------------------------------------------ viewer (static checks)
def test_index_pins_cdn_only():
    html = (WEB / "index.html").read_text()
    urls = re.findall(r"https?://[^\s\"'<>]+", html)
    assert urls, "expected the three.js importmap"
    allowed = ("https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js",
               "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/")
    for u in urls:
        assert u in allowed, u
    js = (WEB / "main.js").read_text()
    assert not re.search(r"https?://", js), "main.js must load three only through the importmap"


def test_mainjs_consumes_render_schema():
    js = (WEB / "main.js").read_text()
    for key in ("arena_half", "st.terrain", "heights", "t.res", "t.half", "water.level", "water.waves", "st.items",
                "st.boxes", "st.spheres", "st.zones", "st.animals", "tangled", "is_trash", "delivered", "held",
                "/api/terrain", '"terrain"', "d.gusts", '"tide"', "offset"):
        assert key in js, key
    for kind in BOX_KINDS - {"boat"} | SPHERE_KINDS | ZONE_KINDS - {"boat"}:
        assert f'"{kind}"' in js, kind
    for t in spec.ANIMAL_TYPES:
        assert re.search(r"\b%s\b" % t, js), t
    for t in spec.ITEM_TYPES:
        assert f'case "{t}"' in js, t
    html = (WEB / "index.html").read_text()
    assert 'id="tide"' in html and 'id="wave-scale"' in html and 'id="btn-gust"' in html


def test_mainjs_world_viewer():
    """World: sea = waves offset by water.level over the whole arena (GPU), drone picker + overview."""
    js = (WEB / "main.js").read_text()
    html = (WEB / "index.html").read_text()
    assert 'data-scene="world"' in html and 'id="drone-picker"' in html and 'id="btn-overview"' in html
    assert 'id="card-agents"' in html
    for key in ('"world"', "makeSea", "updateSea", "uLevel", "water.level", "followDrone", "followIdx",
                "agent_tasks", "agents_done", "m.agents", "2 * h + 10"):
        assert key in js, key
    # the GPU sea uses the same wave formula as waveHeight()
    sea = js[js.index("function makeSea"):js.index("function updateSea")]
    assert "uK[i] * dot(dv, position.xy) - uOmega[i] * uT + uPhase[i]" in sea and "uAmp[i] * cos(ph)" in sea
    assert "skyH + uLevel" in sea
    assert "three@0.186.1" in html


def test_mainjs_ocean_waves_stay_on_gpu():
    """Ocean and world waves share the analytic-normal shader; JS never rebuilds water vertices."""
    js = (WEB / "main.js").read_text()
    assert "makeSea(h, WATER_SEG" in js
    assert "updateSea(oceanGroup.userData.water" in js
    assert "function updateWater" not in js
    assert "waterGeo.computeVertexNormals" not in js


def test_mainjs_smooths_held_payloads_without_frame_allocations():
    """Carried items interpolate at display rate instead of snapping at the server tick rate."""
    js = (WEB / "main.js").read_text()
    assert "target: new THREE.Vector3(), held: false" in js
    assert "for (let i = 0; i < heldItems.length; i++)" in js
    assert "item.position.lerp(item.userData.target, a)" in js
    assert "if (!it.held) g.position.copy(u.target)" in js


def test_mainjs_r186_drone_model_and_shader():
    """The r186 viewer keeps drone detail shared, shader-local, and allocation-light."""
    js = (WEB / "main.js").read_text()
    for key in ("mergeGeometries", "droneBodyGeometry", "droneFrameGeometry", "droneRotorGeometry",
                "dronePart", "onBeforeCompile", "droneRotorMaterial", "forceSinglePass",
                "THREE.Timer", "renderer.setAnimationLoop"):
        assert key in js, key
    assert "THREE.Clock" not in js
    assert "requestAnimationFrame(animate)" not in js
    assert "fd.position.clone()" not in js


def test_mainjs_building_variants_share_geometry_and_shader():
    """Apartments and offices preserve their facade programs in two instanced batches."""
    js = (WEB / "main.js").read_text()
    for key in ("apartmentBuildingGeometry", "officeBuildingGeometry", "apartmentBuildingMaterial",
                "officeBuildingMaterial", '"office" : "apartment"',
                "skyhand-building-facade-r186-v3", "fwidth(gridPosition)", "buildingPart",
                'setAttribute("buildingUv"', 'setAttribute("buildingFacade"',
                "vec2 gridPosition = vBuildingUv", "float frameMask", "float roofOcclusion",
                "buildingFloors: { value: office ? 5 : 4 }",
                "buildingBays: { value: office ? 4 : 3 }"):
        assert key in js, key
    assert "new THREE.InstancedMesh(apartmentBuildingGeometry" in js
    assert "new THREE.InstancedMesh(officeBuildingGeometry" in js
    assert "syncBuildings(st.boxes)" in js
    assert "updateBuildingUniforms" not in js
    assert 'setAttribute("buildingSide"' not in js
    assert "mesh.onBeforeRender" not in js
    assert "facadePosition" not in js
    assert "buildingTime" not in js
    assert "windowTexture" not in js
    assert "g.userData.tex" not in js


def test_mainjs_wave_formula_matches_waves_params():
    waves = pytest.importorskip("skyhand.env.waves")
    keys = set(waves.WaveField(1).params(0))
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
    waves = pytest.importorskip("skyhand.env.waves")
    w = waves.WaveField(1)
    w.scale[0] = 1.7
    p = w.params(0)
    xy = torch.tensor([[[1.3, -2.2], [7.0, 4.0]]])
    t = torch.tensor([3.21])
    ref = w.height(xy, t)[0]
    for j, (x, y) in enumerate(xy[0].tolist()):
        h = sum(p["amp"][i] * math.cos(p["k"][i] * (math.cos(p["dir"][i]) * x + math.sin(p["dir"][i]) * y)
                                       - p["omega"][i] * 3.21 + p["phase"][i]) for i in range(len(p["amp"])))
        assert abs(h - float(ref[j])) < 1e-4


# ------------------------------------------------------------------ real env (skipped until core.py supports the scenes)
def real_session(scene, **kw):
    return DemoSession(scene, num_envs=1, pipeline=P.Pipeline(mission_api={}),
                       make_env=default_make_env("torch", "cpu", lidar=True), seed=3, **kw)


@pytest.mark.parametrize("scene", spec.SCENES)
def test_real_env_frames(scene):
    s = real_session(scene)
    with TestClient(create_app(s)) as c:
        assert c.post("/api/mission", json={"text": EXAMPLES[scene]}).status_code == 200
        assert c.post("/api/approve").json()["flying"] is True
        st = c.get("/api/state").json()["state"]
        check_render_state(st, scene, s.env.num_agents)
        if st["terrain"] is not None:
            assert len(st["terrain"]["heights"]) == st["terrain"]["res"]
        _, frames = _ws_frames(c, 3)
        for f in frames:
            check_render_state(f["state"], scene, s.env.num_agents)
        assert frames[-1]["demo"]["step"] > 0 and len(frames[-1]["state"]["lidar"]["nir"]) > 0


def test_real_env_world_follow_camera_data():
    """What the follow camera / drone picker read: 3 drones with positions, per-drone tasks."""
    s = real_session("world")
    assert s.env.num_agents == spec.SCENE_AGENTS["world"] == 3
    s.compile(SPLIT)
    s.approve()
    for _ in range(3):
        s.tick()
    f = s.frame(include_terrain=False)
    st, demo = f["state"], f["demo"]
    check_render_state(dict(st, terrain=st["terrain"]), "world", 3)
    assert len(st["drones"]) == 3 and all(len(d["pos"]) == 3 and d["alive"] for d in st["drones"])
    assert len({tuple(d["pos"]) for d in st["drones"]}) == 3
    assert demo["num_agents"] == 3 and len(demo["agent_tasks"]) == 3 and demo["agents_done"] == []
    assert st["terrain"]["key"] and s.terrain()["res"] == 128 and len(s.terrain()["heights"]) == 128


def test_real_env_world_per_drone_fallback():
    """Different job per drone: each flies toward its own a0 (max speed from env.max_speed)."""
    from skyhand.env.core import make_env
    env = make_env("world", num_envs=1, seed=3, random_anchors=False, lidar=False)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(SPLIT, "world", env)
    assert m.agents and len({str(j["anchors"]["a0"]) for j in m.agents.values()}) == 3
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    assert P.scripted_for(env).max_speed == env.max_speed == spec.SCENE_MAX_SPEED["world"]
    obs = P.observe(env)
    a0 = obs["anchors"][0, :, 0, :3]
    d0 = a0.norm(dim=-1)
    assert float((a0[0] - a0[1]).norm()) > 1.0 and float((a0[1] - a0[2]).norm()) > 1.0   # own anchors
    best = d0.clone()
    for _ in range(220):
        out = env.step(act(obs, None))
        obs = out.obs
        assert not bool(out.done[0])
        best = torch.minimum(best, obs["anchors"][0, :, 0, :3].norm(dim=-1))
    for ag in range(3):
        assert float(best[ag]) < 0.6 * float(d0[ag]) or float(best[ag]) < 2.0, (ag, d0, best)
    assert float(spec.g_field(obs["g"][0, 0], "speed")[0]) <= env.max_speed * 1.2


def test_real_env_world_entity_names():
    from skyhand.env.core import make_env
    env = make_env("world", num_envs=1, lidar=False)
    names = set(env.entity_names())
    assert {"item_0.handle", "bin", "pad_pickup", "pad_dropoff", "boat", "nofly_0", "nofly_1", "net.corner_3",
            "net.center", "fish_school_0", "whale_0", "drone_0", "drone_2"} <= names
    assert names >= set(FakeEnv("world", seed=2).entity_names()) - {f"obstacle_{k}" for k in range(8)}


def test_real_mission_package_world_split():
    """MissionSpec.agents (when the mission package has it): per-drone bindings, rewards and done."""
    api = P.load_mission_api()
    if "offline_compile" not in api:
        pytest.skip("skyhand.mission.offline_compile not importable")
    from skyhand.env.core import make_env
    env = make_env("world", num_envs=1, seed=3, random_anchors=False, lidar=False)
    pipe = P.Pipeline(mission_api={k: v for k, v in api.items() if k != "compile_mission"})
    m = pipe.compile(SPLIT, "world", env)
    if not m.agents:
        pytest.skip("mission package without per-drone agents: every drone flies the same mission")
    assert set(m.agent_reward_fns) == {0, 1, 2} and m.per_agent_bindings
    ents = [m.per_agent_bindings[ag][0].entity for ag in range(3)]
    assert ents[2] == "boat" and ents[0] != ents[1]
    assert "cooler" in env.entity_descriptions()[ents[1].split(".")[0]]
    pipe.bind(env, m, torch.tensor([0]))
    g = P.observe(env)["g"][0]
    for ag in range(3):
        assert m.agent_reward_fns[ag](g[ag:ag + 1]).shape[0] == 1


def test_real_env_entity_names():
    from skyhand.env.core import make_env
    names = make_env("trash", num_envs=1, lidar=False).entity_names()
    assert {"item_0", "item_0.handle", "bin", "nofly_0"} <= set(names) and "payload" not in names
    names = make_env("delivery", num_envs=1, lidar=False).entity_names()
    assert {"item_0.handle", "pad_pickup", "pad_dropoff", "nofly_0", "nofly_1"} <= set(names)
    names = make_env("ocean", num_envs=1, lidar=False).entity_names()
    assert {"net.corner_0", "boat", "whale_0", "dolphin_1", "turtle_1"} <= set(names)


@pytest.mark.parametrize("text", ["fly to the bin", "pick up the trash bag and put it in the bin"])
def test_real_env_trash_success(text):
    s = real_session("trash")
    s.compile(text)
    s.approve()
    for _ in range(s.max_steps):
        s.tick()
        if not s.flying:
            break
    assert s.status == "success", s.message


@pytest.mark.parametrize("scene", ["trash", "delivery"])
def test_real_env_fallback_controller_reaches_item(scene):
    from skyhand.env.core import make_env
    env = make_env(scene, num_envs=1, seed=5, random_anchors=False, lidar=True)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(EXAMPLES[scene], scene, env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = P.observe(env)
    d0 = float(obs["anchors"][0, 0, 0, :3].norm())
    best = d0
    for _ in range(env.episode_steps):
        obs = env.step(act(obs, None)).obs
        best = min(best, float(obs["anchors"][0, 0, 0, :3].norm()))
        if bool(obs["proprio"][0, 0, 10] > 0.5):
            break
    assert best < 0.5 * d0 and best < 1.0, (d0, best)


def test_real_env_fallback_controller_ocean_grabs_corners():
    from skyhand.env.core import make_env
    env = make_env("ocean", num_envs=1, seed=0, random_anchors=False, lidar=False)
    pipe = P.Pipeline(mission_api={})
    m = pipe.compile(EXAMPLES["ocean"], "ocean", env)
    pipe.bind(env, m, torch.tensor([0]))
    act = pipe.make_controller(env)
    obs = P.observe(env)
    for _ in range(300):
        out = env.step(act(obs, None))
        obs = out.obs
        assert not bool(out.done[0])
    assert bool((obs["proprio"][0, :, 10] > 0.5).any())


@pytest.mark.parametrize("scene,text", [("trash", "pick up the trash bag and put it in the bin"),
                                        ("delivery", EXAMPLES["delivery"])])
def test_real_env_carry_success(scene, text):
    """The scripted fallback completes a carry mission end to end (delivery: 300-700 m at up to 18 m/s)."""
    s = real_session(scene)
    assert s.max_steps == s.env.episode_steps == spec.SCENE_EPISODE_STEPS[scene]
    s.compile(text)
    s.approve()
    for _ in range(s.max_steps):
        s.tick()
        if not s.flying:
            break
    assert s.status == "success", s.message


def test_real_env_controller_uses_env_speed():
    from skyhand.env.core import make_env
    env = make_env("delivery", num_envs=1, lidar=False)
    c = P.scripted_for(env)
    assert c.max_speed == env.max_speed == spec.SCENE_MAX_SPEED["delivery"] and c.max_vz == env.max_vz
    assert c.v_cruise > spec.MAX_SPEED


@pytest.mark.parametrize("scene", spec.SCENES)
def test_real_mission_package_compiles(scene):
    """skyhand.mission (offline parser, no API key needed) compiles the examples into bindable anchors."""
    api = P.load_mission_api()
    if "offline_compile" not in api:
        pytest.skip("skyhand.mission.offline_compile not importable")
    pipe = P.Pipeline(mission_api={k: v for k, v in api.items() if k != "compile_mission"})
    s = DemoSession(scene, num_envs=1, pipeline=pipe, make_env=default_make_env("torch", "cpu", lidar=False), seed=3)
    m = s.compile(EXAMPLES[scene])
    assert m["source"] == P.MISSION_OFFLINE
    names = set(s.env.entity_names())
    for k, v in m["anchors"].items():
        t = v["target"]
        assert not isinstance(t, str) or t.replace("{agent}", "0") in names, (k, t)
    assert s.approve()["flying"] is True
    s.tick()



def test_multi_segment_execution_and_switching():
    """Verify that a multi-segment mission automatically switches z vectors as detectors fire."""
    pipe = P.Pipeline(agent=FakeAgent(8))
    s = make_session("trash", seed=2, pipeline=pipe)
    
    prompt = "pick up the garbage bag, spin in a circle, wiggle up and down, then drop the garbage off"
    data = s.compile(prompt)
    assert "segments" in data
    assert len(data["segments"]) == 4
    assert len(s.pending.segments) == 4
    
    s.approve()
    assert s.flying
    assert s.current_segment_idx == 0
    assert s.z is not None
    z0 = s.z.clone()
    
    # Keep gripper closed during segments 0-2 so FakeEnv retains the item
    s.controller = lambda obs, z: torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0]]])
    
    # Segment 0: Done when holding
    s.env.held[0, 0] = 0
    s.env.grip[0, 0] = True
    s.tick()
    
    assert s.flying
    assert s.current_segment_idx == 1
    assert "Segment 2/4" in s.message
    z1 = s.z.clone()
    assert not torch.allclose(z0, z1)
    
    # Segment 1: Done when speed > 0.8
    s.env.vel[0, 0, 0] = 5.0
    s.tick()
    
    assert s.flying
    assert s.current_segment_idx == 2
    assert "Segment 3/4" in s.message
    z2 = s.z.clone()
    assert not torch.allclose(z1, z2)
    
    # Segment 2: Done when abs(vz) > 0.6
    s.env.vel[0, 0, 2] = 5.0
    s.tick()
    
    assert s.flying
    assert s.current_segment_idx == 3
    assert "Segment 4/4" in s.message
    z3 = s.z.clone()
    assert not torch.allclose(z2, z3)
    
    # Segment 3: Fly over bin and release item to complete delivery
    bin_pos = s.env.zones[0]['pos']
    s.env.drone[0, 0] = torch.tensor([bin_pos[0], bin_pos[1], 2.0])
    s.env.items[0, 0] = torch.tensor([bin_pos[0], bin_pos[1], 2.0])
    s.env.held[0, 0] = 0
    s.env.grip[0, 0] = True
    s.controller = lambda obs, z: torch.tensor([[[0.0, 0.0, 0.0, 0.0, -1.0]]])
    s.tick()
    
    assert not s.flying
    assert s.status == "success"
    assert "all 4 segments complete" in s.message
