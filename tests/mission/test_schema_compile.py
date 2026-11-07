from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from skyhand import spec
from skyhand.mission import MissionSpec, compile_mission, compile_reward, offline_compile, to_bindings
from skyhand.mission.compile import _EXAMPLES, _items, _types_in, build_prompt, parse_and_validate
from skyhand.mission.schema import MissionError, bind_mission, infer_kind, resolve_entity

from mission_fakes import FakeEnv, delivery, ocean, trash_city, trash_no_positions

TRASH = trash_city()
DELIV = delivery()
OCEAN = ocean()
TRASH_ENTS, DELIV_ENTS, OCEAN_ENTS = list(TRASH), list(DELIV), list(OCEAN)


# ------------------------------------------------------------------ schema
def test_schema_unknown_entity():
    data = {"scene": "trash", "anchors": {"a0": "rocket"}, "reward": "-dist(drone, a0)"}
    with pytest.raises((ValidationError, MissionError)):
        MissionSpec.model_validate(data, context={"entities": TRASH_ENTS})
    ms = MissionSpec.model_validate(data)   # no entity list: syntax only
    with pytest.raises(MissionError):
        ms.check_entities(TRASH_ENTS)
    for gone in ("payload", "payload.handle"):   # no more payload entity in the trash scene
        with pytest.raises((ValidationError, MissionError)):
            MissionSpec.model_validate({"scene": "trash", "anchors": {"a0": gone}, "reward": "1"},
                                       context={"entities": TRASH_ENTS})


def test_schema_bad_fields():
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="speed.real")
    with pytest.raises(ValidationError):
        MissionSpec(scene="mars", reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", anchors={"a7": "bin"}, reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", anchors={"a0": [1, 2]}, reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="speed", steps=[{"reward": "__import__('os')"}])
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="payload_is('rocket')")
    for sc in spec.SCENES:
        assert MissionSpec(scene=sc, reward="payload_is('pizza_box')").scene == sc


@pytest.mark.parametrize("scene,ents,alias,target", [
    ("trash", TRASH_ENTS, "truck", "bin"), ("trash", TRASH_ENTS, "garbage truck", "bin"),
    ("trash", TRASH_ENTS, "dumpster", "bin"), ("trash", TRASH_ENTS, "item 3 handle", "item_3.handle"),
    ("trash", TRASH_ENTS, "item3.handle", "item_3.handle"), ("trash", TRASH_ENTS, "Item 2", "item_2"),
    ("trash", TRASH_ENTS, "nofly", "nofly_0"), ("trash", TRASH_ENTS, "no-fly zone", "nofly_0"),
    ("delivery", DELIV_ENTS, "cooler", "item_0"), ("delivery", DELIV_ENTS, "organ", "item_0"),
    ("delivery", DELIV_ENTS, "cooler.handle", "item_0.handle"), ("delivery", DELIV_ENTS, "payload.handle", "item_0.handle"),
    ("delivery", DELIV_ENTS, "hospital", "pad_dropoff"), ("delivery", DELIV_ENTS, "helipad", "pad_dropoff"),
    ("delivery", DELIV_ENTS, "pickup", "pad_pickup"), ("delivery", DELIV_ENTS, "drone", "drone_0"),
    ("ocean", OCEAN_ENTS, "net", "net.center"), ("ocean", OCEAN_ENTS, "ship", "boat"),
    ("ocean", OCEAN_ENTS, "whale", "whale_0"), ("ocean", OCEAN_ENTS, "fish", "fish_school_0"),
])
def test_aliases(scene, ents, alias, target):
    ms = MissionSpec.model_validate({"scene": scene, "anchors": {"a0": alias}, "reward": "1"},
                                    context={"entities": ents})
    assert ms.anchors["a0"] == target
    assert resolve_entity(alias, ents) == target


def test_scene_specific_aliases():
    # "cooler" is item_0 only in delivery; in the trash scene item_0 is whatever is there
    with pytest.raises((ValidationError, MissionError)):
        MissionSpec.model_validate({"scene": "trash", "anchors": {"a0": "cooler"}, "reward": "1"},
                                   context={"entities": TRASH_ENTS})
    with pytest.raises(MissionError):
        resolve_entity("hospital", TRASH_ENTS)
    with pytest.raises(MissionError):
        resolve_entity("truck", DELIV_ENTS)


def test_templates():
    ms = MissionSpec.model_validate({"scene": "ocean", "anchors": {"a0": "net.corner_{agent}"}, "reward": "1"},
                                    context={"entities": OCEAN_ENTS})
    assert [to_bindings(ms, agent=i)[0].entity for i in range(3)] == ["net.corner_0", "net.corner_1", "net.corner_2"]


@pytest.mark.parametrize("name,kind", [
    ("item_3.handle", "handle"), ("item_0.handle", "handle"), ("net.corner_2", "handle"),
    ("net.corner_{agent}", "handle"), ("bin", "zone"), ("boat", "zone"), ("pad_pickup", "zone"),
    ("pad_dropoff", "zone"), ("nofly_0", "zone"), ("nofly_1", "zone"), ("item_3", "point"),
    ("whale_0", "point"), ("fish_school_0", "point"), ("obstacle_3", "point"), ("net.center", "point"),
    ("drone_0", "point"),
])
def test_infer_kind(name, kind):
    assert infer_kind(name) == kind


def test_to_bindings_kinds():
    ms = MissionSpec(scene="ocean", anchors={"a0": "net.corner_2", "a1": "boat", "a2": "whale_0",
                                             "a3": [1, 2, 3]}, reward="1")
    b = to_bindings(ms)
    assert [x.kind for x in b] == ["handle", "zone", "point", "point"]
    assert b[3].point == (1.0, 2.0, 3.0) and b[3].entity is None
    ms = MissionSpec(scene="trash", anchors={"a0": "item_4.handle", "a1": "bin", "a2": "nofly_0", "a3": None},
                     reward="1", anchor_radius={"a3": 2.0})
    assert [x.kind for x in to_bindings(ms)] == ["handle", "zone", "zone", "none"]
    ms = MissionSpec(scene="delivery", anchors={"a0": "cooler.handle", "a1": "hospital", "a2": "pad_pickup"},
                     reward="1")
    b = to_bindings(ms, entities=DELIV_ENTS)
    assert [(x.entity, x.kind) for x in b[:3]] == [("item_0.handle", "handle"), ("pad_dropoff", "zone"),
                                                   ("pad_pickup", "zone")]
    ms2 = MissionSpec(scene="trash", anchors={"a3": [0, 0, 0]}, anchor_radius={"a3": 2.0}, reward="1")
    assert to_bindings(ms2)[3].kind == "zone" and to_bindings(ms2)[3].radius == 2.0
    ms3 = MissionSpec(scene="trash", anchors={"a0": "item_2"}, anchor_kinds={"a0": "zone"}, reward="1")
    assert to_bindings(ms3)[0].kind == "zone"


# ------------------------------------------------------------------ description parsing
def test_parse_items():
    items = {it["name"]: it for it in _items(TRASH)}
    assert len(items) == spec.MAX_ITEMS
    assert items["item_2"]["type"] == "pizza_box" and items["item_2"]["trash"] and items["item_2"]["active"]
    assert items["item_2"]["pos"] == (3.0, 0.0, 0.1) and items["item_2"]["target"] == "item_2.handle"
    assert items["item_1"]["type"] == "backpack" and not items["item_1"]["trash"]
    assert items["item_5"]["type"] == "flower_pot" and not items["item_5"]["trash"]
    assert not items["item_6"]["active"] and items["item_6"]["type"] == "cardboard_box"
    assert not items["item_9"]["active"] and items["item_9"]["type"] is None
    noloc = {it["name"]: it for it in _items(trash_no_positions())}
    assert noloc["item_0"]["pos"] is None and noloc["item_0"]["trash"] and not noloc["item_1"]["trash"]
    # odd but parseable formats: handle-only listing, space-separated type, parentheses / exponents
    odd = _items({"item_7.handle": "trash bag handle at (1, 2e0, 0.5)"})
    assert odd[0]["type"] == "trash_bag" and odd[0]["pos"] == (1.0, 2.0, 0.5) and odd[0]["trash"]


def test_types_in():
    assert _types_in("pick up the pizza box") == ["pizza_box"]
    assert _types_in("grab the cardboard boxes and the bottles") == ["cardboard_box", "bottle"]
    assert _types_in("put the trash bag in the truck") == ["trash_bag"]
    assert _types_in("the backpack, not the bag") == ["backpack", "trash_bag"]
    assert _types_in("pick up the trash") == []


# ------------------------------------------------------------------ Gemini (mocked)
class FakeModels:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents)
        assert config["response_mime_type"] == "application/json"
        txt = self.outputs.pop(0)

        class R:
            text = txt
        return R()


class FakeClient:
    def __init__(self, outputs):
        self.models = FakeModels(outputs)


GOOD = {"anchors": {"a0": "item_2.handle", "a1": "bin", "a2": None, "a3": None},
        "reward": "where(holding, -dist(payload, a1), -dist(drone, a0)) - 2*holding*(1 - payload_is('pizza_box'))",
        "done_when": "distxy(payload, a1) < 1.0 and not holding",
        "explanation": "Pick up the pizza box and drop it in the garbage truck."}


def test_prompt_contents():
    p = build_prompt("pick up the pizza box", "trash", TRASH)
    for s in ("item_2.handle", "pizza_box (trash)", "obstacle_dist", "payload_is", "bycatch", "trash_left",
              "(inactive)", "pad_dropoff", "net.corner_{agent}", "pick up the pizza box", "garbage truck"):
        assert s in p, s


@pytest.mark.parametrize("k", range(len(_EXAMPLES)))
def test_worked_examples_validate(k):
    scene, ents, text, out = _EXAMPLES[k]
    ms = parse_and_validate(json.dumps(out), text, scene, list(ents))
    assert ms.scene == scene and ms.reward == out["reward"]


def test_compile_valid():
    c = FakeClient([json.dumps(GOOD)])
    ms = compile_mission("pick up the pizza box", "trash", TRASH, client=c)
    assert not ms.offline and ms.anchors["a0"] == "item_2.handle" and ms.text == "pick up the pizza box"
    p = c.models.prompts[0]
    assert "item_2.handle" in p and "obstacle_dist" in p and "pick up the pizza box" in p


def test_compile_retry_on_bad_dsl():
    bad = dict(GOOD, reward="__import__('os').system('ls')")
    c = FakeClient(["```json\n" + json.dumps(bad) + "\n```", json.dumps(GOOD)])
    ms = compile_mission("pick up the pizza box", "trash", TRASH, client=c)
    assert not ms.offline and ms.reward == GOOD["reward"]
    assert len(c.models.prompts) == 2 and "rejected" in c.models.prompts[1]


def test_compile_retry_on_bad_item_type():
    bad = dict(GOOD, reward="payload_is('pizza')")
    c = FakeClient([json.dumps(bad), json.dumps(GOOD)])
    ms = compile_mission("pick up the pizza box", "trash", TRASH, client=c)
    assert not ms.offline and "unknown item type" in c.models.prompts[1]


def test_compile_retry_on_unknown_entity():
    bad = dict(GOOD, anchors={"a0": "payload.handle"})
    c = FakeClient([json.dumps(bad), "not json at all", json.dumps(GOOD)])
    ms = compile_mission("pick up the pizza box", "trash", TRASH, client=c)
    assert not ms.offline
    assert "payload.handle" in c.models.prompts[1] and "JSON" in c.models.prompts[2]


def test_compile_falls_back_offline():
    c = FakeClient([json.dumps(dict(GOOD, reward="nope("))] * 3)
    ms = compile_mission("pick up the pizza box", "trash", TRASH, client=c, max_retries=3)
    assert ms.offline and ms.warnings and ms.anchors["a0"] == "item_2.handle"
    with pytest.raises(ValueError):
        compile_mission("x", "trash", TRASH, client=FakeClient(["{}"] * 2), max_retries=2, fallback=False)


def test_compile_no_key_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ms = compile_mission("pick up the pizza box", "trash", TRASH)
    assert ms.offline


# ------------------------------------------------------------------ offline compiler
def _check_offline(scene, ents, text):
    env = FakeEnv(scene, ents)
    ms = offline_compile(text, scene, env.entity_descriptions())
    assert ms.offline and ms.explanation and ms.scene == scene
    g = torch.randn(8, spec.G_DIM) * 3
    exprs = [ms.reward, ms.done_when] + [e for st in ms.steps or [] for e in (st.reward, st.done_when)]
    for e in exprs:
        if e is not None:
            out = compile_reward(e)(g)
            assert out.shape == (8,) and torch.isfinite(out).all()
    bind_mission(env, ms)
    for i in range(len(ms.steps or [])):
        bind_mission(env, ms, step=i)
    for ag in range(env.num_agents):
        env.bind_anchors(torch.arange(env.num_envs), to_bindings(ms, agent=ag, entities=env.entity_names()), agent=ag)
    obs = env.step(torch.zeros(env.num_envs, env.num_agents, spec.ACTION_DIM)).obs
    assert torch.isfinite(compile_reward(ms.reward)(obs["g"])).all()
    return ms, env


@pytest.mark.parametrize("text,item,typ", [
    ("pick up the pizza box and put it in the garbage truck", "item_2", "pizza_box"),
    ("grab the bottle and throw it in the truck", "item_4", "bottle"),
    ("pick up the trash bag", "item_0", "trash_bag"),
    ("pick up some trash", "item_2", "pizza_box"),            # nearest trash; the nearer backpack is skipped
    ("bring item 3 to the truck", "item_3", "pizza_box"),
])
def test_offline_trash_single(text, item, typ):
    ms, env = _check_offline("trash", TRASH, text)
    assert ms.anchors["a0"] == f"{item}.handle" and ms.anchors["a1"] == "bin"
    assert f"payload_is('{typ}')" in ms.reward and "obstacle_dist" in ms.reward
    assert ms.done_when == "distxy(payload, a1) < 1.0 and not holding"
    assert len(ms.steps) == 2 and ms.steps[0].done_when == "holding"
    assert env.bound[0][0].entity == f"{item}.handle" and env.bound[0][0].kind == "handle"
    assert env.bound[0][1].kind == "zone" and "garbage truck" in ms.explanation


def test_offline_trash_explicit_non_trash_and_missing_type():
    ms, _ = _check_offline("trash", TRASH, "pick up the backpack and drop it in the truck")
    assert ms.anchors["a0"] == "item_1.handle" and "payload_is('backpack')" in ms.reward
    ms, _ = _check_offline("trash", TRASH, "pick up the cardboard box")   # only an inactive one exists
    assert ms.anchors["a0"] == "item_2.handle" and ms.warnings
    ms, _ = _check_offline("trash", TRASH, "pick up item 6")               # inactive slot
    assert ms.anchors["a0"] != "item_6.handle" and ms.warnings


def test_offline_trash_avoid_carts_and_nofly():
    ms, _ = _check_offline("trash", TRASH, "grab the bag and drop it in the bin without getting near the moving carts")
    assert ms.anchors["a0"] == "item_0.handle" and "2*exp(-2*obstacle_dist)" in ms.reward and ms.steps
    ms, env = _check_offline("trash", TRASH, "pick up the pizza box, stay out of the no-fly zone")
    assert ms.anchors["a2"] == "nofly_0" and "5*inside(a2)" in ms.reward
    assert all("5*inside(a2)" in st.reward for st in ms.steps)


@pytest.mark.parametrize("text", ["clear the trash pile", "pick up all the trash", "clean up the street"])
def test_offline_pile_multi_step(text):
    ms, env = _check_offline("trash", TRASH, text)
    order = [st.anchors["a0"] for st in ms.steps]
    # trash only (backpack item_1, flower_pot item_5 and inactive item_6..9 skipped), nearest first:
    # item_2 is nearest the drone, then by distance to the truck: item_3, item_0, item_4
    assert order == ["item_2.handle", "item_3.handle", "item_0.handle", "item_4.handle"]
    assert all(st.anchors["a1"] == "bin" for st in ms.steps)
    assert all("payload_is_trash" in st.reward and st.done_when == "distxy(payload, a1) < 1.0 and not holding"
               for st in ms.steps)
    assert ms.anchors["a0"] == "item_2.handle" and ms.done_when == "trash_left < 0.01" and "trash_left" in ms.reward
    assert "backpack" in ms.explanation and "flower pot" in ms.explanation
    for i, item in enumerate(order):
        bind_mission(env, ms, step=i)
        assert env.bound[0][0].entity == item and env.bound[0][0].kind == "handle"


def test_offline_pile_listing_order_without_positions():
    ms, _ = _check_offline("trash", trash_no_positions(), "pick up all the trash")
    assert [st.anchors["a0"] for st in ms.steps] == ["item_0.handle", "item_2.handle", "item_4.handle"]
    assert "trash bin" in ms.explanation


def test_offline_pile_by_type_and_avoid():
    ms, _ = _check_offline("trash", TRASH, "pick up all the pizza boxes and avoid the no-fly zone")
    assert [st.anchors["a0"] for st in ms.steps] == ["item_2.handle", "item_3.handle"]
    assert all(st.anchors["a2"] == "nofly_0" and "5*inside(a2)" in st.reward for st in ms.steps)
    assert "payload_is('pizza_box')" in ms.reward and ms.done_when is None


def test_offline_pile_nothing_left():
    ents = {k: ("(inactive)" if k.startswith("item_") else v) for k, v in TRASH.items()}
    ms, _ = _check_offline("trash", ents, "clear the pile")
    assert not ms.steps and ms.warnings and "hover" in ms.explanation.lower()


@pytest.mark.parametrize("text", ["deliver the organ cooler gently to the hospital",
                                  "pick up the cooler and bring it to the drop-off pad",
                                  "rush the donor heart over"])
def test_offline_delivery(text):
    ms, env = _check_offline("delivery", DELIV, text)
    assert ms.anchors == {"a0": "item_0.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"}
    for term in ("swing", "jerk", "tilt", "5*inside(a2)", "5*inside(a3)", "height < 2", "over_water"):
        assert term in ms.reward, term
    assert ms.done_when == "distxy(payload, a1) < 1.0 and not holding"
    assert len(ms.steps) == 2 and "swing" in ms.steps[1].reward and "inside(a3)" in ms.steps[0].reward
    kinds = [(b.entity, b.kind) for b in env.bound[0]]
    assert kinds == [("item_0.handle", "handle"), ("pad_dropoff", "zone"), ("nofly_0", "zone"), ("nofly_1", "zone")]


def test_offline_delivery_goto_pads():
    ms, _ = _check_offline("delivery", DELIV, "fly to the hospital")
    assert ms.anchors["a0"] == "pad_dropoff" and "inside(a0)" in ms.done_when
    ms, _ = _check_offline("delivery", DELIV, "go back to the pickup pad")
    assert ms.anchors["a0"] == "pad_pickup"


def test_offline_ocean_bycatch():
    ms, env = _check_offline("ocean", OCEAN, "net the school closest to the boat and keep the dolphins out of the net")
    assert ms.anchors["a0"] == "net.corner_{agent}" and ms.anchors["a1"] == "boat"
    assert ms.anchors["a2"] == "fish_school_1"
    assert "bycatch" in ms.reward and "animal_dist" in ms.reward and "not bycatch" in ms.done_when
    assert all("bycatch" in st.reward for st in ms.steps[1:])
    assert [env.bound[ag][0].entity for ag in range(3)] == ["net.corner_0", "net.corner_1", "net.corner_2"]
    assert all(env.bound[ag][0].kind == "handle" for ag in range(3))


def test_offline_ocean_avoid_specific_animal():
    ms, _ = _check_offline("ocean", OCEAN, "haul the net to the boat, avoid the whale")
    assert ms.anchors["a3"] == "whale_0" and "3*exp(-dist(net, a3))" in ms.reward
    assert "10*bycatch" in ms.reward
    ms, _ = _check_offline("ocean", OCEAN, "trawl the fish but don't catch any animals")
    assert ms.anchors["a3"] is None and "10*bycatch" in ms.reward


def test_offline_goto_avoid():
    ms, _ = _check_offline("trash", TRASH, "fly to obstacle 3 and avoid the no-fly zone")
    assert ms.anchors["a0"] == "obstacle_3" and "nofly_0" in ms.anchors.values()
    ms, _ = _check_offline("ocean", OCEAN, "go to the boat")
    assert ms.anchors["a0"] == "boat"
    ms, _ = _check_offline("ocean", OCEAN, "fly to the whale")
    assert ms.anchors["a0"] == "whale_0"
    ms, _ = _check_offline("trash", TRASH, "fly to the garbage truck")
    assert ms.anchors["a0"] == "bin" and "inside(a0)" in ms.done_when
    ms, _ = _check_offline("trash", TRASH, "fly over the flower pot")
    assert ms.anchors["a0"] == "item_5"
    ms, _ = _check_offline("trash", TRASH, "go to (2, -3) at 4 m")
    assert ms.anchors["a0"] == [2.0, -3.0, 4.0]
    ms, _ = _check_offline("delivery", DELIV, "do a backflip")
    assert "not understood" in ms.explanation
