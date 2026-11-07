"""Adapter between the demo server and the (possibly missing) mission / training packages.

Everything external is imported lazily and every piece has a labelled fallback,
so the demo always runs and never misrepresents what is flying:

  policy:  "trained policy"            skyhand.train load(path, device) -> agent.act(obs, z)
           "scripted fallback (env)"   env.expert_action()
           "scripted fallback"         ScriptedController below (fly to a0, grip, fly to a1, drop)
  mission: "Gemini" / "offline parser"  skyhand.mission compile_mission / offline_compile
           "keyword parser"            keyword_compile below
  z:       reward_to_z from skyhand.mission, else the same formula implemented here.

Tests inject fakes through Pipeline(agent=..., mission_api=...).
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import re
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skyhand import spec
from skyhand.spec import AnchorBinding

from . import rewardlang

log = logging.getLogger("skyhand.demo")

POLICY_TRAINED = "trained policy"
POLICY_ENV_EXPERT = "scripted fallback (env expert)"
POLICY_SCRIPTED = "scripted fallback"
MISSION_GEMINI = "Gemini"
MISSION_OFFLINE = "offline parser"
MISSION_KEYWORD = "keyword parser"


# ------------------------------------------------------------------ mission
@dataclass
class CompiledMission:
    text: str
    scene: str
    anchors: Dict[str, Any]                  # "a0" -> entity name | [x, y, z] | None
    reward: str
    done_when: Optional[str]
    explanation: List[str]
    source: str                              # MISSION_* label
    reward_fn: Callable[[torch.Tensor], torch.Tensor] = field(repr=False, default=None)
    done_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = field(repr=False, default=None)
    bindings: List[AnchorBinding] = field(repr=False, default_factory=list)
    raw: Any = field(repr=False, default=None)
    per_agent_bindings: Optional[Dict[int, List[AnchorBinding]]] = field(repr=False, default=None)
    warnings: List[str] = field(default_factory=list)
    # per-drone jobs (world / multi-drone missions where drones do different things):
    # {drone: {"task": str, "anchors": {...}, "reward": str, "done_when": str | None}}; None = one shared mission
    agents: Optional[Dict[int, Dict[str, Any]]] = None
    agent_reward_fns: Optional[Dict[int, Callable]] = field(repr=False, default=None)
    agent_done_fns: Optional[Dict[int, Optional[Callable]]] = field(repr=False, default=None)

    def public(self, descriptions: Optional[Dict[str, str]] = None) -> dict:
        descriptions = descriptions or {}
        out = {"text": self.text, "scene": self.scene, "anchors": _public_anchors(self.anchors, descriptions),
               "reward": self.reward, "done_when": self.done_when, "explanation": self.explanation,
               "source": self.source, "warnings": self.warnings, "agents": None}
        if self.agents:
            out["agents"] = {str(ag): {"task": j.get("task", ""), "reward": j.get("reward", ""),
                                       "done_when": j.get("done_when"),
                                       "anchors": _public_anchors(j.get("anchors") or {}, descriptions, agent=ag)}
                             for ag, j in sorted(self.agents.items())}
        return out

    def tasks(self, n_agents: int) -> List[str]:
        """Short per-drone task labels for the viewer's drone picker."""
        if self.agents:
            return [str(self.agents.get(ag, {}).get("task", "")) for ag in range(n_agents)]
        return [self.text] * n_agents


def _public_anchors(anchors: Dict[str, Any], descriptions: Dict[str, str], agent: Optional[int] = None) -> dict:
    out = {}
    for k, v in anchors.items():
        if v is None:
            continue
        if isinstance(v, str) and "{agent}" in v and agent is not None:
            v = v.replace("{agent}", str(agent))
        if isinstance(v, str) and "{agent}" in v:
            d0 = descriptions.get(v.replace("{agent}", "0"), "")
            desc = f"each drone its own ({v.replace('{agent}', 'i')} for drone i); drone 0: {d0}"
        elif isinstance(v, str):
            desc = descriptions.get(v, "")
        else:
            desc = f"fixed point {[round(float(x), 2) for x in v]}"
        out[k] = {"target": v if isinstance(v, str) else [float(x) for x in v], "description": desc}
    return out


def _binding_for(target, entity_kinds: Dict[str, str]) -> AnchorBinding:
    if target is None:
        return AnchorBinding(kind="none")
    if isinstance(target, str):
        return AnchorBinding(kind=entity_kinds.get(target, infer_kind(target)), entity=target)
    return AnchorBinding(kind="point", point=[float(x) for x in target])


def bindings_from_anchors(anchors: Dict[str, Any], entity_kinds: Dict[str, str]) -> List[AnchorBinding]:
    return [_binding_for(anchors.get(f"a{k}"), entity_kinds) for k in range(spec.N_ANCHORS)]


ZONE_ENTITIES = ("bin", "boat", "pad_pickup", "pad_dropoff")


def infer_kind(name: str) -> str:
    """Anchor kind from an entity name (used when the env does not expose its own kinds)."""
    if name.endswith(".handle") or name.startswith("net.corner_"):
        return "handle"
    if name in ZONE_ENTITIES or name.startswith("nofly_"):
        return "zone"
    return "point"


def entity_kinds(env) -> Dict[str, str]:
    names = list(env.entity_names())
    try:
        return {n: spec.ANCHOR_TYPES[int(k)] for n, k in zip(names, env._ent_kind.tolist())}
    except Exception:
        return {n: infer_kind(n) for n in names}


def observe(env) -> Dict[str, torch.Tensor]:
    """Current obs without stepping (public observe(), else the legacy _obs())."""
    fn = getattr(env, "observe", None) or getattr(env, "_obs")
    return fn()


def _has(text: str, *words) -> bool:
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


_POS_RE = re.compile(r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def _positions(env, names: Sequence[str]) -> Dict[str, Any]:
    """World-0 xy of each name: env._entity_positions() if present, else parsed from descriptions."""
    out: Dict[str, Any] = {}
    try:
        pos = env._entity_positions()[0]
        all_names = list(env.entity_names())
        for n in names:
            out[n] = [float(v) for v in pos[all_names.index(n)][:2]]
        return out
    except Exception:
        pass
    try:
        desc = env.entity_descriptions()
    except Exception:
        return out
    for n in names:
        m = _POS_RE.search(str(desc.get(n, "")))
        if m:
            out[n] = [float(m.group(1)), float(m.group(2))]
    return out


def _closest(env, names: Sequence[str], to: str) -> str:
    names = list(names)
    pos = _positions(env, names + [to])
    if to not in pos:
        return names[0]
    p0 = pos[to]
    return min(names, key=lambda n: math.hypot(pos[n][0] - p0[0], pos[n][1] - p0[1]) if n in pos else 1e9)


def _active(desc: Dict[str, str], name: str, names: Sequence[str]) -> bool:
    return name in names and "inactive" not in str(desc.get(name, "")).lower()


def _nofly_active(env, desc: Dict[str, str], k: int) -> bool:
    names = env.entity_names()
    if f"nofly_{k}" not in names:
        return False
    if f"nofly_{k}" in desc:
        return _active(desc, f"nofly_{k}", names)
    try:
        return bool(env.zone_active[0, 1 + k])
    except Exception:
        return True


# words -> item type (checked in order: "pizza box" before "box")
ITEM_WORDS = (
    ("pizza_box", ("pizza",)),
    ("trash_bag", ("trash bag", "garbage bag", "bin bag", "bag")),
    ("cardboard_box", ("cardboard", "box", "carton")),
    ("bottle", ("bottle",)),
    ("backpack", ("backpack", "rucksack")),
    ("ball", ("ball",)),
    ("flower_pot", ("flower", "pot", "plant")),
    ("cooler", ("cooler", "organ", "heart", "kidney", "liver", "medical", "medicine")),
)
TRASH_WORDS = ("trash", "litter", "garbage", "rubbish", "junk", "waste")
ANIMAL_WORDS = ("whale", "dolphin", "turtle", "animal", "wildlife", "marine life", "bycatch", "creature")


def item_types(env) -> Dict[str, str]:
    """Active item_i -> type for world 0, from entity_descriptions() (fallback: render_state items)."""
    names = [n for n in env.entity_names() if re.fullmatch(r"item_\d+", n)]
    out: Dict[str, str] = {}
    try:
        desc = env.entity_descriptions()
    except Exception:
        desc = {}
    for n in names:
        d = str(desc.get(n, "")).lower()
        if "inactive" in d:
            continue
        for t in sorted(spec.ITEM_TYPES, key=len, reverse=True):
            if t in d or t.replace("_", " ") in d:
                out[n] = t
                break
    if out:
        return out
    try:
        st = env.render_state(0)
        return {it["name"]: it["type"] for it in st.get("items", []) if not it.get("delivered")}
    except Exception:
        return {n: "" for n in names}


def _pick_item(t: str, env, ref: str, exclude: Sequence[str] = (), strict: bool = False) -> Optional[str]:
    """Which item_i the text refers to: explicit index > type word > generic trash > nearest trash.
    exclude: items already taken by teammates. strict: None (instead of another item) when the
    named item / type has no free candidate."""
    types = {n: ty for n, ty in item_types(env).items() if n not in exclude}
    if not types:
        return None if strict else "item_0"
    m = re.search(r"\bitem[\s_#]*(\d+)", t)
    if m:
        if f"item_{m.group(1)}" in types:
            return f"item_{m.group(1)}"
        if strict:
            return None
    want = None
    for typ, words in ITEM_WORDS:
        if _has(t, *words):
            want = typ
            break
    if want is not None:
        cands = [n for n, ty in types.items() if ty == want]
        if cands:
            return _closest(env, cands, ref)
        if strict:
            return None
    trash = [n for n, ty in types.items() if ty in spec.ITEM_TYPES and spec.ITEM_IS_TRASH[spec.ITEM_TYPES.index(ty)]]
    if trash:
        return _closest(env, trash, ref)
    if strict and _has(t, *TRASH_WORDS):
        return None
    return sorted(types)[0]


def _is_trash(env, item: str) -> bool:
    ty = item_types(env).get(item, "")
    return ty in spec.ITEM_TYPES and spec.ITEM_IS_TRASH[spec.ITEM_TYPES.index(ty)]


CARRY = ["-dist(drone, a0)*(1 - holding)", "-dist(payload, a1)", "2*holding"]
CARRY_DONE = "not holding and distxy(payload, a1) < 0.8"    # the drop-off point may be under a truck roof
GENTLE = ["-0.5*swing", "-0.01*jerk"]


def keyword_compile(text: str, scene: str, env) -> CompiledMission:
    """Tiny keyword parser (only used when skyhand.mission is unavailable): one of a few
    templates per scene plus modifiers."""
    if scene == "world":
        return world_compile(text, env)
    t = text.lower()
    names = list(env.entity_names())
    desc = env.entity_descriptions()
    anchors: Dict[str, Any] = {f"a{k}": None for k in range(spec.N_ANCHORS)}
    terms: List[str] = []
    done = None
    me = "drone_0" if "drone_0" in names else names[-1]
    carry_words = ("grab", "pick", "carry", "drop", "deliver", "bring", "take", "move", "collect", "clean", "fetch",
                   "transport", "put", "throw")
    t_verbs = re.sub(r"pick[- ]?up (pad|zone|point|helipad|site)|drop[- ]?off (pad|zone|point|helipad|site)", " ", t)
    reach_only = _has(t, "go to", "fly to", "hover", "visit", "touch", "land", "check on", "inspect") and \
        not _has(t_verbs, *carry_words)
    gentle = _has(t, "smooth", "gentle", "careful", "fragile", "steady", "softly", "no swing")
    avoid = _has(t, "avoid", "obstacle", "car", "traffic", "moving", "safe", "without hitting", "bird", "tree",
                 "building", "collide")
    no_fly = _has(t, "no-fly", "nofly", "no fly", "restricted", "airspace", "hospital")
    if scene in ("trash", "delivery"):
        drop = "bin" if scene == "trash" else "pad_dropoff"
        if reach_only:
            if scene == "delivery" and _has(t, "pickup", "pick-up", "start", "origin", "first pad"):
                target = "pad_pickup"
            elif scene == "delivery" and _has(t, "dropoff", "drop-off", "destination", "hospital", "pad", "helipad"):
                target = "pad_dropoff"
            elif _has(t, "bin", "truck", "dumpster"):
                target = "bin" if "bin" in names else drop
            elif _has(t, "no-fly", "nofly", "no fly"):
                target = "nofly_0"
            elif _has(t, "item", *TRASH_WORDS, *(w for _, ws in ITEM_WORDS for w in ws)):
                target = _pick_item(t, env, me) + ".handle"
            else:
                target = drop
            anchors["a0"] = target
            terms.append("-dist(drone, a0)")
            # zones (bin / truck, pads) may sit inside a truck or on a slope: being over them counts
            done = "inside(a0)" if infer_kind(target) == "zone" and target != "nofly_0" else "dist(drone, a0) < 1.0"
            no_fly = no_fly and target != "nofly_0"
        else:
            item = _pick_item(t, env, me) if scene == "trash" else "item_0"
            anchors["a0"], anchors["a1"] = item + ".handle", drop
            terms += CARRY
            done = CARRY_DONE
            if scene == "trash" and _is_trash(env, item) and \
                    _has(t, *TRASH_WORDS, "only", "leave", "ignore", "not the", "don't", "dont"):
                terms.append("-3*holding*(1 - payload_is_trash)")
            if scene == "delivery":
                gentle = gentle or _has(t, "organ", "cooler", "heart", "kidney", "medical")
                no_fly = no_fly or _has(t, "organ", "safely")
        if gentle:
            terms += GENTLE
        free = [a for a in ("a2", "a3") if anchors[a] is None]
        if no_fly:
            for k in range(2):
                if free and _nofly_active(env, desc, k):
                    a = free.pop(0)
                    anchors[a] = f"nofly_{k}"
                    terms.append(f"-5*inside({a})")
        if avoid:
            terms.append("-2*(obstacle_dist < 1.0)")
        if scene == "delivery" and _has(t, "low", "below the clouds", "terrain"):
            terms.append("-0.2*abs(height - 3)")
    else:
        animals = [n for n in names if n.split("_")[0] in spec.ANIMAL_TYPES]
        schools = [n for n in names if n.startswith("fish_school")]
        if reach_only and not _has(t, "net", "fish", "haul", "catch"):
            want = next((a for a in spec.ANIMAL_TYPES if _has(t, a)), None)
            if want and any(n.startswith(want + "_") for n in animals):
                anchors["a0"] = _closest(env, [n for n in animals if n.startswith(want + "_")], me)
            elif _has(t, "boat"):
                anchors["a0"] = "boat"
            else:
                anchors["a0"] = _closest(env, schools, "boat") if schools else "boat"
            terms.append("-dist(drone, a0)")
            done = "dist(drone, a0) < 1.5"
        else:
            m = re.search(r"school\s*(\d)", t)
            school = f"fish_school_{m.group(1)}" if m and f"fish_school_{m.group(1)}" in names else \
                _closest(env, schools, "boat") if (_has(t, "closest", "nearest") and schools) else \
                (schools[0] if schools else "boat")
            anchors["a0"], anchors["a1"], anchors["a2"] = "net.corner_0", "boat", school
            terms += ["-dist(drone, a0)*(1 - holding)", "2*holding", "-0.5*dist(payload, a2)*holding",
                      "3*fish_in_net", "-dist(payload, a1)*fish_in_net"]
            done = "fish_in_net > 0.2 and dist(payload, a1) < 3.0"
            if _has(t, *ANIMAL_WORDS, "careful", "safe"):
                terms += ["-5*bycatch", "-(animal_dist < 2.0)"]
        if _has(t, "gentle", "careful", "tear", "smooth"):
            terms.append("-2*net_stretch")
        if no_fly and _nofly_active(env, desc, 0):
            anchors["a3"] = "nofly_0"
            terms.append("-5*inside(a3)")
        if _has(t, "avoid", "collide", "safe", "teammate"):
            terms.append("-2*(mate_dist < 1.0)")
    reward = _join_terms(terms)
    return _finish(text, scene, anchors, reward, done, None, MISSION_KEYWORD, env)


# ------------------------------------------------------------------ world: one job per drone
DRONE_RE = re.compile(r"\b(?:drone|uav|quad(?:copter)?)\s*#?\s*(\d)\b")
OTHERS_RE = re.compile(r"\b(?:the\s+)?(?:other|remaining)\s+(?:two\s+|2\s+)?drones?\b|\bthe\s+(?:others|rest)\b|"
                       r"\beveryone else\b")
_LEAD_RE = re.compile(r"^(?:\s|[,;:.\-])*(?:(?:should|will|must|can|has to|is to|then|and|please)\b\s*)*")
_TRAIL_RE = re.compile(r"(?:\s|[,;:.\-])*(?:\b(?:and|while|then)\b(?:\s|[,;:.\-])*)*$")


def _clean_clause(s: str) -> str:
    return _TRAIL_RE.sub("", _LEAD_RE.sub("", s)).strip()


def split_by_drone(text: str, n_agents: int) -> Tuple[str, Dict[int, str]]:
    """("drone 0 grab the bottle, drone 1 fly to the boat, the others hover") ->
    (text for the drones not named, {drone: its own text}). No drone named -> (text, {})."""
    t = text.lower()
    marks = [(m.start(), m.end(), int(m.group(1))) for m in DRONE_RE.finditer(t) if int(m.group(1)) < n_agents]
    if not marks:
        return t.strip(), {}
    shared = t[:marks[0][0]]
    per: Dict[int, str] = {}
    for j, (_, e, ag) in enumerate(marks):
        stop = marks[j + 1][0] if j + 1 < len(marks) else len(t)
        clause = t[e:stop]
        m = OTHERS_RE.search(clause)
        if m:
            shared += " " + clause[m.end():]
            clause = clause[:m.start()]
        per[ag] = (per.get(ag, "") + " " + _clean_clause(clause)).strip()
    return _clean_clause(shared), per


def _world_target(t: str, env, names: Sequence[str], me: str, exclude: Sequence[str]) -> Optional[str]:
    """Reach target named in a world clause."""
    animals = [n for n in names if n.split("_")[0] in spec.ANIMAL_TYPES]
    want = next((a for a in spec.ANIMAL_TYPES if _has(t, a)), None)
    if want and any(n.startswith(want + "_") for n in animals):
        return _closest(env, [n for n in animals if n.startswith(want + "_")], me)
    if _has(t, "pickup", "pick-up", "first pad", "origin clinic"):
        return "pad_pickup"
    if _has(t, "dropoff", "drop-off", "hospital", "clinic", "helipad", "pad"):
        return "pad_dropoff"
    if _has(t, "boat", "ship", "deck"):
        return "boat"
    if _has(t, "fish", "school"):
        schools = [n for n in names if n.startswith("fish_school")]
        return _closest(env, schools, me) if schools else None
    if _has(t, "net"):
        return "net.center" if "net.center" in names else None
    if _has(t, "bin", "truck", "dumpster", "garbage truck"):
        return "bin"
    if _has(t, "no-fly", "nofly", "no fly"):
        return "nofly_0"
    if _has(t, "item", *TRASH_WORDS, *(w for _, ws in ITEM_WORDS for w in ws)):
        it = _pick_item(t, env, me, exclude=exclude, strict=True)
        return it + ".handle" if it else None
    return None


def _world_job(t: str, env, names: Sequence[str], desc: Dict[str, str], agent: int, taken: List[str],
               mods: Dict[str, bool]) -> Dict[str, Any]:
    """One drone's job in the world scene: {"task", "anchors", "reward", "done_when"}."""
    me = f"drone_{agent}" if f"drone_{agent}" in names else names[-1]
    anchors: Dict[str, Any] = {f"a{k}": None for k in range(spec.N_ANCHORS)}
    terms: List[str] = []
    done = None
    carry_words = ("grab", "pick", "carry", "drop", "deliver", "bring", "take", "move", "collect", "clean", "fetch",
                   "transport", "put", "throw")
    t_verbs = re.sub(r"pick[- ]?up (pad|zone|point|helipad|site)|drop[- ]?off (pad|zone|point|helipad|site)", " ", t)
    reach_only = _has(t, "go to", "fly to", "hover", "visit", "touch", "land", "check on", "inspect", "scout",
                      "patrol") and not _has(t_verbs, *carry_words)
    if not t.strip() or _has(t, "stand by", "standby", "wait", "stay", "hold position") and not \
            _has(t_verbs, *carry_words):
        return {"task": "stand by (hover)", "anchors": anchors, "reward": "-abs(height - 3) - 0.1*speed",
                "done_when": None}
    if not reach_only and _has(t, "net", "haul", "trawl", "catch", "fish"):
        schools = [n for n in names if n.startswith("fish_school")]
        m = re.search(r"school\s*(\d)", t)
        school = f"fish_school_{m.group(1)}" if m and f"fish_school_{m.group(1)}" in names else \
            (_closest(env, schools, "boat") if schools else None)
        anchors.update({"a0": "net.corner_{agent}", "a1": "boat", "a2": school})
        terms += ["-dist(drone, a0)*(1 - holding)", "2*holding", "-0.5*dist(payload, a2)*holding",
                  "3*fish_in_net", "-dist(payload, a1)*fish_in_net"]
        done = "fish_in_net > 0.2 and dist(payload, a1) < 3.0"
        if mods["careful"] or _has(t, *ANIMAL_WORDS):
            terms += ["-5*bycatch", "-(animal_dist < 2.0)"]
        if mods["gentle"]:
            terms.append("-2*net_stretch")
        task = f"haul the net (corner {agent}) over {school or 'the fish'} to the boat"
    elif reach_only or not _has(t_verbs, *carry_words, "item", *TRASH_WORDS):
        target = _world_target(t, env, names, me, taken)
        if target is None:
            return {"task": "stand by (hover): no target recognised", "anchors": anchors,
                    "reward": "-abs(height - 3) - 0.1*speed", "done_when": None}
        anchors["a0"] = target
        terms.append("-dist(drone, a0)")
        kind = infer_kind(target)
        done = "inside(a0)" if kind == "zone" and not target.startswith("nofly") else \
            "dist(drone, a0) < 1.5" if target.split("_")[0] in spec.ANIMAL_TYPES + ("fish",) else "dist(drone, a0) < 1.0"
        if target.endswith(".handle"):
            taken.append(target.split(".")[0])
        task = f"fly to {target}"
    else:
        item = _pick_item(t, env, me, exclude=taken, strict=True)
        if item is None:
            return {"task": "stand by (hover): no free item of that kind", "anchors": anchors,
                    "reward": "-abs(height - 3) - 0.1*speed", "done_when": None}
        taken.append(item)
        ty = item_types(env).get(item, "")
        if _has(t, "boat", "ship", "deck"):
            drop = "boat"
        elif ty == "cooler" or _has(t, "pad", "clinic", "hospital", "helipad", "organ"):
            drop = "pad_dropoff"
        else:
            drop = "bin"
        anchors["a0"], anchors["a1"] = item + ".handle", drop
        terms += CARRY
        done = CARRY_DONE
        if _is_trash(env, item) and _has(t, "only", "leave", "ignore", "not the", "don't", "dont"):
            terms.append("-3*holding*(1 - payload_is_trash)")
        if mods["gentle"] or ty == "cooler":
            terms += GENTLE
        task = f"carry {item} ({ITEM_LABEL.get(ty, ty or 'item')}) to {drop}"
    free = [a for a in ("a2", "a3") if anchors[a] is None]
    if mods["no_fly"] or (anchors["a1"] == "pad_dropoff"):
        for k in range(2):
            if free and _nofly_active(env, desc, k):
                a = free.pop(0)
                anchors[a] = f"nofly_{k}"
                terms.append(f"-5*inside({a})")
    if mods["avoid"]:
        terms.append("-2*(obstacle_dist < 1.0)")
    if mods["mates"]:
        terms.append("-2*(mate_dist < 1.0)")
    return {"task": task, "anchors": anchors, "reward": _join_terms(terms), "done_when": done}


ITEM_LABEL = {t: t.replace("_", " ") for t in spec.ITEM_TYPES}
ITEM_LABEL["cooler"] = "organ cooler"


def _subst_agent(anchors: Dict[str, Any], agent: int) -> Dict[str, Any]:
    return {k: (v.replace("{agent}", str(agent)) if isinstance(v, str) else v) for k, v in anchors.items()}


def world_compile(text: str, env) -> CompiledMission:
    """Keyword parser for the world scene. "drone 0 ..., drone 1 ..." gives each drone its own job
    (drones not named share the rest of the text, or stand by); one shared carry text sends every
    drone to a different item of that kind."""
    t = text.lower()
    names = list(env.entity_names())
    desc = env.entity_descriptions()
    n = int(env.num_agents)
    shared, per = split_by_drone(t, n)
    mods = {"gentle": _has(t, "smooth", "gentle", "careful", "fragile", "steady", "softly", "no swing", "tear"),
            "avoid": _has(t, "avoid", "obstacle", "car", "traffic", "moving", "without hitting", "bird", "tree",
                          "building", "collide"),
            "no_fly": _has(t, "no-fly", "nofly", "no fly", "restricted", "airspace"),
            "careful": _has(t, "careful", "safe", *ANIMAL_WORDS),
            "mates": n > 1 and _has(t, "teammate", "each other", "collide", "safe")}
    taken: List[str] = []
    jobs = {ag: _world_job(per[ag] if ag in per else shared, env, names, desc, ag, taken, mods) for ag in range(n)}
    j0 = jobs[0]
    same = all(j["anchors"] == j0["anchors"] and j["reward"] == j0["reward"] and j["done_when"] == j0["done_when"]
               for j in jobs.values())
    out = _finish(text, "world", dict(j0["anchors"]), j0["reward"], j0["done_when"], None, MISSION_KEYWORD, env)
    kinds = entity_kinds(env)
    if same and n > 0:
        out.per_agent_bindings = {ag: bindings_from_anchors(_subst_agent(j0["anchors"], ag), kinds) for ag in range(n)}
        out.explanation = [f"Every drone: {j0['task']}"] + out.explanation
        return out
    out.agents = jobs
    out.per_agent_bindings = {ag: bindings_from_anchors(_subst_agent(j["anchors"], ag), kinds) for ag, j in jobs.items()}
    out.agent_reward_fns = {ag: rewardlang.compile_expr(j["reward"]) for ag, j in jobs.items()}
    out.agent_done_fns = {ag: (rewardlang.compile_expr(j["done_when"]) if j["done_when"] else None)
                          for ag, j in jobs.items()}
    out.explanation = [f"Drone {ag}: {j['task']}" for ag, j in sorted(jobs.items())] + \
        ["Each drone is scored on its own reward program (below); the mission succeeds when every drone with a "
         "goal has reached it."]
    return out


def _join_terms(terms: Sequence[str]) -> str:
    out = ""
    for t in terms:
        t = t.strip()
        if not out:
            out = t
        elif t.startswith("-"):
            out += " - " + t[1:].strip()
        else:
            out += " + " + t
    return out or "0"


def _finish(text, scene, anchors, reward, done, explanation, source, env, reward_fn=None, done_fn=None,
            bindings=None, raw=None) -> CompiledMission:
    shown = {k: (v if isinstance(v, str) else "fixed point") for k, v in anchors.items() if v is not None}
    expl = rewardlang.explain(reward, done, shown, strict=bool(explanation))
    if explanation:
        expl = ([explanation] if isinstance(explanation, str) else list(explanation)) + expl
    return CompiledMission(
        text=text, scene=scene, anchors=anchors, reward=reward, done_when=done, explanation=expl, source=source,
        reward_fn=reward_fn or rewardlang.compile_expr(reward),
        done_fn=done_fn if done_fn is not None else (rewardlang.compile_expr(done) if done else None),
        bindings=bindings or bindings_from_anchors(anchors, entity_kinds(env)), raw=raw)


# ------------------------------------------------------------------ external APIs
def _import_attr(names: Sequence[str], modules: Sequence[str]) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for mod in modules:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in names:
            if n not in found and callable(getattr(m, n, None)):
                found[n] = getattr(m, n)
    return found


MISSION_NAMES = ("compile_mission", "offline_compile", "compile_reward", "to_bindings", "reward_to_z", "cem_refine")


def load_mission_api() -> Dict[str, Any]:
    return _import_attr(MISSION_NAMES, ["skyhand.mission", "skyhand.mission.compile", "skyhand.mission.compiler",
                                        "skyhand.mission.gemini", "skyhand.mission.reward", "skyhand.mission.language",
                                        "skyhand.mission.ztask", "skyhand.mission.zinfer", "skyhand.mission.cem"])


def load_agent(path: Optional[str], device: str = "cpu"):
    """Return the trained agent or None (with a logged reason)."""
    if not path:
        return None
    if not os.path.exists(path):
        log.warning("checkpoint %s not found; using scripted fallback", path)
        return None
    api = _import_attr(["load"], ["skyhand.train", "skyhand.train.agent", "skyhand.train.fb", "skyhand.train.fbcpr"])
    if "load" not in api:
        try:
            from skyhand.train.agent import Agent  # type: ignore
            api["load"] = Agent.load
        except Exception:
            pass
    if "load" not in api:
        log.warning("skyhand.train has no load(); using scripted fallback")
        return None
    try:
        return api["load"](path, device)
    except Exception as e:  # pragma: no cover - depends on other package
        log.warning("loading %s failed (%s); using scripted fallback", path, e)
        return None


def _accepts(fn, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ------------------------------------------------------------------ fallback reward -> z
def reward_to_z_fallback(reward_fn, B, g_samples: torch.Tensor, d: int) -> torch.Tensor:
    """z = sqrt(d) * normalize(mean_i r(g_i) B(g_i))  (PLAN.md 6.3)."""
    with torch.no_grad():
        r = reward_fn(g_samples).float().reshape(-1)
        b = B(g_samples).float().reshape(r.shape[0], -1)
        z = (r.unsqueeze(-1) * b).mean(0)
        return math.sqrt(d) * z / z.norm().clamp_min(1e-8)


def pca_2d(zs: np.ndarray) -> np.ndarray:
    """Project rows of zs [M, d] to 2D with PCA via SVD. Returns [M, 2]."""
    zs = np.asarray(zs, dtype=np.float64)
    if zs.ndim != 2 or zs.shape[0] == 0:
        return np.zeros((0, 2))
    x = zs - zs.mean(0, keepdims=True)
    if zs.shape[0] == 1:
        return np.zeros((1, 2))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2]
    out = x @ comps.T
    if out.shape[1] < 2:
        out = np.concatenate([out, np.zeros((out.shape[0], 2 - out.shape[1]))], 1)
    return out


# Named reference missions for the z map (the three highlighted missions + simple ones).
_CARRY = "-dist(drone, a0)*(1 - holding) - dist(payload, a1) + 2*holding"
REFERENCE_MISSIONS = [
    {"name": "Trash pickup", "scene": "trash", "anchors": {"a0": "item_0.handle", "a1": "bin"},
     "reward": _CARRY + " - 3*holding*(1 - payload_is_trash) - 2*(obstacle_dist < 1.0)"},
    {"name": "Organ delivery", "scene": "delivery",
     "anchors": {"a0": "item_0.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"},
     "reward": _CARRY + " - 0.5*swing - 0.01*jerk - 5*inside(a2) - 5*inside(a3)"},
    {"name": "Net fishing", "scene": "ocean", "anchors": {"a0": "net.corner_0", "a1": "boat", "a2": "fish_school_0"},
     "reward": "-dist(drone, a0)*(1 - holding) + 2*holding + 3*fish_in_net - dist(payload, a1)*fish_in_net - 5*bycatch"},
    {"name": "Go to bin", "scene": "trash", "anchors": {"a0": "bin"}, "reward": "-dist(drone, a0)"},
    {"name": "Fly fast", "scene": "trash", "anchors": {}, "reward": "speed - 2*collision"},
    {"name": "Go to drop-off pad", "scene": "delivery", "anchors": {"a0": "pad_dropoff"}, "reward": "-dist(drone, a0)"},
    {"name": "Hover at 3 m", "scene": "delivery", "anchors": {}, "reward": "-abs(height - 3) - 0.1*speed"},
    {"name": "Go to boat", "scene": "ocean", "anchors": {"a0": "boat"}, "reward": "-dist(drone, a0)"},
    # the world: every task in one map (3 drones)
    {"name": "World: trash to truck", "scene": "world", "anchors": {"a0": "item_0.handle", "a1": "bin"},
     "reward": _CARRY + " - 3*holding*(1 - payload_is_trash) - 2*(obstacle_dist < 1.0)"},
    {"name": "World: net fishing", "scene": "world",
     "anchors": {"a0": "net.corner_0", "a1": "boat", "a2": "fish_school_0"},
     "reward": "-dist(drone, a0)*(1 - holding) + 2*holding + 3*fish_in_net - dist(payload, a1)*fish_in_net - 5*bycatch"},
    {"name": "World: go to clinic pad", "scene": "world", "anchors": {"a0": "pad_dropoff"}, "reward": "-dist(drone, a0)"},
]


# ------------------------------------------------------------------ scripted controller
def lane_of(anc: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Per-agent altitude offset so teammates in transit never share a height. [1, A]"""
    return torch.arange(anc.shape[1], device=anc.device, dtype=height.dtype).unsqueeze(0) * 1.2


class ScriptedController:
    """Fly above a0, descend and grip, carry to above a1, lower and release.

    Uses only actor observations (obs["anchors"], obs["proprio"], obs["lidar"]). Phases per
    [world, agent]: 0 approach a0 at cruise height, 1 descend + grip, 2 carry to
    a1 at cruise height, 3 lower over a1, 4 released / climb and hover.
    trash and delivery carry one item (a0 = item_i.handle, a1 = drop-off); the ocean
    hauls the net (a0 = this drone's corner, a1 = boat, a2 = fish school). In the world each
    drone has its own mode (set_haul): the net haulers haul, the others carry / reach.
    Cruise height is above the terrain / water under the drone; when the near-horizontal
    LiDAR rows see something within a few metres (a building, a hill side, a tree) the drone
    climbs and slows down, so it hops over a city block or a ridge.
    """

    AVOID_RANGE = 4.0          # m, LiDAR return closer than this in a near-horizontal row -> climb
    AVOID_TIME = 1.2           # s of flight: the avoid range grows with the cruise speed
    AVOID_ROWS = 10            # first row with elevation >= ~-5 deg (rows go from -45 to +15 deg)
    STALL_STEPS = 20           # policy steps without descent over the drop-off before releasing anyway
    DROP_CLEAR = 1.0           # m, min clearance above whatever is under the drop-off when releasing

    def __init__(self, scene: str, num_envs: int, num_agents: int, device="cpu",
                 max_speed: Optional[float] = None, max_vz: Optional[float] = None):
        self.scene = scene
        self.num_agents = num_agents
        self.haul = torch.full((num_agents,), scene == "ocean", dtype=torch.bool)
        # action scales of this env (per scene: delivery flies 18 m/s); the transit speed follows them
        self.max_speed = float(max_speed or spec.SCENE_MAX_SPEED.get(scene, spec.MAX_SPEED))
        self.max_vz = float(max_vz or spec.SCENE_MAX_VZ.get(scene, spec.MAX_VZ))
        self.v_cruise = 0.5 * self.max_speed if self.max_speed <= 8 else 0.8 * self.max_speed
        self.vz_cruise = min(2.0 * self.max_vz / 3.0, 0.9 * self.max_vz)
        self.avoid_range = max(self.AVOID_RANGE, self.AVOID_TIME * self.v_cruise)
        # cruise height above the terrain / water; delivery stays above acacias and huts,
        # the world above trees and most of the town (LiDAR climbs over the rest)
        self.cruise_carry = {"trash": 4.5, "delivery": 12.0, "world": 10.0}.get(scene, 2.5)
        self.cruise_haul = 2.5
        self.drop_h = 1.2                              # above the drop-off anchor point
        self.grab_r = 0.08
        self.phase = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
        self.miss = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
        self.stage = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
        self.extra = torch.zeros(num_envs, num_agents, device=device)    # climb-over-obstacle altitude
        self.stall = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)

    @property
    def carry(self) -> bool:
        """True when no drone hauls the net (trash / delivery / an all-carry world mission)."""
        return not bool(self.haul.any())

    @property
    def cruise(self) -> float:
        return self.cruise_carry if self.carry else self.cruise_haul

    def set_haul(self, mask) -> None:
        """Per-drone controller mode: True = haul a net corner, False = carry an item / reach a target."""
        m = torch.as_tensor(list(mask), dtype=torch.bool).reshape(-1)[:self.num_agents]
        if m.numel() < self.num_agents:
            m = torch.cat([m, torch.zeros(self.num_agents - m.numel(), dtype=torch.bool)])
        self.haul = m

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.phase.zero_()
            self.miss.zero_()
            self.stage.zero_()
            self.extra.zero_()
            self.stall.zero_()
        else:
            self.phase[env_ids] = 0
            self.miss[env_ids] = 0
            self.stage[env_ids] = 0
            self.extra[env_ids] = 0
            self.stall[env_ids] = 0

    def _clearance(self, obs: Dict[str, torch.Tensor], height: torch.Tensor) -> torch.Tensor:
        """Vertical distance to the surface under the drone from the downward grid's corner rays."""
        down = obs.get("lidar_down")
        if down is None or down.dim() != 5:
            return torch.full_like(height, 1e3)
        r = down[:, :, 0].float() * (spec.LIDAR_MAX_RANGE / 255.0)                   # [N, A, DV, DH]
        corners = torch.stack([r[..., 0, 0], r[..., 0, -1], r[..., -1, 0], r[..., -1, -1]], -1)
        cos = 1.0 / math.sqrt(1.0 + 2.0 * spec.DOWN_HALF_WIDTH ** 2)
        c = corners.amin(-1) * cos
        off = down[:, :, 0].flatten(2).amax(-1) == 0                                 # all-zero = sensor off
        c = torch.where(off | (corners.amin(-1) >= spec.LIDAR_MAX_RANGE - 0.2), torch.full_like(c, 1e3), c)
        return c.to(height.device)

    def _blocked(self, obs: Dict[str, torch.Tensor], shape) -> torch.Tensor:
        lid = obs.get("lidar")
        if lid is None or lid.dim() != 5 or lid.shape[-3] < 1:
            return torch.zeros(shape, dtype=torch.bool, device=self.phase.device)
        rng = lid[:, :, 0].float() * (spec.LIDAR_MAX_RANGE / 255.0)                # [N, A, V, H]
        active = lid[:, :, 0].flatten(2).amax(-1) > 0                               # all-zero = sensor off
        rows = rng[:, :, min(self.AVOID_ROWS, rng.shape[2] - 1):]
        return (((rows > 0.5) & (rows < self.avoid_range)).flatten(2).any(-1) & active).to(self.phase.device)

    def __call__(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        anc, pro = obs["anchors"], obs["proprio"]
        dev = anc.device
        self.phase, self.miss, self.stage = self.phase.to(dev), self.miss.to(dev), self.stage.to(dev)
        self.extra, self.stall = self.extra.to(dev), self.stall.to(dev)
        rel0, rel1 = anc[..., 0, 0:3], anc[..., 1, 0:3]
        has0, has1 = anc[..., 0, 3] < 0.5, anc[..., 1, 3] < 0.5       # one-hot index 0 is "none"
        vel, grip_closed, holding, height = pro[..., 0:3], pro[..., 9] > 0.5, pro[..., 10] > 0.5, pro[..., 11]
        speed = vel.norm(dim=-1)
        ph = self.phase
        H = self.haul.to(dev).view(1, -1).expand_as(ph)               # per-drone mode
        C = ~H
        any_carry, any_haul = bool(C.any()), bool(H.any())
        full = lambda a, b: torch.where(H, torch.full_like(height, b), torch.full_like(height, a))
        cruise = full(self.cruise_carry, self.cruise_haul)
        min_h = full(0.15, 0.45)
        off = torch.tensor([0.0, 0.0, -spec.GRIPPER_OFFSET[2]], device=dev)
        grip_rel = rel0 + off                                          # where the drone must go to put gripper on a0
        xy0, xy1 = rel0[..., :2].norm(dim=-1), rel1[..., :2].norm(dim=-1)
        # release height: drop_h above a1, but never closer than DROP_CLEAR to whatever is below
        # (a garbage truck roof under the zone point); measured with the corner rays of the
        # downward LiDAR grid, which miss a carried item hanging under the gripper
        clear = self._clearance(obs, height)
        # a large carried item (pizza box, cooler) can fill the corner rays right under the drone: a return
        # closer than the item would hang is the payload, not the drop-off
        clear = torch.where(holding & (clear < 0.8), torch.full_like(clear, 1e3), clear)
        drop_dz = torch.maximum(rel1[..., 2] + self.drop_h, self.DROP_CLEAR - clear)
        # transitions
        ph = torch.where((ph == 0) & has0 & (xy0 < 0.4), torch.ones_like(ph), ph)
        ph = torch.where((ph == 1) & (xy0 > 1.2), torch.zeros_like(ph), ph)
        ph = torch.where((ph <= 1) & holding, torch.full_like(ph, 2), ph)
        ph = torch.where((ph >= 2) & (ph <= 3) & ~holding, torch.where(has0, torch.zeros_like(ph), torch.full_like(ph, 4)), ph)
        if any_carry:
            ph = torch.where(C & (ph == 2) & has1 & (xy1 < 0.35) & (vel[..., :2].norm(dim=-1) < 0.6),
                             torch.full_like(ph, 3), ph)
        # climb over whatever is ahead while in transit (not once over the target: a truck or the
        # buildings next to it would keep the drone climbing instead of descending onto it)
        arrived = torch.where(ph == 0, xy0, xy1) < 3.0
        transit = ((ph == 0) | (ph == 2)) & ~arrived
        blocked = (self._blocked(obs, ph.shape) & transit & C) if any_carry else torch.zeros_like(transit)
        # blocked: aim 1.5 m above the current height (keeps climbing, no wind-up); clear: sink back slowly
        climb = (height - cruise - lane_of(anc, height) + 1.5).clamp_min(0.0)
        self.extra = torch.where(blocked, torch.maximum(self.extra, climb),
                                 (self.extra - 0.02).clamp_min(0.0)).clamp(max=40.0)
        # targets (relative vectors the drone should move along); in transit each agent
        # flies at its own altitude so paths never cross at the same height
        lane = lane_of(anc, height)
        cruise_dz = (cruise + lane + self.extra - height)
        zeros = torch.zeros_like(height)
        t_hover = torch.stack([zeros, zeros, cruise_dz], -1)
        t0 = torch.cat([rel0[..., :2], cruise_dz.unsqueeze(-1)], -1)
        dz1 = torch.maximum(grip_rel[..., 2], min_h - height)
        t1 = torch.cat([grip_rel[..., :2], dz1.unsqueeze(-1)], -1)
        t2 = torch.cat([rel1[..., :2], cruise_dz.unsqueeze(-1)], -1)     # carry: over a1, then lower onto it
        t3 = torch.cat([rel1[..., :2], drop_dz.unsqueeze(-1)], -1)
        if any_haul:
            # ocean haul: move the whole net (its centroid) over a2 (fish), then over a1 (boat),
            # all drones with the same horizontal velocity, once every live drone holds a corner.
            cen = spec.g_field(obs["g"], "net_centroid_rel") if "g" in obs else torch.zeros_like(rel0)
            rel2 = anc[..., 2, 0:3]
            has2 = anc[..., 2, 3] < 0.5
            if any_carry:   # mixed team (world): wait for the drones that haul, ignore the carriers
                team_ok = (holding | C).all(-1, keepdim=True).expand_as(holding)
            else:
                team_ok = holding.all(-1, keepdim=True) if "mates" not in obs else \
                    (holding & ((obs["mates"][..., 6] > 0.5) | ~obs["mates_mask"]).all(-1)).all(-1, keepdim=True).expand_as(holding)
            to_fish = (rel2 - cen)[..., :2]
            to_boat = (rel1 - cen)[..., :2]
            fish_done = (~has2) | (to_fish.norm(dim=-1) < 1.0) | (self.stage == 1)
            self.stage = torch.where(H & fish_done & team_ok & (ph == 2), torch.ones_like(self.stage), self.stage)
            dxy = torch.where((self.stage == 0).unsqueeze(-1), to_fish, to_boat)
            dxy = torch.where(team_ok.unsqueeze(-1), dxy, torch.zeros_like(dxy))
            haul_dz = (cruise - height)
            h2 = torch.cat([dxy, haul_dz.unsqueeze(-1)], -1)
            h0, h1 = t0, t1
            if "mates" in obs:   # keep apart from teammates
                m_rel = obs["mates"][..., 0:3]
                dm = m_rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
                push = (-(m_rel / dm) * (2.5 - dm).clamp_min(0.0) * 3.0 * obs["mates_mask"].unsqueeze(-1).float()).sum(-2)
                h0 = t0 + push
                h1 = t1 + push * (height > min_h + 0.3).unsqueeze(-1).float()
                h2 = h2 + push
            Hx = H.unsqueeze(-1)
            t0, t1 = torch.where(Hx, h0, t0), torch.where(Hx, h1, t1)
            t2, t3 = torch.where(Hx, h2, t2), torch.where(Hx, h2, t3)
        tgt = torch.where((ph == 0).unsqueeze(-1), t0, t_hover)
        tgt = torch.where((ph == 1).unsqueeze(-1), t1, tgt)
        tgt = torch.where((ph == 2).unsqueeze(-1), t2, tgt)
        tgt = torch.where((ph == 3).unsqueeze(-1), t3, tgt)
        tgt = torch.where(((ph == 0) & ~has0).unsqueeze(-1), t_hover, tgt)
        tgt = torch.where((C & (ph == 2) & ~has1).unsqueeze(-1), t_hover, tgt)
        # gripper
        near = (ph == 1) & (grip_rel.norm(dim=-1) < self.grab_r) & (speed < 0.4)
        if any_haul:   # corners may sit at/under the surface: grab from the lowest safe height
            near_h = (ph == 1) & (grip_rel[..., :2].norm(dim=-1) < 0.15) & (speed < 0.5) & \
                ((height < min_h + 0.1) | (grip_rel.norm(dim=-1) < 0.12))   # floating corners: plain reach
            near = torch.where(H, near_h, near)
        self.miss = torch.where(grip_closed & ~holding & (ph == 1), self.miss + 1, torch.zeros_like(self.miss))
        retry = self.miss >= 4
        close = (near & ~retry) | (holding & (ph >= 2) & (ph <= 3))
        # lowering can stall above the drop-off (safety shield, a bin rim): release after 1 s of no progress
        over = (ph == 3) & (xy1 < 0.35) & (speed < 0.35)
        self.stall = torch.where(over & (speed < 0.2), self.stall + 1, torch.zeros_like(self.stall))
        release = over & ((drop_dz.abs() < 0.15) | (self.stall >= self.STALL_STEPS)) & C   # haulers keep the corner
        close = close & ~release
        ph = torch.where(release, torch.full_like(ph, 4), ph)
        self.miss = torch.where(retry, torch.zeros_like(self.miss), self.miss)
        self.phase = ph
        # velocity command
        vmax_xy = torch.where(ph == 1, torch.full_like(height, 1.0), torch.full_like(height, self.v_cruise))
        vmax_xy = torch.where(ph == 3, torch.full_like(height, 1.0), vmax_xy)
        vmax_xy = torch.where(blocked, torch.full_like(height, 0.5), vmax_xy)
        vmax_xy = torch.where(H & (ph >= 2), torch.full_like(height, 1.2), vmax_xy)
        v = 1.2 * tgt
        vxy = v[..., :2]
        n = vxy.norm(dim=-1, keepdim=True)
        # brake in time: speed that can still stop within the remaining distance (~3 m/s^2)
        vmax_xy = torch.minimum(vmax_xy, torch.sqrt(2 * 3.0 * tgt[..., :2].norm(dim=-1)) + 0.3)
        vxy = vxy * torch.clamp(vmax_xy.unsqueeze(-1) / n.clamp_min(1e-6), max=1.0)
        slow = (ph == 1) & (tgt[..., 2] > -2.0)                       # last 2 m of the descent onto a0
        vz_max = torch.where(slow, torch.full_like(height, 0.6), torch.full_like(height, self.vz_cruise))
        vz = torch.maximum(torch.minimum(v[..., 2], vz_max), -vz_max)
        act = torch.zeros(anc.shape[:2] + (spec.ACTION_DIM,), device=dev)
        act[..., 0] = vxy[..., 0] / self.max_speed
        act[..., 1] = vxy[..., 1] / self.max_speed
        act[..., 2] = vz / self.max_vz
        act[..., 4] = torch.where(close, torch.ones_like(height), -torch.ones_like(height))
        return act.clamp(-1.0, 1.0)


def scripted_for(env) -> ScriptedController:
    """ScriptedController with the env's own action scales (env.max_speed / env.max_vz)."""
    return ScriptedController(env.scene, env.num_envs, env.num_agents, env.device,
                              max_speed=getattr(env, "max_speed", None), max_vz=getattr(env, "max_vz", None))


# ------------------------------------------------------------------ pipeline
class Pipeline:
    """One object the server talks to. Inject `agent` / `mission_api` in tests."""

    def __init__(self, agent=None, mission_api: Optional[Dict[str, Any]] = None, device: str = "cpu",
                 use_env_expert: bool = False, z_samples_envs: int = 32, z_samples_steps: int = 60,
                 cem: bool = False):
        self.agent = agent
        self.mission_api = load_mission_api() if mission_api is None else mission_api
        self.device = device
        self.use_env_expert = use_env_expert
        self.z_samples_envs = z_samples_envs
        self.z_samples_steps = z_samples_steps
        self.cem = cem
        self._sample_cache: Dict[Any, torch.Tensor] = {}
        self._haul: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self._zmap_cache: Optional[Dict[str, np.ndarray]] = None
        self._env_expert_ok: Optional[bool] = None

    # -------------------------------------------------- labels
    @property
    def policy_mode(self) -> str:
        if self.agent is not None:
            return POLICY_TRAINED
        if self._env_expert_ok:
            return POLICY_ENV_EXPERT
        return POLICY_SCRIPTED

    @property
    def mission_mode(self) -> str:
        if "compile_mission" in self.mission_api:
            return MISSION_GEMINI if _gemini_key() else MISSION_OFFLINE   # per-mission label comes from spec.offline
        if "offline_compile" in self.mission_api:
            return MISSION_OFFLINE
        return MISSION_KEYWORD

    # -------------------------------------------------- mission
    def compile(self, text: str, scene: str, env) -> CompiledMission:
        api = self.mission_api
        desc = env.entity_descriptions()
        for name in ("compile_mission", "offline_compile"):
            if name not in api:
                continue
            try:
                ms = api[name](text, scene, desc)
                label = MISSION_OFFLINE if (name == "offline_compile" or _get(ms, "offline", False)) else MISSION_GEMINI
                return self._from_spec(ms, text, scene, env, label)
            except Exception as e:
                log.warning("%s failed (%s: %s)", name, type(e).__name__, e)
        return keyword_compile(text, scene, env)

    def _from_spec(self, ms, text, scene, env, label) -> CompiledMission:
        src = _get(ms, "source")
        if isinstance(src, str) and src:
            label = MISSION_GEMINI if "gemini" in src.lower() else MISSION_OFFLINE if "offline" in src.lower() else src
        anchors = dict(_get(ms, "anchors", {}) or {})
        anchors = {k: (v if (v is None or isinstance(v, str)) else list(v)) for k, v in anchors.items()}
        for k in range(spec.N_ANCHORS):
            anchors.setdefault(f"a{k}", None)
        reward = str(_get(ms, "reward"))
        done = _get(ms, "done_when") or None
        api = self.mission_api
        reward_fn = done_fn = bindings = None
        if "compile_reward" in api:
            reward_fn = api["compile_reward"](reward)
            done_fn = api["compile_reward"](done) if done else None
        binder = None
        if "to_bindings" in api:
            ents = env.entity_names()
            per_agent = None
            if _accepts(api["to_bindings"], "agent"):
                tb = api["to_bindings"]
                try:   # name -> description mapping: lets the world resolve "cooler" to the right item
                    descs = env.entity_descriptions()
                    per_agent = {ag: list(tb(ms, agent=ag, entities=descs)) for ag in range(env.num_agents)} \
                        if set(descs) >= set(ents) else None
                except Exception:
                    per_agent = None
                if per_agent is None:
                    per_agent = {ag: list(tb(ms, agent=ag, entities=ents)) for ag in range(env.num_agents)}
            bindings = per_agent[0] if per_agent else list(api["to_bindings"](ms))
            if per_agent:
                binder = per_agent
        warnings = list(_get(ms, "warnings", []) or [])
        out = _finish(text, scene, anchors, reward, done, _get(ms, "explanation"), label, env,
                      reward_fn=reward_fn, done_fn=done_fn, bindings=bindings, raw=ms)
        out.per_agent_bindings = binder
        out.warnings = warnings
        self._agents_from_spec(out, ms, env)
        return out

    def _agents_from_spec(self, out: CompiledMission, ms, env) -> None:
        """Per-drone jobs from MissionSpec.agents (absent in older mission packages: then every drone
        flies the same mission)."""
        agents = _get(ms, "agents")
        stage = getattr(ms, "stage", None)
        if not agents or not callable(stage) or not _accepts(stage, "agent"):
            return
        comp = self.mission_api.get("compile_reward", rewardlang.compile_expr)
        jobs, rfns, dfns = {}, {}, {}
        for ag in range(env.num_agents):
            st = stage(None, agent=ag)
            anchors = {k: (v if (v is None or isinstance(v, str)) else list(v))
                       for k, v in dict(_get(st, "anchors", {}) or {}).items()}
            r, d = str(_get(st, "reward")), _get(st, "done_when") or None
            sub = agents.get(ag) if isinstance(agents, dict) else None
            task = (_get(sub, "explanation") if sub is not None else None) or _get(ms, "explanation") or out.text
            jobs[ag] = {"task": str(task), "anchors": anchors, "reward": r, "done_when": d}
            rfns[ag] = comp(r)
            dfns[ag] = comp(d) if d else None
        out.agents, out.agent_reward_fns, out.agent_done_fns = jobs, rfns, dfns
        out.explanation = [f"Drone {ag}: {j['task']}" for ag, j in sorted(jobs.items())] + list(out.explanation)

    def bind(self, env, mission: CompiledMission, env_ids) -> None:
        """Bind anchors. Mission-package specs bind per agent (their {agent} templates and
        per-drone sub-missions); keyword-parser specs in the ocean give each drone its own net
        corner (corner k -> k + agent). Also records which drones haul the net (see haul_mask)."""
        if mission.per_agent_bindings:
            per_agent = dict(mission.per_agent_bindings)
        else:
            b = mission.bindings
            per_agent = {}
            for ag in range(env.num_agents):
                per = []
                for x in b:
                    if env.num_agents > 1 and x.entity and x.entity.startswith("net.corner_"):
                        c = (int(x.entity.rsplit("_", 1)[1]) + ag) % 4
                        x = AnchorBinding(kind=x.kind, entity=f"net.corner_{c}", radius=x.radius)
                    per.append(x)
                per_agent[ag] = per
        if mission.per_agent_bindings or env.num_agents > 1:
            for ag, b in per_agent.items():
                env.bind_anchors(env_ids, b, agent=ag)
        else:
            env.bind_anchors(env_ids, mission.bindings)
        hauls = [bool(per_agent.get(ag)) and bool(per_agent[ag][0].entity) and
                 str(per_agent[ag][0].entity).startswith("net.corner_") for ag in range(env.num_agents)]
        try:
            self._haul[env] = hauls
        except TypeError:   # env not weak-referenceable
            pass

    def haul_mask(self, env) -> List[bool]:
        """Per drone: True = net-haul controller mode, False = carry / reach. The ocean always hauls;
        in the world it follows the anchors bound last (a0 = net.corner_i -> haul)."""
        if env.scene == "ocean":
            return [True] * env.num_agents
        if env.scene != "world":
            return [False] * env.num_agents
        try:
            return list(self._haul.get(env, [False] * env.num_agents))
        except TypeError:
            return [False] * env.num_agents

    # -------------------------------------------------- policy
    def make_controller(self, env) -> Callable[[Dict[str, torch.Tensor], Optional[torch.Tensor]], torch.Tensor]:
        """Return act(obs, z) for this env."""
        if self.agent is not None:
            agent = self.agent

            def act_trained(obs, z):
                with torch.no_grad():
                    if z is None:
                        z = torch.zeros(int(agent.d), device=env.device)
                    if z.dim() == 2:   # one z per drone (per-drone jobs) -> [N, A, d]
                        z = z.unsqueeze(0).expand(obs["proprio"].shape[0], -1, -1)
                    return agent.act(obs, z, noise=0.0)
            return act_trained
        scripted = scripted_for(env)
        if self.use_env_expert and self._env_expert_ok is not False:
            try:
                env.expert_action()
                self._env_expert_ok = True
            except Exception as e:
                log.info("env.expert_action unavailable (%s: %s); using built-in scripted controller", type(e).__name__, e)
                self._env_expert_ok = False

        def act_scripted(obs, z):
            if self._env_expert_ok:
                try:
                    return env.expert_action()
                except Exception:
                    self._env_expert_ok = False
            scripted.set_haul(self.haul_mask(env))
            return scripted(obs)
        act_scripted.reset = scripted.reset  # type: ignore[attr-defined]
        return act_scripted

    # -------------------------------------------------- z
    def _g_samples(self, scene: str, anchors: Dict[str, Any], make_env, per_agent=None,
                   per_drone: bool = False) -> torch.Tensor:
        """g visited by the scripted controller with these anchors bound: [M, G_DIM], or with
        per_drone=True [M, A, G_DIM] (per_agent: {drone: bindings}, each drone's own anchors)."""
        key = (scene, tuple(sorted((k, str(v)) for k, v in anchors.items() if v is not None)),
               tuple(sorted((ag, str(b)) for ag, b in (per_agent or {}).items())), per_drone)
        if key in self._sample_cache:
            return self._sample_cache[key]
        env = make_env(scene, self.z_samples_envs)
        mission = _finish("", scene, {f"a{k}": anchors.get(f"a{k}") for k in range(spec.N_ANCHORS)},
                          "0", None, None, "", env)
        if per_agent:
            mission.per_agent_bindings = {ag: list(b) for ag, b in per_agent.items() if ag < env.num_agents}
        ids = torch.arange(env.num_envs, device=env.device)
        self.bind(env, mission, ids)
        ctrl = scripted_for(env)
        ctrl.set_haul(self.haul_mask(env))
        obs = observe(env)
        gs = []
        gen = torch.Generator().manual_seed(1)
        for _ in range(self.z_samples_steps):
            act = ctrl(obs) + 0.3 * torch.randn(env.num_envs, env.num_agents, spec.ACTION_DIM, generator=gen).to(env.device)
            out = env.step(act)
            obs = out.obs
            gs.append(obs["g"].reshape(-1, env.num_agents, spec.G_DIM).clone() if per_drone else
                      obs["g"].reshape(-1, spec.G_DIM).clone())
            dead = out.done | out.truncated
            if dead.any():
                rid = torch.nonzero(dead).squeeze(-1)
                obs = env.reset(rid)
                self.bind(env, mission, rid)
                ctrl.reset(rid)
                obs = observe(env)
        g = torch.cat(gs)
        self._sample_cache[key] = g
        return g

    def compute_z(self, reward_fn, scene: str, anchors: Dict[str, Any], make_env,
                  mission: Optional[CompiledMission] = None) -> Optional[torch.Tensor]:
        """z [d] for a shared mission; [A, d] (one z per drone) when the mission gives drones their own
        jobs (mission.agent_reward_fns): drone i's z comes from its own reward on the g its own
        anchors produce."""
        if self.agent is None:
            return None
        if mission is not None and mission.agent_reward_fns:
            g = self._g_samples(scene, anchors, make_env, per_agent=mission.per_agent_bindings,
                                per_drone=True).to(self.device)
            zs = [self._z_from(mission.agent_reward_fns[ag] if ag in mission.agent_reward_fns else reward_fn,
                               g[:, ag]) for ag in range(g.shape[1])]
            return torch.stack(zs)
        g = self._g_samples(scene, anchors, make_env,
                            per_agent=mission.per_agent_bindings if mission is not None else None).to(self.device)
        return self._z_from(reward_fn, g)

    def _z_from(self, reward_fn, g: torch.Tensor) -> torch.Tensor:
        B = self.agent.encode_g
        if "reward_to_z" in self.mission_api:
            try:
                return torch.as_tensor(self.mission_api["reward_to_z"](reward_fn, B, g, int(self.agent.d))).float().reshape(-1)
            except Exception as e:
                log.warning("reward_to_z failed (%s); using built-in formula", e)
        return reward_to_z_fallback(reward_fn, B, g, int(self.agent.d)).reshape(-1)

    def refine_z(self, env, z0, reward_fn, done_fn=None, n_candidates: Optional[int] = None):
        """CEM refinement in clones of world 0 (needs env.num_envs > 1 and --cem)."""
        if not (self.cem and self.agent is not None and "cem_refine" in self.mission_api and env.num_envs > 1):
            return z0, False
        if z0 is not None and z0.dim() == 2:   # per-drone z: CEM refines one shared z only
            return z0, False
        try:
            agent = self.agent
            nc = n_candidates or env.num_envs - 1
            out = self.mission_api["cem_refine"](env, lambda obs, z: agent.act(obs, z, noise=0.0), z0, reward_fn,
                                                 src_env=0, n_candidates=nc, done_fn=done_fn,
                                                 dst_ids=list(range(1, nc + 1)))
            z = out[0] if isinstance(out, tuple) else out
            return torch.as_tensor(z).float().reshape(-1), True
        except Exception as e:
            log.warning("cem_refine failed (%s); keeping z_r", e)
            return z0, False

    def reference_z(self, make_env) -> Dict[str, np.ndarray]:
        if self.agent is None:
            return {}
        if self._zmap_cache is None:
            out = {}
            for m in REFERENCE_MISSIONS:
                try:
                    comp = self.mission_api.get("compile_reward", rewardlang.compile_expr)
                    z = self.compute_z(comp(m["reward"]), m["scene"], m["anchors"], make_env)
                    out[m["name"]] = z.detach().cpu().numpy()
                except Exception as e:
                    log.warning("reference z for %s failed: %s", m["name"], e)
            self._zmap_cache = out
        return self._zmap_cache

    def zmap(self, make_env, current: Optional[torch.Tensor], current_name: str = "your mission") -> dict:
        if self.agent is None:
            return {"available": False, "points": [], "note": "z map available after training"}
        ref = self.reference_z(make_env)
        names = list(ref)
        zs = [ref[n] for n in names]
        n_cur = 0
        if current is not None:
            cur = current.detach().cpu().numpy()
            rows = cur.reshape(-1, cur.shape[-1]) if cur.ndim == 2 else cur.reshape(1, -1)
            for i, r in enumerate(rows):
                names.append(current_name if len(rows) == 1 else f"{current_name} (drone {i})")
                zs.append(r)
            n_cur = len(rows)
        if not zs:
            return {"available": False, "points": [], "note": "no reference missions"}
        p = pca_2d(np.stack(zs))
        scale = max(float(np.abs(p).max()), 1e-6)
        pts = [{"name": n, "x": float(x / scale), "y": float(y / scale), "current": i >= len(names) - n_cur}
               for i, (n, (x, y)) in enumerate(zip(names, p))]
        return {"available": True, "points": pts, "note": ""}
