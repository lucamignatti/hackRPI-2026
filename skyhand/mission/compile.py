"""English mission -> MissionSpec, via Gemini (primary) or a keyword fallback.

Only this module touches google.genai, and only inside _default_client(), so the
rest of the package imports without it. Gemini never sees z: it binds anchors to
scene entities and writes a reward program in the DSL (see dsl.py).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from pydantic import ValidationError

from .. import spec as S
from .dsl import DSLError, compile_reward, describe_vocabulary
from .schema import MissionError, MissionSpec

DEFAULT_MODEL = "gemini-2.5-flash"

# ------------------------------------------------------------------ prompt
_EXAMPLES = [
    ("trash", "pick up the trash bag and drop it in the bin", {
        "anchors": {"a0": "payload.handle", "a1": "bin", "a2": None, "a3": None},
        "reward": "where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 0.5*(obstacle_dist < 0.5) - 0.02*thrust",
        "done_when": "distxy(payload, a1) < 0.6 and not holding",
        "explanation": "Fly to the bag handle, grab it, carry it to the bin and let go, keeping clear of obstacles.",
        "steps": [
            {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
             "explanation": "Grab the bag handle."},
            {"reward": "-dist(payload, a1) + holding", "done_when": "distxy(payload, a1) < 0.6 and not holding",
             "explanation": "Carry the bag to the bin and release it."},
        ],
    }),
    ("trash", "deliver the organ cooler gently, stay out of the no-fly zone", {
        "anchors": {"a0": "payload.handle", "a1": "bin", "a2": "nofly_0", "a3": None},
        "reward": ("where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 0.5*swing - 0.01*jerk - 0.3*tilt "
                   "- 5*inside(a2) - 0.5*(obstacle_dist < 0.5)"),
        "done_when": "distxy(payload, a1) < 0.6 and not holding",
        "explanation": ("Pick up the cooler and carry it to the drop zone as smoothly as possible "
                        "(low swing, low jerk), never entering the no-fly zone."),
    }),
    ("ocean", "drag the net over the fish and haul it to the boat", {
        "anchors": {"a0": "net.corner_{agent}", "a1": "boat", "a2": "fish_school_0", "a3": None},
        "reward": ("where(holding, 2*fish_in_net - 0.3*distxy(net, a2)*(fish_in_net < 0.5) "
                   "- distxy(net, a1)*(fish_in_net > 0.5), -dist(drone, a0)) - 2*net_stretch - 0.5*(mate_dist < 1)"),
        "done_when": "fish_in_net > 0.5 and distxy(net, a1) < 2.5",
        "explanation": ("Each drone grabs its own net corner, the team sweeps the net over the fish school, "
                        "then hauls it to the boat without tearing the net."),
    }),
    ("trash", "go to the point (3, -2) at 4 m and hover", {
        "anchors": {"a0": [3.0, -2.0, 4.0], "a1": None, "a2": None, "a3": None},
        "reward": "-dist(drone, a0) - 0.1*speed*(dist(drone, a0) < 0.5)",
        "done_when": "dist(drone, a0) < 0.3 and speed < 0.3",
        "explanation": "Fly to the point (3, -2) four metres up and hold still there.",
    }),
]

_SCHEMA_DOC = """Return ONLY a JSON object with these keys:
  "anchors": {"a0": ..., "a1": ..., "a2": ..., "a3": ...}  each an entity name from the list, a fixed
             world point [x, y, z] in metres, or null. "{agent}" in a name is replaced by the drone index
             (e.g. "net.corner_{agent}" gives every drone its own corner).
  "anchor_kinds": optional {"aK": "point"|"handle"|"zone"} overrides (normally omit).
  "anchor_radius": optional {"aK": metres} zone radius for fixed points (normally omit).
  "reward": reward expression (higher is better, evaluated every step for every drone).
  "done_when": optional expression, mission complete when > 0.5.
  "explanation": ONE plain-language sentence for a non-expert approving the mission.
  "steps": optional list of sub-missions for long tasks, each {"anchors"?, "reward", "done_when", "explanation"};
           steps run in order, the next starts when done_when fires; omit "anchors" to keep the top-level ones.
Guidelines: keep rewards dense (distance terms), keep weights small (|w| <= 10), penalise inside(no-fly anchor)
when told to avoid a zone, use swing/jerk/tilt penalties for gentle carrying, obstacle_dist for keeping clear.
Only the drone can grab handles (".handle" entities and net corners): grabbing happens by closing the gripper
near a handle, so "holding" tells you whether the payload is in hand."""


def build_prompt(text: str, scene: str, entities: Dict[str, str]) -> str:
    ents = "\n".join(f"  {n}: {d}" for n, d in entities.items())
    ex = []
    for sc, t, out in _EXAMPLES:
        ex.append(f"Scene: {sc}\nMission: {t}\nJSON: {json.dumps(out)}")
    return (
        "You turn a drone mission written in English into a reward program for a pretrained drone policy.\n"
        f"Coordinates: world frame, z up, metres; the arena is {2 * S.ARENA_HALF:.0f} x {2 * S.ARENA_HALF:.0f} m "
        "centred at the origin.\n\n"
        f"{describe_vocabulary()}\n\n{_SCHEMA_DOC}\n\n"
        "Worked examples (entity names differ per scene; use only the names listed for this scene):\n\n"
        + "\n\n".join(ex) +
        f"\n\nNow the real task.\nScene: {scene}\nEntities in this scene:\n{ents}\nMission: {text}\nJSON:"
    )


# ------------------------------------------------------------------ validation
def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    return m.group(1) if m else raw


def parse_and_validate(raw: str, text: str, scene: str, entities: Sequence[str]) -> MissionSpec:
    """Parse model output into a validated MissionSpec. Raises ValueError with a readable message."""
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"output is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("output must be a JSON object")
    data.setdefault("text", text)
    data["scene"] = scene
    data.pop("offline", None)
    try:
        ms = MissionSpec.model_validate(data, context={"entities": list(entities)})
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(str(x) for x in er['loc']) or 'spec'}: {er['msg']}" for er in e.errors())
        raise ValueError(f"invalid mission: {msgs}") from None
    except (MissionError, DSLError) as e:
        raise ValueError(str(e)) from None
    dry_run(ms)
    return ms


def dry_run(ms: MissionSpec) -> None:
    """Evaluate every expression on a zero g batch; raise ValueError if anything breaks."""
    g = torch.zeros(4, S.G_DIM)
    exprs = [("reward", ms.reward), ("done_when", ms.done_when)]
    for i, st in enumerate(ms.steps or []):
        exprs += [(f"steps[{i}].reward", st.reward), (f"steps[{i}].done_when", st.done_when)]
    for what, e in exprs:
        if e is None:
            continue
        try:
            out = compile_reward(e)(g)
        except DSLError as err:
            raise ValueError(f"{what}: {err}") from None
        if out.shape != (4,) or not torch.isfinite(out).all():
            raise ValueError(f"{what}: dry run gave shape {tuple(out.shape)} / non-finite values")


# ------------------------------------------------------------------ Gemini
def _default_client():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return None
    from google import genai   # lazy: only this function imports google.genai
    return genai.Client()


def _generate(client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.2})
    txt = getattr(resp, "text", None)
    if txt is None:
        txt = str(resp)
    return txt


def compile_mission(text: str, scene: str, entities: Dict[str, str], client=None, model: Optional[str] = None,
                    max_retries: int = 3, fallback: bool = True) -> MissionSpec:
    """Compile an English mission with Gemini; validated, with retries on errors.

    entities: name -> description, as from env.entity_descriptions().
    client: object with ``client.models.generate_content(model=, contents=, config=) -> .text``
            (a google.genai.Client); default genai.Client() if GEMINI_API_KEY is set.
    Falls back to offline_compile (spec.offline=True, reason in spec.warnings) when there is
    no API key, the client errors, or every attempt fails validation (unless fallback=False,
    in which case the last error is raised as ValueError / the client exception propagates).
    """
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    names = list(entities)
    reason = None
    if client is None:
        try:
            client = _default_client()
        except Exception as e:  # import / auth problems
            client, reason = None, f"Gemini client unavailable: {e}"
        if client is None and reason is None:
            reason = "no GEMINI_API_KEY set"
    if client is not None:
        base = build_prompt(text, scene, entities)
        prompt = base
        errors: List[str] = []
        for attempt in range(max(1, max_retries)):
            try:
                raw = _generate(client, model, prompt)
            except Exception as e:
                if not fallback:
                    raise
                reason = f"Gemini call failed: {type(e).__name__}: {e}"
                break
            try:
                ms = parse_and_validate(raw, text, scene, names)
                if errors:
                    ms.warnings.append(f"needed {attempt + 1} attempts")
                return ms
            except ValueError as e:
                errors.append(str(e))
                prompt = (base + "\n\nYour previous answer was:\n" + raw.strip()[:4000] +
                          f"\n\nIt was rejected: {e}\nFix the problem and return the corrected JSON only.")
        else:
            if not fallback:
                raise ValueError(f"mission did not validate after {max_retries} attempts: {errors[-1]}")
            reason = f"Gemini output rejected {max_retries} times; last error: {errors[-1]}"
    if not fallback:
        raise RuntimeError(reason or "no Gemini client")
    ms = offline_compile(text, scene, entities)
    if reason:
        ms.warnings.append(reason)
    return ms


# ------------------------------------------------------------------ offline fallback
_POS_RE = re.compile(r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")
_NUM = r"(-?\d+(?:\.\d+)?)"


def _pos(entities: Dict[str, str], name: str) -> Optional[Tuple[float, float, float]]:
    m = _POS_RE.search(entities.get(name, ""))
    return tuple(float(x) for x in m.groups()) if m else None


def _has(t: str, *words: str) -> bool:
    return any(re.search(r"\b" + re.escape(w), t) for w in words)


def _match_entity(phrase: str, entities: Dict[str, str]) -> Optional[str]:
    """Best entity named in a phrase ("the boat", "obstacle 3", "fish", "no-fly zone", "drone 1")."""
    p = phrase.lower()
    m = re.search(r"(obstacle|box|cart|drone|fish school|school|corner)\s*_?#?(\d+)", p)
    if m:
        base = {"box": "obstacle", "cart": "obstacle", "school": "fish_school", "fish school": "fish_school",
                "corner": "net.corner"}.get(m.group(1), m.group(1))
        cand = f"{base}_{m.group(2)}"
        if cand in entities:
            return cand
    table = [
        (("handle",), "payload.handle"), (("bag", "trash", "cooler", "organ", "payload", "package"), "payload"),
        (("bin", "dumpster", "can"), "bin"), (("no-fly", "no fly", "nofly", "restricted"), "nofly_0"),
        (("boat", "ship", "deck"), "boat"), (("net",), "net.center"), (("fish", "school"), "fish_school_0"),
        (("obstacle", "box", "cart"), "obstacle_0"),
    ]
    for words, ent in table:
        if _has(p, *words) and ent in entities:
            return ent
    for name in entities:
        if name.replace("_", " ") in p or name in p:
            return name
    return None


def _point_in(phrase: str, default_z: float = 3.0) -> Optional[List[float]]:
    m = re.search(r"\(?\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)?", phrase)
    if not m:
        return None
    x, y, z = m.group(1), m.group(2), m.group(3)
    h = re.search(_NUM + r"\s*m(?:eters?|etres?)?\s*(?:up|high|altitude)|at\s+" + _NUM + r"\s*m\b", phrase)
    zz = float(z) if z is not None else (float(h.group(1) or h.group(2)) if h else default_z)
    return [float(x), float(y), zz]


def _nearest(entities: Dict[str, str], cands: List[str], ref: Optional[str]) -> str:
    rp = _pos(entities, ref) if ref else None
    if rp is None:
        return cands[0]
    def d(n):
        p = _pos(entities, n)
        return float("inf") if p is None else (p[0] - rp[0]) ** 2 + (p[1] - rp[1]) ** 2
    return min(cands, key=d)


def _nice(name: str) -> str:
    if name.startswith("nofly"):
        return "the no-fly zone"
    if name == "payload.handle":
        return "the payload handle"
    if name in ("bin", "boat", "payload"):
        return "the " + name
    return name.replace(".", " ").replace("_", " ")


def offline_compile(text: str, scene: str, entities: Dict[str, str]) -> MissionSpec:
    """Deterministic keyword-based compiler (no network). Handles trash pickup, gentle cooler/organ
    delivery, net fishing to the boat, "go to X" and "avoid X"; otherwise hovers in place."""
    t = text.lower()
    names = list(entities)
    anchors: Dict[str, object] = {k: None for k in ("a0", "a1", "a2", "a3")}
    penalties: List[str] = []
    expl_extra: List[str] = []
    steps = None
    done = None

    gentle = _has(t, "gentl", "smooth", "careful", "organ", "cooler", "medical", "vaccine", "blood", "fragile",
                  "soft", "steady")
    avoid_obst = _has(t, "collid", "collision", "crash", "bump")
    nav = re.search(r"\b(go|fly|move|head|return|navigate|travel|land|hover)\b", t) is not None
    avoid_part = None
    m_avoid = re.search(r"(?:avoid|away from|stay out of|keep out of|not (?:getting |going )?(?:near|into|through)"
                        r"|without (?:getting |going )?(?:near|into|through)|steer clear of|clear of)\s+([^,.;]+)", t)
    if m_avoid:
        avoid_part = m_avoid.group(1)
    avoid_ent = _match_entity(avoid_part, entities) if avoid_part else None
    if avoid_ent is None and _has(t, "no-fly", "no fly", "nofly", "restricted") and "nofly_0" in entities:
        avoid_ent = "nofly_0"
    if avoid_ent is not None and avoid_ent.startswith("obstacle") and not re.search(r"\d", avoid_part or ""):
        avoid_obst, avoid_ent = True, None   # "avoid the obstacles" -> generic clearance term
    if avoid_part is not None and _has(avoid_part, "obstacle", "cart", "box", "thing"):
        avoid_obst = True

    if scene == "ocean" and "net.corner_0" in entities and (
            _has(t, "net", "haul", "trawl", "catch", "scoop") or (_has(t, "fish") and not nav)):
        schools = [n for n in names if n.startswith("fish_school")] or [None]
        school = schools[0]
        if len(schools) > 1:
            if _has(t, "closest to the boat", "nearest to the boat", "near the boat", "nearest the boat"):
                school = _nearest(entities, schools, "boat")
            else:
                m = re.search(r"school\s*_?(\d+)", t)
                school = f"fish_school_{m.group(1)}" if m and f"fish_school_{m.group(1)}" in entities else \
                    _nearest(entities, schools, "drone_0")
        anchors.update({"a0": "net.corner_{agent}", "a1": "boat" if "boat" in entities else None, "a2": school})
        reward = ("where(holding, 2*fish_in_net - 0.3*distxy(net, a2)*(fish_in_net < 0.5)"
                  " - distxy(net, a1)*(fish_in_net > 0.5), -dist(drone, a0))"
                  " - 2*net_stretch - 0.5*(mate_dist < 1)")
        done = "fish_in_net > 0.5 and distxy(net, a1) < 2.5"
        expl = ("Each drone grabs its own net corner, the team sweeps the net over "
                f"{school.replace('_', ' ') if school else 'the fish'} and hauls the catch to the boat "
                "without tearing the net")
        steps = [
            {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
             "explanation": "Each drone grabs its net corner."},
            {"reward": "holding*(2*fish_in_net - 0.3*distxy(net, a2)) - 2*net_stretch",
             "done_when": "fish_in_net > 0.5", "explanation": "Sweep the net over the fish."},
            {"reward": "holding*(fish_in_net - distxy(net, a1)) - 2*net_stretch",
             "done_when": "distxy(net, a1) < 2.5", "explanation": "Haul the net to the boat."},
        ]
    elif scene == "trash" and "payload.handle" in entities and (
            gentle or _has(t, "pick", "grab", "garbage", "litter", "deliver", "carry", "drop", "collect",
                           "throw", "dispose", "clean") or (_has(t, "trash", "bag") and not nav)):
        anchors.update({"a0": "payload.handle", "a1": "bin"})
        m_dest = re.search(r"(?:to|into|in|at)\s+(?:the\s+)?(?:point\s*)?(\(?\s*-?\d[^a-z]*)", t)
        dest_pt = _point_in(m_dest.group(1), 0.0) if m_dest else None
        if dest_pt is not None:
            anchors["a1"] = dest_pt
        item = "cooler" if _has(t, "cooler", "organ", "vaccine", "blood", "medical") else (
            "bag" if _has(t, "bag", "trash", "garbage", "litter") else "payload")
        actual = str(entities.get("payload", "")).lower()       # describe what is really in the world
        if actual.startswith("cooler"):
            item = "cooler"
        elif actual.startswith("trash bag"):
            item = "bag"
        dest = "the bin" if dest_pt is None else f"the point ({dest_pt[0]:g}, {dest_pt[1]:g})"
        reward = "where(holding, 2 - dist(payload, a1), -dist(drone, a0))"
        done = "distxy(payload, a1) < 0.6 and not holding"
        expl = f"Fly to the {item} handle, grab it, carry it to {dest} and let go"
        if gentle:
            penalties += ["0.5*swing", "0.01*jerk", "0.3*tilt"]
            expl += " as smoothly as possible (low swing and jerk)"
        steps = [
            {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
             "explanation": f"Grab the {item} handle."},
            {"reward": "-dist(payload, a1) + holding" + (" - 0.5*swing - 0.01*jerk" if gentle else ""),
             "done_when": done, "explanation": f"Carry the {item} to {dest} and release it."},
        ]
        if gentle and avoid_ent is None and "nofly_0" in entities:
            avoid_ent = "nofly_0"   # medical deliveries respect the no-fly zone by default
    else:
        target = None
        pt = None
        m_go = re.search(r"(?:go|fly|move|head|return|navigate|travel|land)\s+(?:over\s+|back\s+)?(?:to|toward|towards|"
                         r"above|on|at)?\s*(.+?)(?:\s+(?:and|while|without|but|avoid)\b|[,.;]|$)", t)
        phrase = m_go.group(1) if m_go else t
        pt = _point_in(t)
        if pt is None:
            target = _match_entity(phrase, entities)
            if target == avoid_ent:
                target = None
        if pt is not None:
            anchors["a0"] = pt
            reward = "-dist(drone, a0) - 0.1*speed*(dist(drone, a0) < 0.5)"
            done = "dist(drone, a0) < 0.3"
            expl = f"Fly to the point ({pt[0]:g}, {pt[1]:g}) at {pt[2]:g} m and hold there"
        elif target is not None:
            anchors["a0"] = target
            from .schema import infer_kind
            if infer_kind(target) == "zone":
                reward = "-distxy(drone, a0) - 0.3*abs(height - 3) - 0.1*speed*inside(a0)"
                done = "inside(a0) and speed < 0.5"
                expl = f"Fly over {_nice(target)} and hover above it"
            else:
                reward = "-dist(drone, a0) + 2*exp(-dist(drone, a0)) - 0.5*(height < 1)"
                done = "dist(drone, a0) < 1.0"
                expl = f"Fly close to {_nice(target)}"
        else:
            reward = "-abs(height - 3) - 0.3*speed - 0.2*tilt"
            done = None
            expl = "Mission not understood offline; hover in place at 3 m"

    if avoid_ent is not None:
        from .schema import infer_kind
        slot = "a2" if anchors["a2"] is None else "a3"
        anchors[slot] = avoid_ent
        if infer_kind(avoid_ent) == "zone":
            penalties.append(f"5*inside({slot})")
        else:
            penalties.append(f"3*exp(-dist(drone, {slot}))")
        expl_extra.append(f"avoiding {_nice(avoid_ent)}")
    if avoid_obst:
        penalties.append("2*exp(-2*obstacle_dist)")
        expl_extra.append("keeping clear of obstacles")
    if penalties:
        reward = reward + " - " + " - ".join(penalties)
        if steps:
            for st in steps:
                st["reward"] = st["reward"] + " - " + " - ".join(p for p in penalties if "inside" in p or "dist" in p)
    if expl_extra:
        expl += ", " + " and ".join(expl_extra)
    data = {"text": text, "scene": scene, "anchors": anchors, "reward": reward, "done_when": done,
            "explanation": expl + ".", "steps": steps, "offline": True}
    ms = MissionSpec.model_validate(data, context={"entities": names})
    dry_run(ms)
    return ms
