"""English mission -> MissionSpec, via Gemini (primary) or a keyword fallback.

Only this module touches google.genai, and only inside _default_client(), so the
rest of the package imports without it. Gemini never sees z: it binds anchors to
scene entities and writes a reward program in the DSL (see dsl.py).
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from pydantic import ValidationError

from .. import spec as S
from .dsl import DSLError, compile_reward, describe_vocabulary
from .schema import MissionError, MissionSpec, infer_kind, n_agents_from_entities

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEFAULT_MODEL = "gemini-3.5-flash"

# ------------------------------------------------------------------ prompt
# Each example: (scene, entity excerpt shown to the model, mission text, expected JSON).
_DELIVERY_REWARD = ("where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 0.5*swing - 0.01*jerk - 0.3*tilt "
                    "- 5*inside(a2) - 5*inside(a3) - 0.5*holding*(height < 2)*(distxy(payload, a1) > 3) "
                    "- 0.3*holding*over_water*(height < 3) - 0.5*(obstacle_dist < 0.5)")
_OCEAN_REWARD = ("where(holding, 2*fish_in_net - 0.3*distxy(net, a2)*(fish_in_net < 0.5) "
                 "- distxy(net, a1)*(fish_in_net > 0.5), -dist(drone, a0)) - 2*net_stretch - 0.5*(mate_dist < 1) "
                 "- 5*bycatch + 0.2*min(animal_dist, 5)")

_EXAMPLES = [
    ("trash",
     {"item_0": "trash_bag (trash) at [4.1, -2.0, 0.3]", "item_0.handle": "trash_bag handle (grab here) at [4.1, -2.0, 0.6]",
      "item_1": "backpack at [3.5, -2.2, 0.2]", "item_2": "pizza_box (trash) at [6.0, -2.4, 0.1]",
      "item_2.handle": "pizza_box handle (grab here) at [6.0, -2.4, 0.2]", "item_3": "(inactive)",
      "bin": "garbage truck drop-off at [12.0, 3.0, 0.0]", "drone_0": "drone at [0.0, 0.0, 2.0]"},
     "pick up the pizza box and throw it in the garbage truck", {
         "anchors": {"a0": "item_2.handle", "a1": "bin", "a2": None, "a3": None},
         "reward": ("where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 2*holding*(1 - payload_is('pizza_box')) "
                    "- 0.5*(obstacle_dist < 0.5)"),
         "done_when": "distxy(payload, a1) < 1.0 and not holding",
         "explanation": "Fly to the pizza box (item_2), grab its handle, carry it to the garbage truck and let go.",
         "steps": [
             {"reward": "-dist(drone, a0) + 3*holding - 2*holding*(1 - payload_is('pizza_box'))",
              "done_when": "holding", "explanation": "Grab the pizza box handle."},
             {"reward": "2 - dist(payload, a1) + holding - 0.5*(obstacle_dist < 0.5)",
              "done_when": "distxy(payload, a1) < 1.0 and not holding",
              "explanation": "Carry the pizza box to the garbage truck and release it."},
         ],
     }),
    ("trash",
     {"item_0": "trash_bag (trash) at [4.1, -2.0, 0.3]", "item_0.handle": "trash_bag handle (grab here) at [4.1, -2.0, 0.6]",
      "bin": "garbage truck drop-off at [12.0, 3.0, 0.0]", "drone_0": "drone at [0.0, 0.0, 2.0]"},
     "pick up the garbage bag, spin in a circle, wiggle up and down, then drop the garbage off", {
         "anchors": {"a0": "item_0.handle", "a1": "bin", "a2": None, "a3": None},
         "reward": "where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 0.5*(obstacle_dist < 0.5)",
         "done_when": "distxy(payload, a1) < 1.0 and not holding",
         "explanation": "Pick up the garbage bag, execute spin and wiggle maneuvers while holding it, and drop it in the garbage truck.",
         "steps": [
             {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
              "explanation": "Fly to the garbage bag and grab its handle."},
             {"reward": "speed - 0.5*abs(vz) + holding", "done_when": "speed > 0.8", "max_steps": 60,
              "explanation": "Spin / circle while holding the bag."},
             {"reward": "abs(vz) - 0.5*speed + holding", "done_when": "abs(vz) > 0.6", "max_steps": 60,
              "explanation": "Wiggle up and down with vertical velocity while holding the bag."},
             {"reward": "2 - dist(payload, a1) + holding - 0.5*(obstacle_dist < 0.5)",
              "done_when": "distxy(payload, a1) < 1.0 and not holding",
              "explanation": "Carry the garbage bag to the garbage truck and release it."},
         ],
     }),
    ("trash",
     {"item_0": "trash_bag (trash) at [4.1, -2.0, 0.3]", "item_0.handle": "trash_bag handle (grab here) at [4.1, -2.0, 0.6]",
      "item_1": "backpack at [3.5, -2.2, 0.2]", "item_1.handle": "backpack handle (grab here) at [3.5, -2.2, 0.4]",
      "item_2": "pizza_box (trash) at [6.0, -2.4, 0.1]",
      "item_2.handle": "pizza_box handle (grab here) at [6.0, -2.4, 0.2]", "item_3": "(inactive)",
      "bin": "garbage truck drop-off at [12.0, 3.0, 0.0]", "drone_0": "drone at [0.0, 0.0, 2.0]"},
     "clear the trash pile", {
         "anchors": {"a0": "item_0.handle", "a1": "bin", "a2": None, "a3": None},
         "reward": ("where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 2*holding*(1 - payload_is_trash) "
                    "- 2*trash_left - 0.5*(obstacle_dist < 0.5)"),
         "done_when": "trash_left < 0.01",
         "explanation": ("Carry the trash bag (item_0) and then the pizza box (item_2) to the garbage truck one at "
                         "a time, leaving the backpack where it is."),
         "steps": [
             {"anchors": {"a0": "item_0.handle", "a1": "bin", "a2": None, "a3": None},
              "reward": "where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 2*holding*(1 - payload_is_trash)",
              "done_when": "distxy(payload, a1) < 1.0 and not holding",
              "explanation": "Grab the trash bag (item_0) and drop it in the garbage truck."},
             {"anchors": {"a0": "item_2.handle", "a1": "bin", "a2": None, "a3": None},
              "reward": "where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - 2*holding*(1 - payload_is_trash)",
              "done_when": "distxy(payload, a1) < 1.0 and not holding",
              "explanation": "Grab the pizza box (item_2) and drop it in the garbage truck."},
         ],
     }),
    ("delivery",
     {"item_0": "cooler at [-15.0, -12.0, 3.1]", "item_0.handle": "cooler handle (grab here) at [-15.0, -12.0, 3.4]",
      "pad_pickup": "pickup helipad at [-15.0, -12.0, 2.8], radius 2.0",
      "pad_dropoff": "hospital helipad at [16.0, 14.0, 6.5], radius 2.0",
      "nofly_0": "no-fly zone at [0.0, 2.0, 0.0], radius 5.0", "nofly_1": "no-fly zone at [8.0, -6.0, 0.0], radius 4.0",
      "drone_0": "drone at [-14.0, -12.0, 5.0]"},
     "deliver the donor heart to the hospital, gently", {
         "anchors": {"a0": "item_0.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"},
         "reward": _DELIVERY_REWARD,
         "done_when": "distxy(payload, a1) < 1.0 and not holding",
         "explanation": ("Grab the organ cooler at the pickup pad and carry it smoothly (low swing and jerk) to the "
                         "hospital pad, staying out of both no-fly zones and well above the hills and lakes."),
         "steps": [
             {"reward": "-dist(drone, a0) + 3*holding - 5*inside(a2) - 5*inside(a3)", "done_when": "holding",
              "explanation": "Grab the cooler handle at the pickup pad."},
             {"reward": ("2 - dist(payload, a1) + holding - 0.5*swing - 0.01*jerk - 0.3*tilt - 5*inside(a2) "
                         "- 5*inside(a3) - 0.5*holding*(height < 2)*(distxy(payload, a1) > 3)"),
              "done_when": "distxy(payload, a1) < 1.0 and not holding",
              "explanation": "Carry the cooler gently to the hospital pad and set it down."},
         ],
     }),
    ("ocean",
     {"net.corner_0": "net corner 0 (grab here) at [-2.0, -2.0, 0.0]",
      "net.corner_1": "net corner 1 (grab here) at [2.0, -2.0, 0.0]", "net.center": "net centre at [0.0, 0.0, -0.5]",
      "boat": "boat at [15.0, 0.0, 0.0], radius 3.0", "fish_school_0": "fish school at [-8.0, 6.0, -2.0]",
      "whale_0": "whale at [-6.0, 9.0, -3.0]", "dolphin_0": "dolphin at [4.0, -5.0, -1.0]",
      "drone_0": "drone at [-2.0, -2.0, 2.0]", "drone_1": "drone at [2.0, -2.0, 2.0]"},
     "drag the net over the fish and haul it to the boat, don't catch any whales or dolphins", {
         "anchors": {"a0": "net.corner_{agent}", "a1": "boat", "a2": "fish_school_0", "a3": None},
         "reward": _OCEAN_REWARD,
         "done_when": "fish_in_net > 0.5 and distxy(net, a1) < 2.5 and not bycatch",
         "explanation": ("Each drone grabs its own net corner, the team sweeps the net over the fish school and hauls "
                         "it to the boat without overstretching the net, keeping whales, dolphins and turtles out of it."),
     }),
    ("world",
     {"item_2": "trash_bag (trash) at [-41.0, 12.0, 0.3]",
      "item_2.handle": "trash_bag handle (grab here) at [-41.0, 12.0, 0.6]",
      "item_5": "cooler (organ transport) at [60.0, -80.0, 3.1]",
      "item_5.handle": "cooler handle (grab here) at [60.0, -80.0, 3.4]", "item_9": "backpack (not trash) at [-30.0, 5.0, 0.2]",
      "item_12": "(inactive)", "bin": "garbage truck drop-off, radius 3.2 m at [-55.0, 20.0, 0.0]",
      "pad_pickup": "hospital helipad (pickup), radius 3.0 m at [60.0, -80.0, 2.8]",
      "pad_dropoff": "hospital helipad (drop-off), radius 3.0 m at [-90.0, 110.0, 6.5]",
      "nofly_0": "no-fly zone, radius 15.0 m at [0.0, 40.0, 0.0]", "nofly_1": "no-fly zone, radius 12.0 m at [-60.0, 70.0, 0.0]",
      "boat": "boat deck, delivery zone radius 2.5 m at [120.0, 0.0, 0.5]",
      "drone_0": "drone at [-35.0, 10.0, 3.0]", "drone_1": "drone at [55.0, -75.0, 4.0]",
      "drone_2": "drone at [110.0, 5.0, 3.0]"},
     "drone 0 picks up the trash bag and drone 1 takes the organ to the hospital", {
         "anchors": {"a0": None, "a1": None, "a2": None, "a3": None},
         "reward": "-abs(height - 3) - 0.3*speed - 0.2*tilt",
         "explanation": ("Drone 0 carries the trash bag (item_2) to the garbage truck while drone 1 flies the organ "
                         "cooler (item_5) gently to the drop-off hospital pad around both no-fly zones; drone 2 "
                         "hovers."),
         "agents": {
             "0": {"anchors": {"a0": "item_2.handle", "a1": "bin", "a2": None, "a3": None},
                   "reward": ("where(holding, 2 - dist(payload, a1), -dist(drone, a0)) "
                              "- 2*holding*(1 - payload_is('trash_bag')) - 0.5*(obstacle_dist < 0.5)"),
                   "done_when": "distxy(payload, a1) < 1.0 and not holding",
                   "explanation": "Grab the trash bag (item_2) and drop it in the garbage truck."},
             "1": {"anchors": {"a0": "item_5.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"},
                   "reward": _DELIVERY_REWARD + " - 2*holding*(1 - payload_is('cooler'))",
                   "done_when": "distxy(payload, a1) < 1.0 and not holding",
                   "explanation": "Carry the organ cooler (item_5) gently to the drop-off pad, avoiding the no-fly zones.",
                   "steps": [
                       {"reward": "-dist(drone, a0) + 3*holding - 5*inside(a2) - 5*inside(a3)", "done_when": "holding",
                        "explanation": "Grab the cooler handle at the pickup pad."},
                       {"reward": ("2 - dist(payload, a1) + holding - 0.5*swing - 0.01*jerk - 0.3*tilt "
                                   "- 5*inside(a2) - 5*inside(a3)"),
                        "done_when": "distxy(payload, a1) < 1.0 and not holding",
                        "explanation": "Carry the cooler to the drop-off pad and set it down."},
                   ]},
         },
     }),
    ("trash", {"drone_0": "drone at [0.0, 0.0, 2.0]"},
     "go to the point (3, -2) at 4 m and hover", {
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
           A step that gives "anchors" replaces ALL four anchors, so repeat any no-fly / avoid anchors in it.
  "agents": optional, multi-drone scenes only, when the operator gives DIFFERENT drones DIFFERENT jobs:
            {"<drone index>": {"anchors", "reward", "done_when"?, "explanation", "steps"?}}. A listed drone runs
            its own sub-mission (anchors are per drone, a0..a3 each); drones not listed run the top-level mission
            (make it a hover when they have nothing to do). Omit "agents" when all drones share one job.
Guidelines: keep rewards dense (distance terms), keep weights small (|w| <= 10), penalise inside(no-fly anchor)
when told to avoid a zone, use swing/jerk/tilt penalties for gentle carrying, obstacle_dist for keeping clear.
Only the drone can grab handles (".handle" entities and net corners): grabbing happens by closing the gripper
near a handle, so "holding" tells you whether the payload is in hand. The drone carries ONE item at a time.
Scenes:
  trash: graspable items item_0..item_9, each with item_i.handle. The entity descriptions give each item's type
    and whether it is trash; "(inactive)" items do not exist in this world - never bind them. Bind the handle of
    the item the operator means (the nearest one of that type if several) and carry it to "bin" (a garbage truck
    in a city, a trash bin on open terrain). Never pick up items that are not trash unless explicitly asked.
    "Clear the pile / pick up all the trash" binds a0 to the selector "nearest:trash" (or "nearest:<item type>"),
    which re-targets to the closest matching item every step, and a1 to the drop-off; one mission, no steps. Older form:
    one step per trash item (nearest first), each step binding
    a0 to that item's handle and a1 to bin.
  delivery: item_0 is the organ cooler (handle item_0.handle) waiting at pad_pickup; deliver it to pad_dropoff
    gently (swing / jerk / tilt penalties), avoid nofly_0 and nofly_1, and stay well above the terrain
    (penalise low height while carrying; there are hills, mountains, valleys and lakes, see over_water).
  ocean: a team of drones holds the net corners (net.corner_{agent}); haul fish to the boat without overstretching
    the net (net_stretch; an overstretched corner slips out of its gripper) and keep marine animals (whale_0, dolphin_*, turtle_*) out of the net: penalise bycatch and
    reward animal_dist.
  world: one 400 m map with all of the above and 3 drones (drone_0..drone_2): a town with trash and the garbage
    truck ("bin"), countryside litter, two clinics (pad_pickup, pad_dropoff), the open sea with "boat", the net,
    fish schools and animals, nofly_0 / nofly_1. Each of the three missions above works here with the same rules.
    The organ cooler is NOT item_0: it is the item_i whose description says "cooler"; bind its handle. Never
    bind the cooler as trash. Net fishing always uses all three drones, each on net.corner_{agent} (no "agents").
    When the operator assigns jobs to specific drones ("drone 0 cleans up, drone 1 delivers the organ"), use
    "agents" with one sub-mission per named drone."""


def build_prompt(text: str, scene: str, entities: Dict[str, str]) -> str:
    ents = "\n".join(f"  {n}: {d}" for n, d in entities.items())
    ex = []
    for sc, ex_ents, t, out in _EXAMPLES:
        ex_list = "; ".join(f"{n}: {d}" for n, d in ex_ents.items())
        ex.append(f"Scene: {sc}\nEntities (excerpt): {ex_list}\nMission: {t}\nJSON: {json.dumps(out)}")
    return (
        "You turn a drone mission written in English into a reward program for a pretrained drone policy.\n"
        f"Coordinates: world frame, z up, metres; the {scene} arena is "
        f"{2 * S.SCENE_ARENA_HALF.get(scene, S.ARENA_HALF):.0f} x {2 * S.SCENE_ARENA_HALF.get(scene, S.ARENA_HALF):.0f} m "
        "centred at the origin.\n\n"
        f"{describe_vocabulary()}\n\n{_SCHEMA_DOC}\n\n"
        "Worked examples (entity names and positions differ per world; use only the names listed for this world):\n\n"
        + "\n\n".join(ex) +
        f"\n\nNow the real task.\nScene: {scene}\nEntities in this scene:\n{ents}\nMission: {text}\nJSON:"
    )


# ------------------------------------------------------------------ validation
def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    return m.group(1) if m else raw


def parse_and_validate(raw: str, text: str, scene: str, entities) -> MissionSpec:
    """Parse model output into a validated MissionSpec. Raises ValueError with a readable message.

    entities: names, or a name -> description mapping (lets the world's "cooler" resolve to the right item)."""
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
        ents = entities if isinstance(entities, Mapping) else list(entities)
        ms = MissionSpec.model_validate(data, context={"entities": ents})
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
    for what, e in ms.expressions():
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
                ms = parse_and_validate(raw, text, scene, entities)
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
# Entity descriptions (env.entity_descriptions()) are parsed loosely: the first "[x, y, z]" (or "(x, y, z)")
# is the position, the first ITEM_TYPES word is the item type, "(inactive)" anywhere marks an unused slot,
# "(trash)" / "(not trash)" override the catalogue's is_trash. Without positions, listing order is used.
_F = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_POS_RE = re.compile(r"[\[(]\s*(" + _F + r")\s*,\s*(" + _F + r")\s*,\s*(" + _F + r")\s*[\])]")
_NUM = r"(-?\d+(?:\.\d+)?)"
_INF = float("inf")

# item type -> English phrases naming it (matched longest first, whole words, optional plural)
_TYPE_WORDS = {
    "trash_bag": ("trash bag", "garbage bag", "rubbish bag", "bin bag", "trashbag", "bag of trash", "bag"),
    "pizza_box": ("pizza box", "pizza"),
    "cardboard_box": ("cardboard box", "cardboard", "carton", "box"),
    "bottle": ("plastic bottle", "water bottle", "bottle"),
    "backpack": ("backpack", "rucksack", "school bag"),
    "ball": ("soccer ball", "football", "basketball", "ball"),
    "flower_pot": ("flower pot", "flowerpot", "plant pot", "potted plant", "pot"),
    "cooler": ("organ cooler", "cooler", "organ", "heart", "kidney", "liver", "lung"),
}
_TYPE_PHRASES = sorted(((w, t) for t, ws in _TYPE_WORDS.items() for w in ws if t in S.ITEM_TYPES),
                       key=lambda x: -len(x[0]))


def _pos(entities: Dict[str, str], name: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not name:
        return None
    m = _POS_RE.search(str(entities.get(name, "")))
    return tuple(float(x) for x in m.groups()) if m else None


def _has(t: str, *words: str) -> bool:
    """Prefix match at a word start ("gentl" matches "gently")."""
    return any(re.search(r"\b" + re.escape(w), t) for w in words)


def _word(t: str, *words: str) -> bool:
    """Whole-word match, optional plural ("cart" matches "carts", not "carry")."""
    return any(re.search(r"\b" + re.escape(w) + r"(?:e?s)?\b", t) for w in words)


def _types_in(t: str) -> List[str]:
    """Item types named in English text, in order of appearance ("the pizza box" -> ["pizza_box"])."""
    t = t.lower()
    found = []
    for phrase, typ in _TYPE_PHRASES:
        for m in re.finditer(r"\b" + re.escape(phrase) + r"(?:e?s)?\b", t):
            found.append((m.start(), typ))
            t = t[:m.start()] + "#" * (m.end() - m.start()) + t[m.end():]   # "pizza box" consumes "box"
    seen: List[str] = []
    for _, typ in sorted(found):
        if typ not in seen:
            seen.append(typ)
    return seen


def _item_type(desc: str) -> Optional[str]:
    d = desc.lower()
    best = None
    for typ in S.ITEM_TYPES:
        for form in {typ, typ.replace("_", " ")}:
            m = re.search(r"(?<![a-z])" + re.escape(form) + r"(?![a-z])", d)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), typ)
    return best[1] if best else None


def _items(entities: Dict[str, str]) -> List[dict]:
    """Graspable items item_i (listing order): name, target (handle if listed), type, trash, active, pos."""
    order: List[int] = []
    for n in entities:
        m = re.fullmatch(r"item_(\d+)(?:\.handle)?", n)
        if m and int(m.group(1)) not in order:
            order.append(int(m.group(1)))
    out = []
    for i in order:
        name, handle = f"item_{i}", f"item_{i}.handle"
        d_item, d_handle = str(entities.get(name, "")), str(entities.get(handle, ""))
        both = (d_item + " " + d_handle).lower()
        typ = _item_type(d_item) or _item_type(d_handle)
        if re.search(r"not trash|non[- ]?trash", both):
            trash = False
        elif re.search(r"\(trash\)|\bis trash\b", both):
            trash = True
        else:
            trash = bool(typ is not None and S.ITEM_IS_TRASH[S.ITEM_TYPES.index(typ)])
        out.append({"index": i, "name": name if name in entities else handle,
                    "target": handle if handle in entities else name, "type": typ, "trash": trash,
                    "active": re.search(r"\binactive\b", both) is None,
                    "pos": _pos(entities, name) or _pos(entities, handle)})
    return out


def _d2(p, q) -> float:
    if p is None or q is None:
        return _INF
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2


def _by_distance(items: List[dict], ref) -> List[dict]:
    """Stable sort by distance to ref; listing order when positions are missing."""
    return [it for _, _, it in sorted((_d2(it["pos"], ref), k, it) for k, it in enumerate(items))]


def _tour(items: List[dict], start, dropoff) -> List[dict]:
    """Nearest-first order: the item nearest the drone first, then (the drone is back at the drop-off after each
    delivery) the rest by distance to the drop-off."""
    if not items:
        return []
    first = _by_distance(items, start if start is not None else dropoff)[0]
    rest = [it for it in items if it is not first]
    return [first] + _by_distance(rest, dropoff if dropoff is not None else start)


def _item_label(it: dict) -> str:
    typ = (it["type"] or "item").replace("_", " ")
    return f"the {typ} ({it['name']})"


def _nice(name: str, entities: Dict[str, str], items: Optional[List[dict]] = None) -> str:
    desc = str(entities.get(name, "")).lower()
    for it in items or []:
        if name in (it["name"], it["target"]):
            return _item_label(it)
    if name.startswith("nofly"):
        return "the no-fly zone" if name == "nofly_0" and "nofly_1" not in entities else f"no-fly zone {name[6:]}"
    if name == "bin":
        return "the garbage truck" if "truck" in desc else ("the trash bin" if "bin" in desc else "the drop-off")
    if name == "pad_dropoff":
        return "the hospital pad" if "hospital" in desc else "the drop-off pad"
    if name == "pad_pickup":
        return "the pickup pad"
    m = re.fullmatch(r"(whale|dolphin|turtle)_(\d+)", name)
    if m:
        return f"the {m.group(1)} ({name})"
    if name in ("boat", "payload"):
        return "the " + name
    if name.endswith(".handle"):
        return "the " + name[:-7].replace("_", " ") + " handle"
    return name.replace(".", " ").replace("_", " ")


_NUMBERED = {"item": "item", "obstacle": "obstacle", "building": "obstacle", "box": "obstacle", "car": "obstacle",
             "cart": "obstacle", "drone": "drone", "fish school": "fish_school", "school": "fish_school",
             "corner": "net.corner", "whale": "whale", "dolphin": "dolphin", "turtle": "turtle",
             "nofly": "nofly", "no-fly zone": "nofly", "no fly zone": "nofly", "no-fly": "nofly"}
_WORD_TABLE = [
    (("pickup pad", "pick-up pad", "pickup", "pick-up", "origin", "donor"), "pad_pickup"),
    (("hospital", "drop-off", "dropoff", "destination", "helipad", "landing pad", "pad"), "pad_dropoff"),
    (("garbage truck", "truck", "bin", "dumpster", "trash can", "garbage can"), "bin"),
    (("no-fly", "no fly", "nofly", "restricted"), "nofly_0"),
    (("boat", "ship", "vessel", "deck"), "boat"),
    (("net",), "net.center"), (("fish", "school"), "fish_school_0"),
    (("whale",), "whale_0"), (("dolphin",), "dolphin_0"), (("turtle",), "turtle_0"),
    (("obstacle", "building", "car", "cart", "box", "bench", "hydrant", "wall"), "obstacle_0"),
    (("drone",), "drone_0"),
]


def _match_entity(phrase: str, entities: Dict[str, str], items: Optional[List[dict]] = None,
                  drone: str = "drone_0") -> Optional[str]:
    """Best entity named in a phrase ("the boat", "obstacle 3", "the pizza box", "no-fly zone", "drone 1")."""
    p = phrase.lower()
    keys = "|".join(re.escape(k) for k in sorted(_NUMBERED, key=len, reverse=True))
    m = re.search(r"\b(" + keys + r")s?\s*_?#?(\d+)", p)
    if m:
        cand = f"{_NUMBERED[m.group(1)]}_{m.group(2)}"
        if cand in entities:
            return cand
    if items:
        types = _types_in(p)
        pool = [it for it in items if it["active"] and it["type"] in types]
        if pool:
            return _by_distance(pool, _pos(entities, drone))[0]["name"]
    for words, ent in _WORD_TABLE:
        if ent in entities and _word(p, *words):
            return ent
    for name in entities:
        if re.search(r"\b" + re.escape(name.replace("_", " ")) + r"\b", p) or re.search(
                r"\b" + re.escape(name) + r"\b", p):
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
    rp = _pos(entities, ref)
    if rp is None:
        return cands[0]
    return min(cands, key=lambda n: _d2(_pos(entities, n), rp))


_ANIMAL_WORDS = ("animal", "whale", "dolphin", "turtle", "wildlife", "marine life", "mammal", "bycatch", "creature")
_CARRY_RE = re.compile(r"\b(pick|grab|collect|throw|dispose|clean|carry|carrie|deliver|drop|bring|put|take|fetch|remove|"
                       r"tidy|gather|transport|haul|rush|ferry|ferrie|retrieve)(?:s|es|d|ed|ing)?\b|\bclear\b(?!\s+of)")
_ALL_RE = re.compile(r"\b(all|every|each|everything|pile|piles|heap|entire|whole)\b")
_NAV_RE = re.compile(r"\b(go(?:es)?|fl(?:y|ies)|moves?|heads?|returns?|navigates?|travels?|lands?|hover(?:s)?)\b")
_COOLER_WORDS = ("cooler", "organ", "heart", "kidney", "liver", "lung", "transplant", "medical", "vaccine", "blood")
_HOVER_REWARD = "-abs(height - 3) - 0.3*speed - 0.2*tilt"
# "drone 0 ... and drone 1 ...": a drone named at the start of a clause gets the clause up to the next such drone
_SPLIT_RE = re.compile(r"(?:^|[,;.]\s*|\b(?:and|while|whereas|then|but)\s+)(?:let\s+|have\s+|make\s+)?(?:the\s+)?"
                       r"drone\s*[_#]?\s*(\d+)\b\s*(?:should\s+|will\s+|must\s+|can\s+|is\s+to\s+|to\s+)?", re.I)


def _split_by_drone(text: str, n_agents: int) -> Optional[Dict[int, str]]:
    """{drone index: its clause} for "drone 0 picks up the trash and drone 1 delivers the cooler"; else None."""
    hits = [m for m in _SPLIT_RE.finditer(text) if int(m.group(1)) < n_agents]
    if not hits:
        return None
    out: Dict[int, str] = {}
    for j, m in enumerate(hits):
        end = hits[j + 1].start() if j + 1 < len(hits) else len(text)
        seg = text[m.end():end].strip(" ,;.")
        k = int(m.group(1))
        out[k] = (out[k] + " and " + seg) if k in out else seg
    return out


def _cooler(scene: str, entities: Dict[str, str], items: List[dict]) -> Optional[str]:
    """Anchor target for the organ cooler: item_0.handle in delivery, the item described as a cooler in the world."""
    if scene == "delivery":
        return "item_0.handle" if "item_0.handle" in entities else None
    hits = [it for it in items if it["type"] == "cooler" and it["active"]] or \
        [it for it in items if it["type"] == "cooler"]
    return hits[0]["target"] if hits else None


def _single(text: str, scene: str, entities: Dict[str, str], drone: str = "drone_0") -> dict:
    """One drone's (or the whole team's) mission from English: anchors, reward, done_when, explanation, steps,
    warnings and kind ("ocean" | "delivery" | "trash" | "goto" | "hover")."""
    t = text.lower()
    names = list(entities)
    items = _items(entities)
    active = [it for it in items if it["active"]]
    drone_p, bin_p = _pos(entities, drone), _pos(entities, "bin")
    anchors: Dict[str, object] = {k: None for k in ("a0", "a1", "a2", "a3")}
    penalties: List[str] = []          # added to the top-level reward and to every step
    expl_extra: List[str] = []
    warnings: List[str] = []
    steps: Optional[List[dict]] = None
    done = None
    world = scene == "world"

    gentle = _has(t, "gentl", "smooth", "careful", "organ", "cooler", "medical", "vaccine", "blood", "fragile",
                  "soft", "steady", "heart", "kidney", "transplant")
    avoid_obst = _has(t, "collid", "collision", "crash", "bump")
    nav = _NAV_RE.search(t) is not None
    carry = _CARRY_RE.search(t) is not None
    types_named = _types_in(t)
    m_idx = re.search(r"\bitem\s*_?#?(\d+)", t)

    avoid_part = None
    m_avoid = re.search(r"(?:avoid|away from|stay out of|keep out of|not (?:getting |going )?(?:near|into|through)"
                        r"|without (?:getting |going )?(?:near|into|through)|steer clear of|clear of|"
                        r"(?:don'?t|do not|never) (?:catch|net|hit|tangle|harm|hurt|disturb))\s+([^,.;]+)", t)
    if m_avoid:
        avoid_part = m_avoid.group(1)
    avoid_ent = _match_entity(avoid_part, entities, drone=drone) if avoid_part else None
    avoid_animals = avoid_part is not None and _has(avoid_part, *_ANIMAL_WORDS)
    if avoid_animals and avoid_ent is not None and not re.search(r"\d", avoid_part or ""):
        if _has(avoid_part, "animal", "wildlife", "marine", "mammal", "bycatch", "creature") or \
                len([w for w in ("whale", "dolphin", "turtle") if w in avoid_part]) > 1:
            avoid_ent = None    # generic "animals": bycatch / animal_dist terms cover it
    if avoid_ent is None and _has(t, "no-fly", "no fly", "nofly", "restricted") and "nofly_0" in entities:
        avoid_ent = "nofly_0"
    if avoid_ent is not None and avoid_ent.startswith("obstacle") and not re.search(r"\d", avoid_part or ""):
        avoid_obst, avoid_ent = True, None   # "avoid the obstacles" -> generic clearance term
    if avoid_part is not None and _word(avoid_part, "obstacle", "cart", "car", "box", "thing", "building", "wall"):
        avoid_obst = True

    dest_pt = None
    m_dest = re.search(r"(?:to|into|in|at|on)\s+(?:the\s+)?(?:point\s*)?(\(?\s*-?\d[^a-z]*)", t)
    if m_dest:
        dest_pt = _point_in(m_dest.group(1), 0.0)

    cooler = _cooler(scene, entities, items) if scene in ("delivery", "world") else None
    if scene == "ocean":
        ocean_task = "net.corner_0" in entities and (
            _has(t, "net", "haul", "trawl", "catch", "scoop") or (_has(t, "fish") and not nav))
    elif world:
        ocean_task = "net.corner_0" in entities and (
            _word(t, "net", "trawl", "scoop") or (_has(t, "fish", "catch") and not nav))
    else:
        ocean_task = False
    if scene == "delivery":
        delivery_task = cooler is not None and (
            carry or _has(t, "cooler", "organ", "heart", "kidney", "liver", "transplant", "medical", "package",
                          "payload", "parcel"))
    elif world:
        delivery_task = cooler is not None and (
            _has(t, *_COOLER_WORDS) or (carry and _has(t, "hospital", "clinic", "helipad", "parcel", "package")))
    else:
        delivery_task = False
    trash_task = scene in ("trash", "world") and bool(items) and (
        carry or ((types_named or m_idx or _has(t, "trash", "garbage", "litter", "rubbish", "junk", "waste"))
                  and not nav))

    if ocean_task:
        kind = "ocean"
        schools = [n for n in names if n.startswith("fish_school")] or [None]
        school = schools[0]
        if len(schools) > 1:
            if _has(t, "closest to the boat", "nearest to the boat", "near the boat", "nearest the boat"):
                school = _nearest(entities, schools, "boat")
            else:
                m = re.search(r"school\s*_?(\d+)", t)
                school = f"fish_school_{m.group(1)}" if m and f"fish_school_{m.group(1)}" in entities else \
                    _nearest(entities, schools, drone)
        anchors.update({"a0": "net.corner_{agent}", "a1": "boat" if "boat" in entities else None, "a2": school})
        w_by = 10 if (avoid_animals or _has(t, *_ANIMAL_WORDS)) else 5
        safe = f" - {w_by}*bycatch + 0.2*min(animal_dist, 5)"
        reward = ("where(holding, 2*fish_in_net - 0.3*distxy(net, a2)*(fish_in_net < 0.5)"
                  " - distxy(net, a1)*(fish_in_net > 0.5), -dist(drone, a0))"
                  " - 2*net_stretch - 0.5*(mate_dist < 1)" + safe)
        done = "fish_in_net > 0.5 and distxy(net, a1) < 2.5 and not bycatch"
        expl = ("Each drone grabs its own net corner, the team sweeps the net over "
                f"{school.replace('_', ' ') if school else 'the fish'} and hauls the catch to the boat "
                "without overstretching the net or tangling whales, dolphins or turtles in it")
        steps = [
            {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
             "explanation": "Each drone grabs its net corner."},
            {"reward": "holding*(2*fish_in_net - 0.3*distxy(net, a2)) - 2*net_stretch" + safe,
             "done_when": "fish_in_net > 0.5", "explanation": "Sweep the net over the fish, away from animals."},
            {"reward": "holding*(fish_in_net - distxy(net, a1)) - 2*net_stretch" + safe,
             "done_when": "distxy(net, a1) < 2.5", "explanation": "Haul the net to the boat."},
        ]
    elif delivery_task:
        kind = "delivery"
        dest = "pad_dropoff" if "pad_dropoff" in entities else ("bin" if "bin" in entities else None)
        if dest_pt is not None:
            dest = dest_pt
        anchors.update({"a0": cooler, "a1": dest})
        for slot, nf in (("a2", "nofly_0"), ("a3", "nofly_1")):
            if nf in entities:
                anchors[slot] = nf
                penalties.append(f"5*inside({slot})")
        dest_txt = _nice(dest, entities) if isinstance(dest, str) else f"the point ({dest[0]:g}, {dest[1]:g})"
        gentle_pen = "0.5*swing - 0.01*jerk - 0.3*tilt"
        terrain_pen = "0.5*holding*(height < 2)*(distxy(payload, a1) > 3) - 0.3*holding*over_water*(height < 3)"
        reward = f"where(holding, 2 - dist(payload, a1), -dist(drone, a0)) - {gentle_pen} - {terrain_pen}"
        if world:
            reward += " - 2*holding*(1 - payload_is('cooler'))"
        done = "distxy(payload, a1) < 1.0 and not holding"
        pick = " at the pickup pad" if "pad_pickup" in entities else ""
        which = f" ({cooler.split('.')[0]})" if world else ""
        expl = (f"Grab the organ cooler{which}{pick} and carry it as smoothly as possible (low swing and jerk) to "
                f"{dest_txt}, staying well above the hills and lakes")
        steps = [
            {"reward": "-dist(drone, a0) + 3*holding", "done_when": "holding",
             "explanation": f"Grab the cooler handle{pick}."},
            {"reward": f"2 - dist(payload, a1) + holding - {gentle_pen} - {terrain_pen}", "done_when": done,
             "explanation": f"Carry the cooler gently to {dest_txt} and set it down."},
        ]
        if any(n in entities for n in ("nofly_0", "nofly_1")):
            expl_extra.append("never entering the no-fly zones")
        penalties.append("0.5*(obstacle_dist < 0.5)")
    elif trash_task:
        kind = "trash"
        dest = dest_pt if dest_pt is not None else ("bin" if "bin" in entities else None)
        dest_txt = _nice("bin", entities) if dest == "bin" else (
            f"the point ({dest_pt[0]:g}, {dest_pt[1]:g})" if dest_pt is not None else "the drop-off")
        multi = _ALL_RE.search(t) is not None or (
            re.search(r"\b(clean|tidy)\b|\bclear\b(?!\s+of)", t) is not None and not (types_named or m_idx))
        penalties.append("0.5*(obstacle_dist < 0.5)")
        gentle_pen = " - 0.5*swing - 0.01*jerk" if gentle else ""
        pick_done = "distxy(payload, a1) < 1.0 and not holding"
        chosen: List[dict] = []
        if m_idx and not multi:
            hit = [it for it in items if it["index"] == int(m_idx.group(1))]
            if hit and hit[0]["active"]:
                chosen = hit
            else:
                warnings.append(f"item_{m_idx.group(1)} is not an active item; using the nearest trash instead")
        if not chosen:
            if types_named:
                pool = [it for it in active if it["type"] in types_named]
                if not pool:
                    warnings.append("no active " + " / ".join(x.replace("_", " ") for x in types_named) +
                                    " in this world; using trash items instead")
                    pool = [it for it in active if it["trash"]]
            else:
                pool = [it for it in active if it["trash"]]
            chosen = _tour(pool, drone_p, bin_p) if multi else _by_distance(pool, drone_p)[:1]
        all_trash = multi and not types_named
        if not chosen:
            kind = "hover"
            warnings.append("no matching active item found")
            reward = _HOVER_REWARD
            expl = "No matching item found in this world; hover in place at 3 m"
            penalties = []
        elif not multi:
            it = chosen[0]
            anchors.update({"a0": it["target"], "a1": dest})
            wrong = (f" - 2*holding*(1 - payload_is('{it['type']}'))" if it["type"] else
                     (" - 2*holding*(1 - payload_is_trash)" if it["trash"] else ""))
            reward = "where(holding, 2 - dist(payload, a1), -dist(drone, a0))" + wrong + gentle_pen
            done = pick_done
            lab = _item_label(it)
            expl = f"Fly to {lab}, grab its handle, carry it to {dest_txt} and let go"
            if gentle:
                expl += " as smoothly as possible (low swing and jerk)"
            maneuvers = []
            if _has(t, "spin", "circle", "rotate"):
                maneuvers.append({
                    "reward": "speed - 0.5*abs(vz) + holding",
                    "done_when": "speed > 0.8",
                    "max_steps": 60,
                    "explanation": f"Spin or circle while holding {lab}.",
                })
            if _has(t, "wiggle", "bob", "up and down"):
                maneuvers.append({
                    "reward": "abs(vz) - 0.5*speed + holding",
                    "done_when": "abs(vz) > 0.6",
                    "max_steps": 60,
                    "explanation": f"Wiggle up and down while holding {lab}.",
                })
            steps = [
                {"reward": "-dist(drone, a0) + 3*holding" + wrong, "done_when": "holding",
                 "explanation": f"Grab the handle of {lab}."},
                *maneuvers,
                {"reward": "2 - dist(payload, a1) + holding" + gentle_pen, "done_when": done,
                 "explanation": f"Carry {lab} to {dest_txt} and release it."},
            ]
        else:
            if len(types_named) == 1:
                wrong = f" - 2*holding*(1 - payload_is('{types_named[0]}'))"
            elif all(it["trash"] for it in chosen):
                wrong = " - 2*holding*(1 - payload_is_trash)"
            else:
                wrong = ""
            # one mission with a selector anchor: a0 re-targets to the closest matching item every
            # step (and stays on the item being carried), so the drone works through them all
            sel = f"nearest:{types_named[0]}" if len(types_named) == 1 else "nearest:trash"
            anchors.update({"a0": sel, "a1": dest})
            reward = "where(holding, 2 - dist(payload, a1), -dist(drone, a0))" + wrong + gentle_pen + \
                (" - 2*trash_left" if all_trash else "")
            done = "trash_left < 0.01 and not holding" if all_trash else "not active(a0) and not holding"
            what = types_named[0].replace("_", " ") if len(types_named) == 1 else "trash item"
            expl = f"Keep grabbing the nearest {what} and dropping it in {dest_txt} until none is left"
            skipped = [it for it in items if it["active"] and not (it["trash"] if not types_named else it["type"] in types_named)]
            if skipped:
                kinds = sorted({(x["type"] or "other item").replace("_", " ") for x in skipped})
                expl += ", leaving the " + " and the ".join(kinds) + (" where they are" if len(skipped) > 1
                                                                     else " where it is")
    else:
        kind = "goto"
        target = None
        m_go = re.search(r"(?:go(?:es)?|fl(?:y|ies)|moves?|heads?|returns?|navigates?|travels?|lands?)\s+"
                         r"(?:over\s+|back\s+)?(?:to|toward|towards|above|on|at)?\s*(.+?)"
                         r"(?:\s+(?:and|while|without|but|avoid)\b|[,.;]|$)", t)
        phrase = m_go.group(1) if m_go else t
        pt = _point_in(t)
        if pt is None:
            target = _match_entity(phrase, entities, items, drone=drone)
            if target == avoid_ent:
                target = None
        if pt is not None:
            anchors["a0"] = pt
            reward = "-dist(drone, a0) - 0.1*speed*(dist(drone, a0) < 0.5)"
            done = "dist(drone, a0) < 0.3"
            expl = f"Fly to the point ({pt[0]:g}, {pt[1]:g}) at {pt[2]:g} m and hold there"
        elif target is not None:
            anchors["a0"] = target
            if infer_kind(target) == "zone":
                reward = "-distxy(drone, a0) - 0.3*abs(height - 3) - 0.1*speed*inside(a0)"
                done = "inside(a0) and speed < 0.5"
                expl = f"Fly over {_nice(target, entities, items)} and hover above it"
            else:
                reward = "-dist(drone, a0) + 2*exp(-dist(drone, a0)) - 0.5*(height < 1)"
                done = "dist(drone, a0) < 1.0"
                expl = f"Fly close to {_nice(target, entities, items)}"
        else:
            kind = "hover"
            reward = _HOVER_REWARD
            done = None
            expl = "Mission not understood offline; hover in place at 3 m"

    anchor_sets = [anchors] + [st["anchors"] for st in steps or [] if "anchors" in st]
    if avoid_ent is not None:
        bound = [k for k in ("a0", "a1", "a2", "a3") if anchors[k] == avoid_ent]
        if bound and bound[0] in ("a2", "a3"):
            if not any(f"inside({bound[0]})" in p for p in penalties):
                penalties.append(f"5*inside({bound[0]})" if infer_kind(avoid_ent) == "zone"
                                 else f"3*exp(-dist(drone, {bound[0]}))")
        else:
            free = [k for k in ("a2", "a3") if all(a.get(k) is None for a in anchor_sets)]
            if not free:
                warnings.append(f"no free anchor slot to avoid {avoid_ent}")
            else:
                slot = free[0]
                for a in anchor_sets:
                    a[slot] = avoid_ent
                if infer_kind(avoid_ent) == "zone":
                    penalties.append(f"5*inside({slot})")
                elif scene in ("ocean", "world") and re.match(r"(whale|dolphin|turtle)_\d+$", avoid_ent):
                    penalties.append(f"3*exp(-dist(net, {slot}))")
                else:
                    penalties.append(f"3*exp(-dist(drone, {slot}))")
                expl_extra.append(f"avoiding {_nice(avoid_ent, entities, items)}")
    if avoid_obst:
        penalties.append("2*exp(-2*obstacle_dist)")
        expl_extra.append("keeping clear of obstacles")
    if penalties:
        reward = reward + " - " + " - ".join(penalties)
        for st in steps or []:
            st["reward"] = st["reward"] + " - " + " - ".join(penalties)
    if expl_extra:
        expl += ", " + " and ".join(expl_extra)
    return {"anchors": anchors, "reward": reward, "done_when": done, "explanation": expl + ".", "steps": steps,
            "warnings": warnings, "kind": kind}


def offline_compile(text: str, scene: str, entities: Dict[str, str]) -> MissionSpec:
    """Deterministic keyword-based compiler (no network).

    Handles: trash pickup of one item chosen by type words / item number ("pick up the pizza box" binds the
    nearest active pizza_box), "clear the pile / pick up all the trash" (one step per active trash item, nearest
    first), organ delivery (pad_pickup -> pad_dropoff, gentle, no-fly zones, terrain clearance), ocean net
    fishing to the boat without bycatch, "go to X" and "avoid X"; otherwise hovers in place.
    In the world scene all of these work with the world's entity names (the cooler is the item described as a
    cooler), and "drone 0 ... and drone 1 ..." gives each named drone its own job (MissionSpec.agents; drones not
    named hover). Net fishing always takes the whole team, each drone on its own corner.
    """
    names = list(entities)
    n_ag = n_agents_from_entities(names)
    split = _split_by_drone(text, n_ag) if scene == "world" and n_ag > 1 else None
    data: Dict[str, object] = {"text": text, "scene": scene, "offline": True}
    subs = {k: _single(seg, scene, entities, drone=f"drone_{k}") for k, seg in (split or {}).items()}
    if subs and any(s["kind"] == "ocean" for s in subs.values()):
        subs = {}
        one = _single(text, scene, entities)
        if one["kind"] != "ocean":
            one = _single("net the fish and haul it to the boat", scene, entities)
        one["warnings"].append("net fishing needs the whole team (each drone holds a net corner); "
                               "the per-drone split was ignored")
    elif subs:
        one = None
    else:
        one = _single(text, scene, entities)
    if one is not None:
        data.update({k: one[k] for k in ("anchors", "reward", "done_when", "explanation", "steps", "warnings")})
    else:
        warnings: List[str] = []
        agents = {}
        parts = []
        for k in sorted(subs):
            s = subs[k]
            agents[k] = {f: s[f] for f in ("anchors", "reward", "done_when", "explanation", "steps")}
            warnings += [f"drone {k}: {w}" for w in s["warnings"]]
            parts.append(f"Drone {k}: {s['explanation']}")
        idle = [k for k in range(n_ag) if k not in subs]
        if idle:
            who = " and ".join(f"drone {k}" for k in idle)
            parts.append(f"{who[0].upper() + who[1:]} {'hovers' if len(idle) == 1 else 'hover'} in place at 3 m.")
        data.update({"anchors": None, "reward": _HOVER_REWARD, "done_when": None, "explanation": " ".join(parts),
                     "steps": None, "agents": agents, "warnings": warnings})
    ms = MissionSpec.model_validate(data, context={"entities": entities})
    dry_run(ms)
    return ms
