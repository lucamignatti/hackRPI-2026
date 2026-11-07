# SkyHand

One drone brain, any mission. A zero-shot behavioral foundation model for a gripper drone with a dual-wavelength LiDAR, built at HackRPI 2026 ("In The Clouds"). You type a mission, an LLM turns it into a reward program, the reward becomes a latent task vector z, and the same frozen network flies it. Highlighted missions: NYC trash pickup, organ delivery, and multi-drone net fishing.

The full design, paper notes, and timeline are in [PLAN.md](PLAN.md).

## Layout

| Path | What it is |
|---|---|
| `skyhand/spec.py` | The shared contract: observation keys and shapes, goal-space layout g, anchors, env API. Change it only as a team |
| `skyhand/env/` | Environments. `core.py` is the env and the pure-PyTorch physics backend; `backends/genesis_backend.py` is the Genesis backend. Shared pieces: `controller.py`, `waves.py`, `net.py`, `fish.py`, `lidar.py`, `shield.py`, `expert.py` |
| `skyhand/train/` | FB-CPR learner (off-policy), SF fallback, cross-attention networks, replay buffer, z sampling, `Agent`, training CLI |
| `skyhand/mission/` | Reward DSL, Gemini mission compiler with offline fallback, z inference, CEM refinement, tiny text-to-z model |
| `demo/` | Judge demo: FastAPI server and three.js web viewer |
| `scripts/desktop/` | Desktop training launcher |
| `tests/` | Unit and integration tests for every module |

## Where things run

| Machine | Backend | Use |
|---|---|---|
| Desktop (Ryzen 7900X, RTX 5080) | `torch` and `genesis` | Training, Genesis evaluation, judge demo |
| Laptops (macOS / Linux) | `torch` and `genesis` (CPU or Metal) | Development and tests |

The training path never imports Genesis, so it runs anywhere PyTorch installs. Only `skyhand/env/backends/genesis_backend.py` does, and only when you ask for `backend="genesis"`.

## Setup

### Desktop or laptop
Python 3.12 or newer for Genesis (3.9+ for everything else).
```bash
uv venv --python 3.12 .venv
```
```bash
uv pip install --python .venv/bin/python torch genesis-world numpy pytest fastapi "uvicorn[standard]" httpx google-genai pydantic
```
```bash
uv pip install --python .venv/bin/python -e .
```
On the RTX 5080 install a PyTorch build for CUDA 12.8 or newer.

## Tests
```bash
.venv/bin/python -m pytest -q
```
- Genesis tests skip automatically where `genesis-world` is not installed.
- `tests/test_minimal_deps.py` trains for a few steps with Genesis, Gemini, and the web stack blocked.
- The `slow` marker covers the scripted-expert success tests. Skip them with `-m "not slow"`.

## Training
```bash
python -m skyhand.train.train --scene both --backend torch --device cuda --num-envs 2048 --steps 200000 --out runs/first
```
- `--method sf` switches to the successor-feature fallback with the same code path.
- `--backend genesis` trains on Genesis (desktop only).
- `--set key=value` overrides any field of `skyhand/train/config.py`.
- Outputs go to the run directory: `config.json`, `metrics.csv`, and the `agent.pt` checkpoint.

## Using a checkpoint
```python
from skyhand.train.agent import Agent
from skyhand.env.core import make_env
from skyhand.mission import compile_mission, to_bindings, reward_to_z

agent = Agent.load("runs/first/agent.pt", device="cpu")
env = make_env("trash", num_envs=1, backend="genesis")    # or "torch"
spec = compile_mission("pick up the bag and put it in the bin", "trash", env.entity_descriptions())
env.bind_anchors([0], to_bindings(spec))
z = reward_to_z(spec.reward, agent.encode_g, agent.g_pool)
obs = env.observe()
actions = agent.act(obs, z.expand(1, -1))
```
Gemini needs `GEMINI_API_KEY`. Without it the compiler falls back to an offline keyword parser and marks the mission as offline.

## Judge demo
```bash
.venv/bin/python -m demo.server --port 8000 --host 0.0.0.0 --backend torch --device cpu --checkpoint runs/NAME/agent.pt
```
Open http://localhost:8000. Without `--checkpoint` it flies a scripted controller, and the UI labels every fallback mode so judges are never misled. See [demo/README.md](demo/README.md) for details and troubleshooting. The 3D view loads three.js from jsDelivr, so the table laptop needs internet.

## Work areas

Each area below can be picked up on its own. Every area lists its folders, its tasks, and what "done" means. Changes to `skyhand/spec.py` affect every area, so agree on them as a team.

### Training
**Folders:** `skyhand/train/`, `scripts/desktop/`

1. Time a short run on the 5080 with full-size networks. Try `--num-envs` 1024 and 2048, and `--batch-size` 2048 and 4096. Pick the fastest stable setting.
2. Start the first long run on the trash scene: `--scene trash --method fb`.
3. Watch `metrics.csv`:
   - `fb_loss` and `ortho_loss` should fall and then level off.
   - `d_acc_pos` should rise above 0.5 once the discriminator learns the demos.
   - `collision_rate` should fall.
4. At the 8:30 PM checkpoint, decide FB or SF. The test: given a reach goal, does the drone fly to the target? If not, rerun with `--method sf`.
5. Train `--scene both` overnight.

**Done when:** a checkpoint flies typed reach and pick-and-place missions in the trash scene, and success rates are written down for both scenes.

### Environments
**Folders:** `skyhand/env/`, `tests/env/`

1. Watch the scripted expert in the demo viewer (`--env-expert`) in both scenes, and note anything that looks wrong.
2. Raise ocean expert delivery above its current 38–41%. Its rollouts are the training demonstrations.
3. Run one checkpoint on both backends and compare success rates. The difference is the sim-to-sim gap reported to judges.
4. Measure step time with LiDAR on at 2048 worlds. If it is the bottleneck, lower the resolution in `skyhand/spec.py`.
5. Add a test for every physics change.

**Done when:** both scenes are stable at training scale, the ocean expert delivers fish reliably, and the sim-to-sim gap is measured.

### Mission layer
**Folders:** `skyhand/mission/`, `tests/mission/`

1. Set `GEMINI_API_KEY`, then compile 10 varied missions per scene. Fix the prompt where Gemini writes bad rewards or picks wrong entities.
2. On a real checkpoint, check that `reward_to_z` gives sensible z: missions that mean the same thing should land close together on the z map.
3. Time CEM refinement on the desktop. Keep it only if it finishes in a few seconds.
4. Extend the offline parser to any mission the demo script relies on, so the demo survives losing the network.

**Done when:** typed missions compile reliably with and without Gemini, and their z values fly the intended behaviour on a real checkpoint.

### Judge demo
**Folders:** `demo/`, `tests/demo/`

1. Run the demo with a real checkpoint in both scenes and fix anything that breaks or looks wrong.
2. Rehearse the demo script in PLAN.md section 9 with someone acting as a judge.
3. Test every perturbation control mid-flight: gusts, waves, dropped obstacles, disabled drones, moving the bag.
4. Prepare the table setup: the desktop, the laptop, a cable, and internet for the three.js CDN.

**Done when:** a stranger can type a mission and watch it fly without help, and the fallbacks are rehearsed.

### Submission
1. Record backup videos of all three missions.
2. Write the Devpost: pitch, the tracks you're entering, how it works, results, and limitations.
3. Make the slides for the demo script.

**Done when:** the Devpost is submitted before 9:00 AM and the videos are uploaded.

### Ground rules
- Work on your own branch and merge small changes often. Keep `main` passing tests.
- Run `.venv/bin/python -m pytest -q -m "not slow"` before every push.
- Post checkpoint paths and metric screenshots in the team chat, not just "it works".

## What to go over together

Spend the first 30 minutes reading these as a group, in this order:

1. **PLAN.md sections 1, 3, and 6.** The pitch, the decisions, and how a reward becomes z.
2. **`skyhand/spec.py`.** Observation keys, the goal-space vector g, anchors, and the env API.
3. **`SkyEnv.step()` in `skyhand/env/core.py`.** One policy step: grip, shield, physics substeps, LiDAR, events.
4. **`fb_measure_loss` and `FBCPR.update` in `skyhand/train/fb_cpr.py`.** The core learning rule.
5. **`skyhand/mission/dsl.py` and `infer_z.py`.** What Gemini is allowed to write, and how its reward becomes z.
6. **`demo/server.py`.** How a typed mission turns into a flying drone on screen.

### Known weak spots to check in each area

**Environments**
- Grasping is simplified: a handle within reach and the gripper closing counts as a grab.
- Ocean physics is tuned for a proof of concept: a light net, shallow fish schools, and caught fish riding with the net.
- In the Genesis backend, obstacle sizes are fixed per env instance, not per world.
- Genesis' own raycaster returns the drone's body, so both backends use the shared torch ray caster.

**Training**
- Never run at full size for long. On an M4 Pro it managed only 70–90 env steps per second.
- These choices go beyond the literal plan, and each one is worth a look:
  - The style and penalty terms are scaled by the size of Fᵀz (`scale_reg`, Motivo's convention).
  - The target action comes from the current actor, not a target actor.
  - The orthonormality loss carries a constant offset, which leaves its gradients unchanged.
- The replay buffer needs about 12 GB at 2M transitions. Use `--buffer-device cpu` if the GPU runs short.

**Mission layer**
- Gemini has never been called live. The default model is `gemini-2.5-flash`; override it with `GEMINI_MODEL`.
- The offline parser only covers the highlighted missions and simple "go to" and "avoid" requests.
- CEM refinement steps every world in the env, not only its candidates.

**Demo**
- The 3D view loads three.js from jsDelivr, so the table laptop needs internet.
- With no checkpoint, the ocean mission usually ends in a timeout, because the fallback rarely nets enough fish.
- Every fallback is labelled on screen. Keep it that way, so judges never see a scripted controller passed off as the model.

### Checkpoints
| Time | What must be true |
|---|---|
| 8:30 PM | FB or SF chosen; the policy reaches a typed goal |
| 11:30 PM | Typed mission to z to flight works end to end on a real checkpoint |
| Overnight | Long runs on both scenes; sleep in shifts |
| 6:00 AM | Demo rehearsed, backup videos recorded |
| 9:00 AM | Devpost submitted |

## Status at hand-off (Sat Nov 7, afternoon)

**Built and tested**
- Full suite: 155 tests pass, plus one expected failure that documents the Genesis raycaster issue.
- Minimal stack (Python 3.9, PyTorch 2.0.1, no Genesis): 63 tests pass, with the Genesis tests skipped.
- End to end with a checkpoint: mission compiles, z is inferred, and the policy acts on both backends. The demo server runs in trained-policy mode with a z map.

**Scripted expert success**, used for demonstrations, measured on the torch backend over 32 worlds:

| Scene | Result |
|---|---|
| Trash | 94% grasp, 88% deliver to the bin, 6% crash |
| Ocean (3 seeds) | 38–41% deliver fish to the boat, 44–47% catch fish, 9–13% end early |

**Measured cost.** A full-size training run on an M4 Pro GPU managed only 70–90 env steps per second, and network updates dominate. Measure on the training GPUs before choosing `--num-envs`, `--batch-size`, and `--updates-per-step`.

**Not done yet**
- No real training run has happened, so there is no trained checkpoint. `runs/tiny-check` is a 20-step plumbing test.
- Gemini has not been called live.
- The Genesis raycaster returns the drone's own body, so both backends use the shared torch ray caster (see PLAN.md 5.0).
