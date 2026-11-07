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
