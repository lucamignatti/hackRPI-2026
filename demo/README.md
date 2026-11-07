# SkyHand judge demo

A FastAPI server runs the simulator and streams frames over a WebSocket. A three.js page
(no build step) draws the scene and sends judge actions back.

## Run

```bash
# from the repo root
.venv/bin/python -m demo.server --port 8000 --host 0.0.0.0 --backend torch --device cpu \
    --checkpoint runs/x/agent.pt
# then open http://<desktop-ip>:8000 on the judging laptop
```

Flags:

| flag | meaning |
|---|---|
| `--backend torch\|genesis` | physics backend, passed straight to `make_env` |
| `--device cpu\|cuda` | torch device for the env and the agent |
| `--checkpoint PATH` | trained agent (`skyhand.train.agent.Agent.load`). Leave it out to fly the scripted fallback |
| `--scene trash\|ocean` | starting scene |
| `--cem` / `--cem-candidates 64` | refine z with CEM in cloned worlds after Approve (needs a checkpoint; uses 1 + candidates worlds) |
| `--env-expert` | the fallback flies `env.expert_action()` instead of the built-in controller in `demo/pipeline.py` |
| `--no-lidar` | skip LiDAR scans (faster; the point cloud is empty) |
| `--speed 1.0` | sim speed multiplier (1 = real time, 20 Hz policy) |
| `--seed N` | fixed world layout |

Set `GEMINI_API_KEY` to compile missions with Gemini. Without it, the mission package's offline parser is used.

## What judges do

1. **Scene**: Trash arena or Ocean.
2. **Mission**: type anything, or click an example, then **Compile**. The card shows the anchors
   (a0..a3, with the entity each one is bound to), the plain-language explanation, the raw reward
   program and the done condition. Nothing flies yet.
3. **Approve & fly** (or **Cancel**). This is the human-in-the-loop step. With a checkpoint, z is computed
   from the reward (and optionally refined with CEM), then the frozen policy flies it.
4. **Mess with it** while it flies:
   - **Wind gust**: a random 5 to 7 m/s gust for 2.5 s.
   - **Drop obstacle (click)**: then click the ground. One box drops there.
   - **Move bag (click)** (trash): then click the ground. The bag moves there; the drone must re-find it.
   - **Wave height** slider (ocean): scales every wave's amplitude live.
   - **Disable drone i** (ocean): knocks one fishing drone out.
   - **Stop** / **Reset world**. After a reset, the last mission can be approved again.
5. The status pill shows `flying`, `success`, `failure` (crash, net torn, out of bounds) or `timeout` (60 s).
   Gauges show the current reward (the reward program on `g` of drone 0), payload swing, shield
   interventions, time, and wind.

The **z map** (bottom right) is a PCA of z for named reference missions (trash pickup, organ delivery,
net fishing and a few simple ones) plus the judge's mission. Without a checkpoint it reads
"z map available after training".

## Honest labels

The top bar always shows which mode is running, so the demo never misrepresents itself:

- `policy: trained policy` means the FB agent from `--checkpoint` is flying.
  `policy: scripted fallback` is the hand-written controller (fly to a0, grip, carry to a1, release;
  in the ocean each drone grabs its own net corner and the team hauls the net over the fish and then to the boat).
  `scripted fallback (env expert)` is `env.expert_action()`. Whenever a fallback flies, a yellow banner
  says so on the 3D view.
- `compiler: Gemini` means Gemini produced the reward. `offline parser` is the mission package's
  rule-based compiler (no API key, or Gemini failed; the reason is shown under the card).
  `keyword parser` is the tiny built-in parser used only when `skyhand.mission` cannot be imported.

## Files

- `server.py`: FastAPI app, `DemoSession` (env, policy, mission, z), 20 Hz sim loop in a worker thread, WebSocket hub.
- `pipeline.py`: adapter to `skyhand.mission` / `skyhand.train` with lazy imports and fallbacks,
  the scripted controller, reward-to-z fallback, reference missions, PCA.
- `rewardlang.py`: whitelisted reward-expression evaluator and plain-language explainer (fallback only).
- `web/index.html`, `web/main.js`, `web/style.css`: the viewer. three.js 0.160.0 from jsDelivr via an importmap.

API: `GET /` · `GET /api/health` · `GET /api/state` (one frame) · `GET /api/info` · `GET /api/zmap` ·
`POST /api/scene {scene}` · `POST /api/mission {text}` · `POST /api/approve` · `POST /api/cancel` ·
`POST /api/stop` · `POST /api/reset` · `POST /api/perturb {kind, ...}` with kinds `drop_obstacle {pos}`,
`gust {vec, duration}`, `waves {scale}`, `move_payload {pos}`, `disable_drone {agent}` · `WS /ws`
(messages `{"type": "frame", "state": render_state(0), "demo": {...}}`).

Tests: `.venv/bin/python -m pytest tests/demo -q`.

## Troubleshooting

- **Blank page or "offline, retrying"**: the laptop cannot reach the server. Check the desktop's IP, that the
  server runs with `--host 0.0.0.0`, and the firewall on port 8000. `curl http://<ip>:8000/api/health` should answer.
- **Black 3D view, panel fine**: three.js loads from `cdn.jsdelivr.net`. With no internet at the table, the page
  cannot draw the scene. Use a phone hotspot, or fall back to the recorded videos / the Genesis viewer.
- **Choppy motion**: start with `--no-lidar`, or check the server log for slow ticks (Genesis on CPU is slow; use `--device cuda`).
- **Checkpoint did not load**: the log says why and the top bar shows `scripted fallback`. Check the path and that
  `skyhand.train` imports.
- **Compiler shows `offline parser` although you set a key**: the note under the card gives Gemini's error.
- **Mission does something odd**: read the reward card before approving, Cancel and rephrase.
