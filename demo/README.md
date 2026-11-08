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
| `--scene trash\|delivery\|ocean\|world` | starting scene |
| `--cem` / `--cem-candidates 64` | refine z with CEM in cloned worlds after Approve (needs a checkpoint; uses 1 + candidates worlds) |
| `--env-expert` | the fallback flies `env.expert_action()` instead of the built-in controller in `demo/pipeline.py` |
| `--no-lidar` | skip LiDAR scans (faster; the point cloud is empty; the fallback can no longer climb over buildings) |
| `--domain-rand` / `--gusts` (and `--no-...`) | env options; left out, the env defaults apply (domain randomisation and gusts are **off**). The Wind gust button only appears when gusts are on (they need `--domain-rand --gusts`) |
| `--wind`, `--dynamic-wind`, `--rand-dynamics` (and `--no-...`) | more env options, passed to `make_env` only when given |
| `--fake-env` | kinematic preview env (`demo/fake_env.py`, no physics) for working on the viewer when `skyhand/env` is broken. The top bar says `sim: kinematic preview` |
| `--speed 1.0` | sim speed multiplier (1 = real time, 20 Hz policy) |
| `--seed N` | fixed world layout |

Set `GEMINI_API_KEY` to compile missions with Gemini. Without it, the mission package's offline parser is used.
Set `ELEVENLABS_API_KEY` to enable the optional **Speak** button, which records one mission (up to 30 seconds),
transcribes it with ElevenLabs Scribe v2, and puts the text in the mission box for review. Voice input never
auto-compiles or starts a flight. Browser microphone access requires `localhost` or HTTPS; a judging laptop
opening plain `http://<desktop-ip>:8000` will retain typed input but its browser will disable the microphone.

## What judges do

1. **Scene** (trash and ocean are 50 x 50 m; delivery is 600 x 600 m; world is 400 x 400 m; from `frame.arena_half`):
   - **Trash**: each world is a city (street grid of buildings, cars driving, trash piled at the curb, a
     garbage truck as the drop-off, entity `bin`) or open terrain with scattered litter and a bin. Items
     `item_0..item_9` are trash (round trash bag, flat pizza box, cardboard cube, thin bottle; drawn in drab
     colours with an **orange ring** on the ground) or not trash (backpack, ball, flower pot; clean bright colours).
     Hover an item for its name and type; the item an anchor is bound to gets a label.
   - **Organ delivery**: long-range, fast delivery between remote clinics. Carry the organ cooler (`item_0`)
     from the blue pickup pad (`pad_pickup`) to the green drop-off pad (`pad_dropoff`), 300 to 700 m away, at up
     to 18 m/s, across hills, mountains, valleys and lakes with beaches, past clinic
     buildings, acacias, rocks, and the red no-fly cylinders
     (`nofly_0`, `nofly_1`). The camera follows the drone here (toggle **Follow drone**); the speed gauge shows
     speed against the scene's limit. Episodes are 90 s.
   - **Ocean**: four drones, one per net corner, haul a net over a fish school and back to the boat among whales (dark blue),
     dolphins (grey) and turtles (green). An animal caught in the net turns red and the **bycatch** gauge says which.
   - **World**: one 400 m map with everything at once and three drones (up to 12 m/s): hills, lakes and a coast
     into open sea (water is the tide level plus waves everywhere), a town (buildings, cars, curbside trash, the
     garbage truck `bin`), countryside litter and non-trash items, two clinics with `pad_pickup` / `pad_dropoff`, the boat / net / fish / animals at sea, trees, rocks and no-fly zones. Each drone can get
     its own job: "drone 0 pick up the pizza box and drop it in the garbage truck, drone 1 deliver the organ
     cooler to the clinic, drone 2 fly to the boat" (drones not named share the rest of the text, e.g. "the
     other drones collect trash"; one shared carry text sends every drone to a different item). The card lists
     each drone's anchors and reward; the mission succeeds when every drone with a goal has met its own done
     condition, and a drone that is done holds position. "Net the fish ..." uses the drones on the net.
2. **Mission**: type anything, click an example, or press **Speak** and dictate a command, then **Compile**.
   Voice transcription fills the same editable text box so the operator can correct it before compilation. The card shows the anchors
   (a0..a3, with the entity each one is bound to), the plain-language explanation, the raw reward
   program and the done condition. Nothing flies yet.
3. **Approve & fly** (or **Cancel**). This is the human-in-the-loop step. With a checkpoint, z is computed
   from the reward (and optionally refined with CEM), then the frozen policy flies it.
4. **Mess with it** while it flies:
   - **Drop obstacle (click)**: then click the ground (or the terrain). One box drops there.
   - **Move item / Move cooler (click)** (trash, delivery): then click the ground. The item the mission's a0 is
     bound to (else `item_0`) moves there; the drone must re-find it.
   - **Tide** slider (worlds with lakes, and the world's sea): raises or lowers the water level by up to 1.5 m; beaches flood.
   - **Wave height** slider (ocean, world): scales every wave's amplitude live.
   - **Wind gust** (only when the env runs with gusts on): a random 5 to 7 m/s gust for 2.5 s.
   - **Disable drone i** (ocean, world): knocks one drone out.
   - **Follow drone** / **Overview**: the camera follows the selected drone (on by default in delivery and the
     world) or frames the whole map. In multi-drone scenes the **drone picker** (Drone 0 / 1 / 2, coloured like
     the drones; keys 1-3, O for overview) chooses which drone to follow, and the line under it shows that
     drone's job (and "done" once it has met its goal). Orbit / zoom still work while following.
   - **Stop** / **Reset world**. After a reset, the last mission can be approved again.
5. The status pill shows `flying`, `success`, `failure` (crash, out of bounds) or `timeout`
   (the env's `episode_steps`: 45 s in trash and ocean, 90 s in delivery).
   Gauges show the current reward (the reward program on `g` of drone 0; with per-drone jobs the mean of each
   drone's own reward), payload swing, shield
   interventions, time, wind, and per scene the items delivered (trash, delivery) or bycatch (ocean).
   Carry missions (a0 = an item handle, a1 = `bin` / `pad_dropoff`) also end with success on the env's
   `delivered` event.

The **z map** (bottom right) is a PCA of z for named reference missions (trash pickup, organ delivery,
net fishing and a few simple ones) plus the judge's mission. Without a checkpoint it reads
"z map available after training".

## Honest labels

The top bar always shows which mode is running, so the demo never misrepresents itself:

- `policy: trained policy` means the FB agent from `--checkpoint` is flying.
  `policy: scripted fallback` is the hand-written controller (fly to a0, grip, carry to a1, release;
  it flies at the env's own speed limits (`env.max_speed`, `env.max_vz`), cruise height follows the terrain
  (12 m in delivery), and when the near-horizontal LiDAR rows see something within 4 m it
  climbs over it, so it hops over city blocks and ridges; in the ocean each drone grabs its own net corner and
  the team hauls the net over the fish and then to the boat; in the world each drone flies its own job, net
  haulers in haul mode and the others carrying or reaching, cruising 10 m above the terrain).
  With a checkpoint and per-drone jobs, z is computed per drone from its own reward and the policy gets
  z as `[N, A, d]` (the z map shows one point per drone; CEM is skipped for per-drone z).
  `scripted fallback (env expert)` is `env.expert_action()`. Whenever a fallback flies, a yellow banner
  says so on the 3D view.
- `compiler: Gemini` means Gemini produced the reward. `offline parser` is the mission package's
  rule-based compiler (no API key, or Gemini failed; the reason is shown under the card).
  `keyword parser` is the tiny built-in parser used only when `skyhand.mission` cannot be imported.
  It knows item types ("pizza box", "backpack", "trash", "litter"), `bin` / truck, the pads, no-fly zones,
  fish schools, the boat and the animals.
- `sim: kinematic preview` appears only with `--fake-env`.

## Files

- `server.py`: FastAPI app, `DemoSession` (env, policy, mission, z), 20 Hz sim loop in a worker thread, WebSocket hub.
- `pipeline.py`: adapter to `skyhand.mission` / `skyhand.train` with lazy imports and fallbacks,
  the scripted controller, reward-to-z fallback, reference missions, PCA.
- `rewardlang.py`: whitelisted reward-expression evaluator (`dist`, `distxy`, `inside`, ...) and plain-language explainer (fallback only).
- `fake_env.py`: kinematic stand-in env with the spec's API and render-state schema (`--fake-env`, tests).
- `web/index.html`, `web/main.js`, `web/style.css`: the viewer. three.js 0.186.1 (r186) from jsDelivr via an importmap.
  City buildings alternate shared apartment and office silhouettes with procedural PBR facades; each
  silhouette is rendered as one instanced batch, so a full city uses two facade draw calls and no facade textures.
  The terrain heightfield becomes a mesh coloured by height (sand at the water line, grass, rock, snow on
  peaks; asphalt in a city), rebuilt only when the terrain changes (the world's 128 x 128 grid is built once
  per terrain version). Lakes are a flat plane at `water.level`, the ocean is the animated wave surface, and
  the world's sea is the wave surface offset by `water.level`, a 256 x 256 grid over the whole arena displaced
  on the GPU (the terrain hides it on land). Ocean waves use the same analytic-normal GPU shader on a smaller
  96 x 96 grid, with no per-frame JavaScript vertex or normal rebuild.

API: `GET /` · `GET /api/health` · `GET /api/state` (one frame, terrain heights included) · `GET /api/info` ·
`GET /api/terrain` (current heightfield) · `GET /api/zmap` ·
`POST /api/scene {scene}` · `POST /api/mission {text}` · `POST /api/transcribe` (raw audio body) · `POST /api/approve` · `POST /api/cancel` ·
`POST /api/stop` · `POST /api/reset` · `POST /api/perturb {kind, ...}` with kinds `drop_obstacle {pos}`,
`gust {vec, duration}`, `waves {scale}`, `tide {offset}`, `move_payload {pos, item?}`, `disable_drone {agent}` · `WS /ws`
(messages: `hello` (info), `terrain` (full heightfield, once on connect), then
`{"type": "frame", "state": render_state(0), "demo": {...}}`). Frames carry `terrain` as
`{version, res, half, key}` and add `heights` only on the first frame after the terrain changed; the
viewer fetches `/api/terrain` if it ever sees a `key` it has not built.

Tests: `.venv/bin/python -m pytest tests/demo -q`. Most run against `fake_env.py`; the `real_env` ones
skip a scene whose `make_env` fails or builds a world outside the arena.

## Troubleshooting

- **Blank page or "offline, retrying"**: the laptop cannot reach the server. Check the desktop's IP, that the
  server runs with `--host 0.0.0.0`, and the firewall on port 8000. `curl http://<ip>:8000/api/health` should answer.
- **Black 3D view, panel fine**: three.js loads from `cdn.jsdelivr.net`. With no internet at the table, the page
  cannot draw the scene. Use a phone hotspot, or fall back to the recorded videos / the Genesis viewer.
- **Choppy motion**: terrain LiDAR uses a bounded ray marcher. Multi-drone scans are staggered to avoid one large
  tick; `--no-lidar` remains the fastest diagnostic. Genesis on CPU is slow; use `--device cuda` when available.
- **Checkpoint did not load**: the log says why and the top bar shows `scripted fallback`. Check the path and that
  `skyhand.train` imports.
- **Compiler shows `offline parser` although you set a key**: the note under the card gives Gemini's error.
- **Speak button disabled**: microphone capture requires `localhost` or HTTPS. Also set `ELEVENLABS_API_KEY`
  on the demo server; the key is never sent to the browser.
- **Mission does something odd**: read the reward card before approving, Cancel and rephrase.
