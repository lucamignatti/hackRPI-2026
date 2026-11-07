# SkyHand: one drone brain, any mission

HackRPI 2026, theme "In The Clouds". Plan revised Sat Nov 7, 2:05 PM (codebase built; see README.md for status). Devpost closes **Sun 9:00 AM**, edits until 11:00 AM.

## 1. Pitch

A single behavioral foundation model for a gripper drone that sees through a dual-wavelength LiDAR. It is pretrained without any task reward. At test time you type a mission in English, it becomes a reward, the reward becomes a latent task vector z, and the same frozen network flies the mission.

Two small, dynamic proof-of-concept scenes, three highlighted missions:

| Mission | Scene | Track | What it shows |
|---|---|---|---|
| Trash pickup | Arena with moving obstacles | Best Sustainability | Find the bag, grab its handle, carry it to the bin past moving obstacles in gusty wind |
| Organ delivery | Same arena, a cooler instead of a bag | Best Healthcare | Same skills, different reward: gentle carry, low swing, avoid a no-fly zone. No extra env work |
| Net fishing | Ocean with waves, a net, fish schools | Multi-agent showcase | Drones grab net corners and haul a school of fish to a boat; green LiDAR sees the fish underwater |

Judge demo: judges type their own mission into a web app, approve the reward it compiles to, and watch the same weights fly it. They can also throw obstacles, gusts, and waves at it live (section 9).

Also targets Best ML/Data Science, Best AI Hack, Best Ethical AI Agent (safety shield, validated reward programs, human approval), Best In The Clouds, and MLH Best Use of Gemini API.

## 2. What we take from each paper

| Paper | What we use |
|---|---|
| Dayan 1993, successor representation | Value = expected future state occupancy times reward. Everything below builds on it |
| Barreto et al. 2017, successor features | Reward as φ·w and generalized policy improvement. Basis of our fallback |
| Barreto et al. 2018, SF + GPI in deep RL | Off-policy successor-feature TD, bootstrapping with the target policy's own greedy action |
| Borsa et al. 2018, USFA | Conditioning on a policy vector z; any batch of transitions trains for every z |
| Blier, Tallec, Ollivier 2021 | The TD algorithm for successor measures in continuous spaces and the finite-rank F·B form |
| Touati & Ollivier 2021, FB | The FB loss, B orthonormality, z_r = E[r(s) B(s)], z = B(goal), and feeding B only the reward-relevant part of the state |
| Touati, Rapin, Ollivier 2023 | FB is the most reliable zero-shot method. Twin F, TD3 smoothing, z mixing, embedding z together with the state, hyperparameters |
| Tirinzoni et al. 2025, Meta Motivo (FB-CPR) | Online off-policy FB with parallel environments and a discriminator that pulls the policy toward unlabeled demonstrations |
| BFM-Zero 2025 | Privileged critics with a sensor-only actor, a penalty critic for physical limits, refining z at test time with search |

All of these are **off-policy**: F, B, and the actor learn from a replay buffer filled by policies running other values of z.

## 3. Decisions and honest constraints

- **Two physics backends behind one env API.** The training path must not depend on Genesis. So the env has a **pure-PyTorch backend** that runs anywhere PyTorch runs, and a **Genesis backend** for the desktop and demo. Both share every piece of task logic (controller, waves, net, fish, payload, shield, anchors, goal space, LiDAR model); only drone rigid-body dynamics and contacts differ. Training uses the torch backend by default; evaluating the same checkpoint on the Genesis backend measures the sim-to-sim gap.
- **Method: FB-CPR, online and off-policy**, with demonstrations from a scripted expert instead of motion capture.
- **Fallback with the same code: successor features with hand-built features.** Decide at the 8:30 PM checkpoint.
- **Training needs only torch and numpy.** The whole training path imports only torch and numpy, targets Python 3.9+ and PyTorch 2.0+, and is tested with Genesis, Gemini, and the web stack blocked (tests/test_minimal_deps.py).
- **Simplified grasp.** Closing the gripper near a handle at low speed welds it to the gripper. Standard in RL, stated on Devpost.
- **LiDAR and water.** Ordinary near-infrared LiDAR mostly reflects off water. Bathymetric LiDAR uses green 532 nm light and reaches a few meters in clear water, but is heavy and expensive today. We simulate a dual-wavelength unit with randomized turbidity and say so openly.
- **Horizon is no longer a problem.** Both scenes are about 30 m across, which the policy covers within its few-second lookahead at γ = 0.99 and 20 Hz. No route planner is needed.
- **What "general" means.** Any reward written over the goal-space variables in section 6 becomes a z with no retraining. FB only captures the part of a reward that correlates with B, and behaviors the replay buffer never covered will not appear.

## 4. Architecture

```
 English mission
      │
      ├─► Gemini ──► anchors + reward program (validated DSL) ──┐
      │                                                         ▼
      └─► tiny text→z model (distilled, stretch) ──►  z_r = √d · normalize(mean_i r(g_i) B(g_i))
                                                                │  over replay-buffer states
                                                                ▼
                                                   CEM refinement of z in copied env worlds
                                                                │
 ┌───────────── Genesis scene, n_envs worlds in parallel ───────┼──────────────────────────┐
 │ Genesis: rigid bodies, contacts, welds, raycaster LiDAR, cameras                         │
 │ our env code (torch, batched): controller, waves, buoyancy, net, fish, water LiDAR,      │
 │                                moving obstacles, wind, shield, anchors, g, expert        │
 └──────────────▲──────────────────────────────────────────────┬───────────────────────────┘
                │ actions = π(actor tokens, z)                   │ transitions → GPU replay buffer
                │                                                ▼
        actor π (LiDAR) ◄── F×2, B, Q_R, Q_D, discriminator D (privileged) ◄── scripted-expert demos
                                                                 │
                                   demo server ──► web viewer for judges (three.js)
```

## 5. Environments (Genesis plus our env code)

### 5.00 Environment redesign (Sat afternoon)
The scenes were rebuilt; skyhand/spec.py and README.md are the current reference.
- **trash**: each world is a city (a tight street grid: narrow 4.5 m streets between blocks packed with 6–18 m buildings, cars driving lanes, trash piled at the curb NYC-style, a garbage truck as drop-off, hydrants and benches) or natural terrain (rolling hills, lakes with beaches and a tide, trees, rocks, birds) with scattered litter. Items: trash (bag, pizza box, cardboard box, bottle) and non-trash (backpack, ball, flower pot), shaped to be distinguishable on LiDAR. One item is carried at a time.
- **delivery**: long-range, fast organ delivery between two remote clinics 300–700 m apart in a 600 m arena (hills, mesas, a river valley, lakes, a clinic building at each pad, acacias, vultures, wildlife-reserve no-fly zones). Up to 18 m/s, 90 s episodes; the cooler must be set down gently on the pad.
- **ocean**: four drones, one per net corner, plus whales, dolphins, and turtles; an animal touching the held net is tangled (bycatch event and g field). Fish dart and change course at random. Drones crash if they touch the net, the net cannot pass through the boat hull or seabed, and it never tears: an overstretched corner slips out of its gripper (slip and release events).
- **world**: the generic pretraining world, a 400 m map with 3 drones that contains all of the above at once: countryside with hills and lakes, a coastline into open sea (tide plus waves), a town with curbside trash and a garbage truck, two clinics with helipads and the cooler, the boat, net, fish, and animals offshore, trees, rocks, birds, no-fly zones. Each drone follows its own anchors, so a world episode can mix single-drone item tasks with a team net haul. FB pretrains here; the other three scenes are focused presets for evaluation and the demo.
- Domain randomisation is off by default. The Genesis backend now integrates drone rigid bodies only.

### 5.0 Status of the spikes (done)
- Custom drone body: a Genesis box with external wrenches flies, hovers, and tracks velocity (tests/env/test_genesis.py).
- Per-world grasp: implemented in shared torch code (handle within reach + gripper closing), so it behaves the same in both backends; no Genesis weld needed.
- Swinging payload: shared position-based pendulum; carried bags visibly swing.
- Net: torch mass-spring mesh (Genesis cloth was not needed). The perimeter floats, the interior sags into a bag; caught fish ride with the net.
- LiDAR: Genesis' raycaster returns the drone's own body at ~7 cm in genesis-world 1.4.5, so both backends use the shared torch ray caster against the same primitives. Genesis' raycaster is behind `genesis_lidar=True`.
- Obstacle sizes in the Genesis backend are fixed per env instance (geometry is fixed at scene build); positions still vary per world.

### 5.1 First-hour spikes (original plan, kept for reference)
| Question | Test | Fallback |
|---|---|---|
| Custom drone body flies | Load our quadrotor URDF, apply thrust and torque with `apply_links_external_wrench` each step | None needed, this is standard |
| Per-world grasp | `add_weld_constraint` / `delete_weld_constraint` with `envs_idx` on a subset of worlds | Spring-damper force between gripper and handle, applied as wrenches on both |
| Swinging bag | Bag as two links joined by a ball joint | Rigid weld, no swing; drop the swing term |
| Net as Genesis cloth | PBD cloth with `n_envs > 1`, corners re-pinned to gripper positions each step | Our own torch mass-spring net (section 5.5), the expected path |
| LiDAR cost | `gs.sensors.Lidar` with 64 × 16 rays at 2,048 worlds | Fewer rays, scans at 10 Hz |

### 5.2 Drone, shared by both scenes
- **Body:** our own URDF, a quadrotor of about 2 kg with a gripper link underneath. We do not use `gs.morphs.Drone`, because its docs say it has no collision checking.
- **Action, 5 values in [-1, 1]:** desired velocity (vx, vy, vz), yaw rate, gripper command (closed if above 0).
- **Controller in torch:** velocity error → desired acceleration → collective thrust and attitude → body torques. Applied as a wrench every physics step, since Genesis applies external wrenches for one step only.
- **Timing:** physics dt 10 ms, policy at 20 Hz (5 physics steps per action), LiDAR at 10 Hz.
- **Grasp:** when the gripper closes within a few cm of a handle at low relative speed, weld the handle to the gripper in that world. Opening deletes the weld.
- **Domain randomization:** mass, drag, controller gains, wind, sensor noise, payload mass.

### 5.3 Dual-wavelength LiDAR
- Genesis `gs.sensors.Lidar` with a spherical pattern of 64 azimuth × 16 vertical rays, mounted on the drone body. A small downward grid pattern covers the gripper.
- **NIR channel:** the Genesis hit distance, cut short where the ray crosses the wave surface. The surface crossing is computed analytically in torch from the wave field.
- **Green channel:** passes the water surface. Rays are also tested against fish spheres and net nodes in torch. Intensity falls as exp(-2·k·depth) with turbidity k randomized per episode, and returns vanish past a max depth.
- Output: a [3, 16, 64] image (NIR range, green range, green intensity) with noise and dropout, stored as uint8.

### 5.4 Trash scene (also used for organ delivery)
- Flat ground about 30 × 30 m, a bin, and 4 to 8 box obstacles.
- **Dynamic elements:**
  - Two or three obstacles move on looping paths at walking speed, like pedestrians or carts.
  - Wind with gusts applied as a force on the drone and the payload.
  - The bag is a real rigid body. It can tip, slide when bumped, and swing once carried.
- Organ delivery swaps the bag for a cooler and adds a no-fly zone. The reward differs, the env does not.
- Randomized each episode: layout, bag pose, bin position, obstacle paths.

### 5.5 Ocean scene
- **Waves:** a sum of 3 to 4 Gerstner waves evaluated in torch. Height, slope, and surface velocity are available at any point. Wave height is randomized and adjustable live in the demo.
- **Buoyancy and drag:** applied as wrenches on floating bodies (net floats, the boat) from submerged depth and relative water velocity.
- **Net:** a torch mass-spring grid of about 8 × 8 nodes with position-based constraints, water drag, buoyancy on the top edge, and weight on the bottom edge. Corner nodes are pinned to grippers that hold them. The pull on each held corner is applied back to that drone as a force. The net tears past a stretch limit.
- **Fish schools:** 2 or 3 schools of about 20 fish each, simulated as boids in torch: cohesion, alignment, separation, and fleeing from nearby drones and net nodes. A fish counts as caught when it ends up inside the net's volume as the net closes and lifts.
- **Boat:** a floating rigid body that bobs on the waves. Its deck is the delivery zone.
- **Team:** 3 or 4 drones per world, each a separate entity in the same Genesis scene.

### 5.6 Anchors: how a task points at the world
**Update (Sat evening):** anchors stay a fixed set of 4 slots per drone. A slot can name a fixed entity, a point, or a *selector* such as `nearest:trash`, `nearest:pizza_box`, `nearest:net_corner`, `nearest:fish_school`, or `nearest:animal`, which resolves every step to the closest match (and stays on an item while it is carried). Variable numbers of objects are handled this way and by the nearest-entity tokens, not by a variable slot count, which would break B's fixed input, bias the buffer average that gives z, and blow up pretraining coverage. The goal space has an `anchor_active` flag per slot, so an unused slot is no longer indistinguishable from an anchor at the drone; reward programs can read it with `active(ak)`.

Each world exposes **K = 4 anchors**: points or regions the task cares about, such as "the handle to grab", "where to deliver", and "the zone to avoid". Anchors are part of the observation. During pretraining they are bound at random each episode to random objects and free-space points. At test time the language layer binds them to named scene objects.

### 5.7 Safety shield
Applied in the env after the policy acts. A velocity command that would enter a no-fly zone or an obstacle's safety margin within 0.5 s is projected onto the boundary. Every intervention is counted and exposed as an event.

### 5.8 Scripted expert (demonstrations for the policy regularizer)
A privileged state machine that does the basic skills competently:
- Takeoff, waypoint flight, hover.
- Approach above a handle, descend, close, lift, fly to an anchor, descend, release. Plus a gentle-carry variant.
- Net: each drone flies to its corner, grabs it, the team lifts and flies toward the boat in formation.

It produces state-only trajectories, which play the role motion capture plays in Motivo. They also cover FB's weak point, exploration, by showing the buffer what grasping looks like.

### 5.9 Env API
```python
env = SkyEnv(scene="trash" | "ocean", n_envs=2048, device="cuda")
out = env.reset(env_ids)
env.bind_anchors(anchor_ids)          # random in pretraining, from language at test time
out = env.step(actions)               # dict of torch tensors
# out: actor_tokens, priv_tokens, g, done, events, shield_hits
env.copy_state(src_env, dst_env_ids)  # Genesis entity states + our torch state, for CEM over z
acts = env.expert_action()            # scripted expert
```

## 6. Goal space g and the reward language

### 6.1 Goal space g (input to B, the discriminator, and reward programs)
B sees only the reward-relevant part of the state, as the FB paper recommends. F and the actor see everything. All positions are relative to the drone or to anchors.

| Group | Variables |
|---|---|
| Drone | position, velocity, speed, tilt, height above ground or waves, thrust use |
| Gripper | closed, holding, held mass |
| Payload | position, velocity, swing rate, jerk |
| Anchors 0 to 3 | position relative to drone, position relative to payload, inside-zone flag |
| Obstacles | distance to nearest obstacle, its closing speed |
| Team and ocean | net centroid, max net stretch, fish inside net, nearest-teammate distance |
| Events | collision, shield intervention |

### 6.2 Reward programs
Gemini never sees z. It binds anchors and writes a reward program over g:
```json
{
  "anchors": { "a0": "bag.handle", "a1": "bin", "a2": "nofly_1" },
  "reward": "-dist(payload, a1) + 3*holding - 0.2*swing - 5*inside(a2) - 0.05*thrust",
  "done_when": "holding and dist(payload, a1) < 0.5"
}
```
- Allowed: the variables in 6.1, `dist`, `inside`, `abs`, `min`, `max`, `clamp`, `exp`, comparisons, arithmetic.
- Parsed by a whitelisted expression parser and compiled to batched torch operations. No `eval` of model output.
- The operator, or the judge, sees the anchors and reward in plain language and approves before launch.

### 6.3 From reward to z
1. Score 10,000 to 100,000 replay-buffer states with the reward program, including demo states.
2. `z_r = sqrt(d) * normalize( mean_i r(g_i) * B(g_i) )`, the correlation between reward and each B dimension.
3. Shortcuts: reach a goal with z = B(g_goal). Reach while avoiding with z = B(g_goal) − λ·Σ B(g_forbidden) (FB paper, eq. 7).
4. **Refinement:** copy the current world into 256 worlds, roll out candidate z values sampled around z_r, score with the reward program, keep the best (cross-entropy method, 3 to 5 rounds).

### 6.4 Tiny text-to-z model (stretch)
Have Gemini generate about 2,000 missions with reward programs, compute each z with 6.3, and fine-tune a small sentence encoder on cosine loss to z. It runs offline in milliseconds but must be retrained whenever B changes. Gemini stays the primary path.

### 6.5 Chaining
The language layer can emit a queue of reward programs, such as "grab the bag" then "drop it in the bin", and recompute z when `done_when` fires.

## 7. Learning: FB-CPR, online and off-policy

### 7.1 Networks
| Network | Inputs | Output |
|---|---|---|
| F₁, F₂ (twin) | privileged tokens + actor tokens + action, z | R^d |
| B | g | R^d, L2-normalized to √d |
| Actor π | actor tokens (proprio, anchors, teammates, LiDAR), z | 5 actions, tanh |
| Discriminator D | g, z | probability that (g, z) came from a demo |
| Q_D (style critic) | same as F | scalar |
| Q_R (penalty critic) | same as F | scalar |

Critics see privileged state and the actor sees LiDAR, as in BFM-Zero. Cut line: give the actor privileged tokens too.

### 7.2 Losses (b = batch, bars = target networks, a' = π(s', z) + clipped noise)
- **FB measure loss** (Touati, Rapin, Ollivier 2023 eq. 8, twin targets):
  `L_FB = mean_{i≠j} ( F_k(s_i,a_i,z_i)ᵀ B(g'_j) − γ·min_l F̄_l(s'_i,a'_i,z_i)ᵀ B̄(g'_j) )² − 2·mean_i F_k(s_i,a_i,z_i)ᵀ B(g'_i)`
- **Auxiliary diagonal loss** (eq. 9 / 42, weight 1/d): `mean_i ( F_k(s_i,a_i,z_i)ᵀ z_i − B(g'_i)ᵀ Cov⁺ z_i − γ·min_l F̄_l(s'_i,a'_i,z_i)ᵀ z_i )²`, with B and Cov held fixed. Trains Fᵀz directly on the z it was sampled with.
- **Orthonormality** (eq. 25 and 26): `L_ortho = mean_{i,j} ( B(g_i)ᵀ B(g_j) )² − mean_i ‖B(g_i)‖² − mean_j ‖B(g_j)‖²`, weight 100 to start.
- **Discriminator** (Motivo eq. 9): demo states paired with their segment encoding z_τ = mean B(g) are positives. Policy states with their sampled z are negatives. Gradient penalty 10.
- **Style critic Q_D:** TD on reward log(D/(1−D)).
- **Penalty critic Q_R:** TD on the sum of penalties for collision, shield intervention, excessive tilt, jerk, and energy.
- **Actor:** `L_π = −mean[ min_l F_l(s,π(s,z),z)ᵀ z + λ_D·Q_D + λ_R·Q_R ]`, with λ_D = 0.05 and λ_R = 0.02 to start.

### 7.3 z sampling and data collection
- Each world keeps its z for 100 policy steps (5 s), then resamples. Worlds are staggered.
- Mixture: 50% demo encodings z_τ, 25% B(g) of random buffer states, 25% uniform on the √d sphere.
- Exploration noise σ = 0.2 on actor outputs.
- Warm start: fill the buffer with expert rollouts and random actions.
- Expert worlds: after warm start, a fixed 10% of training worlds keep flying the scripted expert, so grasp and carry transitions with actions keep entering the buffer. They are flagged and excluded from the discriminator's policy negatives.
- Demonstrations: the expert demo env keeps running between collections (the first collection is a full episode), so demo segments cover grasping and carrying, not only the approach.
- Each loop: one env step for all worlds, then G gradient steps on batch b. Start with G = 2 and b = 4096.
- Train one model on both scenes by splitting worlds between them, so a single z space covers trash, organ, and fishing.

### 7.4 Starting hyperparameters
| Setting | Value | Source |
|---|---|---|
| d (z dimension) | 256 | BFM-Zero ablation |
| γ | 0.99 at 20 Hz | Section 3 |
| Optimizer | Adam | All FB papers |
| lr F, actor, Q_D, Q_R | 1e-4 | Touati 2023, Motivo |
| lr B, D | 1e-5 | Motivo, BFM-Zero |
| Target update | Polyak 0.005 | Motivo |
| Target smoothing noise | σ 0.2, clip 0.3 | Touati 2023 |
| Replay buffer | 2M transitions | Section 10 memory note |
| Reward labels for z inference | 10,000 | Touati 2023 |

### 7.5 Multi-agent net fishing
- One team z shared by all drones. Each drone also gets its corner as its own anchor.
- Actor and F share weights across drones, with teammates as tokens. B and the discriminator see team variables: net centroid, stretch, fish inside.
- The scripted net expert supplies demonstrations.
- Cut line: two drones holding a net with fixed slack, scooping a single school.

### 7.6 Fallback: successor features
If FB-CPR has not learned reach and grasp by 8:30 PM, freeze B as a hand-built φ(g) from section 6.1 and train ψ in place of F with the projected SF loss:
`L_SF = mean( ψ(s,a,z)ᵀ z − φ(g')ᵀ z − γ·ψ̄(s',a',z)ᵀ z )²`
Infer z by least-squares regression of the reward program on φ. Everything else stays the same.

### 7.7 Evaluation
- Success rate, time to complete, payload swing, collisions, shield interventions, fish caught.
- Held-out reward programs and held-out layouts, to back the generality claim with numbers.
- Compare against TD3 trained directly on one mission's reward, the standard reference in the zero-shot RL papers.

## 8. Network architecture

All networks except B and D share one design: **z-conditioned query tokens that cross-attend to state tokens**. This matches the finding in Touati 2023 that z must be processed together with the state.

### 8.1 Tokens
| Token | Count | Content |
|---|---|---|
| proprio | 1 | velocity, attitude, angular rate, gripper, holding, last action |
| anchors | 4 | relative position, type embedding, zone size |
| teammates | up to 3 | relative position and velocity, held corner |
| LiDAR patches | 32 | 4×8 patches of the 16×64 image, linear embed + learned 2D position (azimuth wraps) |
| downward grid | 4 | patches of the gripper-facing rays |
| privileged entities | up to 16 | exact states of obstacles, payload, net summary, fish school centers (critics only) |
| action | 1 | F, Q_D, and Q_R only |

### 8.2 Block, as specified
- **Pre-norm with non-affine LayerNorm** on the query stream and the context tokens.
- **Cross-attention** from queries to context tokens, with **only keys normalized** (non-affine LayerNorm per head on K, none on Q).
- **SwiGLU MLP with 4× hidden:** `W₂( silu(W₁x) ⊙ W₃x )`, W₁ and W₃ project width → 4·width.
- **Per-feature sigmoid gate, range (0, 2), identity at init**, on both residual branches, computed from the branch input: `y ← y ⊙ 2·sigmoid(W_g·x̂ + b_g)` with W_g and b_g zero-initialized.

Tested: the block below runs, keeps shapes, and every gate starts at exactly 1. About 1.18M parameters per block at width 256.

```python
LN = lambda d: nn.LayerNorm(d, elementwise_affine=False)

class Gate(nn.Module):                       # per-feature, range (0, 2), identity at init
    def __init__(s, w):
        super().__init__(); s.lin = nn.Linear(w, w)
        nn.init.zeros_(s.lin.weight); nn.init.zeros_(s.lin.bias)
    def forward(s, x, y): return y * 2 * torch.sigmoid(s.lin(x))

class Block(nn.Module):
    def __init__(s, w, heads):
        super().__init__()
        s.h, s.dh = heads, w // heads
        s.nq, s.nc, s.nm, s.nk = LN(w), LN(w), LN(w), LN(w // heads)
        s.q, s.k, s.v, s.o = (nn.Linear(w, w) for _ in range(4))
        s.w1, s.w3, s.w2 = nn.Linear(w, 4*w), nn.Linear(w, 4*w), nn.Linear(4*w, w)
        s.ga, s.gm = Gate(w), Gate(w)
    def forward(s, q, ctx):                  # q: [B, Nq, w], ctx: [B, Nc, w]
        x, c = s.nq(q), s.nc(ctx)
        Q = s.q(x).unflatten(-1, (s.h, s.dh)).transpose(1, 2)
        K = s.nk(s.k(c).unflatten(-1, (s.h, s.dh))).transpose(1, 2)   # K-norm only
        V = s.v(c).unflatten(-1, (s.h, s.dh)).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V).transpose(1, 2).flatten(-2)
        q = q + s.ga(x, s.o(a))
        x = s.nm(q)
        return q + s.gm(x, s.w2(F.silu(s.w1(x)) * s.w3(x)))
```

### 8.3 Queries, heads, and sizes
- Queries: 8 learned base tokens plus a per-query linear projection of z (normalized to √d). In F, Q_D, and Q_R every query also gets a projection of the action, so the action does not depend on one context token to be seen.
- Head: mean over queries, non-affine LayerNorm, linear to R^d for F, a scalar for Q_D and Q_R, or 5 tanh actions for the actor.
- B: 3 SwiGLU MLP blocks over g (same pre-norm, 4× hidden, gates), linear to d, L2-normalized to √d. D: same style over g concatenated with z, sigmoid output.

| Network | Width | Heads | Blocks |
|---|---|---|---|
| F₁, F₂ | 256 | 4 | 4 |
| Actor | 256 | 4 | 3 |
| Q_D, Q_R | 256 | 4 | 3 |
| B, D | 256 | n/a | 3 MLP blocks |

## 9. Judge demo

### 9.1 What judges do
1. Pick a scene: trash or ocean.
2. Type any mission. Examples: "grab the bag and drop it in the bin without getting near the moving carts", "carry the cooler as smoothly as possible", "net the school closest to the boat".
3. See the anchors and reward Gemini produced, in plain language, and press **Approve**. This is the human-in-the-loop step for the Ethical AI track.
4. Watch z being computed and refined, then the drone flies it live.
5. Mess with it while it flies: click to drop an obstacle, drag the bag, raise a wind gust, raise the waves, or knock out one fishing drone.

### 9.2 What the screen shows
- The 3D scene in the browser: drones, payload, obstacles, animated wave surface, net, and fish.
- The LiDAR point cloud, NIR returns in grey and green returns in green, so fish under the surface visibly light up.
- A z map: a 2D projection of z with points for known missions and the judge's new mission, showing where it landed.
- Live gauges: reward, payload swing, shield interventions.

### 9.3 How it runs
- The desktop runs Genesis headless with a few display worlds plus 256 worlds for z refinement.
- A small FastAPI server streams state over a WebSocket at 30 Hz: poses, net nodes, fish, LiDAR points.
- A three.js page renders it and sends judge actions back. Waves are drawn from the same Gerstner parameters, so they match the physics.
- The page runs on a laptop at the table, connected to the desktop over the local network or a direct cable.
- Backup: recorded videos of all three missions, plus the Genesis viewer if the web page fails.

## 10. Infrastructure

### 10.1 Desktop (Ryzen 7900X + RTX 5080): training and demo machine
- Linux or WSL2 strongly preferred over native Windows.
- Python 3.12 or newer (Genesis requires at least 3.12 and below 3.15).
- PyTorch with CUDA 12.8 or newer (the 5080 requires it), then `genesis-world`. Pin the exact Genesis version on day one, since its API changes between releases.

### 10.2 Laptops
Genesis runs on Apple Metal and CPU, so everyone can build and debug env code locally with a few worlds.

### 10.3 Memory budget for the replay buffer
- One LiDAR image is 3,072 values, 3 KB as uint8. Next observations are stored by index.
- 2M transitions ≈ 6 GB of LiDAR, which fits on the 5080's 16 GB beside the networks and Genesis.
- If more is needed, keep the buffer in pinned host memory and copy batches over.

### 10.4 Repo layout
```
env/
  drone.urdf          quadrotor with gripper link
  sky_env.py          SkyEnv: shared API, anchors, g, tokens, shield, copy_state
  controller.py       velocity → wrench
  lidar.py            Genesis raycaster + water and fish channels
  scenes/trash.py     arena, moving obstacles, wind, bag and cooler
  scenes/ocean.py     Gerstner waves, buoyancy, net, boids, boat
  expert.py           scripted expert
train/
  nets.py  buffer.py  fb_cpr.py  sf_fallback.py  zsample.py  configs/
mission/
  dsl.py  compile.py  infer_z.py  text2z/
demo/
  server.py           FastAPI + WebSocket, judge actions
  web/                three.js viewer
scripts/
PLAN.md
```

## 11. Work areas

The work is split into independent areas (training, environments, mission layer, judge demo, submission), each with folders, tasks, and a definition of done. See "Work areas" in README.md. The team assigns them.

The interfaces are fixed in skyhand/spec.py; change them only together, since every module and test depends on it.

## 12. Timeline (now to submission)

| Time | Milestone | Done when |
|---|---|---|
| 1:30 – 2:30 PM | M0 setup and spikes | **Done**: codebase, both backends, tests. Next: start the first real training run on the desktop |
| 2:30 – 5:30 PM | M1 flying | Drone, controller, trash scene, grasp, anchors, g, scripted expert. FB learns goal reaching with z = B(g_goal) |
| 5:30 – 8:30 PM | M2 grasping | Discriminator on expert demos. Reward programs for grasp, carry, and drop produce working z. Ocean scene runs with waves, net, and fish |
| **8:30 PM** | **Checkpoint** | **If FB has not learned reach and grasp, switch to the SF fallback** |
| 8:30 – 11:30 PM | M3 language and demo | Gemini to DSL to z to CEM, end to end. Demo app shows a live trash mission. Fishing training starts |
| 11:30 PM – 3:30 AM | M4 overnight | Long runs on the desktop: LiDAR actor, both scenes, multi-agent net. Sleep in shifts |
| 3:30 – 6:00 AM | M5 polish | Held-out evaluation, judge controls, z map, rehearse with typed missions. Text-to-z if time allows |
| 6:00 – 8:00 AM | M6 write-up | Devpost text, backup videos, slides |
| 8:00 – 9:00 AM | Buffer | Submit by 9:00. Edits allowed until 11:00 |

### Cut lines, in this order
1. Tiny text-to-z model: Gemini path only.
2. Judge perturbation controls: keep typed missions and live flight.
3. Net fishing: two drones, fixed-slack net, one school.
4. LiDAR actor: give the actor privileged tokens, show LiDAR as a visualization, say so honestly.
5. FB-CPR: SF fallback at the 8:30 PM checkpoint.
6. Web demo: Genesis viewer plus recorded videos.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Per-world welds or cloth do not work batched | Spike in the first hour, spring-damper grasp and torch net as fallbacks |
| Genesis API changes or bugs | Pin one version, keep custom physics in our own torch code |
| Env step too slow at thousands of worlds | Profile at M1. Fewer LiDAR rays, 10 Hz scans, fewer worlds for the ocean scene |
| FB training unstable or slow | Twin F with min targets, orthonormality weight, low lr for B, SF fallback with the same code |
| Grasping never discovered | Expert demos in the buffer and discriminator, anchors on handles, shaped reward programs |
| z_r poor for a mission | Include demo states in the scoring set, CEM refinement, chain shorter programs |
| Replay buffer exceeds GPU memory | uint8 LiDAR, index-based next state, pinned host memory |
| LLM writes a bad reward program | Whitelisted DSL, schema validation, retry, human approval |
| Demo network fails at the table | Direct cable, recorded videos, Genesis viewer |

## 14. Open questions

- Team size and who owns which role.
- Desktop OS: Linux, WSL2, or Windows.
- Gate placement: this plan gates both residual branches from the branch input. Confirm, or limit the gate to the attention output.
- SwiGLU width: full 4× hidden for both gate and up projections, not the 8/3 width.
