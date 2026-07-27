# PolyZero — a superhuman Polytopia bot

## Context

Goal: train a bot that plays *The Battle of Polytopia* better than any human, on a school AI training workstation (spec TBD — design scales from 1 GPU to multi-GPU via config), and eventually have it drive the real Steam client via screen capture + cursor control.

Research findings that shape the plan:

- **[GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes)** (AIIDE 2020) — open-source Java Polytopia reimplementation with a forward model and MCTS/RHEA/OSLA baseline agents. Frozen at ~2020 rules (predates the 2022 "Path of the Ocean" naval rework and current tribes/balance); its Py4J Python bridge ([polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl)) caps out at ~hundreds of steps/sec. We use it as a **rules reference oracle and differential-test target**, not as our engine.
- **Midjiwan's built-in bots are rule-based heuristics that cheat**: difficulty scales via bonus star income (Crazy = 5 stars/turn vs a player's 2), not smarter play. Beating them on merit is a mid-project milestone; "better than any human" requires learned self-play — the same conclusion AlphaStar/OpenAI Five reached for comparable games.
- **Polytopia's ToS forbids bots/automation.** Midjiwan is historically AI-research-friendly (DeepMind AndroidEnv partnership; reportedly considered an official bot API after a 2026 community project).

User decisions (confirmed):
1. **Build our own engine** matching current Steam rules; Tribes as reference only.
2. **Python + NumPy/Numba** engine stack (Rust-port escape hatch noted below).
3. Real-game phase: **single-player vs built-in bots first**; any multiplayer vs humans is gated on written permission from Midjiwan.
4. Hardware unknown → profile on the machine; all scale knobs in config.

## Approach in one paragraph

Build a fast, current-rules Python game engine with a copyable `GameState` + forward model (M0–M1); train a ~10M-param fully-convolutional policy/value network with **PPO self-play** using masked auto-regressive action heads (M2); scale strength with an **AlphaStar-lite league** (checkpoint pool + exploiter + curriculum from tiny-map 1v1 mirror matches up to large maps, all tribes, FFA) (M3–M4); optionally boost eval-time strength with policy-guided **determinized IS-MCTS** on the forward model; finally add a **vision + mouse-control layer** that parses the Steam client's screen into the same observation format and plays single-player vs Crazy bots (M5).

## Phase 1 — Engine (`polyengine`)

A from-scratch simulator of current Steam rules. Python package, performance-critical inner loops (combat resolution, movement/pathfinding, legality-mask generation, observation encoding) as Numba `@njit` kernels over flat NumPy arrays; game orchestration in plain Python.

- **State:** `GameState` = flat per-tile arrays (terrain, resource, improvement, owner, unit slots) + per-player structs (stars, tech bitmask, per-player **fog memory**: last-seen snapshot of every tile + staleness counter — the engine does the remembering, matching exactly what the real client shows). Cheap `copy()` is non-negotiable: it powers MCTS baselines, the IS-MCTS booster, and debugging.
- **Determinism:** every game is `(map_seed, ruleset_version, tribe_config)` → fully reproducible; all stochasticity (map gen, ruins) from one per-game PCG64 stream.
- **Ruleset as data:** unit stats, tech tree, tribe configs, combat constants in versioned TOML/JSON files — balance patches become data edits, and a `tribes-2020-compat` ruleset enables differential testing vs the Tribes engine.
- **Vector env:** `VecGame` steps N games across worker processes (multiprocessing + shared-memory NumPy buffers), returning batched observation tensors and legality bitmaps.
- **Throughput target:** ≥ 5k micro-steps/s/core including observation encoding (Numba makes this realistic); ≥ 100k steps/s aggregate on a 32-core box. If profiling on the school machine shows the env starving the GPU, port only `poly-core`'s kernels to Rust/PyO3 behind the same API — the plan isolates them for exactly this reason.
- **Scope order:** 4 base tribes, 11×11 maps, Domination mode first. Special tribes (Aquarion, Elyrion, Polaris, Cymanti — they modify core rules) and Path-of-the-Ocean naval come in M4 only after the base engine is golden-tested.

**Rule-correctness validation** (the step everyone skips and regrets):
1. Table-driven golden tests from community-documented formulas (combat: `damage = ceil(atkForce/(atkForce+defForce) · atk · 4.5)` with force = stat·HP/maxHP·bonuses, etc. — the Polytopia wiki documents all of it).
2. Differential replay vs Tribes in 2020-compat mode: identical action scripts through both engines, diff state hashes every step.
3. Property-based tests (hypothesis): stars never negative, fog never reveals unseen tiles, every legal action applies without error, incremental score matches recomputation.
4. **Steam replay fixtures:** screen-record ~30 short real games, transcribe to action scripts, assert the engine reproduces every visible number (HP, stars, level-up choices). The only test of *current-rules* fidelity; directly de-risks Phase 3.

## Phase 2 — Agent

**Observation** (identical contract for sim and future screen parser — deliberate constraint):
- Spatial tensor ~96 planes × H × W, all from the acting player's perspective (opponents in fixed relative slots): terrain (8), resources (7), improvements (~18), units (~36: type one-hot, relative owner, HP fraction, veteran, moved/attacked, status), cities/territory (~12), fog (visible, explored, staleness scalar) — unit/city planes rendered from the player's **fog memory**, not ground truth; normalized x,y coords.
- Global vector ~180 dims: stars, income, turn, tech one-hot, per-opponent public score & diplomacy state, tribe one-hots, map size, mode.
- Variable map sizes: fully-convolutional net, size-bucketed batches (12/16/20/32 padding + off-board mask plane).

**Action space** — one micro-action per RL step; a turn is a sequence ending in END_TURN. Four auto-regressive masked heads (the AlphaStar pattern): action-type (~12) → source tile (per-tile pointer via 1×1 conv) → target tile (pointer, conditioned on type+source) → sub-type (~48: unit to train / improvement / tech / ability / **city level-up reward choice** — those are real decisions and first-class actions). Engine exposes `legal_types() / legal_sources(t) / legal_targets(t, src)` bitmaps. Rejected: flat joint space (~10M logits on big maps) and whole-turn macro-actions.

**Algorithm — PPO self-play.** AlphaZero/MuZero rejected: fog of war breaks perfect-information tree search, and in-loop MCTS multiplies compute ~100-800× per move — wrong trade on a workstation when we own a cheap perfect simulator. Feed-forward net first (fog-memory planes carry most hidden state); add a small GRU over the global vector only if eval shows weak scouting behavior. Reward: terminal ±1 (Domination), small Δscore shaping annealed to zero over the first ~30% of training; 100-turn cap with score tiebreak. Each player's trajectory contains only their own decision points; GAE runs across turn boundaries.

**Network:** fully-conv ResNet trunk, 12 blocks × 160 ch with squeeze-excitation, global vector injected via FiLM at blocks 1/5/9; per-tile pointer heads at native resolution; value head = global-avg-pool + MLP with PopArt; auxiliary heads (final-score and opponent-tech prediction, 0.25 weight) as free representation shaping and a fog-inference probe. ~10M params to start, scalable to ~40M. PPO config sketch: 1024–4096 parallel games, rollout 128, minibatch 8192, 3 epochs, lr 3e-4 cosine, γ=0.999, GAE λ=0.95, per-head entropy 0.01 annealed.

**Eval-time booster (optional, after M3):** determinized IS-MCTS — sample K=8–16 hidden-state completions consistent with fog memory (`sample_hidden(rng)` in the engine), short PUCT searches (~200 sims) with the trained policy as prior and value head at leaves, aggregate root visits. Typically +100–300 Elo without touching training; can later distill back into the policy (expert-iteration-lite) if self-play plateaus.

## Phase 3 — League, curriculum, evaluation

- **League (AlphaStar-lite, workstation-sized):** one main learner; frozen checkpoint pool (snapshot every ~2×10⁷ steps, cap ~40, Elo-spread-preserving eviction); one **main exploiter** seat once scripted baselines are beaten (reinitialized whenever it hits 70% vs main; trains only vs current main to find degenerate strategies). Opponent sampling: 50% mirror self, 35% PFSP weighted p(1−p), 10% hardest (1−p)², 5% scripted baselines.
- **Curriculum — win-rate gates, not schedules:** (1) 11×11 1v1 mirror-Imperius on mirrored seeds → (2) random base-tribe pairings → (3) maps 11–20, all standard-rules tribes → (4) special tribes + naval rework → (5) 30×30 and 3–4 player FFA. Keep ~20% of games at earlier stages forever (anti-forgetting).
- **Baseline ladder:** `RandomLegal` → `Greedy-OSLA` → `ScriptedBot` (hand-written Midjiwan-style heuristic, plus a "Crazy-cheat" +star-income variant as the Phase-4 proxy target) → `MCTS-1k/10k` on our forward model → frozen checkpoints → humans on the replay viewer.
- **Protocol:** mirrored seed pairs (same map, swapped seats) to halve spawn-luck variance; ≥400 games per pairing; checkpoint promotion via SPRT (p=0.5 vs 0.55, α=β=0.05); anchored Elo (Random=0) fit nightly. Behavioral dashboards (game length, tech timing, kill ratios, expansion rate) to catch reward-hacking (e.g., turtling to the turn cap) before Elo does.

## Phase 4 — Playing the real Steam client (`polypilot`)

- **Perception:** screen capture (`mss`/DXGI) → tile-grid registration → template matching + a small CNN classifier for terrain/units/buildings (Polytopia's clean flat 2D art makes this tractable) + OCR for stars/HP/scores → emit the *same observation tensors* the sim produces.
- **Reconciliation:** maintain a `GameState` mirror in our engine; after each detected opponent turn, diff parsed screen vs predicted state — divergence = rules bug (feed back into Steam replay fixtures) or parse error.
- **Control:** map chosen micro-action → click sequences (`pyautogui`/win32 SendInput) with UI-state verification between clicks; human-speed pacing.
- **Scope & ethics:** single-player vs built-in Crazy bots only — "beats Crazy difficulty without the star cheat, through the real client" is the headline demo. **Multiplayer vs humans is designed but not enabled until Midjiwan grants written permission** (contact via polytopia.io — genuinely plausible given their DeepMind collaboration and interest in a bot API).

## Repo structure

```
polytopia-zero/
├── polyengine/            # game engine (Python + Numba)
│   ├── core/              #   state.py, actions.py, rules/, mapgen.py, fog.py, score.py
│   ├── kernels/           #   @njit hot loops: combat, movement, masks, obs encoding
│   ├── data/              #   versioned ruleset TOMLs: units, techs, tribes
│   ├── vec.py             #   VecGame multiprocess batch env
│   └── agents/            #   Random, OSLA, ScriptedBot, MCTS/IS-MCTS baselines
├── polynet/               # trunk, FiLM, auto-regressive masked heads (PyTorch)
├── polytrain/             # PPO learner, rollout buffer, PopArt, hydra configs
├── polyleague/            # checkpoint pool, PFSP sampler, exploiter mgmt, Elo
├── polyeval/              # ladder runner, SPRT, mirrored-seed pairing, dashboards
├── polypilot/             # phase 4: capture/, perception/, control/, reconcile.py
├── viewer/                # TS canvas replay viewer (engine dumps JSON replays)
├── tests/golden/          # wiki-formula tables, Steam fixtures, Tribes diff scripts
└── scripts/               # train.py, eval_ladder.py, bench_throughput.py
```

Most load-bearing invariant: **`polyengine.core.state.GameState` + the ruleset data files are the single source of truth** consumed by the trainer, baselines, search booster, viewer, and Phase-4 reconciler. Freeze its serialization and legality API earliest; review it hardest.

## Milestones (each ends in something watchable)

| # | Deliverable | Est. |
|---|---|---|
| M0 | Engine core: 4 base tribes, 11×11, mapgen, fog+memory, combat, tech, Domination; golden tests; ASCII renderer. *Watch: random-vs-random game printed turn by turn.* | 3–4 wk |
| M1 | `VecGame` + Numba kernels hit throughput target; web replay viewer; Random/OSLA/Scripted baselines; Tribes differential tests. *Watch: bots fight in the browser.* | 2–3 wk |
| M2 | PPO trainer + network + masking; mirror-Imperius self-play; beats ScriptedBot >90%, MCTS-1k >60%. *Watch: agent replays, first Elo curve.* | 4–5 wk |
| M3 | League + curriculum stages 2–3 + SPRT eval harness. *Watch: league Elo dashboard; you lose to it in the viewer.* | 4–6 wk |
| M4 | Special tribes + naval rework golden-tested; **map-generation fidelity** (see below); curriculum 4–5; IS-MCTS booster; FFA. *Watch: it beats you with Cymanti.* | 6–8 wk |
| M5 | `polypilot` vision+control; plays Steam single-player, beats Crazy bots. *Watch: the end-goal demo.* | 6–10 wk |

≈ 6–8 months solo, ~4 with two people (engine and training parallelize after M0 freezes the state format).

## Target hardware and how this workload actually scales

Two machines: the dev laptop (RTX PRO 1000, 8 GB; 16-core hybrid CPU) and the
training workstation (4× RTX PRO 6000 Blackwell, 96 GB each = 384 GB VRAM;
Threadripper PRO up to 96 cores / 192 threads; up to 2 TB DDR5). `polytrain/profiles.py`
holds both as named profiles (`--profile laptop|workstation|auto`) so no run-shape
constants are scattered through the training code.

**Standard large-language-model scaling advice does not transfer to this workload,
and following it literally would waste effort.** The distinction that matters: an
LLM is memory-bound on *parameters*; this is a reinforcement-learning agent that is
compute-bound on *generating its own data*.

| LLM guidance | Why it doesn't apply here | What we do instead |
|---|---|---|
| FSDP / DeepSpeed ZeRO-3 sharding | Exists to fit a model that won't fit on one GPU. Ours is 6.2M params (24.6 MB); even the scaled-up workstation net is 55.6M (222 MB) — **0.23% of one 96 GB card** | Plain data-parallel (DDP) *if* gradients ever dominate; today they don't |
| NVIDIA Transformer Engine, FP8/FP4 | TE fuses transformer attention/linear kernels. Our trunk is convolutional (3×3 convs + squeeze-excitation), so those kernels don't apply | BF16 autocast as the default win. FP8 on convs is plausible on Blackwell but must be validated against a BF16 reference — RL value heads are numerically touchier than a language-model loss |
| `num_workers`, `pin_memory` | Those are `DataLoader` settings for a *fixed dataset*. We have no dataset; every sample is produced by self-play as we train | `env_workers`: processes stepping the game engine |
| Maximise global batch size | Partly right, but PPO is on-policy — a batch is only useful while it reflects the current policy. Huge batches mean stale data, not free speed | Scale parallel *environments* so a large batch is still fresh |

**What actually unlocks the workstation, in order of value:**

1. **96 CPU cores.** Rollout generation is CPU-bound (engine stepping, mask
   building, observation encoding) — measured at 93% CPU / 7% GPU. This is the
   present bottleneck and it scales close to linearly with cores.
2. **4 GPUs for concurrent league seats.** The league wants a main learner plus
   exploiters (Phase 3). One GPU per seat is simpler and more useful than
   sharding one small model four ways.
3. **384 GB VRAM to grow the network ~50×.** Only pays off *alongside* far more
   parallel environments; otherwise a bigger net just makes each rollout step
   slower and starves the learner.

**Known gap to close before that box arrives:** the current multiprocess rollout
puts a whole policy copy in each worker, which measured *worse* than
single-process GPU for large nets (CPU inference collapses; several processes
sharing one GPU merely contend). The architecture that scales to 96 cores + 4 GPUs
is workers doing env/encode/mask over shared memory while a dedicated inference
process batches GPU forwards. That is the one piece of engineering the workstation
genuinely requires — see the table in `polytrain/parallel.py` for the measurements
that justify it.

## Deferred: map-generation fidelity (M4)

Current `mapgen.py` produces *plausible* maps, not Polytopia's. It is good enough for
M2–M3 because training runs on small maps where the differences barely bind, and because
the agent's observation contract does not depend on how tiles were placed. Before any
result is claimed to transfer to the real game — and certainly before M5's Steam pilot —
generation has to match the real algorithm, including at least:

- **Cities never spawn on the map border**, and respect minimum spacing from each other.
- **Lighthouses in the four corners** (and their role in revealing the map / ocean routes).
- Per-tribe terrain weighting applied as the real generator does (tribes already carry
  spawn weights in `tribes.toml`, but the placement algorithm is ours, not theirs).
- Village count and distribution scaling with map size; capital placement rules.
- Ruins, resource placement rules, and the land/water ratio by map type.

Why it is safe to defer: mis-generated maps bias *what positions the agent sees*, not the
rules it learns. The risk is a policy overfit to unrealistic openings (e.g. border cities),
which matters exactly when maps get large — which is the M4 curriculum stage. Doing it then
also lets it be validated the same way the ruleset was: against captured real games rather
than by hand.

## Verification

- **Engine:** `pytest tests/` (golden formulas, property tests), `scripts/bench_throughput.py` for steps/sec, Tribes differential replays, then the 30 Steam replay fixtures as the fidelity gate.
- **Training:** first smoke test = PPO on 5×5 toy config overfits to beat RandomLegal within minutes; then M2/M3 win-rate gates with SPRT; watch replays in the viewer for sane behavior (expansion, scouting, focus-fire).
- **Phase 4:** parser accuracy vs sim-rendered screenshots (label-free: render known states, screenshot, parse, diff); then a full recorded Crazy-bot game with zero reconciliation divergences.

## First implementation step

Scaffold the repo, write `GameState` + ruleset TOMLs for the 4 base tribes, and implement mapgen + movement + combat with the wiki-formula golden tests — i.e., start M0.
