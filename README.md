# PolyZero

Training a bot to play *The Battle of Polytopia* better than any human.

**Status: M2 in progress — the PPO self-play stack (observation encoder, masked auto-regressive heads, network, trainer) is built and training runs end to end on GPU.** The full research + architecture plan lives in [PLAN.md](PLAN.md).

## Quickstart

```bash
# One-time: create the venv and install (Windows / PowerShell shown; adapt for your OS)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

# Run the golden + property tests
.venv\Scripts\python -m pytest

# Watch a random-vs-random game print turn by turn
.venv\Scripts\python scripts/demo_random.py --size 11 --seed 7 --show-every 25

# Record a game and open it in the visual replay viewer
.venv\Scripts\python scripts/record_replay.py --p0 scripted --p1 random --open

# Baseline ladder win-rate matrix
.venv\Scripts\python scripts/arena.py --agents random scripted osla --games 20

# Throughput: single core, then multiprocess self-play
.venv\Scripts\python scripts/bench_throughput.py
.venv\Scripts\python scripts/bench_throughput.py --parallel --games 640
```

### Watching games

`scripts/record_replay.py` writes a `.json` replay plus a **self-contained `.html` viewer**
(no server, no dependencies — open it from disk).

The viewer's rule is that **nothing the engine knows is hidden**:

- **Board** — terrain, resources (fruit / crop / game / metal / fish / whale), territory tints,
  cities (level, walls, capital ring) and units (type, HP bar, veteran ring, dimmed once spent).
- **Tech tree** — every tech for every player, marked researched / available / locked, with the
  *live* star cost recomputed per frame from that player's city count and Literacy discount.
- **Tile inspector** — click any tile for its terrain, resource, territory, city (level/pop/
  walls/capital), unit (HP, veteran, kills, whether it can still move or act), and which players
  have explored it.
- **Action log** — every micro-action, colour-coded by player, click to jump.
- **Fog selector** — re-renders the board as a chosen player actually saw it, from the recorded
  per-player explored masks.
- **"Not simulated yet" panel** — lists the mechanics the engine does not model, so an empty
  log never reads as "nothing happened".

Scrub with the slider or ← / → / space.

### What's landed (M2 so far)

```bash
# Install the training extras (cu128+ is required for Blackwell GPUs)
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# Tiny-map smoke run: should climb against RandomLegal within minutes
.venv\Scripts\python -u -m polytrain.train --smoke

# A small real run
.venv\Scripts\python -u -m polytrain.train --size 9 --games 32 --rollout 128 --iters 200
```

- **Observation encoder** (`polyengine/obs.py`) — 51 spatial planes + a global vector,
  Numba-compiled. Unit and city planes render from the player's **fog memory**, never ground
  truth, and opponent stars are not observable — the same information a human sees on screen,
  which is what makes the M5 vision layer able to produce identical tensors. Pads to a fixed
  size with an off-board mask plane for size-bucketed batching.
- **Action masks** (`polyengine/masks.py`) — `legal_types / legal_sources / legal_targets /
  legal_subtypes` for the four auto-regressive heads. Cross-checked against `enumerate_actions`
  on every decision point of random games, so the policy provably cannot emit an illegal action.
- **Network** (`polynet/`) — fully-convolutional ResNet trunk with squeeze-excitation and FiLM
  injection of the global vector; pointer heads (1×1 conv, one logit per tile) for source and
  target; PopArt value head; auxiliary final-score and opponent-tech heads. One set of weights
  serves every map size.
- **PPO trainer** (`polytrain/`) — self-play rollouts with GAE chained **per (game, player)**,
  since a player's own decisions are not contiguous in a turn-based game; clipped surrogate loss,
  summed per-head entropy, annealed material shaping on top of terminal ±1.

- **Snapshot opponent pool** (`polytrain/pool.py`) — a fraction of iterations are played against
  frozen past selves, so the policy has to stay good against old styles instead of cycling.

#### Results so far (6×6 smoke config, 0.25M-param net, ~14 min on an RTX PRO 1000)

| run | vs RandomLegal | vs ScriptedBot (100 games) |
|---|---|---|
| mirror self-play, final (iter 150) | 100% | **47%** |
| + snapshot pool, final (iter 300) | 100% | **69%** |
| + snapshot pool, best (iter 160) | 100% | **88%** |

The first run reached 91.7% against ScriptedBot at iteration 100 and fell back to 47% by 150 —
confirmed with a 100-game eval, so drift, not noise. That is textbook self-play strategy cycling:
mirror self-play only optimises against the *current* opponent, so nothing stops the policy
unlearning how to beat an earlier style. The snapshot pool is the standard fix and roughly
doubled sustained strength. A best-by-win-rate checkpoint is now saved alongside `latest.pt`,
because self-play win rate is not a monotone proxy for strength.

**The M2 gate (>90% vs ScriptedBot, >60% vs MCTS-1k) is not met yet** — 88% at best on the toy
config. Remaining M2 work: a scaled-up run at the real config (`--size 9`, full 160×12 net), the
MCTS-1k gate, and the Elo curve. The residual best-vs-final gap (88% → 69%) is exactly what the
full PFSP league in M3 is designed to close.

### What's landed (M1)

- **Current ruleset** (`ruleset_version = "current-2026.07"`) — 23-tech tree with the live
  `tier × cities + 4` research formula and Philosophy/Literacy's 33 % discount; the naval roster
  (Raft / Scout / Rammer / Bomber / Juggernaut); Diplomacy with Cloak + Dagger; unit skills and a
  `naval` flag carried as data.
- **Numba grid kernels** (`polyengine/kernels/grid_kernels.py`) — fog/visibility, movement
  Dijkstra, and attack-target scans compiled; **5.2k micro-steps/s/core** (M1 target ≥ 5k, was
  ~3.0k in M0).
- **Vector env** (`polyengine/vec.py`) — `SyncVecGame` for the M2 PPO rollout worker, plus
  `parallel_rollout` for multiprocess mass self-play.
- **Baseline ladder** — `ScriptedBot` (rule-based), `GreedyOSLA` (one-step lookahead), and
  `MCTSAgent` (UCT on the forward model) over a shared heuristic, plus `scripts/arena.py`.
  Measured on 9×9: `random` 0 % → `osla` ~71 % → `scripted` ~79 % overall win rate, and
  MCTS(30 sims) beats random 4/4. MCTS is an **eval-time** baseline, not a training component —
  it costs ~0.16 s/move at 30 sims (~50 s/game), so keep `--games` small in the arena.
- **Replay + fidelity harness** — `polyengine.replay` records every micro-action;
  `polyengine.testing.differential` re-simulates a replay and localises the first divergent
  frame/field. This is the gate that will pin the ruleset against captured Steam games.

### What's landed (M0 core)

- `polyengine.core.state.GameState` — flat-array, cheaply-copyable state with deterministic
  hashing, dict (de)serialization, and per-player **fog memory** (`mem_*` last-seen snapshots).
- **Ruleset-as-data** — versioned TOMLs in `polyengine/data/` (terrain, units, techs, tribes,
  combat constants) loaded into flat NumPy lookup arrays.
- **Forward model** — mapgen, 8-connected movement/pathfinding, the wiki combat formula
  (Numba kernel, golden-tested), city capture, simplified economy, Domination win/turn-cap.
- **Legality/action contract** — the four-field `Action` (type → source → target → sub-type)
  that the Phase-2 auto-regressive heads will mask against.
- `RandomAgent` baseline, ASCII renderer (god's-eye + fogged views), and a golden +
  property + determinism suite (`pytest`) — now **89 tests**.

Still simplified pending later milestones: the resource/building economy, the full tech-effect
logic, and special tribes (scheduled M2–M4 in [PLAN.md](PLAN.md)). Naval and Cloak/Dagger
**stats and tech gating are current data**, but their *mechanics* (ports, embarking,
infiltration) are wired up in M4 — those units are flagged non-trainable until then.

Concretely **not yet simulated** (the viewer lists these too): buildings (lumber hut, farm,
mine, port, temple, market), resource harvesting, city level-up rewards (workshop / explorer /
walls / giant), roads and trade routes, naval embarking, and diplomacy relations. Cities
currently grow through a placeholder economy rather than the real population/level loop.

### Capture takes a turn

Moving a unit onto a village or city does **not** take it. Capture is a separate action
(`A_CAPTURE`) that requires the unit to still be standing there at the start of a later turn,
and it consumes that unit's turn. This is governed by `capture_requires_fresh_unit` /
`capture_ends_unit_turn` in `data/combat.toml` and pinned by `tests/test_capture.py`.

### Ruleset fidelity

The TOMLs encode the current Steam rules to the best of available knowledge, and the tree
invariants (tiers, prerequisites, cost formula, rosters) are pinned by `tests/test_rules_current.py`.
Exact per-unit stat fidelity is deliberately *not* trusted to hand-entry: the authority is
`polyengine.testing.differential`, which replays a captured Steam game through the engine and
names the first frame and field that disagree. Feeding it real captures is an M3 task.

The short version:

1. **Engine** — a from-scratch Python + Numba simulator of the *current* Steam ruleset (the academic [Tribes](https://github.com/GAIGResearch/Tribes) engine is frozen at 2020 rules and too slow through its Java bridge; we keep it as a differential-testing oracle). Copyable `GameState`, forward model, versioned ruleset data files, multiprocess vector env.
2. **Agent** — PPO self-play with masked auto-regressive action heads (AlphaStar-style type → source tile → target tile → sub-type), ~10M-param fully-convolutional ResNet with FiLM-injected global state, fog-of-war handled via engine-maintained "last-seen" memory planes. AlphaStar-lite league (checkpoint pool + PFSP + exploiter) and a win-rate-gated curriculum from tiny-map 1v1 mirrors up to all tribes / big maps / FFA. Optional determinized IS-MCTS booster at eval time.
3. **Real client** — a vision (screen capture + template matching/CNN + OCR) and mouse-control layer that plays the actual Steam game in **single-player vs the built-in Crazy bots**. Multiplayer vs humans stays disabled unless Midjiwan grants permission (their ToS forbids automation; they are historically AI-research-friendly).

See [PLAN.md](PLAN.md) for the milestone schedule (M0–M5), repo layout, throughput targets, and rule-validation strategy.
