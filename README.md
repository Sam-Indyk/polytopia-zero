# PolyZero

Training a bot to play *The Battle of Polytopia* better than any human.

**Status: M1 complete — current-ruleset data, Numba kernels, vector env, baseline ladder, and a replay viewer.** The full research + architecture plan lives in [PLAN.md](PLAN.md).

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
(no server, no dependencies — open it from disk). Canvas board with territory tints, cities,
units and HP; scrub with the slider or ← / → / space; click any entry in the action log to jump;
and a **fog dropdown** that re-renders the board as a chosen player actually saw it, straight
from the recorded per-player explored masks.

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
