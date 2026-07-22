# PolyZero

Training a bot to play *The Battle of Polytopia* better than any human.

**Status: M0 in progress — engine core is live and green.** The full research + architecture plan lives in [PLAN.md](PLAN.md).

## Quickstart

```bash
# One-time: create the venv and install (Windows / PowerShell shown; adapt for your OS)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

# Run the golden + property tests
.venv\Scripts\python -m pytest

# Watch a random-vs-random game print turn by turn (the M0 watch target)
.venv\Scripts\python scripts/demo_random.py --size 11 --seed 7 --show-every 25

# Rough single-core throughput probe
.venv\Scripts\python scripts/bench_throughput.py
```

### What's landed (M0 core)

- `polyengine.core.state.GameState` — flat-array, cheaply-copyable state with deterministic
  hashing, dict (de)serialization, and per-player **fog memory** (`mem_*` last-seen snapshots).
- **Ruleset-as-data** — versioned TOMLs in `polyengine/data/` (terrain, units, techs, tribes,
  combat constants) loaded into flat NumPy lookup arrays.
- **Forward model** — mapgen, 8-connected movement/pathfinding, the wiki combat formula
  (Numba kernel, golden-tested), city capture, simplified economy, Domination win/turn-cap.
- **Legality/action contract** — the four-field `Action` (type → source → target → sub-type)
  that the Phase-2 auto-regressive heads will mask against.
- `RandomAgent` baseline, ASCII renderer (god's-eye + fogged views), and a **39-test** golden +
  property + determinism suite (`pytest`).

Still simplified pending later milestones: the resource/building economy, the full tech-effect
logic, naval/oceans, and special tribes (all scheduled M1–M4 in [PLAN.md](PLAN.md)).

The short version:

1. **Engine** — a from-scratch Python + Numba simulator of the *current* Steam ruleset (the academic [Tribes](https://github.com/GAIGResearch/Tribes) engine is frozen at 2020 rules and too slow through its Java bridge; we keep it as a differential-testing oracle). Copyable `GameState`, forward model, versioned ruleset data files, multiprocess vector env.
2. **Agent** — PPO self-play with masked auto-regressive action heads (AlphaStar-style type → source tile → target tile → sub-type), ~10M-param fully-convolutional ResNet with FiLM-injected global state, fog-of-war handled via engine-maintained "last-seen" memory planes. AlphaStar-lite league (checkpoint pool + PFSP + exploiter) and a win-rate-gated curriculum from tiny-map 1v1 mirrors up to all tribes / big maps / FFA. Optional determinized IS-MCTS booster at eval time.
3. **Real client** — a vision (screen capture + template matching/CNN + OCR) and mouse-control layer that plays the actual Steam game in **single-player vs the built-in Crazy bots**. Multiplayer vs humans stays disabled unless Midjiwan grants permission (their ToS forbids automation; they are historically AI-research-friendly).

See [PLAN.md](PLAN.md) for the milestone schedule (M0–M5), repo layout, throughput targets, and rule-validation strategy.
