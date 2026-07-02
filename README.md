# PolyZero

Training a bot to play *The Battle of Polytopia* better than any human.

**Status: planning phase — no code yet.** The full research + architecture plan lives in [PLAN.md](PLAN.md).

The short version:

1. **Engine** — a from-scratch Python + Numba simulator of the *current* Steam ruleset (the academic [Tribes](https://github.com/GAIGResearch/Tribes) engine is frozen at 2020 rules and too slow through its Java bridge; we keep it as a differential-testing oracle). Copyable `GameState`, forward model, versioned ruleset data files, multiprocess vector env.
2. **Agent** — PPO self-play with masked auto-regressive action heads (AlphaStar-style type → source tile → target tile → sub-type), ~10M-param fully-convolutional ResNet with FiLM-injected global state, fog-of-war handled via engine-maintained "last-seen" memory planes. AlphaStar-lite league (checkpoint pool + PFSP + exploiter) and a win-rate-gated curriculum from tiny-map 1v1 mirrors up to all tribes / big maps / FFA. Optional determinized IS-MCTS booster at eval time.
3. **Real client** — a vision (screen capture + template matching/CNN + OCR) and mouse-control layer that plays the actual Steam game in **single-player vs the built-in Crazy bots**. Multiplayer vs humans stays disabled unless Midjiwan grants permission (their ToS forbids automation; they are historically AI-research-friendly).

See [PLAN.md](PLAN.md) for the milestone schedule (M0–M5), repo layout, throughput targets, and rule-validation strategy.
