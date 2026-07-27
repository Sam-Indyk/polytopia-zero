# PolyZero — orientation for the next session

A from-scratch engine for *The Battle of Polytopia* plus a PPO self-play agent.
Goal (PLAN.md): beat the game's built-in Crazy bots through the real Steam client.

**Read this file, then `docs/ARCHITECTURE.md` if you're changing engine internals.**
`PLAN.md` holds the milestone schedule and the long-term design.

---

## Environment

```bash
.venv\Scripts\python.exe -m pytest            # 247 tests, ~4 min
.venv\Scripts\python.exe -m polytrain.train_league --smoke --iters 50
```

Always go through `.venv\Scripts\python.exe`. Python 3.12, numpy 2.4, numba 0.66,
torch 2.11+cu128 (the cu128 build is **required** — the dev GPU is Blackwell,
compute capability 12.0; earlier CUDA builds won't run on it).

Two gotchas that will waste your time otherwise:
- Numba needs the MS VC++ redistributable (`winget install Microsoft.VCRedist.2015+.x64`).
- The console is cp1252. Non-ASCII in `print()` raises `UnicodeEncodeError`.
  Keep CLI output ASCII or set `PYTHONIOENCODING=utf-8`.

## Layout

| Package | What it is |
|---|---|
| `polyengine/core/` | Rules: state, actions, combat, movement, fog, **economy**, mapgen |
| `polyengine/data/*.toml` | The ruleset **as data** — units, techs, terrain, improvements, combat |
| `polyengine/kernels/` | Numba `@njit` hot loops (combat, fog, pathfinding) |
| `polyengine/obs.py`, `masks.py` | What the network sees / what it may legally do |
| `polyengine/testing/` | `differential` (replay determinism) + `wiki_check` (rule fidelity) |
| `polynet/` | PyTorch: conv trunk, auto-regressive masked heads, PopArt value |
| `polytrain/` | PPO, rollouts, snapshot pool, PFSP league, eval protocol, profiles |
| `scripts/` | CLI entry points (record replays, arena, benchmarks, wiki fetch) |

## Four invariants — don't break these

1. **The observation may only contain what a human can see.** `obs.py` renders
   units/cities from the acting player's *fog memory* (`mem_*`), never ground
   truth, and opponent stars are not observable. M5's screen-reading agent must
   be able to produce identical tensors from pixels; if the policy learns to use
   hidden information, that whole phase collapses. Tests in `tests/test_obs.py`.

2. **Masks and `enumerate_actions` must describe the same set.** The policy
   samples from `masks.py`; the engine accepts `enumerate_actions`. Drift means
   either illegal actions (crash) or a silent capability ceiling.
   `tests/test_masks.py` checks equivalence at every decision point of real games.

3. **Rules come from the wiki, not from memory.** Every stat in `data/*.toml` is
   checked by `scripts/check_ruleset.py` against cached wiki pages (164 checks,
   0 mismatches). Change a number → re-run it.

4. **Games are deterministic.** `(map_seed, ruleset_version)` fully determines a
   game. `polyengine/testing/differential.py` re-simulates recorded replays and
   localises the first divergent frame *and field*.

## Getting the real rules

Fandom blocks generic scraping (WebFetch gets HTTP 402) **but its MediaWiki
`api.php` responds fine.** `scripts/fetch_wiki.py` uses it; pages are cached in
`.wikicache/` (committed, so you don't need network access).

```bash
.venv\Scripts\python.exe scripts/fetch_wiki.py --all   # refresh the cache
.venv\Scripts\python.exe scripts/check_ruleset.py      # diff TOMLs vs wiki
```

Two rules were wrong for a while and each badly distorted training, so treat
"the agent is doing something weird" as a rules bug until proven otherwise:
- Capture was a side effect of moving. It is its own action: a unit must **start
  its turn** on the city, and capturing ends that unit's turn.
- Cities had no **unit-support cap**. A city supports `level + 1` units. Without
  it, star income converted straight into army and the agent flooded the map
  with 75 warriors while ignoring the tech tree.

## Current state

Engine is complete for base-tribe land play: movement, combat, capture, tech,
and the full population economy (harvests, buildings, city levels, level-up
reward choices). **Not simulated yet:** markets/roads/city connections, naval
embarking (the Port building exists; boarding does not), cloak infiltration and
diplomacy, special tribes. The viewer lists these so an empty log never reads as
"nothing happened".

Best trained agent (`checkpoints_11x11/`, committed), 100 mirrored games on 11×11:

| | vs Random | vs Greedy-OSLA (held out) | vs ScriptedBot |
|---|---|---|---|
| `best.pt` (iter 275) | 100% | 96% | 55% ±5% |
| `latest.pt` (iter 700) | 98% | 82% | 58% ±5% |

It plays properly — expands, buys population, levels cities, respects capacity —
but **it plateaued after ~iteration 275**. See "What to do next".

`best.pt` is mislabelled: it was picked on a 16-game eval reading 68.8%, which at
100 games is 55%. Selecting the max over many noisy evals overestimates. Don't
trust a checkpoint label; re-measure with `polytrain.evaluate.play_match`.

## What to do next

The plateau is most likely **sparse terminal reward**: with 96-step rollouts and
11×11 games running ~80 turns, only ~6–9 episodes finish per iteration, so 700
iterations saw maybe 5,000 win/loss signals. In rough order of expected value:

1. **Longer rollouts** (`--rollout 256`–512) so games finish inside a segment.
   One flag, ~30 min to test. Try this first.
2. **More parallel games**, which needs the rollout rewrite below.
3. **A bigger network** — 0.57M params is small for a 64-plane 11×11 board.

**The one piece of engineering the target hardware genuinely needs:** the current
multiprocess collector (`polytrain/parallel.py`) puts a full policy copy in every
worker. Measured, that's a win for tiny nets (9.6k vs 3.8k steps/s) and *worse*
for large ones (660 vs 2,660). Training currently uses **one core of sixteen**
(measured 7% total CPU, GPU 25% idle-ish) because `train_league.py` is
single-process. The architecture that scales is workers doing env/encode/mask
over shared memory with one process batching GPU inference. See the table in
`polytrain/parallel.py` and the hardware section of `PLAN.md`.

Deferred by explicit decision: **map-generation fidelity** (cities never on the
border, corner lighthouses, real village spacing). Fine while maps are small;
required before M5. Rationale is in `PLAN.md`.

## Traps that have already bitten

- **Checkpoints get clobbered.** A 2-iteration flag test once overwrote a
  300-iteration run because it used the default `--out`. Always pass an explicit
  `--out` for throwaway runs.
- **`torch.multinomial` on a NaN row returns an out-of-range index** rather than
  raising, which later faults inside an unrelated `gather` as an async CUDA
  assert with a misleading traceback. Guarded now in `polynet/net.py`.
- **`viewer.py` embeds JS/CSS in a Python string.** Escapes are treacherous:
  `\25b8` in CSS was silently parsed as a Python octal escape. After editing,
  regenerate a replay and render it (headless Edge works) before trusting it.
- **Elo resets on `--resume`** — the league pool rebuilds empty, so ratings
  across a resume boundary aren't comparable. Weights restore correctly.
- `_fill_by_id` in `rules.py` tolerates missing optional TOML keys; `from_dict`
  derives dtypes from `_allocate()` so adding a state array can't silently break
  serialization (it did once).

## Watching games

```bash
.venv\Scripts\python.exe scripts/record_replay.py --net checkpoints_11x11/best.pt \
    --p0 net --p1 scripted --size 11 --open
```

Writes a **self-contained HTML** viewer — no server. It has a built-in key
explaining every glyph, a click-anywhere tile inspector, per-player tech trees,
and a fog selector that re-renders the board as one player actually saw it.
Replays are gitignored (an 1,800-frame 11×11 game is ~8 MB); regenerate them.
