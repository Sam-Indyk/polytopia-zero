# PolyZero — orientation for the next session

A from-scratch engine for *The Battle of Polytopia* plus a PPO self-play agent.
Goal (PLAN.md): beat the game's built-in Crazy bots through the real Steam client.

**Read this file, then `docs/ARCHITECTURE.md` if you're changing engine internals.**
`PLAN.md` holds the milestone schedule and the long-term design.

---

## Environment

```bash
.venv\Scripts\python.exe -m pytest            # 270 tests, ~4 min
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
   checked by `scripts/check_ruleset.py` against cached wiki pages (169 checks,
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

Several rules were wrong for a while and each distorted training, so treat "the
agent is doing something weird" as a rules bug until proven otherwise:
- Capture was a side effect of moving. It is its own action: a unit must **start
  its turn** on the city, and capturing ends that unit's turn.
- Cities had no **unit-support cap**. A city supports `level + 1` units. Without
  it, star income converted straight into army and the agent flooded the map
  with 75 warriors while ignoring the tech tree.
- **City defence was wrong in three ways at once** (fixed 28 Jul). A city
  fortifies *only its owner's* units, and only those with the **`fortify`
  skill** — a skill that was present in `units.toml` but read by no code at all.
  A besieging enemy used to inherit the defender's walls, which made a captured
  city *harder* to retake than open ground. Unwalled cities gave nothing when
  they should give 1.5x. And the wall multiplier was **1.5 instead of 4**.
- **Parks produced no income**, though they are a level-up reward the agent can
  choose — so it was being taught that a good reward was a bad one.
- **Siege did not exist.** A city with an enemy unit standing on it produces no
  stars. Without this the agent could neither apply nor break the central
  pressure mechanic of the game.

The last three all survived 164 wiki checks because **nothing was checking
`combat.toml`'s multipliers**. `wiki_check.check_combat` now parses the wiki's
damage-formula sentence and pins them; anything the combat maths reads from data
belongs in that check.

## Current state

Engine is complete for base-tribe land play: movement, combat, capture, tech,
and the full population economy (harvests, buildings, city levels, level-up
reward choices), plus **siege** (an enemy unit on a city zeroes its income).

**Not simulated yet**, in roughly the order they are worth doing — all verified
missing against the cached wiki, all CPU-only work:
- **Roads / Bridges** (Roads, T2) and **city connections**: each connection to
  the capital adds **+1 population to both** the capital and the connected city.
  Population is the only route to city levels, so this is a compounding economy
  the agent simply cannot access.
- **Market** (Trade, T3): 2 stars/turn per *level* of adjacent Windmill, Sawmill
  or Forge, max level 8. All three partner buildings already exist.
- **Naval embarking**: the unit data is there (`naval = true`, `carry`) but
  boarding is not. `check_ruleset.py` reports 4 hulls with placeholder HP until
  this lands.
- Cloak infiltration and diplomacy, special tribes.

The viewer lists these so an empty log never reads as "nothing happened".

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
iterations saw maybe 5,000 win/loss signals.

**A run addressing this is in flight** (28 Jul, `checkpoints_big/`, gitignored):
`128x10 = 3.44M params` (6× the old net), `--games 64 --rollout 192` on 11×11 —
12,288 steps/iteration, 4× the old run, and ~21 episodes/iteration instead of
6–9. Measured 25.5s/iteration (7.2s rollout + 18.3s update), 5.5 GB of 8.5 GB
peak VRAM. Watch it with `Get-Content runs_big.log -Tail 20`.

The sizing measurement that drove those numbers, and which should be redone
before any future scale-up:

| net | params | rollout | update | peak VRAM |
|---|---|---|---|---|
| 64×6 (old) | 0.58M | 1.87s | 2.23s | 1.8 GB |
| 96×8 | 1.57M | 2.07s | 3.07s | 3.0 GB |
| **128×10** | **3.37M** | **2.09s** | **4.39s** | **4.7 GB** |
| 160×12 | 6.17M | 2.10s | 6.97s | 6.8 GB |

The headline: **rollout time is flat across net sizes** — it is CPU-bound, so a
bigger network is nearly free during collection and costs only in the PPO update.
That is why growing the net was cheap and why item 1 below is still the ceiling.

1. **Parallel rollout** is now the binding constraint (see below).
2. **A bigger network still** — 160×12 fits, but only pays off with more
   parallel environments; otherwise the update dominates.

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
- **THIS MACHINE BUGCHECKS UNDER SUSTAINED GPU LOAD.** Four crashes on
  27–28 Jul, every one with *byte-identical* bugcheck parameters:
  `HYPERVISOR_ERROR 0x00020001 (0x28, 0x1, 0x29b92701, 0xfc801000)`, each
  alongside a WHEA **fatal hardware error** (Event ID 1) and corrected
  **PCI Express AER** errors naming the GPU (`PCI\VEN_10DE&DEV_2DB8`, the
  RTX PRO 1000, Dell subsystem `1028`). It presents as HYPERVISOR_ERROR rather
  than the usual `0x124` because VBS is enabled. This is **not a bug in this
  repo** — our workload is just a reliable stress test. Time-to-crash is
  variable (2h43m once, 9 minutes another time), so only the *signature* is
  deterministic, not the timing.

  **Ruled out — do not re-investigate these:**
  - *Wake from sleep.* Two crashes followed a Modern Standby exit within 15
    min, which looked compelling, but a later crash happened with no standby
    transition for 18 minutes beforehand. Correlation was coincidence.
  - *PCIe ASPM.* Already `Off` on AC before any of the crashes.
  - *GPU power limit.* Not settable on this laptop part (`nvidia-smi` reports
    `N/A`; default 50 W, max 65 W).

  **Still untried, in order of likelihood:** NVIDIA driver update/rollback
  (was 32.0.15.9608, 30 Mar 2026), Dell BIOS update, warranty claim. Note the
  PCIe link negotiates **x4 width on a Gen5 link**, which is worth raising with
  Dell alongside the AER errors.

  *Because of this, long runs must be crash-resilient — see below.*
- **Elo used to reset on `--resume`.** Fixed: `League.state_dict()` /
  `load_state_dict()` persist the pool and Elo to `<out>/league.pt`, and
  `--resume` now also restores the Adam optimiser state and RNG. Ratings are
  comparable across a resume boundary again.
- `_fill_by_id` in `rules.py` tolerates missing optional TOML keys; `from_dict`
  derives dtypes from `_allocate()` so adding a state array can't silently break
  serialization (it did once).

## Long runs on a machine that crashes

Because of the bugcheck above, `train_league.py` is built to lose minutes, not
nights. Four mechanisms, all of them tested:

| flag / file | what it does |
|---|---|
| `--save-every N` (default 5) | writes `latest.pt` every N iterations, not just at eval |
| `--auto-resume` | picks up `<out>/latest.pt` if present; safe to pass on a fresh run |
| `checkpoint.save_atomic` | write-to-`.tmp` + `fsync` + `os.replace`, and rotates a `.bak`, so a crash mid-save cannot tear the file |
| `checkpoint.load_resumable` | falls back to `.bak` if the primary is truncated; returns `None` rather than dying |

`scripts/train_overnight.ps1` wraps all that in a restart loop, and
`scripts/Register-Overnight.ps1` registers a **logon** scheduled task so the loop
survives the reboot that follows a bugcheck:

```
crash -> reboot -> logon -> task -> supervisor -> --auto-resume
```

Two power settings matter and are already applied (`powercfg`, AC): idle
standby and hibernate are both **never**. A 10-minute idle sleep once cost a run
68 minutes mid-training. The supervisor additionally holds `ES_SYSTEM_REQUIRED`
for its lifetime, so the machine stays awake even if a Windows update resets the
power plan.

**Remove the task when the run is done**, or training starts on every login:

```bash
powershell -ExecutionPolicy Bypass -File scripts/Register-Overnight.ps1 -Remove
```

Checkpoint promotion now uses a **held-out** gate: with `--league-scripted`,
ScriptedBot is a training opponent, so `best.pt` is selected on the OSLA win rate
instead (`blob["gate_name"]` records which). Selecting on an opponent you train
against measures sparring, not strength.

## Watching games

```bash
.venv\Scripts\python.exe scripts/record_replay.py --net checkpoints_11x11/best.pt \
    --p0 net --p1 scripted --size 11 --open
```

Writes a **self-contained HTML** viewer — no server. It has a built-in key
explaining every glyph, a click-anywhere tile inspector, per-player tech trees,
and a fog selector that re-renders the board as one player actually saw it.
Replays are gitignored (an 1,800-frame 11×11 game is ~8 MB); regenerate them.
