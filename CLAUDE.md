# PolyZero — orientation for the next session

A from-scratch engine for *The Battle of Polytopia* plus a PPO self-play agent.
Goal (PLAN.md): beat the game's built-in Crazy bots through the real Steam client.

**Read this file, then `docs/ARCHITECTURE.md` if you're changing engine internals.**
`PLAN.md` holds the milestone schedule and the long-term design.

---

## Environment

```bash
.venv\Scripts\python.exe -m pytest            # 491 tests, ~7 min
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
   checked by `scripts/check_ruleset.py` against cached wiki pages (177 checks,
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

- **Movement was approximated.** Mountains carried `move_cost = 2` as a stand-in
  for "halts movement", which both overcharged them and let units walk straight
  *through* forests. Rough terrain is now enterable-but-not-traversable, and
  **zone of control** — a unit that steps beside an enemy stops dead — did not
  exist at all despite shaping every tactical exchange in the real game.

The defence bugs survived 164 wiki checks because **nothing was checking
`combat.toml`'s multipliers**. `wiki_check.check_combat` now parses the wiki's
damage-formula sentence and pins them; anything the combat maths reads from data
belongs in that check.

The wiki contradicts itself on the **Market rate** — `Resources` and `Trade` say
2 stars per adjacent building level, `Buildings` says 1. **1 is correct**,
confirmed by the project owner. `check_ruleset.py` reports the disagreement as a
note; do not "fix" it back to 2.

## Current state

Engine is complete for base-tribe land play: movement, combat, capture, tech,
and the full population economy (harvests, buildings, city levels, level-up
reward choices), **siege**, the **roads/trade economy**, and **naval embarking**.

Movement now implements the real rules rather than approximations: costs run in
**half-moves** because a road step costs 0.5, rough terrain (forest, mountain) is
*enterable but not traversable*, and **zone of control** stops a unit that steps
next to an enemy (`creep` bypasses it). A Rider gets exactly the wiki's 2 -> 4
tiles on roads.

Also implemented: **healing** (4 HP in friendly territory, 2 elsewhere, applied
automatically to any unit that took no action — "if a unit does not take any
action in a given turn, it will automatically heal"), **Disband** (Free Spirit,
refunds half cost), **naval hull upgrades** (Raft -> Scout/Rammer/Bomber; landing
drops the upgrade), **Diplomacy** (Embassy on a rival capital paying both tribes,
plus Capital Vision), and the **Cloak** — see below.

Terrain defence is tech-gated as the wiki describes: forest needs Archery,
mountain Climbing, and shallow water / ocean Aquatism.

**Tasks and monuments** are in: all seven non-tribe tasks (Pacifist, Wealth,
Explorer, Killer, Network, Metropolis, Genius) are checked once a turn and
*latched* — a monument once earned is not taken back when the condition lapses.
Each monument is free to place, yields 3 population and 400 points, and only one
may be built per task. **Pirates** too: a Dagger on a Port becomes one and
reverts to a Dagger on landing.

**Map generation follows the real algorithm** (wiki: Map Generation), in its
order — capitals, villages, terrain, resources, lighthouses:

- **All six sizes**: tiny 11, small 14, normal 16, large 18, huge 20, massive 30.
  `size=None` picks what Domination would for the opponent count (1 -> 11,
  2 -> 14, 3 -> 16, 4+ -> 18).
- **Six water amounts** via `map_type`: drylands 0-10% ... waterworld 90-100%.
  Until this landed the generator produced **no water at all**, which silently
  made Ports, embarking, Bridges, Pirates and Aquatism unreachable. On a
  `continents` map the Port is now the most-built improvement.
- **Terrain is per tribe.** The map is partitioned between capitals and each
  region generated at its owner's rates. Base (Luxidoor) is 48% field / 38%
  forest / 14% mountain, and tribe multipliers apply in the documented order:
  mountain first (rescaling forest and field), then forest (rescaling field),
  then field is the remainder.
- **Resources only spawn within two tiles of a city**, and a tile *adjacent* to
  one is ~3x richer than one two away (18% fruit vs 6%). Expansion is how you
  reach resources at all.
- **Capitals go one per quadrant** and never within two tiles of the border;
  quadrants alone were not enough, since two capitals either side of a shared
  border landed 2 tiles apart, so each is now placed furthest from those already
  down. Villages keep 2 tiles from the edge and 2 from each other.
  **Continents and Pangea are the two types the wiki exempts from quadrants**:
  there capitals are drawn uniformly from the tiles at least `size // 4` (never
  fewer than 6) from every capital already placed. Taking the *furthest* tile
  there, as the quadrant path does, silently reimposes quadrants — with two
  players it returns opposite corners every time.
- **Which village passes run depends on the map type**, per the wiki's feature
  table: suburbs (up to two beside each capital) on Lakes and Archipelago,
  pre-terrain villages on those plus Water World at the documented density
  coefficient, post-terrain villages everywhere, and **tiny island villages** —
  villages with no land neighbour at all — on Continents and Pangea.
- **Ruins** (4/5/7/9/11/23 by size) spawn on any terrain but shallow water, never
  beside a capital or another ruin, and **starfish** at one per 25 water tiles,
  never beside a city, lighthouse or another starfish. Both are examined by the
  same action — see below.

All **12 regular tribes** are in, with their real starting tech, unit and stars
(Luxidoor has no starting tech and a level-3 capital instead). The spawn
multipliers are load-bearing, not decoration: Bardur land grows **zero** crop,
Vengir is metal-rich and fruit-barren, Xin-xi's 1.5x mountain and 1.5x metal
compound.

Still simplified, and deliberately: Continents uses smoothed noise rather than
real seeded landmass growth, Pangea is not centre-seeded, capitals are not
preferentially coastal, and a ruin never shares a tile with a resource (the wiki
allows it, but the resource is meant to be *invisible* until the ruin is cleared,
and we have no way to hide it from the observation).

## EXAMINE: ruins and starfish are one action

`A_EXAMINE` covers both, because the game gives them one mechanic: "Examining a
ruin requires a unit to **start the turn** on the ruin and consumes the unit's
actions for that turn (like capturing a village)", and a starfish is harvested
"much like if it were examining a ruin". Starfish need **Navigation** and pay 8
stars. Ruin rewards are drawn uniformly from those whose condition holds:

| reward | effect | condition |
|---|---|---|
| Resources | 10 stars | always |
| Scrolls of Wisdom | a free tech | something is researchable |
| Population | 3 population to the best capital | you own a capital |
| Explorer | reveals the map around the ruin | a tile in the 5x5 box is dark |
| New Friends / Rammer | veteran Swordsman on land, veteran Rammer carrying a Warrior on water | by terrain |

Two of those were silently unreachable when first written, which is the kind of
bug that shows up as "the agent ignores ruins" rather than as a crash:

- **Explorer was never in the pool at all.** It is listed in `RUIN_REWARDS` and
  was simply never appended.
- **The unit reward could never once be paid.** A unit must be standing on the
  ruin to examine it, so the spawn tile was always occupied and the code fell
  back to stars — every single time. The wiki's own trivia has the answer: "If
  the ruin provides a unit as a reward, the examining unit is **pushed** in a
  forced unit spawn." `push_unit` in `actions.py` does that.

## Forced unit spawns have a push order

Two things must appear on a tile that is usually already occupied: a ruin's unit
reward (the examiner is standing on it by definition) and a city's **super-unit**
reward (you garrison the city you are levelling). Both go through
`push_unit`, and the direction is not arbitrary:

> prioritise the direction the unit last moved, and back up to the closest
> direction to that if something is already there; a unit that has never moved
> is pushed towards the centre of the map.

So `state.unit_facing` records the compass index of each unit's last move — net
displacement, since the engine keeps no path — and `push_order` walks outwards
from it: facing, +1, −1, +2, −2, … with the direction it came from tried last.
Clockwise wins each tie; that part is our choice, not an observed rule.

The Giant used to be **silently dropped** if the city tile was taken, which made
the strongest reward in the game a trap: you picked it and got nothing, with
nothing in the log to say why. An *enemy* on the city is still not ours to shove,
so a besieged city leaves the reward unpaid.

**Not simulated yet:** special tribes only (excluded by choice). The viewer lists
this so an empty log never reads as "nothing happened".

**There is no trained agent right now, by design.** Every checkpoint predating
28 Jul is architecturally dead: the observation went 64 -> **79** planes and the
action head 8 -> **11** (Disband, Upgrade, Examine), so nothing will load. They
were deleted rather than left in the repo to tempt someone. Training starts from
scratch, and the previous numbers (55% vs ScriptedBot at 11x11) are not
comparable to anything measured against the current rules.

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

## Training across map types

Rollouts and evals both **rotate map types** — `--map-types` on
`train_league.py`, default `drylands,lakes,continents,archipelago`, picked per
game by seed in `rollout._make_game` and per pair in `evaluate.play_match`. An
agent that only learned dry maps now shows it in the eval line instead of looking
fine until someone puts it on water.

**The map type is never told to the network.** It reads water tile by tile out of
the terrain planes, through fog, and has to conclude for itself that this world
wants Ports and boats rather than Riders and roads. There was briefly a "wetness"
scalar in the global vector; it is gone, because it is a shortcut M5's
screen-reading agent could not reproduce and it makes the strategy switch a
lookup instead of an inference. `tests/test_obs.py` pins the global vector as
identical between a drylands and a water-world map of the same seed.

Deferred by explicit decision: the remaining **map-generation fidelity** gaps
(Continents' smoothed noise instead of seeded landmass growth, Pangea not
centre-seeded, capitals not preferentially coastal). Fine while maps are small;
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

## What "winning" means

One definition, used everywhere:

> destroy every opponent, or lead on **score** when the turn cap hits.

`Game.score` is Polytopia's published formula (wiki Score page): 5 points per
star of unit cost (super units 50), 20 per territory tile, 5 per explored tile,
100 per city plus 50 per level above 1, 250 per Park, 400 per monument, 100 per
temple, 100 per **tier** of each technology. Stockpiled stars score **nothing**.
The wiki supplies its own anchor — "level 1 city + 9 tiles territory = 280
points" — which `tests/test_score_and_bot.py` pins.

`Game.winner()` and `rollout._score_winner` both use it. They used to be two
different functions and disagreed on **15%** of turn-capped games; that is now
0%, measured the same way.

`agents/heuristic.py` is the *static* eval for baselines and PPO shaping. It is
the score lead over the strongest opponent, plus a **Domination tilt** whose
weights come from the game's own Domination rating rather than from taste:

    battle skills     (enemy killed + 1) / (friendly lost + enemy killed + 1)
    tribes destroyed  eliminated / total opponents (with a continuous
                      "pressure" term so a search has a slope to climb)
    speed skills      100% at 10 turns per enemy

`DOMINATION_WEIGHT = 200` caps the whole tilt at about two cities: enough to
break an even position toward the aggressive line, not enough to buy one kill
with an army. Battle skill being a *ratio* is the point — a raw kill count would
call trading two units for one progress.

## Stealth: how the Cloak actually works

Getting this wrong is easy and the failure is silent, so the rules in full:

- A Cloak is **visible when trained**, and hides the moment it *moves*. It is
  revealed again by capturing a village (and, unimplemented, exploring a ruin or
  breaking ice).
- An adjacent enemy unit gets an **eye symbol on itself** — not the Cloak's
  position. The wiki's illustration is three neighbouring units each showing an
  eye; the player triangulates. The observation carries this as the
  `cloak_sense` plane, flagged on OUR units.
- Walking into a hidden Cloak **reveals it and cancels the move, without
  spending the action**. So the move must be *legal to attempt*, and the mover
  may then attack what it found.

The trap: it is not enough to scrub the Cloak from the observation. Every rule
the enemy's move generation touches has to be blind to it too. A hidden Cloak
used to block its tile and project zone of control, which carved a five-tile
hole in the enemy's reachable set — and the policy samples from that mask, so it
could read the Cloak's exact position out of its own action space with the
observation never mentioning it. `hidden_from()` / `occupied_mask()` in
`movement.py` exist for this, and `attack_targets` filters the same way.
`tests/test_cloak_and_actions.py` asserts the enemy's legal moves are
byte-identical with and without a hidden Cloak nearby.

## Bot difficulty

`ScriptedBot(difficulty=...)` follows what the wiki says separates Midjiwan's
bots: aggression and income, not a different program. Easy/normal/hard/crazy
receive 1/2/3/5 stars per turn, so `income_bonus` carries the surplus over the 2
a level-1 capital already makes; `ScriptedBot.crazy()` is the M5 target. Call
`bot.apply_handicap(game, seat)` once before play, or the handicap is not
applied. Measured over 24 mirrored games: normal beats Random 100%, crazy beats
normal 58%, hard beats easy 67%.

## Watching games

```bash
.venv\Scripts\python.exe scripts/record_replay.py --net checkpoints_11x11/best.pt \
    --p0 net --p1 scripted --size 11 --map-type continents --open
```

Writes a **self-contained HTML** viewer — no server. It has a built-in key
explaining every glyph, a click-anywhere tile inspector, per-player tech trees,
and a fog selector that re-renders the board as one player actually saw it.
Replays are gitignored (an 1,800-frame 11×11 game is ~8 MB); regenerate them.
