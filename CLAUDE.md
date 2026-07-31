# PolyZero — orientation for the next session

A from-scratch engine for *The Battle of Polytopia* plus a PPO self-play agent.
Goal (PLAN.md): beat the game's built-in Crazy bots through the real Steam client.

**Read this file, then `docs/ARCHITECTURE.md` if you're changing engine internals.**
`PLAN.md` holds the milestone schedule and the long-term design.

---

## Environment

```bash
.venv\Scripts\python.exe -m pytest            # 675 tests, ~9 min
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
   checked by `scripts/check_ruleset.py` against cached wiki pages (237 checks,
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
next to an enemy (`hide` bypasses it — see the skill table below). A Rider gets exactly the wiki's 2 -> 4
tiles on roads.

Also implemented: **healing** (4 HP in friendly territory, 2 elsewhere, applied
automatically to any unit that took no action — "if a unit does not take any
action in a given turn, it will automatically heal"), **Disband** (Free Spirit,
refunds half cost), **naval hull upgrades** (Raft -> Scout/Rammer/Bomber; landing
drops the upgrade), **Diplomacy** (Embassy on a rival capital paying both tribes,
plus Capital Vision), and the **Cloak** — see below.

Terrain defence is tech-gated as the wiki describes: forest needs Archery,
mountain Climbing, and shallow water / ocean Aquatism.

## Terrain does four things, not one

Easy to forget, because only the first is obvious:

1. **Movement cost / rough terrain**, and mountains need Climbing to enter.
2. **Defence bonus**, tech-gated as above.
3. **Sight.** "Mountain: all units on this terrain have a sight radius of two
   tiles." Taking the high ground is worth a turn, and `rules.terrain_sight`
   feeds the fog kernel alongside `unit_sight` (a unit takes the better of the
   two, so a Scout on a mountain still sees 5x5, not 7x7).
4. **Hidden resources.** "Crop is hidden until Organization or Farming has been
   researched"; Climbing "reveals metal". This is an **invariant 1** rule, not
   decoration — the encoder was showing crop and metal the acting player could
   not see, so the policy could learn to expand toward resources that are not on
   its screen and M5's screen reader could never reproduce the tensor. `obs.py`
   scrubs the plane *and* sets the `R_NONE` plane, because an all-zero column
   would itself mark the tile as interesting. Ground truth is untouched;
   `C.HIDDEN_RESOURCE_TECHS` is the table.

Two movement rules from the Movement page that are equally easy to miss:

- **Costs round up.** "Even if you have 0.5 movement points remaining, you can
  still move onto a tile costing 1 or even 3." So any movement left is enough to
  enter one more tile; the unit simply arrives spent.
- **Roads do not work in enemy territory.** "This only applies to Roads in
  friendly territory or in neutral territory." An invader gets no use of the
  defender's network — which is what stops a Rider crossing an empire in a turn,
  and why roads at home are an asset rather than a liability. `road_like_mask`
  takes the moving player for this reason; cities are road nodes and obey the
  same rule.

## Unit skills are the rules, and they were nearly all missing

**`.wikicache/Unit Skills.wiki` is the reference. Read it before touching
combat.** Eleven skills sat in `units.toml` and were read by *no code at all*
(31 Jul) — a failure with no symptom, because the game still runs and the agent
still trains; the Catapult is simply a better unit than the real one.

| skill | who | what it does |
|---|---|---|
| `dash` | warrior, archer, rider, swordsman, knight, cloak, dagger, scout, rammer, pirate | may attack *after moving* |
| `escape` | rider | may move *after attacking* — but not attack again if it also has dash |
| `persist` | knight | attacks again immediately after a kill, no limit |
| `stiff` | catapult, mind bender, cloak, raft, bomber, juggernaut | does not retaliate when attacked |
| `surprise` | dagger, pirate | its attacks draw no retaliation |
| `splash` | bomber | damages enemies adjacent to the target |
| `stomp` | juggernaut | damages every adjacent enemy when it moves |
| `convert` | mind bender | attacking takes the unit instead of hurting it |
| `heal` | mind bender | Heal Others: +4 HP to every adjacent friendly |
| `scout` | cloak, scout | sees 5x5 instead of 3x3 |
| `independent` | dagger, pirate | costs no city support slot |
| `static` | giant, cloak, dagger, all hulls | can never become a veteran |
| `fortify` | warrior, archer, rider, defender, knight | gets the city defence bonus |
| `creep` | cloak | ignores rough terrain **except mountains** |
| `hide` | cloak | invisible after moving; ignores ZoC and walks *through* enemies |
| `carry` / `water` | hulls | derived into `unit_naval` / `unit_hull_hp_varies` |

Two of those needed the wiki to settle a contradiction:

- **`dash` was inverted.** Every unit could attack after moving, which is the
  skill's *absence* being the special case. A Catapult must now choose between
  repositioning and firing.
- **`creep` was wrong twice over.** The Movement page says creep bypasses zone
  of control; the Unit Skills page says it ignores terrain "except mountains"
  and that ZoC "cannot be negated". The skill's own history decides it: Sneak
  originally did terrain, was removed, Creep replaced it, and Sneak returned as
  the ZoC skill. The Cloak's ZoC immunity comes from **`hide`**, which also lets
  it move *through* enemy units.

`carry` and `water` used to be duplicated as separate `naval` / `hull_hp_varies`
TOML flags; they are now derived from the skill list, because two sources of
truth for one rule is exactly how a skill ends up read by nobody.
`tests/test_unit_skills.py` ends with a guard that fails if any skill in the data
is not mentioned anywhere in `polyengine/`, and `wiki_check.check_unit_skills`
diffs the whole skill-to-unit assignment against the wiki's table.

## Veterans: a promotion is a free heal you can bank

"A unit can be promoted to a veteran after it kills three enemy units (through
attacks or retaliation). Promoting a unit increases its maximum health by five
and then heals the unit completely."

It is **an action, not an automatic consequence**, and it **costs the unit
nothing** — it does not spend the move or the attack. That is what makes the
standard play work, confirmed by the project owner: leave a unit unpromoted,
let it take damage, then promote to get a full heal *and still fight this turn*.
Promoting at full health throws the heal away. `A_PROMOTE`, `can_promote`, and a
`promote_ready` observation plane so the policy can see a banked promotion the
way the client's star icon shows it.

Before this, `unit_kills` was counted and read by nothing, nothing ever
promoted, and `effective_max_hp` never added the +5 — so the veteran flag was
inert, including on the veteran Swordsman a ruin hands out.

## Tile abilities and the four temples

`A_TILE_ABILITY` carries four sub-types, each behind its own technology, all
acting on land you already own:

| ability | tech | effect |
|---|---|---|
| Clear Forest | Forestry | forest -> field, **+1 star** |
| Grow Forest | Spiritualism | field -> forest, 5 stars |
| Burn Forest | Construction | *empty* forest -> field with crop, 5 stars |
| Destroy | Chivalry | removes a building **and the population it paid**, free |

There are **four temples**, not one: Temple (field, Free Spirit, 20),
**Mountain** (Meditation, 20), **Forest** (Spiritualism, **15**), **Water**
(Aquatism, 20). Until these landed, Spiritualism unlocked *nothing whatsoever*
and Meditation only the Pacifist task. A temple is worth 100 points when built
and gains 50 every three turns to a maximum of 300 — the page's prose says "every
two turns" but its own table gives three-turn bands, reconciled by "including
the turn of their construction". `state.improvement_turn` records when each
building went down; `economy.temple_score` reads it.

The **Explorer** (level-2 reward, and a ruin reward) is now the walking figure
the game has: it takes 15 steps towards the nearest fog, revealing as it goes,
and cannot cross mountain, water or ocean without Climbing, Sailing or
Navigation. It used to be an instant radius-4 blob, which is a different shape —
a real Explorer strings out along the frontier.

**Tasks and monuments** are in: all seven non-tribe tasks (Pacifist, Wealth,
Explorer, Killer, Network, Metropolis, Genius) are checked once a turn and
*latched* — a monument once earned is not taken back when the condition lapses.
Each monument is free to place, yields 3 population and 400 points, and only one
may be built per task. **Pirates** too: a Dagger on a Port becomes one and
reverts to a Dagger on landing.

**Map generation follows the real algorithm** (wiki: Map Generation). Read that
page and `polyengine/core/mapgen.py`'s module docstring together; the second is a
transcription of the first. A second source is worth knowing about:
**QuasiStellar/Polytopia-Map-Generator** (`app.py`) is a community port of the
pre-Moonrise generator, and where it overlaps the wiki it agrees, so it settles
the things the wiki only describes in prose — the exact 1/3 outer-city falloff,
the randomised flood fill that hands each tribe its land, the guaranteed
resources beside certain capitals.

**The game has two pipelines and so do we**, which is the single most important
thing to know before touching this file:

    drylands / lakes / archipelago / waterworld
        capitals (quadrants) -> suburbs -> pre-terrain villages -> terrain and
        water -> post-terrain villages -> resources -> ruins, starfish

    continents / pangea
        land -> villages -> *convert* villages to capitals -> terrain ->
        island villages -> resources -> ruins, starfish

"Villages are added after all the land is made, that's why Continents and Pangea
don't use quadrants for capital placement." On those two the capitals are chosen
out of the villages already on the ground: furthest apart, preferring the coast,
one per landmass. Measured, that gives coastal capitals ~90% of the time and
capitals on different continents 100% of the time.

- **All six sizes**: tiny 11, small 14, normal 16, large 18, huge 20, massive 30.
  `size=None` picks what Domination would for the opponent count (1 -> 11,
  2 -> 14, 3 -> 16, 4+ -> 18).
- **Landmasses are grown, not thresholded.** `_grow_blobs` seeds N points and
  accretes tiles one at a time, never letting two blobs touch, so "continents
  ... must be at least one tile apart" holds by construction. Each map type is
  then a choice of how many blobs and how big: one large mass per player on
  continents (30–200 tiles each), a single centre-seeded mass on pangea, many
  small ones on archipelago, islets on waterworld — and for drylands and lakes
  it is *water* that is grown, as lakes inside the land, with the border ring
  forced to land ("a border of land bridges will always generate on the edges").

  The generator used to threshold smoothed noise instead, and the result was
  *speckle*: a "continents" map came out as a lace of one-tile isthmuses you
  could walk across, so on the one map type built around ocean, Ports, Bridges
  and boats had nothing to do. `tests/test_mapgen.py` now asserts you genuinely
  cannot walk from one capital to the other on continents.
- **Water depth is an edge rule.** "Ocean ... does not share any *edges* with
  land tiles" — corners do not count, so a tile diagonally off a headland is
  still deep ocean. Getting this wrong shrinks the ocean and quietly changes
  what Aquatism and hull movement are worth.
- **Terrain is per tribe, in exact proportions.** The land is partitioned by a
  randomised flood fill from the capitals (each tribe taking turns, which is
  self-balancing and gives ragged natural borders), and each region is then
  dealt its terrain by *count*, not by per-tile sampling: "since update 2.0.58,
  terrain will always be generated in accordance with the percentage rates".
  Base (Luxidoor) is 48% field / 38% forest / 14% mountain, and tribe
  multipliers apply in the documented order: mountain first (rescaling forest
  and field), then forest (rescaling field), then field is the remainder.
  The result is stored per tile as **`state.tile_tribe`** — the biome is real
  map state, fixed at generation, and quite separate from `owner`, which changes
  hands all game. The viewer tints it and names it in the tile inspector.
- **Resources only spawn within two tiles of a city**, and a tile *adjacent* to
  one is ~3x richer than one two away (18% fruit vs 6%). Expansion is how you
  reach resources at all. **Starfish are the documented exception** — they
  "spawn on any water tiles, and are scattered across the map", which is what
  makes sailing out for one worth the turn.
- **Some tribes are promised a start.** Kickoo gets two water tiles with a fish
  in each *even on drylands*; Imperius two fruit, Bardur two game, Zebasi one
  crop, all adjacent to the capital. Without this a Bardur start on an unlucky
  map has nothing to hunt, which is its entire early economy.
- **Tribe water multipliers are deliberately NOT applied.** Kickoo's "2.0x
  water" and Aquarion's "1.5x" are documented as bugged in the real game — "this
  effect is bugged and does not currently apply to map generation" — and
  Oumaji's 0.5x is "found in the source code but ... not translated into the
  world generation". `tribes.toml` keeps the numbers; `mapgen.py` does not read
  them. Do not "fix" this.
- **Capitals go one per quadrant** on the four types the wiki says use quadrants,
  never within two tiles of the border, and on land if the quadrant has any.
  Quadrants alone were not enough, since two capitals either side of a shared
  border landed 2 tiles apart, so each is placed furthest from those already
  down. Villages keep 2 tiles from the edge (1 for the pre-terrain pass) and 2
  from each other.
- **A map only seats so many players.** "Tiny maps support a maximum of nine
  players, while other maps support up to 16" — `max_players_for_size`, enforced
  by `new_game`, which refuses rather than dealing a map it cannot fill. Below
  tiny (sizes the game never uses but our smoke runs do) the cap scales by area
  at the official rate of ~13 tiles per player, so a 6x6 seats two.
  `_ensure_capitals` is the backstop underneath that: both placement routes can
  still come up short on a cramped map, and a player without a capital has no
  city and no unit — dead before the first move, game terminal at turn 0. That
  is not hypothetical; it showed up as a training smoke run whose rollouts
  collapsed from 1024 steps to 64.
- **Capitals share a continent when the map is crowded.** "Capitals are placed on
  different land masses, *space permitting*" — with two players on a normal map
  they always get their own, but four cannot each have a 30-tile continent, and
  8 of 10 seeds put at least two of them on the same one. The "you must sail to
  reach the enemy" assertion in `tests/test_mapgen.py` is therefore scoped to
  two players on purpose. Do not generalise it.
- **Which village passes run depends on the map type**, per the wiki's feature
  table, transcribed as `SUBURB_TYPES` / `PRE_TERRAIN_COEFFICIENT` /
  `ISLAND_VILLAGE_TYPES` / `QUADRANT_TYPES` and pinned by a test: suburbs (up to
  two beside each capital) on Lakes and Archipelago, pre-terrain villages on
  those plus Water World at the documented density coefficient, post-terrain
  villages everywhere, and **tiny island villages** — villages with no land
  neighbour at all — on Continents, Pangea *and Water World*.
- **Ruins** (4/5/7/9/11/23 by size) spawn on any terrain but shallow water, never
  beside a capital or another ruin, with at most a third of them on deep ocean;
  **starfish** at one per 25 water tiles, never beside a city, lighthouse or
  another starfish. Both are examined by the same action — see below.

All **12 regular tribes** are in, with their real starting tech, unit and stars
(Luxidoor has no starting tech and a level-3 capital instead). The spawn
multipliers are load-bearing, not decoration: Bardur land grows **zero** crop,
Vengir is metal-rich and fruit-barren, Xin-xi's 1.5x mountain and 1.5x metal
compound.

Still simplified, and deliberately: a ruin never shares a tile with a resource
(the wiki allows it, but the resource is meant to be *invisible* until the ruin
is cleared, and we have no way to hide it from the observation), and the exact
"weighted randomness" the wiki mentions for quadrant capitals is a uniform draw
among the furthest tiles.

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

## Forced unit spawns, the un-siege, and the Houdini

Read `.wikicache/Forced Unit Spawns.wiki` before touching this; it specifies the
whole mechanic and every clause below is load-bearing.

Two things must appear on a tile that is usually already occupied: a ruin's unit
reward (the examiner is standing on it by definition) and a city's **super-unit**
reward. Both go through `push_unit`, and `state.unit_facing` — the compass index
of each unit's last move, by net displacement since the engine keeps no path —
decides where the occupant goes:

| the occupant | pushed |
|---|---|
| friendly, has moved | the way it was going |
| **enemy, has moved** | **the opposite way — back where it came from** |
| ranged, has attacked | the direction it last shot |
| never moved | towards the centre of the map |
| standing on the exact centre | south |

then, if that tile is blocked, "counterclockwise and then clockwise one tile at
a time" — so facing, −1, +1, −2, +2, … with the reverse direction tried last.

**The enemy reversal is the un-siege.** "The forced spawn of super units are
commonly used in besieged cities to push out the invader": the besieger walked
*onto* your city, so its recorded direction sends it straight back out and the
siege breaks without a shot. Taking the super-unit reward on a besieged city is
a standard play, and the engine has to offer it.

**If there is nowhere to go, the unit is destroyed** — "it will be removed from
the game entirely". The community calls this a **Houdini**: ring your besieged
city with eight units, level up, take the Giant, and the enemy Giant standing on
the city simply vanishes. It is the cheapest way in the game to delete a super
unit.

Crucially it is *not* scored as combat: "this does not count as a kill for the
Gate of Power, nor does it count as an attack for the Altar of Peace." So a
Houdini moves neither `kills` nor `losses` and a Pacifist streak survives it. Get
that wrong and the agent learns to farm monuments with a mechanic the real game
does not credit.

The Giant used to be **silently dropped** if the city tile was taken, which made
the strongest reward in the game a trap: you picked it and got nothing, with
nothing in the log to say why.

**Not simulated yet:** special tribes only (excluded by choice). The viewer lists
this so an empty log never reads as "nothing happened".

**There is no trained agent right now, by design.** Every checkpoint predating
28 Jul is architecturally dead: the observation went 64 -> **82** planes and the
action head 8 -> **14** (Disband, Upgrade, Examine, Promote, Heal Others, Tile
Ability), so nothing will load. They
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

The map-generation fidelity gaps that were deferred here are **closed** (30 Jul):
landmasses are grown rather than thresholded, Pangea is centre-seeded, and
capitals prefer the coast. See the map-generation section above.

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
