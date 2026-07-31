# Architecture

How the pieces fit, and *why* they're shaped this way. `CLAUDE.md` is the quick
orientation; this is the reference for changing internals. `PLAN.md` is the
long-term design and milestone schedule.

---

## The shape of the system

```
polyengine  ── rules, state, legality ──┐
                                        ├─→ obs.py / masks.py ─→ polynet ─→ polytrain
polyengine/data/*.toml ── the ruleset ──┘        (tensors)      (policy)     (PPO)
                                        └─→ replay.py ─→ viewer.py (watch it)
```

Three properties drive nearly every design decision:

1. **`GameState.copy()` must be cheap.** It powers MCTS baselines, the OSLA
   agent, and any future search. Hence flat NumPy arrays and no Python objects
   per tile or unit.
2. **The observation contract is frozen early** and shared with the future
   screen-reading agent (M5). That constrains what the encoder may look at.
3. **Rules live in data, not code**, so a balance patch is a TOML edit and a
   second ruleset (e.g. a 2020-rules variant for cross-checking) drops in behind
   the same API.

---

## polyengine

### `core/state.py` — GameState

Square `size × size` grid; tile index `y * size + x`. Polytopia allows one unit
per tile, so unit data is stored **per tile** rather than as slot lists. Every
field is a flat NumPy array, so `copy()` is a handful of memcpys and
`state_hash()` is deterministic.

Array groups (`_TILE_ARRAYS`, `_PLAYER_ARRAYS`, `_FOG_ARRAYS`) are declared once
and drive `copy`, `state_hash`, `to_dict`, and `from_dict` generically. **Adding
a state array means adding it to `_TILE_ARRAYS` and `_allocate()`, and nothing
else** — `from_dict` reads dtypes back off a freshly allocated instance rather
than a hand-maintained table. That table existed once and silently broke
round-trip hashing when new arrays appeared.

Notable fields:
- `unit_home` — tile of the city supporting each unit. This is what makes army
  size scarce (`level + 1` per city).
- `improvement` — buildings (harvests consume their resource and leave nothing).
- `mem_unit_type`, `mem_city_owner`, … — **per-player fog memory**: the last-seen
  snapshot of every explored tile, plus `fog_staleness`. The engine does the
  remembering, exactly as the real client does.

### `core/rules.py` — ruleset as data

Parses `data/*.toml` into id-indexed NumPy arrays. Ids in `constants.py` must
match the `id` fields in the TOMLs; the loader asserts they're dense. All five
files carry a `ruleset_version` and the loader refuses a mismatch.

`_fill_by_id` tolerates keys an entry omits (so optional fields like
`hull_hp_varies` don't require touching every row).

### `core/movement.py` + `kernels/grid_kernels.py` — half-moves

Costs are carried in **half-moves** (a plain step is 2, a road step is 1),
because roads cost 0.5 and the Dijkstra runs on integers. Three rules are
expressed as *"you may enter but not expand from"* nodes rather than as costs:

- rough terrain (forest, mountain) without a road on it — `creep` exempts a unit
  from all of it *except mountains*
- zone of control — any tile adjacent to an enemy unit, unless the mover has
  `hide` or `sneak`
- the source tile is always expandable, so a unit that begins its turn inside a
  zone of control can still leave

`hide` additionally lets a unit move *through* enemy units without stopping on
them, which the kernel gets by being told nothing blocks and having occupied
tiles filtered out of the destinations afterwards.

Passability is per **tile**, not per terrain, because a land unit may step onto
water if and only if a Port is there to carry it.

### `core/economy.py` — the population chain

```
resources + buildings → population → city level → unit capacity
        ↑                                       ↘ +1 star income per level
   city connections
```

**City connections** are the second source of population: a road/bridge/port
chain from the capital grants +1 population to *both* ends. Because a chain can
break, `city_conn_pop` records what each city was granted so
`recompute_connections` is idempotent and can hand population back —
`remove_population` de-levels a city when it drops below a threshold.

**Markets** are deliberately outside `city_income`: they belong to the tile, not
a city, and the wiki notes their income survives effects that zero a city's
output. A besieged city still earns from its Markets.

**Tasks** (`task_satisfied` / `update_tasks`) are standing conditions on the
whole empire, evaluated once per turn and latched into a per-player bitmask.
Latching matters: the conditions are *current* state ("have a level 5+ city"),
but a monument once earned is never withdrawn, so the bit must be sticky rather
than recomputed. `monument_built` is a second bitmask enforcing one monument
per task.

Cities never grow on their own. Population is bought: harvest fruit/game/fish
(2★ → 1 pop), lumber hut (3★ → 1), farm and mine (5★ → 2), port (7★ → 1), temple
(20★ → 1); sawmill/windmill/forge scale with adjacent partner buildings. Level
*n* costs *n* population and grants +1 income, +1 unit capacity, +50 score, and a
**choice of two rewards**.

That reward choice is a first-class action (`A_CHOOSE_REWARD`) and, like the real
game's popup, it *blocks every other action* until resolved — `enumerate_actions`
returns only reward choices while one is pending.

### `core/actions.py` — the action contract

One micro-action per RL step; a turn is a sequence ending in `END_TURN`. Four
fields — `(atype, src, dst, subtype)` — mapping onto the network's four heads.

| type | source | target | sub-type |
|---|---|---|---|
| `END_TURN` | – | – | – |
| `MOVE` | unit | tile | – |
| `ATTACK` | unit | tile | – |
| `CAPTURE` | unit | = src | – |
| `TRAIN` | city | – | unit id |
| `RESEARCH` | – | – | tech id |
| `BUILD` | tile | – | improvement id |
| `CHOOSE_REWARD` | city | – | reward id |
| `DISBAND` | unit | = src | – |
| `UPGRADE` | naval unit | – | hull unit id |
| `EXAMINE` | unit | = src | – |
| `PROMOTE` | unit | = src | – |
| `HEAL_OTHERS` | unit | = src | – |
| `TILE_ABILITY` | tile | – | ability id |

`ATTACK` is overloaded: for a unit with the `infiltrate` skill (the Cloak) it
targets an adjacent enemy *city* and resolves as infiltration. The wiki is
explicit that this is the Cloak's attack rather than a separate action, so it
costs no extra action type and no extra network head.

`EXAMINE` is shared the same way: a ruin and a starfish have one mechanic between
them — "a unit to start the turn on the ruin", consuming its actions, and
starfish are harvested "much like if it were examining a ruin" — so they are one
action type rather than two.

`enumerate_actions` is the ground truth for legality. `masks.py` must agree with
it exactly (invariant 2 in `CLAUDE.md`).

### `kernels/` — Numba

`@njit(cache=True)` functions over the flat arrays: the combat formula, fog
recomputation, movement Dijkstra, attack-target scans. Pure functions of their
array arguments — no engine imports — so they compile fast and are testable in
isolation. The pure-Python paths they replaced are the reference behaviour.

---

## The observation and action interface

### `obs.py` — 82 planes + a 60-value global vector

| planes | contents |
|---|---|
| 5 / 7 / 27 | terrain, resource, improvement (one-hot) |
| 4 | territory: own / enemy / neutral / **off-board** |
| 9 | city: present, own, enemy, level, pop, walls, capital, workshop, support-used |
| 17 + 6 | unit type one-hot, then own/enemy/hp/veteran/moved/acted |
| 1 | cloak_sense: an adjacent hidden Cloak, flagged on OUR unit |
| 1 | promote_ready: one of OUR units has an unclaimed veteran promotion |
| 3 | fog: visible, explored, staleness |
| 3 | geometry: x, y, distance to edge |

Everything is **relative to the acting player** (own vs enemy slots), so
self-play stays symmetric. Unit and city planes come from `mem_*`, and HP/status
only appear for *currently visible* tiles. `pad_to` bottom-right pads to a fixed
size for size-bucketed batching, with the off-board plane marking the padding.

The global vector is deliberately thin: own status, believed opponent status,
turn progress, map size. In particular it carries **no summary of the map type or
how wet the world is**. Training rotates map types and the policy is expected to
work out from the terrain planes that an archipelago wants Ports and boats while
drylands wants Riders and roads — the same inference a player makes from the
board. An aggregate "wetness" number would be a shortcut with no counterpart on a
real screen, so it is not offered. `tests/test_obs.py` pins this: two maps of the
same seed and size, one drylands and one water world, must produce byte-identical
global vectors.

### `masks.py` — four conditional heads

`legal_types → legal_sources(type) → legal_targets(type, src) → legal_subtypes(type, src)`.

`NEEDS_SOURCE / NEEDS_TARGET / NEEDS_SUBTYPE` say which heads apply to which
action type; heads that don't apply contribute exactly zero log-prob and zero
entropy, so the joint remains a proper distribution over legal actions.

`SUBTYPE_DEPENDS_ON_SOURCE` matters: `BUILD` and `CHOOSE_REWARD` sub-types depend
on *which* tile/city was picked, so `ActionMasks.subtypes()` caches on
`(atype, src)` for those and `(atype, -1)` otherwise.

`ActionMasks` caches per decision point. `legal_types()` goes *through* the cache
so `act()` doesn't recompute source masks it already built.

---

## polynet

`net.py`:
- **Trunk**: stem conv → N residual blocks (GroupNorm, squeeze-excitation),
  fully convolutional so one set of weights serves every map size. The global
  vector is injected by **FiLM** (per-channel scale/shift) at selected depths
  rather than being concatenated as constant planes.
- **Pointer heads**: source and target are 1×1 convs producing one logit per
  tile at native resolution — they transfer across map sizes. Target is
  conditioned on the chosen type *and* on the source tile's own feature vector.
- **Value head**: global-average-pool (masked, ignoring padding) + **PopArt**, so
  the reward scale can drift (shaping annealing, changing opponents) without
  destabilising the trunk.
- **Aux heads**: final-score regression and opponent-tech prediction at 0.25
  weight — free representation shaping, and the tech head doubles as a probe of
  what the net infers through fog.

`policy.py` bridges NumPy engine state to torch. Two entry points sharing one
implementation so acting and training can't disagree about the factorisation:
`act()` (sample) and `evaluate()` (re-score stored actions). A test asserts
`evaluate` reproduces `act`'s log-probs exactly.

**Performance note:** `act()` pulls sampled heads to host *once* per step.
Indexing a CUDA tensor per game (`int(atype[i])`) forces a device sync each time
and was the single largest cost in the rollout (29.7 ms → 10.5 ms when fixed).

---

## polytrain

### `rollout.py` — GAE across turn boundaries

The subtlety this file exists for: in a turn-based game a player's own decisions
are **not contiguous in time**. Each transition is tagged `(game, player)` and
advantages are chained along those per-player sequences, so the value target for
one of P0's decisions is P0's *next* decision, however many opponent actions
intervened.

Reward is terminal ±1, plus optional small potential-based shaping on the
material heuristic, annealed to zero early so the final objective is pure
win/loss.

With an `opponent` set, only the learner's seat is stored — the opponent shapes
the environment but its moves are not the learner's decisions.

### `league.py` — PFSP (M3)

Mirror self-play **cycles**: a 6×6 run hit 91.7% vs ScriptedBot at iteration 100
and fell to 47% by 150 (confirmed over 100 games — drift, not noise). A uniform
snapshot pool fixed most of it; prioritised fictitious self-play fixed the rest
by spending games where they carry signal:

- `p(1−p)` — contested opponents (maximal at 50%)
- `(1−p)²` — opponents currently beating the learner

Mixture per `PLAN.md`: 50% self / 35% even / 10% hard / 5% scripted. Win rates
**decay**, so a snapshot the learner used to beat becomes eligible again as the
policy drifts. Eviction preserves Elo *spread*, not recency — a pool of recent
snapshots is a pool of near-identical opponents.

Snapshots, Elo and decayed win rates round-trip through `League.state_dict()` to
`<out>/league.pt`. Without that, every crash-restart silently dropped the pool
back to pure mirror self-play — which is exactly the cycling failure the league
exists to prevent.

### `evaluate.py` — measurement protocol

**Mirrored seed pairs** (same map, both seats) cancel spawn luck. Win rates come
with standard errors. SPRT (p₀=0.50 vs p₁=0.55, α=β=0.05) decides checkpoint
promotion without fixed-N waste.

This matters more than it sounds: a 16-game eval has ±12% standard error, which
was enough to hide a real regression until it was re-measured at 100 games.

### `profiles.py` — dev box vs workstation

Named hardware configurations (`--profile smoke|laptop|workstation|auto`).
Profiles never override a flag passed explicitly. The docstring explains why
standard LLM scaling advice (FSDP/ZeRO sharding, Transformer Engine, DataLoader
workers) doesn't transfer to this workload — the model is 24 MB, not 70 B params,
and there is no dataset because every sample is generated by self-play.

---

## Verification strategy

Four layers, weakest to strongest:

1. **Unit and property tests** — 675 of them, including hypothesis properties.
2. **Golden tests** — the combat formula pinned against wiki-documented cases.
3. **Wiki reconciliation** (`scripts/check_ruleset.py`) — every unit stat, tech
   tier/prerequisite, and improvement cost diffed against cached wiki pages.
   237 checks. This found six real errors on first run, including two entirely
   missing technologies, and later caught eleven unit skills that were present
   in the data and read by no code.
4. **Differential replay** (`polyengine/testing/differential.py`) — re-simulates
   a recorded action sequence and reports the first divergent frame *and field*.
   Today it guards determinism; pointed at a captured Steam game it becomes the
   fidelity gate that settles any remaining rules question.

Layer 4 is the one that matters most long-term and is the least exercised: it
has no real Steam captures yet.

---

## Known limitations

- **Rollout is single-process** (`train_league.py`), using ~1 of 16 cores. This is
  now the binding constraint: rollout wall-time is essentially independent of net
  size (see the sizing table in `CLAUDE.md`), because it is CPU-bound.
- **The multiprocess collector doesn't scale to large nets** — full policy copy
  per worker; measured table in `polytrain/parallel.py`.
- **The dev box bugchecks under sustained GPU load** — a hardware fault, not a
  code bug (WHEA fatal error + PCIe AER on the NVIDIA link). Long runs therefore
  go through `scripts/train_overnight.ps1`, and checkpointing is atomic with
  `.bak` fallback. Details and mitigations in `CLAUDE.md`.
- **Map generation** follows the wiki's algorithm, in both of its pipelines —
  grown landmasses shaped per map type, per-tribe biomes by randomised flood
  fill, exact terrain proportions, the two-tile resource rule, quadrant capitals
  (except on Continents and Pangea, where capitals are promoted from villages),
  all six sizes, ruins, starfish, island villages and the suburb / pre-terrain /
  post-terrain village passes. `mapgen.py`'s module docstring is the map to it.
  Still simplified: a ruin never hides a resource under it (the wiki allows that;
  we do not place the two on one tile, which avoids leaking the hidden resource
  through the observation), and the "weighted randomness" for quadrant capitals
  is a uniform draw among the furthest tiles.
- **Special tribes** are excluded by decision. Their stats and tech gating are
  correct data; the mechanics are not implemented and the units are flagged
  non-trainable.
- **Replay files are large** — full board state per frame. Delta-encoding the
  tile arrays would cut them ~10–50×.
