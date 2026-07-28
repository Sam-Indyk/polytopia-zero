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

### `core/economy.py` — the population chain

```
resources + buildings → population → city level → unit capacity
                                                ↘ +1 star income per level
```

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

`enumerate_actions` is the ground truth for legality. `masks.py` must agree with
it exactly (invariant 2 in `CLAUDE.md`).

### `kernels/` — Numba

`@njit(cache=True)` functions over the flat arrays: the combat formula, fog
recomputation, movement Dijkstra, attack-target scans. Pure functions of their
array arguments — no engine imports — so they compile fast and are testable in
isolation. The pure-Python paths they replaced are the reference behaviour.

---

## The observation and action interface

### `obs.py` — 64 planes + a 60-value global vector

| planes | contents |
|---|---|
| 5 / 7 / 11 | terrain, resource, improvement (one-hot) |
| 4 | territory: own / enemy / neutral / **off-board** |
| 9 | city: present, own, enemy, level, pop, walls, capital, workshop, support-used |
| 16 + 6 | unit type one-hot, then own/enemy/hp/veteran/moved/acted |
| 3 | fog: visible, explored, staleness |
| 3 | geometry: x, y, distance to edge |

Everything is **relative to the acting player** (own vs enemy slots), so
self-play stays symmetric. Unit and city planes come from `mem_*`, and HP/status
only appear for *currently visible* tiles. `pad_to` bottom-right pads to a fixed
size for size-bucketed batching, with the off-board plane marking the padding.

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

1. **Unit and property tests** — 247 of them, including hypothesis properties.
2. **Golden tests** — the combat formula pinned against wiki-documented cases.
3. **Wiki reconciliation** (`scripts/check_ruleset.py`) — every unit stat, tech
   tier/prerequisite, and improvement cost diffed against cached wiki pages.
   164 checks. This found six real errors on first run, including two entirely
   missing technologies.
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
- **Map generation isn't faithful** to Polytopia's (see `PLAN.md`, deferred M4).
- **Naval, cloak/diplomacy, special tribes**: stats and tech gating are correct
  data, mechanics are not implemented. Those units are flagged non-trainable.
- **Replay files are large** — full board state per frame. Delta-encoding the
  tile arrays would cut them ~10–50×.
