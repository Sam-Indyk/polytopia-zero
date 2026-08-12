# Training runs 2–7 — what each one established

Runs and checkpoints are gitignored, so this is the only surviving record. Written
12 Aug 2026, at the end of the session that produced runs 5, 6 and 7, immediately
before moving to different hardware. **Nothing in `checkpoints_run*/` or
`runs_run*.log` will exist in the next session** — the numbers below are all that
is left of them.

Read this with `CLAUDE.md` (rules, invariants, engine state) and `PLAN.md`
(milestones). This file only covers *training*: what was run, what it measured,
and which explanations survived contact with the data.

---

## The scoreboard

`Scripted` is ScriptedBot at **crazy** over 32 games — the M5 target and the most
informative column. `wr vs own snapshots` is the learner's win rate against
members of its own league pool, which is the single best "is it still improving?"
signal: a policy that is climbing beats its past selves comfortably, one that is
merely drifting sits near 50%.

| run | net | distribution | best vs Scripted | final | wr vs own snapshots (1st half → 2nd) |
|---|---|---|---|---|---|
| 2 | 96×8 | 11×11, 4 map types | — | — | 56.4% → 54.9% |
| 3 | 96×8 | staged | 18/32 @525 | 6/32 @1575 | 77.5% → 65.9% |
| **4** | 96×8 | **staged, narrow first** | **22/32 @550** | 13/32 @1275 | **79.5% → 64.5%** |
| 5 | 96×8 | 6 map types + 12 tribes from it 1 | 7/32 @50 | 4/32 @675 | 61.1% → 55.4% |
| 6 | 96×8 | same, exploration annealed | 11/32 @150 | 6/32 @575 | 59.7% → 54.5% |
| 7 | 96×8 | gradual widening (new) | — | it 40 when the box died | — |

**Run 4 is the high-water mark of the project: 22/32 (69%) against a Crazy bot at
eval@550.** Nothing since has come close, and run 7 — the run that should — had
managed 40 iterations when the machine bugchecked.

---

## The central finding: widening the world all at once erases it

This cost two full runs to establish and is the most important thing here.

Run 4 trained on **drylands + lakes only** through iteration ~780, then its
curriculum widened to all six map types at a single stage boundary:

| run 4 | it 600–799 (2 map types) | it 800–999 (6 map types) |
|---|---|---|
| vs Scripted | **19.8/32** | 10.2/32 |
| Elo | **+125.6** | −58.6 |
| wr vs own snapshots | 76% | 59% |

It never recovered over the next 500 iterations. Widening did not *add* five
regimes to the one it had — it **overwrote** the one it had.

Runs 5 and 6 then started wide, on all six map types and twelve tribes from
iteration 1, and never learned at all: ~55% against their own snapshots,
falling, for 600+ iterations each.

### Why the "map type is free" argument was wrong

The change was made on this reasoning: `PolicyEncoder.encode` pads a batch to
`max(sizes)`, so **board size** costs throughput; map type and tribe do not
change tensor shape, therefore staging them buys nothing.

That conflated **computational** cost with **statistical** cost. Map type is free
in throughput and expensive in sample efficiency — six regimes on one budget is a
sixth of the data each (7,680 steps/iteration ÷ 6 ≈ 1,280 per regime, against
3,840 when narrow). At 1.79M parameters the gradients *interfere* rather than
accumulate: waterworld wants Ports and embarking, drylands wants Riders and
roads, and a trunk that size trades one for the other.

The fix, now in `polytrain/curriculum.py`: `standard` and `ffa` add **one map
type per stage**, ordered by how much water they put on the board (pangea → 
continents → archipelago → waterworld), and `standard` gained stages so that
**one axis moves per boundary** — a new map type never lands together with a
size or seat change. When run 4 broke, four axes moved at once, which is exactly
why it took two more runs to work out which. Pinned by
`test_standard_moves_one_axis_per_boundary`.

---

## Explanations that were tested and did *not* survive

Recording these matters as much as the finding, because each looked convincing.

### "Exploration collapsed, so the agent never discovers tech"

**Partly true, not the cause.** The type head had genuinely collapsed — at run 4
iteration 1000, RESEARCH held a *median* probability of 0.0007 and EXAMINE 0.007,
while the aggregate entropy floor was satisfied by hedging between MOVE, ATTACK
and TRAIN. A scalar measure of spread cannot distinguish that from healthy
exploration, which is why a per-**action** floor was added alongside the
per-head entropy floor.

The intervention worked on its own terms, dramatically:

| probe (waterworld self-play) | run 4 @1000 | run 5 @100 |
|---|---|---|
| median P(RESEARCH) | 0.0007 | **0.0159** (23×) |
| mean P(EXAMINE) | 0.007 | **0.199** (28×) |
| techs held at game end | 6.8 | **14.8** |
| type-head entropy | 0.24 | **0.56** |

And run 5 at iteration 100 beat run 4's `snap@100` head-to-head **29–19**. But
none of it converted: run 5 flatlined for the next 575 iterations and lost
**11–37** to run 4's `snap@650`. Run 6 isolated the variable by annealing the
exploration pressure and failed identically. **Exploration was never the binding
constraint on strength.**

### "The intrinsic bonus was permanent, so it changed the objective"

**Real bug, correctly diagnosed, still not the cause.** Making `TypeNovelty`'s
counts decay (to fix "an abandoned verb keeps its historical count forever")
destroyed the guarantee that makes count-based exploration safe: cumulative
counts give `β/√N → 0`, so the intrinsic reward vanishes and the optimal policy
is asymptotically the true one. Decaying counts reach a *steady state*, so the
bonus is permanent.

The algebra says what a permanent bonus buys. Expected bonus per step is

```
Σᵢ sᵢ · β/√(sᵢ·T)  =  β/√T · Σᵢ√sᵢ        T = 1/(1−decay)
```

which is **maximised when the action-type shares are uniform** — a direct payment
for playing a random mix of verbs. Measured: **+0.58 per episode against a
terminal reward of ±1**, versus +0.03 under cumulative counts.

Fixed by annealing all three exploration pressures (novelty β, action floor,
the raised entropy floor) to **exactly zero** over 40% of the run via
`explore_scale`. Unlike `shaping_for`, none of them may keep a floor — the
shaping term's floor is only defensible because a potential-based term provably
cannot change the optimal policy.

But **run 6 applied this fix and still did not learn**, which is what redirected
the investigation to the curriculum. `TypeNovelty.per_episode_scale()` now exists
so this class of failure is visible: it appeared in no log line and was obvious
the moment the number was computed.

### "The critic is broken"

**False, and worth knowing.** Measured explained variance of `V(s)` against
returns:

| | explained variance | mean \|advantage\| |
|---|---|---|
| run 6 (6 map types) | **+0.841** | 0.195 |
| run 4 (narrow, snap@650) | **+0.884** | 0.172 |

The value function predicts 84% of return variance even on the wide
distribution. The learning signal was fine; the policy could not act on it. This
rules out "more parameters won't help because the advantages are noise".

**`value_loss` is computed in `ppo.py` but never logged** — worth adding, along
with explained variance, since nobody had looked at it in seven runs.

---

## Sizing: what 100M parameters would take

Measured on CPU (parameter counts exact; VRAM extrapolated from the 160×12
anchor in `CLAUDE.md`):

| channels × blocks | params | weights + Adam + grads | activations @ mb 1024 |
|---|---|---|---|
| 96×8 (all runs here) | 1.8M | 0.03 GB | ~3 GB (measured) |
| 128×10 | 3.6M | 0.05 GB | ~4.7 GB (measured) |
| 160×12 | 6.4M | 0.10 GB | ~6.8 GB (measured) |
| 256×16 | 20.5M | 0.30 GB | — |
| 384×20 | 56.1M | 0.84 GB | — |
| **512×20** | **99.0M** | **1.48 GB** | **~36 GB (est.)** |
| 640×20 | 154.0M | 2.30 GB | — |

**512×20 is the ~100M config.** Model state is comfortable at 1.5 GB; the
constraint is activations, which scale with channels × blocks × minibatch. On an
8.1 GB card it needs the minibatch cut from 1024 to roughly **128**.

Speed is the harder problem. Update time scales with parameters (6.97s at 6.4M →
~110s at 99M), and the measured "rollout time is flat across net sizes" finding
**does not extrapolate** — it held up to 6.4M because the GPU forward was
negligible beside CPU env stepping, which stops being true at 99M. Estimate
~2 min/iteration against ~19s today, so a 4,000-iteration run is **5–6 days of
continuous GPU load**.

### Two constraints, not one

1. **Capacity** — can the trunk represent six regimes at once? Fixed by parameters.
2. **Samples per regime** — fixed only by more parallel games.

99M parameters against 7,680 steps/iteration is a very high parameter-to-data
ratio (AlphaZero: ~46M against millions of games). Scaling parameters without
scaling data is the standard way to get a large model that learns *worse*. The
prerequisite is throughput, and the blocker is documented in `CLAUDE.md`:
`train_league.py` is single-process and uses **one core of sixteen**.
`polytrain/parallel.py` puts a full policy copy in every worker, which measures
*worse* for large nets (660 vs 2,660 steps/s). The architecture that scales is
workers doing env/encode/mask over shared memory with one process batching GPU
inference.

**Important caveat on capacity:** nothing measured here shows 1.79M was the
binding constraint. Run 4 was beating a Crazy bot 19.8/32 and climbing at 76%
against its own snapshots when the curriculum broke it. Capacity is plausibly the
*next* ceiling; it was not this one.

---

## Hardware: this machine is not a reliable training host

Five bugchecks, all with the **byte-identical** signature already documented in
`CLAUDE.md`:

```
HYPERVISOR_ERROR 0x00020001 (0x28, 0x1, 0x29b92701, 0xfc801000)
```

Most recent: 12 Aug 2026 13:54, while starting a 160×12 (6.4M) run — 6.8 GB peak
on an 8.1 GB card. Time-to-crash across all five ranges from **9 minutes to
2h43m**, so only the signature is deterministic, not the timing. Two nights of
training were lost to reboots during this session.

`scripts/Register-Overnight.ps1` registers a logon task that closes the
crash → reboot → logon → `--auto-resume` loop. **It was never registered**, which
is why both nights were lost rather than recovered. Register it on any machine
that shows this fault (and remove it when the run ends, or training starts on
every login).

---

## Checkpoint inventory at time of writing (all now lost)

| dir | best | latest | league pool |
|---|---|---|---|
| `checkpoints_run2` | it 1925 | it 3095 | 20 snapshots, elo −630..+849 |
| `checkpoints_run3` | it 1950 | it 1985 | 20 snapshots, elo −615..+162 |
| **`checkpoints_run4`** | **it 1225** | it 1295 | 20 snapshots, elo −526..+68 |
| `checkpoints_run5` | it 300 | it 695 | 20 snapshots, elo −456..−232 |
| `checkpoints_run6` | it 100 | it 580 | 20 snapshots, elo −480..−154 |
| `checkpoints_run7` | it 25 | it 40 | 1 snapshot |

Run 4's `snap@650` was the strongest policy the project produced and was used as
the benchmark opponent for the head-to-heads above. It is gone. **Runs 8+ start
from scratch**, and the first milestone to re-reach is *run 4's eval@550: 22/32
against Crazy*.

Note that `best.pt` is selected on the fixed gate (`crazy@11x11-fixed`, six map
types), so run 4's `best.pt` at iteration 1225 is a *post-collapse* checkpoint
and was **not** its strongest policy on the narrow distribution.

---

## Where run 7 was going, and how to judge it

Run 7 is the first run on the gradual curriculum. Config:

```bash
scripts/train_overnight.ps1 -Out checkpoints_run7 -Curriculum standard `
  -Channels 96 -Blocks 8 -Games 48 -Rollout 160 -Iters 4000 -MaxTurns 0 `
  -Log runs_run7.log
```

Stage schedule at 4,000 iterations (`with_horizon` spreads stages evenly, so
stage length is `iters / n_stages` and the presets' `until` values are
decorative):

| iteration | world |
|---|---|
| 1 | 11×11 · 2 map types · 2 players · 2 tribes |
| 500 | + all 12 tribes |
| 1000 | + pangea (3 map types) |
| 1500 | + continents (4) |
| 2000 | + sizes 11/11/14, seats 2/2/3 |
| 2500 | + archipelago (5) |
| 3000 | + waterworld (6) |
| 3500 | + sizes 11/14/16, seats 2/3/4 |

**How to tell if it is working, in order of usefulness:**

1. **wr vs own snapshots** should sit at 70–86% while climbing, as run 4 did.
   Below ~60% means drift, not progress.
2. **vs Scripted** should reach ~20/32 by iteration ~550 to match run 4.
3. At each map-type boundary, watch for the run-4 collapse signature. Because one
   axis moves per boundary now, a drop is *attributable*.

**Do not compare eval lines across runs.** This misled the session twice: run 4's
eval@100 covered two land map types and run 5's covered six, so "OSLA 28 vs 22"
was a different exam, not a regression. The only valid cross-run comparison is a
**head-to-head at matched iteration count on a fixed map set** — the script
pattern is in the session history and is worth rebuilding as
`scripts/head_to_head.py`.

Equally: the per-map-type buckets on the eval line are **4–6 games each**. A
`waterworld 0/4` is noise. This was over-read once here and generated an entire
false diagnosis.

---

## Recommended order of work on the new machine

1. **Parallel rollout** (`polytrain/parallel.py` rewrite). The prerequisite for
   everything else, and it needs no GPU — buildable anywhere.
2. **Re-run run 7's config** to confirm the gradual curriculum reaches run 4's
   22/32 and then keeps going through the first map-type boundaries.
3. **Then scale the net**, with samples scaled alongside it. 256×16 (20.5M) is
   the sane next step from 1.79M; 512×20 (99M) only once throughput supports it.
4. Log `value_loss` and explained variance per iteration.
5. Register the overnight task if the new box shows any instability.
