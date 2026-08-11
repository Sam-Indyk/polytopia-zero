"""What kind of game to play at iteration N.

Training used to vary exactly one thing about the world — the map type — and
hold everything else fixed at an 11x11 duel between Imperius and Bardur. An
agent trained that way has never seen a large board, has never met ten of the
twelve tribes, and has never been in a game where diplomacy means anything,
because a treaty with your only opponent is just a slower way to draw.

This module owns the four axes and how they widen:

    size      how big the board is
    map type  drylands ... waterworld
    players   2 for a duel, up to 16 for a free-for-all
    tribes    which of the twelve are dealt

**Size is chosen per iteration, everything else per game.** That is not a
stylistic choice. `PolicyEncoder.encode` pads every board in a batch up to the
largest one in it (`pad = max(sizes)`), so a single 18x18 among 64 games makes
all 64 cost 18x18. Rotating size per game would put the largest size in nearly
every batch and quietly pay large-map compute for small-map games. Map type,
seat count and tribe do not change tensor shape, so those vary freely.

Why a curriculum and not just maximum variety from iteration 0: the diagnosed
cause of the last plateau was **sparse terminal reward**, and every axis you
widen makes each situation rarer. PLAN.md says the same thing in its own words
— "curriculum from tiny-map 1v1 mirror matches up to large maps, all tribes,
FFA". So the default ramps, and the wide end is opt-in.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from polyengine.core.mapgen import (MAP_SIZES, MAP_TYPES,
                                    max_players_for_size)

ALL_MAP_TYPES = tuple(sorted(MAP_TYPES))
# The only board sizes the real game deals: tiny 11, small 14, normal 16,
# large 18, huge 20, massive 30 (wiki, Map Generation). Earlier presets used 6
# and 9 because they were fast, which trains on boards Polytopia never
# produces -- and board size is not a free parameter, it decides how far apart
# capitals start and therefore the whole shape of the opening.
CANONICAL_SIZES = tuple(sorted(MAP_SIZES.values()))
# The twelve regular tribes. Special tribes modify core rules and are out of
# scope for the engine, so they are out of scope here too.
ALL_TRIBES = ("ai_mo", "bardur", "hoodrick", "imperius", "kickoo", "luxidoor",
              "oumaji", "quetzali", "vengir", "xin_xi", "yadakk", "zebasi")


@dataclass(frozen=True)
class GameSpec:
    """Everything `Game.new` needs. The turn cap is left to the engine, which
    scales it from size and seat count."""
    size: int
    map_type: str
    tribes: tuple[str, ...]
    turn_cap_scale: float = 1.0

    @property
    def n_players(self) -> int:
        return len(self.tribes)


@dataclass(frozen=True)
class Stage:
    """One band of iterations and the worlds it draws from."""
    until: int                       # last iteration in this stage; -1 = forever
    sizes: tuple[int, ...]
    map_types: tuple[str, ...] = ALL_MAP_TYPES
    players: tuple[int, ...] = (2,)
    tribes: tuple[str, ...] = ALL_TRIBES
    mirror: bool = False             # both seats get the same tribe
    # Multiplier on the engine's turn cap. The cap is a training device, so the
    # curriculum owns it. Early on a shorter cap recycles games quickly and
    # keeps terminal rewards arriving while the agent is too weak to finish
    # anything; later a generous one lets a real game play out rather than
    # being cut off mid-conquest. Episode starvation killed both previous runs
    # (0.5 episodes/iteration, 63% of iterations finishing nothing), so this is
    # the dial that matters most.
    turn_cap_scale: float = 1.0

    def __post_init__(self):
        for m in self.map_types:
            if m not in MAP_TYPES:
                raise ValueError(f"unknown map type {m!r}")
        for t in self.tribes:
            if t not in ALL_TRIBES:
                raise ValueError(f"unknown tribe {t!r}")
        if not self.sizes or not self.players:
            raise ValueError("a stage needs at least one size and player count")


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[Stage, ...]
    name: str = "custom"

    def stage_for(self, iteration: int) -> Stage:
        for st in self.stages:
            if st.until < 0 or iteration <= st.until:
                return st
        return self.stages[-1]

    def size_for(self, iteration: int) -> int:
        """One size per iteration — see the module docstring on batch padding."""
        st = self.stage_for(iteration)
        return st.sizes[iteration % len(st.sizes)]

    def spec(self, iteration: int, index: int, seed: int = 0) -> GameSpec:
        """The world for game `index` of iteration `iteration`."""
        st = self.stage_for(iteration)
        size = self.size_for(iteration)
        # Explicit integer mix rather than hash() on a tuple: reproducibility
        # here is a property of the run, not of the interpreter's hashing.
        rng = random.Random((seed * 1_000_003 + iteration) * 9176 + index)
        map_type = st.map_types[(iteration + index) % len(st.map_types)]

        # A board only seats so many; asking for more is an error in `new_game`
        # rather than something to discover at 3am in a training log.
        n = st.players[(iteration + index) % len(st.players)]
        n = max(2, min(n, max_players_for_size(size)))

        if st.mirror:
            one = rng.choice(st.tribes)
            tribes = tuple([one] * n)
        elif n <= len(st.tribes):
            tribes = tuple(rng.sample(st.tribes, n))
        else:                        # more seats than tribes: repeats allowed
            tribes = tuple(rng.choice(st.tribes) for _ in range(n))
        return GameSpec(size=size, map_type=map_type, tribes=tribes,
                        turn_cap_scale=st.turn_cap_scale)

    def describe(self, iteration: int) -> str:
        st = self.stage_for(iteration)
        return (f"{self.name}: size {'/'.join(map(str, st.sizes))} · "
                f"{len(st.map_types)} map types · "
                f"{'/'.join(map(str, st.players))} players · "
                f"{len(st.tribes)} tribes"
                + (" · mirror" if st.mirror else ""))


# --- presets ------------------------------------------------------------------
# Every stage is a MIXTURE, not a replacement. Stage 4 of the previous
# curriculum contained only 14x14 and 16x16 boards, and the measured
# consequence was fatal: **0.5 episodes per iteration, with 63% of iterations
# finishing no game at all** across 48 games in flight. A learner cannot be
# taught by a reward it never receives, and strength peaked at iteration 1,925
# and then decayed for 1,100 iterations.
#
# So the later stages keep the small boards in the mix and merely shift weight
# toward the large ones. Sizes are drawn per iteration (see `size_for`), so a
# mixture means "this fraction of iterations", and the fast games keep terminal
# rewards flowing while the slow ones teach the hard positions.
#
# Sizes are all canonical. `weights` gives the relative frequency of each size.

PRESETS: dict[str, Curriculum] = {
    # What training did before any of this existed. Kept as a control.
    "legacy": Curriculum(name="legacy", stages=(
        Stage(until=-1, sizes=(11,), players=(2,),
              map_types=("drylands", "lakes", "continents", "archipelago"),
              tribes=("imperius", "bardur")),
    )),

    # Duels throughout, widening in board size and tribe variety. Tiny boards
    # never leave the mix.
    "duel": Curriculum(name="duel", stages=(
        Stage(until=199, sizes=(11,), players=(2,), turn_cap_scale=0.35,
              map_types=("drylands", "lakes"), tribes=("imperius", "bardur")),
        Stage(until=599, sizes=(11,), players=(2,), turn_cap_scale=0.5),
        Stage(until=1199, sizes=(11, 11, 14), players=(2,), turn_cap_scale=0.7),
        Stage(until=-1, sizes=(11, 14, 14, 16), players=(2,)),
    )),

    # The default. Duels first, then crowds -- three seats is the smallest game
    # in which a treaty can pay for itself. Small boards stay in every stage.
    # Only the axes with a *mechanical* reason are staged. Board size is one:
    # `PolicyEncoder.encode` pads a batch to `max(sizes)`, so one 18x18 among 48
    # games makes all 48 cost 18x18. Seat count is another, for a different
    # reason -- more players means longer games means fewer terminal rewards per
    # iteration, and episode starvation has killed two runs here.
    #
    # Map type and tribe have neither. Map type does not change the tensor shape
    # at all, and the spatial planes are binary own/enemy so sixteen tribes
    # encode exactly like two. They were staged by analogy with size, and run 4
    # showed what that cost: stage one was `drylands, lakes` -- both land -- and
    # it ran 780 iterations. The type head collapsed onto its floor at iteration
    # 100 and warrior-rush set in, so when water finally arrived the policy had
    # no exploration left to adapt with.
    #
    # The premise of a curriculum is that the easy task's optimum is a stepping
    # stone to the hard one. Warrior-rush-on-drylands is not a stepping stone to
    # naval play, it is a competing attractor -- and there was nothing to bridge
    # anyway, since the agent beats Random 32/32 on every map type from early on.
    "standard": Curriculum(name="standard", stages=(
        Stage(until=199, sizes=(11,), players=(2,), turn_cap_scale=0.35),
        Stage(until=599, sizes=(11,), players=(2,), turn_cap_scale=0.5),
        Stage(until=1199, sizes=(11, 11, 14), players=(2, 2, 3), turn_cap_scale=0.7),
        Stage(until=1999, sizes=(11, 14, 14, 16), players=(2, 3, 3, 4), turn_cap_scale=0.85),
        Stage(until=-1, sizes=(11, 14, 16, 16, 18), players=(2, 3, 4, 4, 6)),
    )),

    # The wide end: up to sixteen tribes on a massive board. Even here a fifth
    # of iterations are 11x11 duels, purely to keep episodes finishing.
    "ffa": Curriculum(name="ffa", stages=(
        Stage(until=199, sizes=(11,), players=(2,), turn_cap_scale=0.35),
        Stage(until=599, sizes=(11,), players=(2, 2, 3), turn_cap_scale=0.5),
        Stage(until=1199, sizes=(11, 11, 14), players=(2, 3, 3, 4), turn_cap_scale=0.7),
        Stage(until=1999, sizes=(11, 14, 14, 16), players=(2, 3, 4, 4, 6), turn_cap_scale=0.85),
        Stage(until=-1, sizes=(11, 14, 16, 18, 20), players=(2, 4, 6, 8, 12, 16)),
    )),
}


def parse(spec: str) -> Curriculum:
    """A preset name, or an inline `size:players:maptypes` stage list.

        standard
        11:2 , 14:2/3 , 16:2/3/4          -> three equal-length stages
        11:2:drylands/lakes               -> map types pinned for that stage

    Inline stages split the run evenly between them, which is almost always
    what you want when trying a shape out rather than tuning one.
    """
    spec = spec.strip()
    if spec in PRESETS:
        return PRESETS[spec]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty curriculum {spec!r}")
    stages = []
    for i, part in enumerate(parts):
        bits = part.split(":")
        sizes = tuple(int(x) for x in bits[0].split("/"))
        players = tuple(int(x) for x in bits[1].split("/")) if len(bits) > 1 else (2,)
        maps = tuple(bits[2].split("/")) if len(bits) > 2 else ALL_MAP_TYPES
        stages.append(Stage(until=-1, sizes=sizes, players=players, map_types=maps))
    return Curriculum(name=spec, stages=tuple(stages))


def with_horizon(cur: Curriculum, iters: int) -> Curriculum:
    """Spread a curriculum's stages evenly across `iters` iterations.

    A preset's `until` values assume a long run; a 200-iteration experiment
    using them would never leave stage one and would silently be a duel-only
    run wearing the name "ffa". This rescales instead."""
    n = len(cur.stages)
    if n == 1:
        return cur
    step = max(1, iters // n)
    # `replace` rather than re-listing the fields: the hand-written version
    # silently dropped `turn_cap_scale` the moment it was added, so every
    # rescaled curriculum quietly ran at the full cap. Copying by construction
    # means a new Stage field can never be lost here again.
    stages = tuple(
        replace(st, until=(-1 if i == n - 1 else (i + 1) * step - 1))
        for i, st in enumerate(cur.stages))
    return Curriculum(name=cur.name, stages=stages)
