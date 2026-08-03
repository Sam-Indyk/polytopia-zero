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
from dataclasses import dataclass, field

from polyengine.core.mapgen import MAP_TYPES, max_players_for_size

ALL_MAP_TYPES = tuple(sorted(MAP_TYPES))
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
        return GameSpec(size=size, map_type=map_type, tribes=tribes)

    def describe(self, iteration: int) -> str:
        st = self.stage_for(iteration)
        return (f"{self.name}: size {'/'.join(map(str, st.sizes))} · "
                f"{len(st.map_types)} map types · "
                f"{'/'.join(map(str, st.players))} players · "
                f"{len(st.tribes)} tribes"
                + (" · mirror" if st.mirror else ""))


# --- presets ------------------------------------------------------------------
# `until` values are iteration indices, chosen so each stage gets a few hundred
# iterations at the measured ~25s/iteration of the 128x10 config.

PRESETS: dict[str, Curriculum] = {
    # What training did before any of this existed. Kept so a run can be
    # reproduced, and so "did the curriculum help?" has a control.
    "legacy": Curriculum(name="legacy", stages=(
        Stage(until=-1, sizes=(11,), players=(2,),
              map_types=("drylands", "lakes", "continents", "archipelago"),
              tribes=("imperius", "bardur")),
    )),

    # The default. Duels throughout, widening in board size and tribe variety.
    # Diplomacy stays inert here, which is honest: in a 1v1 a treaty is a draw
    # you both agreed to.
    "duel": Curriculum(name="duel", stages=(
        Stage(until=99, sizes=(9, 11), players=(2,),
              map_types=("drylands", "lakes"), tribes=("imperius", "bardur")),
        Stage(until=299, sizes=(11,), players=(2,)),
        Stage(until=699, sizes=(11, 14), players=(2,)),
        Stage(until=-1, sizes=(14, 16), players=(2,)),
    )),

    # Duels first, then crowds. Three players is the smallest game in which a
    # treaty can pay for itself, so that is where diplomacy starts to matter.
    "standard": Curriculum(name="standard", stages=(
        Stage(until=99, sizes=(9, 11), players=(2,),
              map_types=("drylands", "lakes"), tribes=("imperius", "bardur")),
        Stage(until=299, sizes=(11,), players=(2,)),
        Stage(until=599, sizes=(11, 14), players=(2, 3)),
        Stage(until=999, sizes=(14, 16), players=(2, 3, 4)),
        Stage(until=-1, sizes=(16, 18), players=(2, 3, 4, 6)),
    )),

    # The wide end: everything the engine can deal, up to sixteen tribes.
    # Expect slow, diffuse credit assignment — a win among sixteen is rare.
    "ffa": Curriculum(name="ffa", stages=(
        Stage(until=99, sizes=(9, 11), players=(2,),
              map_types=("drylands", "lakes"), tribes=("imperius", "bardur")),
        Stage(until=299, sizes=(11,), players=(2, 3)),
        Stage(until=599, sizes=(14,), players=(3, 4)),
        Stage(until=999, sizes=(16, 18), players=(4, 6, 8)),
        Stage(until=-1, sizes=(18, 20, 30), players=(4, 8, 12, 16)),
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
    stages = tuple(
        Stage(until=(-1 if i == n - 1 else (i + 1) * step - 1),
              sizes=st.sizes, map_types=st.map_types, players=st.players,
              tribes=st.tribes, mirror=st.mirror)
        for i, st in enumerate(cur.stages))
    return Curriculum(name=cur.name, stages=stages)
