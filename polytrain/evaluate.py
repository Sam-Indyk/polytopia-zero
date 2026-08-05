"""Evaluation protocol: mirrored seed pairs, anchored Elo, and SPRT promotion.

Two problems with naive win-rate evaluation, both of which bit the M2 runs:

* **Spawn luck.** A 16-game eval had ~±12% standard error, enough that a real
  regression (91.7% → 47%) was indistinguishable from noise until re-measured
  with 100 games. Playing *mirrored seed pairs* — the same map twice with the
  seats swapped — cancels most of the map-quality variance, so a given number of
  games buys a much tighter estimate.

* **When to promote.** Fixed-N testing either wastes games on obvious cases or
  under-powers close ones. SPRT (sequential probability ratio test) stops as soon
  as the evidence is decisive, which is what PLAN.md specifies for checkpoint
  promotion: p=0.5 vs p=0.55 at alpha=beta=0.05.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from polyengine.agents.heuristic import evaluate as material
from polyengine.core.game import Game


def play_pair(agent_a, agent_b, seed: int, size: int, max_steps: int = 800,
              map_type: str = "drylands", tribes=None,
              n_players: int = 2) -> tuple[float, ...]:
    """Play one map once per seat, rotating A through all of them.

    In a duel this is the mirrored pair it always was: the same map twice with
    the seats swapped, which cancels terrain and spawn luck so a given number of
    games buys a much tighter estimate. With more seats the same idea generalises
    — A takes each seat in turn and B fills the rest — so a lucky corner of the
    map is credited to every agent equally rather than to whoever drew it.

    Returns A's score in each game, so the caller can average however it likes.
    """
    scores = []
    for a_seat in range(n_players):
        game = Game.new(size=size, seed=seed, map_type=map_type,
                        tribes=list(tribes) if tribes else None)
        seats = game.state.n_players
        if a_seat >= seats:
            break
        agents = {p: (agent_a if p == a_seat else agent_b) for p in range(seats)}
        steps = 0
        while not game.is_terminal() and steps < max_steps:
            game.step(agents[game.current_player].act(game))
            steps += 1
        if game.is_terminal():
            w = game.winner()
            scores.append(1.0 if w == a_seat else (0.5 if w < 0 else 0.0))
        else:
            # Unfinished: fall back to the static evaluation, which is a score
            # lead over the *strongest* opponent and so already means the right
            # thing with more than two tribes on the board.
            m = material(game, a_seat)
            scores.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))
    return tuple(scores)


@dataclass
class MatchResult:
    games: int
    score: float                # total points for A
    pairs: int
    # (axis, value) -> (points, games). The average is what gets logged; this is
    # what tells you *where* the agent is weak, which is the whole reason for
    # rotating size, map type and seat count in the first place.
    buckets: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.score / self.games if self.games else 0.0

    def breakdown(self, axis: str) -> list[tuple]:
        """[(value, win_rate, games)] for one axis, ordered by value."""
        rows = [(k[1], pts / max(1, n), n)
                for k, (pts, n) in self.buckets.items() if k[0] == axis]
        return sorted(rows, key=lambda r: str(r[0]))

    def format_breakdown(self) -> str:
        parts = []
        for axis in ("size", "players", "map"):
            rows = self.breakdown(axis)
            if len(rows) > 1:          # a single bucket says nothing new
                parts.append(axis + " " + " ".join(
                    f"{v}:{self._frac(wr * n)}/{n}" for v, wr, n in rows))
        return " | ".join(parts)

    @staticmethod
    def _frac(x: float) -> str:
        return f"{x:.1f}".rstrip("0").rstrip(".")

    @property
    def record(self) -> str:
        """Points out of games — "29.5/32" — which is what was actually played.

        This replaces a percentage with a binomial standard error, which was
        worse than no error bar at all. The binomial formula assumes each game
        is an independent draw; these are the *same fixed maps* every eval, so
        it describes the wrong quantity in both directions. Measured on five
        identical repeats (same weights, same maps): the Random match returned
        the same number all five times against a quoted +-3.1%, while Scripted
        swung 56.2%-75.0% against a quoted +-8.8%.

        A raw fraction cannot mislead like that. It shows the resolution (32
        games move in steps of 1/32), and it makes saturation obvious: 31/32
        says "one specific map is lost" rather than "96.9%, give or take".
        """
        pts = f"{self.score:.1f}".rstrip("0").rstrip(".")
        return f"{pts}/{self.games}"

    @property
    def elo_diff(self) -> float:
        p = min(max(self.win_rate, 1e-4), 1 - 1e-4)
        return -400.0 * math.log10(1.0 / p - 1.0)


DEFAULT_MAP_TYPES = ("drylands", "lakes", "continents", "archipelago")


def play_match(agent_a, agent_b, n_pairs: int, size: int, seed: int = 12345,
               max_steps: int = 800,
               map_types=DEFAULT_MAP_TYPES,
               sizes=None, players=(2,), tribes=None) -> MatchResult:
    """Mirrored-seed match: `n_pairs` maps, each played from every seat.

    Map types rotate across the pairs, so a win rate averages over the kinds of
    world the agent will actually meet rather than over drylands alone. An agent
    that only knows how to play dry maps now shows it here instead of looking
    fine right up to the point someone puts it on water.

    `sizes` and `players` rotate the same way. That matters because a single
    averaged win rate is exactly the wrong summary for an agent trained on a
    curriculum: "62%" can be 90% on the small duels it has seen a thousand times
    and 30% on the crowded boards it barely knows, and the average hides the one
    number you needed. `MatchResult.buckets` keeps the breakdown.
    """
    sizes = tuple(sizes) if sizes else (size,)
    players = tuple(players) if players else (2,)
    total, games = 0.0, 0
    buckets: dict[tuple, list[float]] = {}
    # Every (map, size, seats) combination, in a fixed shuffled order.
    #
    # Advancing each axis with `i % len(axis)` looked fine and was wrong: two
    # axes of the same length step together and never separate. On the first
    # real run `sizes=(11,14)` with `players=(2,3)` reported
    # "size 11:88% 14:38% | players 2:88% 3:38%" — the same numbers twice,
    # because 14 was only ever played with 3 seats. It read as two findings and
    # was one fact, with no way to tell the board from the crowd.
    #
    # Walking the product in order fixes the aliasing but biases any eval
    # shorter than the product, which is the normal case: 6 maps x 2 sizes x 4
    # seat counts is 48 combinations and `--eval-pairs` is 16. Shuffling once
    # with a fixed seed makes a short eval a *sample* across all three axes
    # rather than a prefix of the first one.
    combos = [(m, s, p) for m in map_types for s in sizes for p in players]
    random.Random(0xE7A1).shuffle(combos)
    for i in range(n_pairs):
        kind, sz, np_ = combos[i % len(combos)]
        pool = None
        if tribes:
            # Deal distinct tribes per pair, deterministically in the seed.
            rng = random.Random(seed * 1_000_003 + i)
            pool = (rng.sample(list(tribes), np_) if np_ <= len(tribes)
                    else [rng.choice(list(tribes)) for _ in range(np_)])
        scores = play_pair(agent_a, agent_b, seed + i, sz, max_steps,
                           map_type=kind, tribes=pool, n_players=np_)
        total += sum(scores)
        games += len(scores)
        buckets.setdefault(("size", sz), []).extend(scores)
        buckets.setdefault(("map", kind), []).extend(scores)
        buckets.setdefault(("players", np_), []).extend(scores)
    return MatchResult(games=games, score=total, pairs=n_pairs,
                       buckets={k: (sum(v), len(v)) for k, v in buckets.items()})


# --- SPRT ----------------------------------------------------------------------
@dataclass
class SPRTResult:
    decision: str               # "accept" | "reject" | "continue"
    llr: float
    lower: float
    upper: float
    games: int

    @property
    def decided(self) -> bool:
        return self.decision != "continue"


def sprt(wins: float, games: int, p0: float = 0.50, p1: float = 0.55,
         alpha: float = 0.05, beta: float = 0.05) -> SPRTResult:
    """Sequential test of H1(p=p1) against H0(p=p0) from a running score.

    Draws count as half a win, which is why `wins` is a float. Returns "accept"
    when the candidate is convincingly better, "reject" when it is not, and
    "continue" while the evidence is inconclusive.
    """
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    if games <= 0:
        return SPRTResult("continue", 0.0, lower, upper, 0)

    # Log-likelihood ratio under a binomial model with fractional successes.
    w = min(max(wins, 1e-9), games - 1e-9)
    losses = games - w
    llr = (w * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0)))

    if llr >= upper:
        decision = "accept"
    elif llr <= lower:
        decision = "reject"
    else:
        decision = "continue"
    return SPRTResult(decision, llr, lower, upper, games)


def sprt_match(agent_a, agent_b, size: int, max_pairs: int = 200,
               seed: int = 999, p0: float = 0.50, p1: float = 0.55,
               max_steps: int = 800) -> tuple[SPRTResult, MatchResult]:
    """Play mirrored pairs until SPRT decides or `max_pairs` is exhausted."""
    total = 0.0
    games = 0
    result = SPRTResult("continue", 0.0, 0.0, 0.0, 0)
    for i in range(max_pairs):
        s0, s1 = play_pair(agent_a, agent_b, seed + i, size, max_steps)
        total += s0 + s1
        games += 2
        result = sprt(total, games, p0, p1)
        if result.decided:
            break
    return result, MatchResult(games=games, score=total, pairs=games // 2)
