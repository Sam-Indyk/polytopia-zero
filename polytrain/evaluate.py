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
from dataclasses import dataclass

from polyengine.agents.heuristic import evaluate as material
from polyengine.core.game import Game


def play_pair(agent_a, agent_b, seed: int, size: int, max_steps: int = 800
              ) -> tuple[float, float]:
    """Play one map twice with swapped seats. Returns A's score in each game."""
    scores = []
    for a_seat in (0, 1):
        game = Game.new(size=size, seed=seed)
        agents = {a_seat: agent_a, 1 - a_seat: agent_b}
        steps = 0
        while not game.is_terminal() and steps < max_steps:
            game.step(agents[game.current_player].act(game))
            steps += 1
        if game.is_terminal():
            w = game.winner()
            scores.append(1.0 if w == a_seat else (0.5 if w < 0 else 0.0))
        else:
            m = material(game, a_seat)
            scores.append(1.0 if m > 0 else (0.5 if m == 0 else 0.0))
    return scores[0], scores[1]


@dataclass
class MatchResult:
    games: int
    score: float                # total points for A
    pairs: int

    @property
    def win_rate(self) -> float:
        return self.score / self.games if self.games else 0.0

    @property
    def stderr(self) -> float:
        """Standard error of the win rate (binomial approximation)."""
        if self.games < 2:
            return float("nan")
        p = self.win_rate
        return math.sqrt(max(p * (1 - p), 1e-9) / self.games)

    @property
    def elo_diff(self) -> float:
        p = min(max(self.win_rate, 1e-4), 1 - 1e-4)
        return -400.0 * math.log10(1.0 / p - 1.0)


def play_match(agent_a, agent_b, n_pairs: int, size: int, seed: int = 12345,
               max_steps: int = 800) -> MatchResult:
    """Mirrored-seed match: `n_pairs` maps, each played from both seats."""
    total = 0.0
    for i in range(n_pairs):
        s0, s1 = play_pair(agent_a, agent_b, seed + i, size, max_steps)
        total += s0 + s1
    return MatchResult(games=2 * n_pairs, score=total, pairs=n_pairs)


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
