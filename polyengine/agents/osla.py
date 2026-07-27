"""Greedy one-step-lookahead (OSLA) baseline.

For each legal action, apply it to a copy of the game and score the result with
the shared heuristic from the acting player's perspective; take the argmax. Ties
broken by a seeded RNG so mirror matches don't deadlock into identical lines.
"""
from __future__ import annotations

import numpy as np

from polyengine.agents.heuristic import evaluate
from polyengine.core import constants as C
from polyengine.core.actions import Action


class GreedyOSLA:
    def __init__(self, seed: int = 0):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def act(self, game) -> Action:
        player = game.current_player
        actions = game.legal_actions()
        if not actions:
            return Action(C.A_END_TURN)
        best_val, best = -1e18, []
        for a in actions:
            sim = game.clone()
            sim.step(a)
            v = evaluate(sim, player)
            if v > best_val + 1e-9:
                best_val, best = v, [a]
            elif abs(v - best_val) <= 1e-9:
                best.append(a)
        return best[int(self.rng.integers(len(best)))]
