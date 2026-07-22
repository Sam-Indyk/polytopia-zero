"""RandomLegal — the bottom rung of the baseline ladder (anchored Elo = 0).

Picks uniformly among legal actions, with a small bias against ending the turn
while other actions remain, so random games actually do things (move, fight,
expand) instead of passing immediately.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import Action, enumerate_actions


class RandomAgent:
    def __init__(self, seed: int = 0, end_turn_bias: float = 0.15):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.end_turn_bias = end_turn_bias

    def act(self, game) -> Action:
        actions = enumerate_actions(game.state, game.current_player, game.rules)
        non_end = [a for a in actions if a.atype != C.A_END_TURN]
        if not non_end:
            return Action(C.A_END_TURN)
        if self.rng.random() < self.end_turn_bias:
            return Action(C.A_END_TURN)
        return non_end[int(self.rng.integers(len(non_end)))]
