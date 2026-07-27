"""Static state evaluation shared by the OSLA / Scripted / MCTS baselines.

Returns a scalar from `player`'s perspective: own material minus the strongest
opponent's. "Material" blends city levels (economy + win condition), unit value
weighted by current HP, tech, territory, and a little stockpiled stars. It is a
deliberately simple, fast, cheat-y (full-information) heuristic — good enough to
anchor the baseline ladder, not the learned value function.
"""
from __future__ import annotations

from polyengine.core import constants as C


def _material(game, player: int) -> float:
    s = game.state
    rules = game.rules
    owned_city = (s.city_owner == player) & (s.city_level > 0)
    val = 15.0 * float(s.city_level[owned_city].sum())
    val += 3.0 * bin(int(s.tech[player])).count("1")
    val += 0.2 * float(s.stars[player])
    val += 0.5 * float((s.owner == player).sum())
    for t in range(s.n_tiles):
        if s.unit_owner[t] == player and s.unit_type[t] != C.U_NONE:
            ut = s.unit_type[t]
            hp_frac = float(s.unit_hp[t]) / float(rules.unit_max_hp[ut])
            val += (float(rules.unit_cost[ut]) + 2.0) * hp_frac
    return val


def evaluate(game, player: int) -> float:
    n = game.state.n_players
    opp = max((_material(game, p) for p in range(n) if p != player), default=0.0)
    return _material(game, player) - opp
