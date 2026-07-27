"""ScriptedBot — a hand-written Midjiwan-style heuristic policy.

Fast, rule-based (no full-state eval), with a fixed priority order:
  1. Capture an enemy/neutral city if a unit can move onto it.
  2. Make a favourable attack (lethal, or deal >= damage taken).
  3. Train a unit when stars allow and the army is small.
  4. Research a cheap useful tech when flush with stars.
  5. March a unit toward the nearest enemy city/unit.
  6. Otherwise end the turn.

The plan also calls for a "Crazy-cheat" variant (bonus star income) as the
Phase-4 proxy target — that's just this policy with an income multiplier, wired
in when the league needs it.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.core.rules import Ruleset
from polyengine.kernels.combat_kernel import damage_numbers


class ScriptedBot:
    def __init__(self, seed: int = 0, min_army: int = 4):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.min_army = min_army

    def act(self, game) -> Action:
        s = game.state
        rules = game.rules
        player = game.current_player
        actions = game.legal_actions()
        by_type: dict[int, list[Action]] = {}
        for a in actions:
            by_type.setdefault(a.atype, []).append(a)

        # 1. Capture: a MOVE landing on a city this player doesn't own.
        captures = [a for a in by_type.get(C.A_MOVE, [])
                    if s.city_level[a.dst] > 0 and s.city_owner[a.dst] != player]
        if captures:
            return self._pick(captures)

        # 2. Favourable attack.
        best_atk, best_score = None, 0.0
        for a in by_type.get(C.A_ATTACK, []):
            score = self._attack_score(s, rules, a)
            if score > best_score:
                best_atk, best_score = a, score
        if best_atk is not None:
            return best_atk

        # 3. Train when the army is small and we can afford it.
        army = int(((s.unit_owner == player) & (s.unit_type != C.U_NONE)).sum())
        trains = by_type.get(C.A_TRAIN, [])
        if trains and army < self.min_army:
            trains.sort(key=lambda a: rules.unit_cost[a.subtype], reverse=True)
            return trains[0]

        # 4. Research when flush.
        research = by_type.get(C.A_RESEARCH, [])
        if research and int(s.stars[player]) >= 8:
            return self._pick(research)

        # 5. March toward the nearest objective.
        march = self._march(game, player, by_type.get(C.A_MOVE, []))
        if march is not None:
            return march

        return Action(C.A_END_TURN)

    def _attack_score(self, s, rules: Ruleset, a: Action) -> float:
        at, dt = s.unit_type[a.src], s.unit_type[a.dst]
        from polyengine.core.combat import defense_bonus_for
        bonus = defense_bonus_for(s, a.dst, rules)
        atk_res, def_res = damage_numbers(
            float(rules.unit_attack[at]), int(s.unit_hp[a.src]),
            int(rules.unit_max_hp[at]), float(rules.unit_defense[dt]),
            int(s.unit_hp[a.dst]), int(rules.unit_max_hp[dt]), bonus, rules.accuracy)
        lethal = atk_res >= int(s.unit_hp[a.dst])
        ranged = int(rules.unit_range[at]) > 1
        retaliation = 0 if (lethal or ranged) else def_res
        # Reward damage dealt and kills; penalise damage taken.
        return atk_res + (10.0 if lethal else 0.0) - retaliation

    def _march(self, game, player: int, moves: list[Action]) -> Action | None:
        if not moves:
            return None
        s = game.state
        targets = [t for t in range(s.n_tiles)
                   if (s.unit_type[t] != C.U_NONE and s.unit_owner[t] != player)
                   or (s.city_level[t] > 0 and s.city_owner[t] != player)]
        if not targets:
            return self._pick(moves)
        best, best_d = None, 1 << 30
        for a in moves:
            d = min(s.chebyshev(a.dst, t) for t in targets)
            if d < best_d:
                best_d, best = d, a
        return best

    def _pick(self, actions: list[Action]) -> Action:
        return actions[int(self.rng.integers(len(actions)))]
