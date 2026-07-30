"""ScriptedBot — a hand-written policy modelled on Polytopia's own bots.

The wiki describes what actually separates Midjiwan's difficulty levels:

    "The difficulty level determines the aggressiveness and starting income of
     bot opponents. Bots of higher difficulties are more aggressive. For
     example, easy bots will almost always be peaceful until provoked, while
     crazy bots may be immediately hostile. The difficulty also determines the
     income bots receive from their capital city ... Easy bots receive one star
     per turn (SPT), normal bots (and human players) receive two, hard bots
     receive three, and crazy bots receive five."

So difficulty here is two knobs, not a different program: `aggression` (how
readily it spends units on attacks, and how large an army it keeps) and an
income handicap written to `state.income_bonus`. `ScriptedBot.crazy()` is the
M5 target opponent: full aggression and +3 SPT.

Priority order, roughly "resolve, then fight, then spend, then move":

  0. Answer a pending level-up reward (the engine blocks on it anyway).
  1. Capture a city we are standing on. Nothing else is worth more.
  2. Take a favourable fight — including a Cloak's infiltrate, which is an
     attack whose target is a city.
  3. Sit on an undefended enemy city: siege zeroes its income and sets up a
     capture next turn.
  4. Spend: monuments (free), roads that actually complete a city connection,
     then the cheapest population per star.
  5. Train toward an army target that scales with aggression.
  6. March — but leave badly hurt units at home, because doing nothing IS
     recovering, and marching them one tile closer forfeits the heal.

It plays with full information, like the real bots do. It is a baseline and a
sparring partner, not a model of correct play.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.core.combat import defense_bonus_for, effective_max_hp
from polyengine.core.rules import Ruleset
from polyengine.kernels.combat_kernel import damage_numbers

# difficulty -> (aggression 0..1, extra stars per turn). The SPT figures are the
# wiki's, minus the 2 that a level-1 capital already produces.
DIFFICULTIES = {
    "easy": (0.15, -1),
    "normal": (0.45, 0),
    "hard": (0.70, 1),
    "crazy": (1.00, 3),
}


class ScriptedBot:
    def __init__(self, seed: int = 0, difficulty: str = "normal",
                 min_army: int | None = None):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}; "
                             f"choose from {sorted(DIFFICULTIES)}")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.difficulty = difficulty
        self.aggression, self.income_bonus = DIFFICULTIES[difficulty]
        # A more aggressive bot keeps a bigger standing army.
        self.min_army = (min_army if min_army is not None
                         else int(round(2 + 5 * self.aggression)))

    @classmethod
    def crazy(cls, seed: int = 0) -> "ScriptedBot":
        """The M5 target opponent: maximum aggression and +3 SPT."""
        return cls(seed=seed, difficulty="crazy")

    def apply_handicap(self, game, player: int) -> None:
        """Grant this bot's difficulty income. Call once, before play starts."""
        game.state.income_bonus[player] = self.income_bonus

    # --- policy ---------------------------------------------------------------
    def act(self, game) -> Action:  # noqa: C901 - a priority list reads better flat
        s, rules, player = game.state, game.rules, game.current_player
        by_type: dict[int, list[Action]] = {}
        for a in game.legal_actions():
            by_type.setdefault(a.atype, []).append(a)

        # 0. A pending reward blocks everything else.
        if by_type.get(C.A_CHOOSE_REWARD):
            return self._best_reward(by_type[C.A_CHOOSE_REWARD], rules)

        # 1. Capture what we are standing on.
        if by_type.get(C.A_CAPTURE):
            return self._pick(by_type[C.A_CAPTURE])

        # 1b. Examine the ruin (or harvest the starfish) we are standing on.
        # Free value, and it has to happen before the march or the unit walks
        # off the tile it needs to *start* a turn on to claim anything.
        if by_type.get(C.A_EXAMINE):
            return self._pick(by_type[C.A_EXAMINE])

        # 2. Attacks, infiltration included.
        attack = self._best_attack(s, rules, by_type.get(C.A_ATTACK, []))
        if attack is not None:
            return attack

        # 3. Step onto an enemy city nobody is defending.
        siege = [a for a in by_type.get(C.A_MOVE, [])
                 if s.city_level[a.dst] > 0 and s.city_owner[a.dst] != player]
        if siege:
            return self._pick(siege)

        # 4. Spend stars before moving, so anything bought can still act.
        build = self._best_build(s, rules, player, by_type.get(C.A_BUILD, []))
        if build is not None:
            return build

        army = int(((s.unit_owner == player) & (s.unit_type != C.U_NONE)).sum())
        trains = by_type.get(C.A_TRAIN, [])
        if trains and army < self.min_army:
            return max(trains, key=lambda a: (rules.unit_attack[a.subtype]
                                              + rules.unit_defense[a.subtype],
                                              -rules.unit_cost[a.subtype]))

        research = by_type.get(C.A_RESEARCH, [])
        if research and int(s.stars[player]) >= 8:
            return self._pick(research)

        # 5. Over target and Free Spirit researched: cash in the weakest unit.
        disband = by_type.get(C.A_DISBAND, [])
        if disband and army > self.min_army + 2:
            return min(disband, key=lambda a: int(s.unit_hp[a.src]))

        # 6. March.
        march = self._march(game, player, by_type.get(C.A_MOVE, []), rules)
        if march is not None:
            return march

        return Action(C.A_END_TURN)

    # --- pieces ---------------------------------------------------------------
    def _best_attack(self, s, rules: Ruleset,
                     attacks: list[Action]) -> Action | None:
        best, best_score = None, 0.0
        for a in attacks:
            score = self._attack_score(s, rules, a)
            if score > best_score:
                best, best_score = a, score
        return best

    def _attack_score(self, s, rules: Ruleset, a: Action) -> float:
        at = int(s.unit_type[a.src])

        # Infiltration consumes the Cloak but takes the city's income, wounds the
        # garrison and spawns daggers — worth more against a big city.
        if "infiltrate" in rules.unit_skills[at]:
            return 12.0 + 4.0 * float(s.city_level[a.dst])

        dt = int(s.unit_type[a.dst])
        if dt == C.U_NONE:
            return 0.0
        atk_res, def_res = damage_numbers(
            float(rules.unit_attack[at]), int(s.unit_hp[a.src]),
            effective_max_hp(s, a.src, rules),
            float(rules.unit_defense[dt]), int(s.unit_hp[a.dst]),
            effective_max_hp(s, a.dst, rules),
            defense_bonus_for(s, a.dst, rules), rules.accuracy)

        lethal = atk_res >= int(s.unit_hp[a.dst])
        ranged = int(rules.unit_range[at]) > 1
        retaliation = 0.0 if (lethal or ranged) else float(def_res)
        suicidal = retaliation >= int(s.unit_hp[a.src])

        value = float(atk_res) + (10.0 if lethal else 0.0) - retaliation
        if suicidal and not lethal:
            # Dying to the reply is a losing trade even when the damage looks
            # fine: Domination scores kills as a RATIO against our own losses.
            # A timid bot refuses; an aggressive one will still take it.
            value -= 15.0 * (1.0 - self.aggression)
        return value * (0.4 + 0.6 * self.aggression)

    def _best_build(self, s, rules: Ruleset, player: int,
                    builds: list[Action]) -> Action | None:
        if not builds:
            return None
        # Monuments are free and yield 3 population; never pass one up.
        monuments = [a for a in builds if rules.imp_requires_task[a.subtype] >= 0]
        if monuments:
            return monuments[0]

        # A road is only worth it when it completes a connection to the capital,
        # which is +1 population to BOTH ends. Decided by trying it, not guessing.
        roads = [a for a in builds if rules.imp_connector[a.subtype]]
        if roads and int(s.stars[player]) >= 8:
            connecting = self._connecting_road(s, rules, player, roads)
            if connecting is not None:
                return connecting

        priced = min(((self._pop_cost(s, a, rules), i, a)
                      for i, a in enumerate(builds)), key=lambda t: t[:2])
        return priced[2] if priced[0] < 1e8 else None

    def _connecting_road(self, s, rules: Ruleset, player: int,
                         roads: list[Action]) -> Action | None:
        """A road that raises the number of cities connected to the capital."""
        from polyengine.core.economy import connected_cities
        before = len(connected_cities(s, player, rules))
        for a in roads[:12]:            # bounded: this copies state per candidate
            probe = s.copy()
            probe.improvement[a.src] = a.subtype
            if len(connected_cities(probe, player, rules)) > before:
                return a
        return None

    def _march(self, game, player: int, moves: list[Action],
               rules: Ruleset) -> Action | None:
        if not moves:
            return None
        s = game.state
        # "If a unit does not take any action in a given turn, it will
        # automatically heal" — so a badly hurt unit standing in our own
        # territory is better left alone than walked one tile closer.
        healthy = [a for a in moves
                   if not (int(s.unit_hp[a.src]) * 2 < effective_max_hp(s, a.src, rules)
                           and int(s.owner[a.src]) == player)]
        moves = healthy or moves

        targets = [t for t in range(s.n_tiles)
                   if (s.unit_type[t] != C.U_NONE and s.unit_owner[t] != player
                       and s.unit_owner[t] != C.NEUTRAL)
                   or (s.city_level[t] > 0 and s.city_owner[t] != player)]
        if not targets:
            return self._pick(moves)
        # Step onto a ruin for free if one is on the way: examining costs a turn
        # but pays a tech or a veteran Swordsman, and "examine ruins that
        # otherwise may be taken by enemies".
        onto_ruin = [a for a in moves if s.improvement[a.dst] == C.IMP_RUIN]
        if onto_ruin:
            return self._pick(onto_ruin)
        return min(moves, key=lambda a: min(s.chebyshev(a.dst, t) for t in targets))

    def _pop_cost(self, state, action: Action, rules: Ruleset) -> float:
        """Stars per population for this build — lower is better value."""
        from polyengine.core.economy import improvement_population
        pop = improvement_population(state, action.src, action.subtype, rules)
        if pop <= 0:
            return 1e9
        return int(rules.imp_cost[action.subtype]) / pop

    def _best_reward(self, rewards: list[Action], rules: Ruleset) -> Action:
        """Prefer compounding rewards (income, population) over one-off ones."""
        order = {"income": 0, "population": 1, "stars": 2, "border": 3,
                 "walls": 4, "super_unit": 5, "explore": 6, "score": 7}
        return min(rewards,
                   key=lambda a: order.get(rules.reward_effect[a.subtype], 9))

    def _pick(self, actions: list[Action]) -> Action:
        return actions[int(self.rng.integers(len(actions)))]
