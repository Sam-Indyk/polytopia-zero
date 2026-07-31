"""Game — thin orchestration over GameState + the action rules.

Owns turn flow (income, city growth, per-turn unit reset, fog refresh), the
legality surface the agents/masks consume, and the Domination win condition.

Economy (see core/economy.py): each owned city yields `level` stars per turn,
+1 if it is the capital and +1 with a Workshop. Cities do NOT grow on their own
— population is bought through harvests and buildings, and city level caps how
many units a city can support (`level + 1`). That scarcity is what stops star
income turning directly into an unbounded army.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import Action, apply_action, enumerate_actions
from polyengine.core.economy import (city_income, player_income,
                                     recompute_connections, temple_score,
                                     update_tasks)
from polyengine.core.combat import recover_idle_units
from polyengine.core.diplomacy import roll_turn
from polyengine.core.fog import recompute_fog
from polyengine.core.mapgen import new_game
from polyengine.core.rules import Ruleset, load_ruleset
from polyengine.core.state import GameState

# Point values from the wiki's Score page. Kept as named constants because they
# are now the *definition of winning* at the turn cap, not incidental weights.
UNIT_SCORE_PER_STAR = 5
SUPER_UNIT_SCORE = 50
TERRITORY_SCORE = 20
EXPLORED_SCORE = 5
CITY_BASE_SCORE = 100
CITY_LEVEL_SCORE = 50          # per level above 1
PARK_SCORE = 250
TECH_SCORE_PER_TIER = 100
# Temples score dynamically -- see economy.temple_score. This is the id list the
# score function uses to route them away from the flat improvement table.
TEMPLE_IDS = np.array(sorted(C.TEMPLES), dtype=np.int8)


class Game:
    def __init__(self, state: GameState, rules: Ruleset):
        self.state = state
        self.rules = rules
        self._attacked_this_turn: dict[int, bool] = {}
        self._begin_turn(state.current_player)

    @classmethod
    def new(cls, size: int | None = 11, tribes: list[str] | None = None,
            seed: int = 0, ruleset: str = "current",
            map_type: str = "drylands") -> "Game":
        """A fresh game. `size=None` picks what Domination would for this many
        opponents; `map_type` selects a wetness band (see mapgen.MAP_TYPES)."""
        rules = load_ruleset(ruleset)
        state = new_game(rules, size=size, tribes=tribes, seed=seed,
                         map_type=map_type)
        return cls(state, rules)

    def clone(self) -> "Game":
        """A deep copy that shares the (immutable) ruleset. No turn side effects.

        Bypasses __init__ so cloning does NOT re-run _begin_turn (which would
        double-collect income). This is what the OSLA / MCTS baselines lean on.
        """
        g = Game.__new__(Game)
        g.state = self.state.copy()
        g.rules = self.rules
        g._attacked_this_turn = dict(self._attacked_this_turn)
        return g

    # --- queries ------------------------------------------------------------
    @property
    def current_player(self) -> int:
        return self.state.current_player

    def legal_actions(self) -> list[Action]:
        return enumerate_actions(self.state, self.current_player, self.rules)

    def owns_city(self, player: int) -> bool:
        return bool(((self.state.city_owner == player) &
                     (self.state.city_level > 0)).any())

    def is_terminal(self) -> bool:
        return self._alive_count() <= 1 or self.state.turn >= C.TURN_CAP

    def winner(self) -> int:
        """Winning player id, or -1 if not yet decided / drawn."""
        alive = [p for p in range(self.state.n_players) if self.owns_city(p)]
        if len(alive) == 1:
            return alive[0]
        if self.state.turn >= C.TURN_CAP and alive:
            return max(alive, key=self.score)
        return -1

    def score(self, player: int) -> int:
        """Polytopia's own scoring formula, from the wiki's Score page.

        Four categories, verbatim:

          Army and Territory
            * "Units: 5 points per star of a unit's cost. Super units are worth
               50 points each. (They have a theoretical cost of 10 stars.)"
            * "Territory: 20 points per tile in the player's territory."
            * "Exploration: 5 points per explored tile."
          Cities
            * "Cities: 100 points plus 50 points per level above level 1."
            * "Park: 250 points each."
          Monuments and Temples
            * "Monuments: 400 points each."
            * "Temples: 100 points plus 100 points per level above level 1"
          Science
            * "Technology: 100 points per tier of the technology."

        Note what is NOT here: stockpiled stars score nothing. The previous
        version added them, along with a flat 5 per unit and 20 per tech, and
        counted cities as 100 *per level* rather than 100 plus 50 per level
        above the first. The wiki offers a checkable anchor -- "level 1 city +
        9 tiles territory = 280 points" -- which `tests/test_score.py` pins.
        """
        s, r = self.state, self.rules
        total = 0

        # --- army and territory ---
        units = (s.unit_owner == player) & (s.unit_type != C.U_NONE)
        for tile in np.flatnonzero(units):
            utype = int(s.unit_type[tile])
            # A hull scores as its passenger: "Ship and battleship upgrades do
            # not give extra score."
            carried = int(s.unit_carried[tile])
            if carried != C.U_NONE:
                utype = carried
            if utype == C.U_GIANT:
                total += SUPER_UNIT_SCORE
            else:
                total += UNIT_SCORE_PER_STAR * int(r.unit_cost[utype])
        total += TERRITORY_SCORE * int((s.owner == player).sum())
        total += EXPLORED_SCORE * int(s.fog_explored[player].sum())

        # --- cities ---
        owned = (s.city_owner == player) & (s.city_level > 0)
        n_cities = int(owned.sum())
        total += CITY_BASE_SCORE * n_cities
        total += CITY_LEVEL_SCORE * int((s.city_level[owned] - 1).sum())
        total += PARK_SCORE * int(s.city_park[owned].sum())

        # --- monuments and temples (score travels with the builder) ---
        built = np.flatnonzero(s.improvement_owner == player)
        if len(built):
            imps = s.improvement[built]
            # A temple's worth grows with age -- 100 at construction, +50 every
            # two turns to a maximum of 300 -- so it cannot come from the flat
            # per-improvement table the way a monument's 400 does.
            is_temple = np.isin(imps, TEMPLE_IDS)
            total += int(r.imp_score[imps[~is_temple]].sum())
            for tile in built[is_temple]:
                total += temple_score(s, int(tile))

        # --- science ---
        for tech in range(r.n_techs):
            if s.has_tech(player, tech):
                total += TECH_SCORE_PER_TIER * int(r.tech_tier[tech])
        return total

    # --- stepping -----------------------------------------------------------
    def step(self, action: Action) -> dict:
        if action.atype == C.A_ATTACK:
            self._attacked_this_turn[self.current_player] = True
        info = apply_action(self.state, action, self.rules)
        if action.atype == C.A_END_TURN:
            self._advance()
        else:
            recompute_fog(self.state, self.current_player)
        return info

    def _advance(self) -> None:
        s = self.state
        start = s.current_player
        n = s.n_players
        next_player = None
        for step in range(1, n + 1):
            nxt = (start + step) % n
            if self.owns_city(nxt):
                next_player = nxt
                break
        if next_player is None:
            return  # nobody owns a city: game is terminal, leave state as-is
        if next_player <= start:
            s.turn += 1  # wrapped back to an earlier seat -> new round
        s.current_player = next_player
        if not self.is_terminal():
            self._begin_turn(next_player)

    def _begin_turn(self, player: int) -> None:
        s = self.state
        # Update elimination status.
        for p in range(s.n_players):
            if not self.owns_city(p):
                s.alive[p] = 0
        # War lapses on its own: "to be at war, either side must have attacked
        # the other side at most a turn ago. After that, the relation goes back
        # to neutral." Ageing the record here is what makes that true.
        roll_turn(s, player)
        # Income only. Cities no longer grow for free: population is bought via
        # harvests and buildings, which is what makes unit capacity scarce.
        # Roads built or lost since this player's last turn change what is
        # connected, and connections are worth population -- so settle the
        # network before taking income.
        recompute_connections(s, player, self.rules)
        s.stars[player] += player_income(s, player, self.rules)
        # Units that did nothing last turn recover, which must happen while the
        # move/attack flags still describe last turn -- so before the reset.
        recover_idle_units(s, player, self.rules)
        # Pacifist counts whole turns without an attack. `_attacked_this_turn`
        # is set by apply_action and cleared here, so the counter measures the
        # turn that just ended.
        if self._attacked_this_turn.get(player):
            s.calm_turns[player] = 0
        else:
            s.calm_turns[player] += 1
        self._attacked_this_turn[player] = False
        update_tasks(s, player, self.rules)
        # Reset this player's units for the new turn.
        mine = (s.unit_owner == player) & (s.unit_type != C.U_NONE)
        s.unit_moved[mine] = 0
        s.unit_attacked[mine] = 0
        recompute_fog(s, player)


    def _alive_count(self) -> int:
        return sum(1 for p in range(self.state.n_players) if self.owns_city(p))
