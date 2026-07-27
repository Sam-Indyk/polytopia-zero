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

from polyengine.core import constants as C
from polyengine.core.actions import Action, apply_action, enumerate_actions
from polyengine.core.economy import city_income, player_income
from polyengine.core.fog import recompute_fog
from polyengine.core.mapgen import new_game
from polyengine.core.rules import Ruleset, load_ruleset
from polyengine.core.state import GameState


class Game:
    def __init__(self, state: GameState, rules: Ruleset):
        self.state = state
        self.rules = rules
        self._begin_turn(state.current_player)

    @classmethod
    def new(cls, size: int = 11, tribes: list[str] | None = None,
            seed: int = 0, ruleset: str = "current") -> "Game":
        rules = load_ruleset(ruleset)
        state = new_game(rules, size=size, tribes=tribes, seed=seed)
        return cls(state, rules)

    def clone(self) -> "Game":
        """A deep copy that shares the (immutable) ruleset. No turn side effects.

        Bypasses __init__ so cloning does NOT re-run _begin_turn (which would
        double-collect income). This is what the OSLA / MCTS baselines lean on.
        """
        g = Game.__new__(Game)
        g.state = self.state.copy()
        g.rules = self.rules
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
        s = self.state
        owned = (s.city_owner == player) & (s.city_level > 0)
        city_score = int(s.city_level[owned].sum()) * 100
        city_score += int(s.city_park[owned].sum()) * 250
        n_units = int(((s.unit_owner == player) & (s.unit_type != C.U_NONE)).sum())
        tech_score = bin(int(s.tech[player])).count("1") * 20
        return city_score + n_units * 5 + tech_score + int(s.stars[player])

    # --- stepping -----------------------------------------------------------
    def step(self, action: Action) -> dict:
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
        # Income only. Cities no longer grow for free: population is bought via
        # harvests and buildings, which is what makes unit capacity scarce.
        s.stars[player] += player_income(s, player)
        # Reset this player's units for the new turn.
        mine = (s.unit_owner == player) & (s.unit_type != C.U_NONE)
        s.unit_moved[mine] = 0
        s.unit_attacked[mine] = 0
        recompute_fog(s, player)


    def _alive_count(self) -> int:
        return sum(1 for p in range(self.state.n_players) if self.owns_city(p))
