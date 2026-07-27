"""City capture is its own action, not a side effect of moving.

Regression tests for the rule that moving a unit onto a village/city does NOT
take it: the unit must still be standing there at the start of a later turn and
spend that turn capturing.
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import Action, can_capture, capture_targets, enumerate_actions
from polyengine.core.game import Game


def setup_unit_next_to_city(seed=3, size=9):
    """Place a fresh player-0 warrior adjacent to a city player 0 doesn't own."""
    game = Game.new(size=size, seed=seed)
    s = game.state
    city = next(t for t in range(s.n_tiles)
                if s.city_level[t] > 0 and s.city_owner[t] != 0)
    adj = next(n for n in s.neighbors(city)
               if s.unit_type[n] == C.U_NONE and s.terrain[n] == C.T_FIELD)
    s.unit_type[adj] = C.U_WARRIOR
    s.unit_owner[adj] = 0
    s.unit_hp[adj] = 10
    s.unit_moved[adj] = 0
    s.unit_attacked[adj] = 0
    s.current_player = 0
    return game, city, adj


def refresh(state, tile):
    """Simulate the unit's turn resetting."""
    state.unit_moved[tile] = 0
    state.unit_attacked[tile] = 0


def test_moving_onto_a_city_does_not_capture_it():
    game, city, adj = setup_unit_next_to_city()
    before = int(game.state.city_owner[city])
    game.step(Action(C.A_MOVE, adj, city))
    assert int(game.state.city_owner[city]) == before, \
        "moving onto a city must not capture it"


def test_capture_is_illegal_on_the_turn_the_unit_arrives():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    assert not can_capture(game.state, city, game.rules)
    assert not [a for a in enumerate_actions(game.state, 0, game.rules)
                if a.atype == C.A_CAPTURE]


def test_capture_becomes_legal_the_following_turn():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    assert can_capture(game.state, city, game.rules)
    assert capture_targets(game.state, 0, game.rules) == [city]


def test_capture_transfers_the_city_and_ends_the_unit_turn():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    game.step(Action(C.A_CAPTURE, city, city))
    s = game.state
    assert int(s.city_owner[city]) == 0
    assert int(s.owner[city]) == 0
    assert s.unit_moved[city] and s.unit_attacked[city], \
        "capturing must consume the unit's turn"


def test_capturing_claims_neutral_neighbours_as_territory():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    neutral = [n for n in game.state.neighbors(city)
               if game.state.owner[n] == C.NEUTRAL]
    game.step(Action(C.A_CAPTURE, city, city))
    assert all(game.state.owner[n] == 0 for n in neutral)


def test_cannot_capture_your_own_city():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    game.step(Action(C.A_CAPTURE, city, city))
    refresh(game.state, city)
    assert not can_capture(game.state, city, game.rules)


def test_applying_an_illegal_capture_raises():
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))   # unit has moved this turn
    with pytest.raises(ValueError, match="illegal capture"):
        game.step(Action(C.A_CAPTURE, city, city))


def test_capture_requires_a_unit_on_the_tile():
    game, city, adj = setup_unit_next_to_city()
    assert not can_capture(game.state, city, game.rules)   # nobody there yet


def test_ruleset_exposes_capture_constants():
    game, _, _ = setup_unit_next_to_city()
    assert game.rules.capture_requires_fresh_unit is True
    assert game.rules.capture_ends_unit_turn is True
