"""Action masks must describe exactly the same set as `enumerate_actions`.

The masks are what the policy samples from; `enumerate_actions` is what the
engine will accept. If they ever disagree, the agent can emit an illegal action
(crash) or is silently barred from legal ones (a hidden capability ceiling), so
this equivalence is checked exhaustively on random self-play states.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyengine import masks as M
from polyengine.agents.random_agent import RandomAgent
from polyengine.core import constants as C
from polyengine.core.actions import enumerate_actions
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()


def actions_from_masks(state, player, rules) -> set[tuple]:
    """Expand the four masked heads into the full legal action set."""
    out = set()
    types = M.legal_types(state, player, rules)
    for atype in range(C.N_ACTION_TYPES):
        if not types[atype]:
            continue
        if atype == C.A_END_TURN:
            out.add((atype, -1, -1, -1))
            continue
        if atype == C.A_RESEARCH:
            for sub in np.flatnonzero(M.legal_subtypes(state, player, rules, atype)):
                out.add((atype, -1, -1, int(sub)))
            continue
        for src in np.flatnonzero(M.legal_sources(state, player, rules, atype)):
            src = int(src)
            if atype == C.A_TRAIN:
                for sub in np.flatnonzero(
                        M.legal_subtypes(state, player, rules, atype)):
                    out.add((atype, src, -1, int(sub)))
            else:
                for dst in np.flatnonzero(
                        M.legal_targets(state, player, rules, atype, src)):
                    out.add((atype, src, int(dst), -1))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_masks_match_enumerate_actions_through_a_game(seed):
    game = Game.new(size=9, seed=seed)
    agents = [RandomAgent(seed=seed * 3 + p) for p in range(game.state.n_players)]
    for _ in range(60):
        if game.is_terminal():
            break
        s, p = game.state, game.current_player
        expected = {(a.atype, a.src, a.dst, a.subtype)
                    for a in enumerate_actions(s, p, R)}
        assert actions_from_masks(s, p, R) == expected
        game.step(agents[p].act(game))


def test_end_turn_is_always_legal():
    game = Game.new(size=9, seed=5)
    assert M.legal_types(game.state, 0, R)[C.A_END_TURN]


def test_capture_target_is_the_source_tile():
    from tests.test_capture import refresh, setup_unit_next_to_city
    from polyengine.core.actions import Action
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    targets = M.legal_targets(game.state, 0, R, C.A_CAPTURE, city)
    assert targets[city] and targets.sum() == 1


def test_train_sources_are_empty_cities_only():
    game = Game.new(size=9, seed=2)
    s = game.state
    s.stars[0] = 50
    src = M.legal_sources(s, 0, R, C.A_TRAIN)
    for tile in np.flatnonzero(src):
        assert s.city_owner[tile] == 0 and s.city_level[tile] > 0
        assert s.unit_type[tile] == C.U_NONE


def test_train_subtypes_respect_stars_and_tech():
    game = Game.new(size=9, seed=2)
    s = game.state
    s.stars[0] = 2                       # only a Warrior is affordable
    sub = M.legal_subtypes(s, 0, R, C.A_TRAIN)
    assert sub[R.unit_id["warrior"]]
    assert not sub[R.unit_id["catapult"]]
    assert not sub[R.unit_id["raft"]], "naval hulls are not trainable"


def test_research_subtypes_respect_prerequisites():
    game = Game.new(size=9, seed=2)
    s = game.state
    s.stars[0] = 200
    sub = M.legal_subtypes(s, 0, R, C.A_RESEARCH)
    for tid in np.flatnonzero(sub):
        req = int(R.tech_requires[tid])
        assert req < 0 or s.has_tech(0, req)
        assert not s.has_tech(0, int(tid)), "already-known tech offered again"


def test_no_stars_means_no_train_or_research():
    game = Game.new(size=9, seed=2)
    s = game.state
    s.stars[0] = 0
    types = M.legal_types(s, 0, R)
    assert not types[C.A_TRAIN] and not types[C.A_RESEARCH]


def test_head_applicability_sets_are_consistent():
    assert C.A_END_TURN not in M.NEEDS_SOURCE
    assert C.A_RESEARCH not in M.NEEDS_SOURCE
    assert M.NEEDS_TARGET <= M.NEEDS_SOURCE          # target implies a source
    assert C.A_TRAIN in M.NEEDS_SUBTYPE and C.A_RESEARCH in M.NEEDS_SUBTYPE


def test_subtype_dim_covers_units_and_techs():
    assert M.subtype_dim(R) >= R.n_units
    assert M.subtype_dim(R) >= R.n_techs


def test_action_masks_cache_is_consistent():
    game = Game.new(size=9, seed=8)
    am = M.ActionMasks(game.state, 0, R)
    assert np.array_equal(am.types(), am.types())
    assert np.array_equal(am.sources(C.A_MOVE),
                          M.legal_sources(game.state, 0, R, C.A_MOVE))
