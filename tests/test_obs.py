"""Observation encoding — shape, perspective invariance, and the fog contract.

The load-bearing property here is that the encoder may only use what the acting
player can see. If ground truth ever leaks into a plane, the agent learns to
rely on information the M5 screen parser cannot supply.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()


def test_layout_is_contiguous_and_matches_n_planes():
    layout = O.plane_layout(R)
    cursor = 0
    for name, (start, count) in layout.items():
        assert start == cursor, f"{name} block is not contiguous"
        cursor += count
    assert cursor == O.n_planes(R)


def test_spatial_shape_and_range():
    game = Game.new(size=11, seed=1)
    sp = O.encode_spatial(game.state, 0, R)
    assert sp.shape == (O.n_planes(R), 11, 11)
    assert sp.dtype == np.float32
    assert sp.min() >= 0.0 and sp.max() <= 1.0


def test_terrain_and_resource_are_one_hot():
    game = Game.new(size=11, seed=2)
    sp = O.encode_spatial(game.state, 0, R)
    t0, tc = O.plane_layout(R)["terrain"]
    r0, rc = O.plane_layout(R)["resource"]
    assert np.all(sp[t0:t0 + tc].sum(axis=0) == 1.0)
    assert np.all(sp[r0:r0 + rc].sum(axis=0) == 1.0)


def test_padding_sets_the_offboard_mask():
    game = Game.new(size=9, seed=3)
    sp = O.encode_spatial(game.state, 0, R, pad_to=16)
    off = O.plane_layout(R)["territory"][0] + 3
    assert sp.shape[1:] == (16, 16)
    assert sp[off, :9, :9].sum() == 0.0             # real board unmasked
    assert sp[off].sum() == 16 * 16 - 9 * 9         # exactly the padding


def test_padding_smaller_than_map_is_rejected():
    game = Game.new(size=11, seed=1)
    with pytest.raises(ValueError, match="smaller than map size"):
        O.encode_spatial(game.state, 0, R, pad_to=9)


# --- the fog contract ---------------------------------------------------------
def test_unexplored_tiles_reveal_no_units_or_cities():
    game = Game.new(size=11, seed=5)
    s = game.state
    sp = O.encode_spatial(s, 0, R)
    layout = O.plane_layout(R)
    u0, uc = layout["unit_type"]
    c0, cc = layout["city"]
    unexplored = s.fog_explored[0] == 0
    assert unexplored.any(), "test needs some unexplored tiles"
    for i in np.flatnonzero(unexplored):
        y, x = divmod(int(i), s.size)
        assert sp[u0:u0 + uc, y, x].sum() == 0.0, "unit leaked through fog"
        assert sp[c0:c0 + cc, y, x].sum() == 0.0, "city leaked through fog"


def _played_game(seed=7, size=11, steps=40):
    """Play a while so fog memory has accumulated remembered-but-unseen tiles."""
    from polyengine.agents.random_agent import RandomAgent
    game = Game.new(size=size, seed=seed)
    agents = [RandomAgent(seed=seed * 5 + p) for p in range(game.state.n_players)]
    for _ in range(steps):
        if game.is_terminal():
            break
        game.step(agents[game.current_player].act(game))
    return game


def _remembered_unseen(state, player):
    return np.flatnonzero((state.fog_explored[player] == 1)
                          & (state.fog_visible[player] == 0))


def test_a_played_game_has_remembered_but_unseen_tiles():
    """Precondition for the memory tests below — fail loudly if fog stops working."""
    game = _played_game()
    assert _remembered_unseen(game.state, 0).size > 0


def test_encoder_uses_memory_not_ground_truth():
    """A unit that moves out of sight stays where it was last seen."""
    game = _played_game(seed=7)
    s = game.state
    remembered = _remembered_unseen(s, 0)
    assert remembered.size > 0, "expected remembered-but-unseen tiles"
    tile = int(remembered[0])
    s.mem_unit_type[0][tile] = C.U_WARRIOR
    s.mem_unit_owner[0][tile] = 1
    s.unit_type[tile] = C.U_NONE          # ground truth: nothing there
    sp = O.encode_spatial(s, 0, R)
    u0 = O.plane_layout(R)["unit_type"][0]
    y, x = divmod(tile, s.size)
    assert sp[u0 + C.U_WARRIOR, y, x] == 1.0, "encoder ignored fog memory"


def test_hp_and_status_only_for_currently_visible_units():
    """Remembered-but-unseen units expose type/owner but not live HP."""
    game = _played_game(seed=9)
    s = game.state
    hidden = _remembered_unseen(s, 0)
    assert hidden.size > 0, "expected remembered-but-unseen tiles"
    tile = int(hidden[0])
    s.mem_unit_type[0][tile] = C.U_WARRIOR
    s.mem_unit_owner[0][tile] = 1
    s.unit_hp[tile] = 7
    sp = O.encode_spatial(s, 0, R)
    a0 = O.plane_layout(R)["unit_attrs"][0]
    y, x = divmod(tile, s.size)
    assert sp[a0 + 1, y, x] == 1.0          # known to be an enemy
    assert sp[a0 + 2, y, x] == 0.0          # but HP is not revealed


def test_perspective_is_relative_to_the_acting_player():
    game = Game.new(size=11, seed=11)
    own = O.plane_layout(R)["territory"][0]
    sp0 = O.encode_spatial(game.state, 0, R)
    sp1 = O.encode_spatial(game.state, 1, R)
    # P0's "own territory" plane must equal P1's "enemy territory" plane.
    assert np.array_equal(sp0[own], sp1[own + 1])
    assert np.array_equal(sp0[own + 1], sp1[own])


# --- global vector ------------------------------------------------------------
def test_global_vector_shape_and_tech_bits():
    game = Game.new(size=9, seed=4)
    s = game.state
    tech = R.tech_id["riding"]
    s.grant_tech(0, tech)
    gv = O.encode_global(s, 0, R)
    assert gv.shape == (O.global_dim(R),)
    assert gv[4 + tech] == 1.0


def test_global_vector_hides_opponent_stars():
    game = Game.new(size=9, seed=4)
    s = game.state
    s.stars[1] = 99
    stride = 4 + R.n_techs
    gv = O.encode_global(s, 1, R)          # opponent block is index `stride`
    assert gv[stride + 0] == 0.0, "opponent stars must not be observable"


def test_the_global_vector_says_nothing_about_the_map_type():
    """Two very different worlds, one identical global vector.

    Training rotates map types, and it is tempting to hand the net a wetness
    summary so it can tell an archipelago from drylands at a glance. That would
    be a shortcut with no counterpart on a real screen: a player reads water
    tile by tile, through fog, and infers the rest. The spatial planes carry the
    terrain; the global vector must stay silent about it.
    """
    dry = Game.new(size=11, seed=11, map_type="drylands")
    wet = Game.new(size=11, seed=11, map_type="waterworld")
    water_dry = int((~R.terrain_is_land[dry.state.terrain]).sum())
    water_wet = int((~R.terrain_is_land[wet.state.terrain]).sum())
    assert water_wet > water_dry * 5, "the two maps must actually differ"

    a = O.encode_global(dry.state, 0, R)
    b = O.encode_global(wet.state, 0, R)
    assert np.array_equal(a, b)

    # ... while the spatial planes plainly do differ, which is where the policy
    # is supposed to learn "build Ports here, Riders there" from.
    sp_dry = O.encode_spatial(dry.state, 0, R)
    sp_wet = O.encode_spatial(wet.state, 0, R)
    assert not np.array_equal(sp_dry, sp_wet)


def test_encode_returns_both_tensors():
    game = Game.new(size=9, seed=6)
    sp, gv = O.encode(game.state, 0, R)
    assert sp.shape[0] == O.n_planes(R) and gv.shape[0] == O.global_dim(R)
