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
    """Own/enemy are relative slots, so one player's "mine" is the other's
    "theirs" — **on tiles both have actually seen.**

    This used to compare the two planes over the whole board, and only passed
    because the territory planes read ground-truth `owner`: each player was
    being shown the full extent of the other's empire through unexplored fog.
    Scoped to shared knowledge, it tests perspective rather than the leak.
    """
    from polyengine.agents.scripted import ScriptedBot

    # The two tribes start on opposite sides and share no explored ground at
    # all on turn 0, so the property is vacuous until they have met.
    game = Game.new(size=11, seed=11)
    bots = [ScriptedBot(seed=1), ScriptedBot(seed=2)]
    s = game.state
    for _ in range(2000):
        both_seen = (s.fog_explored[0] == 1) & (s.fog_explored[1] == 1)
        if both_seen.sum() > 5 or game.is_terminal():
            break
        game.step(bots[game.current_player].act(game))
    own = O.plane_layout(R)["territory"][0]
    sp0 = O.encode_spatial(s, 0, R)
    sp1 = O.encode_spatial(s, 1, R)
    both = ((s.fog_explored[0] == 1) & (s.fog_explored[1] == 1)
            ).reshape(s.size, s.size)
    assert both.any(), "test needs some commonly explored ground"
    assert np.array_equal(sp0[own][both], sp1[own + 1][both])
    assert np.array_equal(sp0[own + 1][both], sp1[own][both])


def test_territory_is_not_visible_through_unexplored_fog():
    """Invariant 1. Borders under the clouds are not on a player's screen, and
    the M5 vision layer could never reproduce them from pixels.

    Cities and units have been rendered from `mem_*` since the start; territory
    was reading ground truth, so from turn 0 every player could see exactly how
    far every empire reached across a map it had never walked.
    """
    game = Game.new(size=11, seed=11)
    s = game.state
    own = O.plane_layout(R)["territory"][0]
    sp = O.encode_spatial(s, 0, R)
    dark = (s.fog_explored[0] == 0).reshape(s.size, s.size)
    assert dark.any(), "test needs unexplored ground to be meaningful"
    # Enemy territory exists on the board...
    assert (s.owner == 1).any()
    # ...but none of the three ownership planes fires on an unexplored tile.
    for slot in range(3):
        assert not sp[own + slot][dark].any(), (
            f"territory plane {slot} leaked through unexplored fog")


def test_remembered_territory_is_memory_not_live_truth():
    """Once seen, a border stays as last seen until you look again.

    The other half of the contract. Gating on `explored` alone would still have
    told the player the instant a distant tile changed hands, which is not
    something any screen shows."""
    from polyengine.agents.scripted import ScriptedBot
    from polyengine.core.fog import recompute_fog

    game = Game.new(size=11, seed=11)
    bots = [ScriptedBot(seed=1), ScriptedBot(seed=2)]
    s = game.state
    tile = -1
    for _ in range(3000):
        if game.is_terminal():
            break
        game.step(bots[game.current_player].act(game))
        recompute_fog(s, 0)
        # Explored by P0, currently out of sight, and P1's ground.
        cand = np.flatnonzero((s.fog_explored[0] == 1) & (s.fog_visible[0] == 0)
                              & (s.owner == 1))
        if len(cand):
            tile = int(cand[0])
            break
    assert tile >= 0, "never found remembered enemy territory out of sight"

    own = O.plane_layout(R)["territory"][0]
    y, x = divmod(tile, s.size)
    assert O.encode_spatial(s, 0, R)[own + 1, y, x] == 1.0

    # The border recedes with nobody watching. Ground truth moves; memory does
    # not. (It is flipped to neutral rather than to P0, because a player's own
    # territory is always visible — assigning it would reveal the tile and make
    # the test vacuous.)
    s.owner[tile] = C.NEUTRAL
    recompute_fog(s, 0)
    assert s.fog_visible[0][tile] == 0, "tile became visible; test is void"
    sp = O.encode_spatial(s, 0, R)
    assert sp[own + 1, y, x] == 1.0, "memory tracked a change it could not see"
    assert sp[own + 2, y, x] == 0.0, "neutral plane fired on remembered ground"


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
