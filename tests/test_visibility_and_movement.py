"""Rules found by a full-engine audit (31 Jul), each invisible until looked for.

Three come from `.wikicache/Terrain.wiki` and `.wikicache/Movement.wiki`:

    "Crop is hidden until Organization or Farming has been researched."
    "Climbing ... reveals metal, a resource used by the Mine."
    "Mountain: all units on this terrain have a sight radius of two tiles."
    "When calculating movement cost of a tile, the game rounds up - meaning even
     if you have 0.5 movement points remaining, you can still move onto a tile
     costing 1 or even 3."
    "[Road movement] only applies to Roads in friendly territory or in neutral
     territory, but not when moving onto Roads in enemy territory."

The hidden-resource one is an **invariant 1** matter, not a nicety: the encoder
was showing crop and metal the acting player had no way to see, so a policy could
learn to expand toward resources that are not on its screen — and M5's screen
reader could never reproduce that tensor.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.fog import recompute_all
from polyengine.core.game import Game
from polyengine.core.movement import reachable
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11
MID = 5 * N + 5


def blank(seed=1, all_tech=True):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.terrain[t] = C.T_FIELD
        s.resource[t] = C.R_NONE
        s.improvement[t] = C.IMP_NONE
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.owner[t] = C.NEUTRAL
    s.tech[0] = 0
    if all_tech:
        for k in range(R.n_techs):
            s.grant_tech(0, k)
    s.current_player = 0
    return game, s


def put(s, tile, utype=C.U_WARRIOR, player=0):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    return tile


# --- hidden resources (invariant 1) -------------------------------------------------
class TestHiddenResources:
    def resource_planes(self, s, tile, rid):
        sp = O.encode_spatial(s, 0, R)
        lo, _ = O.plane_layout(R)["resource"]
        y, x = tile // N, tile % N
        return float(sp[lo + rid, y, x]), float(sp[lo + C.R_NONE, y, x])

    def setup_tile(self, s, tile, rid):
        s.terrain[tile] = C.T_MOUNTAIN if rid == C.R_METAL else C.T_FIELD
        s.resource[tile] = rid
        s.owner[tile] = 0
        put(s, MID)                      # someone to see it with
        recompute_all(s)

    @pytest.mark.parametrize("rid,techs", sorted(C.HIDDEN_RESOURCE_TECHS.items()))
    def test_it_is_invisible_without_the_technology(self, rid, techs):
        _, s = blank(all_tech=False)
        self.setup_tile(s, MID + 1, rid)
        shown, none = self.resource_planes(s, MID + 1, rid)
        assert shown == 0.0, f"{R.resource_names[rid]} visible without {techs}"
        assert none == 1.0, "the one-hot must fall back to 'no resource'"

    @pytest.mark.parametrize("rid,techs", sorted(C.HIDDEN_RESOURCE_TECHS.items()))
    def test_either_technology_reveals_it(self, rid, techs):
        for tech in techs:
            _, s = blank(all_tech=False)
            s.grant_tech(0, R.tech_id[tech])
            self.setup_tile(s, MID + 1, rid)
            shown, none = self.resource_planes(s, MID + 1, rid)
            assert shown == 1.0, f"{tech} should reveal {R.resource_names[rid]}"
            assert none == 0.0

    def test_visible_resources_are_unaffected(self):
        """Fruit, game and fish are on screen from turn one."""
        for rid, terrain in ((C.R_FRUIT, C.T_FIELD), (C.R_GAME, C.T_FOREST),
                             (C.R_FISH, C.T_WATER)):
            _, s = blank(all_tech=False)
            self.setup_tile(s, MID + 1, rid)
            s.terrain[MID + 1] = terrain
            recompute_all(s)
            shown, _ = self.resource_planes(s, MID + 1, rid)
            assert shown == 1.0, R.resource_names[rid]

    def test_the_resource_one_hot_stays_a_one_hot(self):
        """Concealing crop must leave the column a one-hot, not an all-zero
        hole that would itself mark the tile as interesting.

        Scoped to explored ground: an unexplored tile is legitimately all-zero
        now, because the resource planes are rendered from fog memory rather
        than from ground truth."""
        _, s = blank(all_tech=False)
        self.setup_tile(s, MID + 1, C.R_CROP)
        sp = O.encode_spatial(s, 0, R)
        lo, n = O.plane_layout(R)["resource"]
        explored = 0
        for tile in range(s.n_tiles):
            column = sp[lo:lo + n, tile // N, tile % N]
            if s.fog_explored[0][tile]:
                explored += 1
                assert column.sum() == pytest.approx(1.0), tile
            else:
                assert column.sum() == 0.0, tile
        assert explored, "nothing was explored; the test proves nothing"

    def test_the_engine_still_knows_the_resource_is_there(self):
        """Hiding is a *rendering* rule. The ground truth is untouched, and
        mapgen's two-tile placement rule still holds."""
        _, s = blank(all_tech=False)
        self.setup_tile(s, MID + 1, C.R_CROP)
        assert int(s.resource[MID + 1]) == C.R_CROP


# --- sight from a mountain ------------------------------------------------------------
class TestMountainSight:
    def seen(self, s):
        s.fog_explored[0][:] = 0
        recompute_all(s)
        return int(s.fog_explored[0].sum())

    def test_a_unit_on_a_mountain_sees_five_by_five(self):
        _, s = blank()
        s.terrain[MID] = C.T_MOUNTAIN
        put(s, MID)
        assert self.seen(s) == 25

    def test_the_same_unit_on_a_field_sees_three_by_three(self):
        _, s = blank()
        put(s, MID)
        assert self.seen(s) == 9

    def test_a_scout_gains_nothing_extra_from_the_high_ground(self):
        """Both give radius two, and a unit takes the better of the two rather
        than the sum."""
        _, s = blank()
        s.terrain[MID] = C.T_MOUNTAIN
        put(s, MID, C.U_SCOUT)
        assert self.seen(s) == 25


# --- movement details -------------------------------------------------------------------
class TestMovementRounding:
    def test_half_a_move_still_enters_an_expensive_tile(self):
        """A Warrior has one move. A road step costs half, and the mountain
        beyond costs a full move it no longer has -- but "the game rounds up"."""
        _, s = blank()
        w = put(s, MID)
        for t in (MID, MID + 1, MID + 2):
            s.owner[t] = 0
        s.improvement[MID] = C.IMP_ROAD
        s.improvement[MID + 1] = C.IMP_ROAD
        s.terrain[MID + 2] = C.T_MOUNTAIN
        assert MID + 2 in reachable(s, w, R)

    def test_a_spent_unit_cannot_continue_past_the_tile_it_scraped_into(self):
        """Rounding up buys the step, not a free extra one."""
        _, s = blank()
        w = put(s, MID)
        for t in range(s.n_tiles):
            s.owner[t] = 0
        s.improvement[MID] = C.IMP_ROAD
        s.improvement[MID + 1] = C.IMP_ROAD
        s.terrain[MID + 2] = C.T_MOUNTAIN
        assert MID + 3 not in reachable(s, w, R)


class TestRoadsInEnemyTerritory:
    def line_of_roads(self, s, owner_of_the_road):
        for t in (MID, MID + 1, MID + 2, MID + 3, MID + 4):
            s.improvement[t] = C.IMP_ROAD
        s.owner[MID] = 0
        for t in (MID + 1, MID + 2, MID + 3, MID + 4):
            s.owner[t] = owner_of_the_road
        return put(s, MID, C.U_RIDER)

    def furthest(self, s, src):
        row = [t for t in reachable(s, src, R) if t // N == MID // N and t % N > MID % N]
        return max((t % N for t in row), default=MID % N)

    def test_our_own_roads_carry_a_rider_four_tiles(self):
        _, s = blank()
        src = self.line_of_roads(s, 0)
        assert self.furthest(s, src) == MID % N + 4

    def test_an_enemys_roads_do_not(self):
        """"not when moving onto Roads in enemy territory" -- an invader gets
        plain movement, which is what stops a Rider crossing an empire."""
        _, s = blank()
        src = self.line_of_roads(s, 1)
        assert self.furthest(s, src) == MID % N + 2

    def test_neutral_roads_still_work(self):
        """"in friendly territory or in neutral territory"."""
        _, s = blank()
        src = self.line_of_roads(s, C.NEUTRAL)
        assert self.furthest(s, src) == MID % N + 4

    def test_an_enemy_city_is_not_a_road_node_either(self):
        """A city counts as a road tile, so it has to obey the same ownership
        rule. Measured as a comparison, because the rounding-up rule means the
        slower unit still scrapes onto one more tile — it just gets no further.
        """
        def furthest_across(city_owner):
            _, s = blank()
            for t in (MID, MID + 1):
                s.improvement[t] = C.IMP_ROAD
                s.owner[t] = 0
            s.city_level[MID + 2] = 1
            s.city_owner[MID + 2] = city_owner
            s.owner[MID + 2] = city_owner
            for t in (MID + 3, MID + 4):
                s.improvement[t] = C.IMP_ROAD
                s.owner[t] = city_owner
            src = put(s, MID, C.U_RIDER)
            return self.furthest(s, src)

        assert furthest_across(1) < furthest_across(0)
