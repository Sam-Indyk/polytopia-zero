"""Veteran promotion, the four tile abilities, temples, and the Explorer.

Veterans, quoted from `.wikicache/Units.wiki`:

    "A unit can be promoted to a veteran after it kills three enemy units
     (through attacks or retaliation). Promoting a unit increases its maximum
     health by five and then heals the unit completely. Therefore, a player may
     choose to not immediately promote a unit when it becomes eligible to do so
     because the promotion can be used as a 'free' heal later."

That last sentence is why promotion is an action rather than an automatic
consequence, and why it costs the unit nothing: the project owner confirms the
promotion heal does not spend the turn, which is what makes holding it back a
real play rather than a curiosity.

Tile abilities, each from its own technology page:

    Clear Forest (Forestry)     "removes a forest and gives the player one star"
    Grow Forest (Spiritualism)  "turns a field tile into a forest tile at a cost
                                 of 5 stars"
    Burn Forest (Construction)  "replaces an empty Forest with a field with crop
                                 at a cost of 5 stars"
    Destroy (Chivalry)          "removes a building and its benefits at no cost"

Temples (`.wikicache/Temples.wiki`): four of them, one per terrain, "worth 100
points at time of construction ... with each level, its point value is increased
by 50 points", levelling every two turns to a maximum of five.
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    can_promote,
    can_use_tile_ability,
    enumerate_actions,
)
from polyengine.core.combat import effective_max_hp, resolve_attack
from polyengine.core.economy import can_build, temple_level, temple_score
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11
MID = 5 * N + 5


def blank(seed=1):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.improvement_turn[t] = -1
        s.resource[t] = C.R_NONE
        s.terrain[t] = C.T_FIELD
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.owner[t] = C.NEUTRAL
    for k in range(R.n_techs):
        s.grant_tech(0, k)
        s.grant_tech(1, k)
    s.stars[0] = 50
    s.current_player = 0
    return game, s


def put(s, tile, utype=C.U_WARRIOR, player=0, hp=None, kills=0):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = hp if hp is not None else int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    s.unit_veteran[tile] = 0
    s.unit_kills[tile] = kills
    return tile


# --- veterans ----------------------------------------------------------------------
class TestVeterans:
    def test_three_kills_are_needed(self):
        _, s = blank()
        for kills in range(5):
            tile = put(s, MID, C.U_WARRIOR, kills=kills)
            assert can_promote(s, tile, 0, R) == (kills >= 3), kills

    def test_promotion_adds_five_max_hp(self):
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=3)
        base = effective_max_hp(s, tile, R)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        assert effective_max_hp(s, tile, R) == base + 5

    def test_promotion_heals_completely(self):
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, hp=1, kills=3)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        assert int(s.unit_hp[tile]) == effective_max_hp(s, tile, R)
        assert int(s.unit_hp[tile]) == int(R.unit_max_hp[C.U_WARRIOR]) + 5

    def test_promotion_costs_the_unit_nothing(self):
        """The whole point of holding it back: promote *and* still fight.

        Confirmed by the project owner -- "promote later on ... to regain the
        health without taking up an action".
        """
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, hp=2, kills=3)
        put(s, MID + 1, C.U_WARRIOR, player=1, hp=99)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        assert not s.unit_moved[tile] and not s.unit_attacked[tile]
        assert any(a.atype == C.A_ATTACK and a.src == tile
                   for a in enumerate_actions(s, 0, R))

    def test_a_promotion_can_be_banked_and_spent_later(self):
        """Eligible at full health, still eligible after taking damage."""
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=3)
        assert can_promote(s, tile, 0, R)
        s.unit_hp[tile] = 3                       # took a beating in between
        assert can_promote(s, tile, 0, R)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        assert int(s.unit_hp[tile]) == int(R.unit_max_hp[C.U_WARRIOR]) + 5

    def test_a_unit_promotes_only_once(self):
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=9)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        assert not can_promote(s, tile, 0, R)

    def test_kills_accumulate_from_real_combat(self):
        _, s = blank()
        a = put(s, MID, C.U_SWORDSMAN)
        for tile in (MID + 1, MID - 1, MID + N):
            put(s, tile, C.U_WARRIOR, player=1, hp=1)
            s.unit_attacked[a] = 0
            s.unit_moved[a] = 0
            resolve_attack(s, a, tile, R)
            a = next(t for t in range(s.n_tiles)
                     if int(s.unit_type[t]) == C.U_SWORDSMAN)
        assert int(s.unit_kills[a]) == 3
        assert can_promote(s, a, 0, R)

    def test_a_veteran_really_is_tougher(self):
        _, s = blank()
        plain = put(s, MID, C.U_WARRIOR)
        vet = put(s, MID + 2, C.U_WARRIOR, kills=3)
        apply_action(s, Action(C.A_PROMOTE, vet, vet), R)
        assert effective_max_hp(s, vet, R) > effective_max_hp(s, plain, R)

    def test_it_is_offered_by_the_action_space_and_the_masks(self):
        """Random play almost never reaches three kills on one unit, so this
        action type shows up in no integration run. Both the enumerator and the
        masks are pinned here instead, or a break in either would be invisible.
        """
        from polyengine.masks import ActionMasks
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=3)
        assert any(a.atype == C.A_PROMOTE and a.src == tile
                   for a in enumerate_actions(s, 0, R))
        m = ActionMasks(s, 0, R)
        assert m.types()[C.A_PROMOTE]
        assert m.sources(C.A_PROMOTE)[tile]

    def test_promotion_survives_a_serialization_round_trip(self):
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=3)
        apply_action(s, Action(C.A_PROMOTE, tile, tile), R)
        clone = type(s).from_dict(s.to_dict(), R)
        assert clone.unit_veteran[tile]
        assert effective_max_hp(clone, tile, R) == effective_max_hp(s, tile, R)


# --- tile abilities -----------------------------------------------------------------
class TestTileAbilities:
    def own(self, s, tile, terrain=C.T_FIELD):
        s.owner[tile] = 0
        s.terrain[tile] = terrain
        return tile

    def test_clear_forest_pays_a_star(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        before = int(s.stars[0])
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_CLEAR_FOREST), R)
        assert int(s.terrain[tile]) == C.T_FIELD
        assert int(s.stars[0]) == before + 1

    def test_grow_forest_costs_five(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FIELD)
        before = int(s.stars[0])
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_GROW_FOREST), R)
        assert int(s.terrain[tile]) == C.T_FOREST
        assert int(s.stars[0]) == before - 5

    def test_burn_forest_leaves_a_crop(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        before = int(s.stars[0])
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_BURN_FOREST), R)
        assert int(s.terrain[tile]) == C.T_FIELD
        assert int(s.resource[tile]) == C.R_CROP
        assert int(s.stars[0]) == before - 5

    def test_burn_forest_needs_an_empty_forest(self):
        """"replaces an *empty* Forest with a field with crop"."""
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        s.resource[tile] = C.R_GAME
        assert not can_use_tile_ability(s, tile, C.TA_BURN_FOREST, 0, R)

    def test_destroy_removes_a_building_and_its_population(self):
        _, s = blank()
        city = MID
        s.city_level[city] = 2
        s.city_owner[city] = 0
        s.owner[city] = 0
        tile = self.own(s, MID + 1, C.T_FOREST)
        apply_action(s, Action(C.A_BUILD, tile, -1, C.IMP_LUMBER_HUT), R)
        pop = int(s.city_pop[city])
        assert int(s.improvement[tile]) == C.IMP_LUMBER_HUT
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_DESTROY), R)
        assert int(s.improvement[tile]) == C.IMP_NONE
        assert int(s.city_pop[city]) < pop, "the population it paid goes too"

    def test_destroy_is_free(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        apply_action(s, Action(C.A_BUILD, tile, -1, C.IMP_LUMBER_HUT), R)
        before = int(s.stars[0])
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_DESTROY), R)
        assert int(s.stars[0]) == before

    def test_ruins_and_lighthouses_are_not_buildings_to_destroy(self):
        _, s = blank()
        for feature in (C.IMP_RUIN, C.IMP_LIGHTHOUSE):
            tile = self.own(s, MID, C.T_FIELD)
            s.improvement[tile] = feature
            assert not can_use_tile_ability(s, tile, C.TA_DESTROY, 0, R)

    @pytest.mark.parametrize("ability,tech", sorted(C.TILE_ABILITY_TECH.items()))
    def test_each_ability_needs_its_technology(self, ability, tech):
        _, s = blank()
        s.tech[0] = 0                                  # forget everything
        tile = self.own(s, MID, C.T_FOREST)
        s.improvement[tile] = (C.IMP_LUMBER_HUT if ability == C.TA_DESTROY
                               else C.IMP_NONE)
        assert not can_use_tile_ability(s, tile, ability, 0, R)
        s.grant_tech(0, R.tech_id[tech])
        if ability == C.TA_GROW_FOREST:
            s.terrain[tile] = C.T_FIELD
        assert can_use_tile_ability(s, tile, ability, 0, R)

    def test_abilities_are_confined_to_our_own_territory(self):
        _, s = blank()
        tile = MID
        s.terrain[tile] = C.T_FOREST
        s.owner[tile] = 1
        assert not can_use_tile_ability(s, tile, C.TA_CLEAR_FOREST, 0, R)
        s.owner[tile] = C.NEUTRAL
        assert not can_use_tile_ability(s, tile, C.TA_CLEAR_FOREST, 0, R)
        s.owner[tile] = 0
        assert can_use_tile_ability(s, tile, C.TA_CLEAR_FOREST, 0, R)

    def test_a_city_tile_is_never_reshaped(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        s.city_level[tile] = 1
        s.city_owner[tile] = 0
        assert not can_use_tile_ability(s, tile, C.TA_CLEAR_FOREST, 0, R)

    def test_grow_and_burn_need_the_stars(self):
        _, s = blank()
        s.stars[0] = 4
        tile = self.own(s, MID, C.T_FIELD)
        assert not can_use_tile_ability(s, tile, C.TA_GROW_FOREST, 0, R)
        s.stars[0] = 5
        assert can_use_tile_ability(s, tile, C.TA_GROW_FOREST, 0, R)

    def test_the_abilities_appear_in_the_action_space(self):
        _, s = blank()
        tile = self.own(s, MID, C.T_FOREST)
        offered = {a.subtype for a in enumerate_actions(s, 0, R)
                   if a.atype == C.A_TILE_ABILITY and a.src == tile}
        assert C.TA_CLEAR_FOREST in offered
        assert C.TA_BURN_FOREST in offered

    def test_grow_then_burn_is_a_round_trip_to_a_crop(self):
        """The documented combo: 5 stars to grow, 5 to burn, and a crop field
        where there was bare ground."""
        _, s = blank()
        tile = self.own(s, MID, C.T_FIELD)
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_GROW_FOREST), R)
        apply_action(s, Action(C.A_TILE_ABILITY, tile, -1, C.TA_BURN_FOREST), R)
        assert int(s.terrain[tile]) == C.T_FIELD
        assert int(s.resource[tile]) == C.R_CROP


# --- temples ------------------------------------------------------------------------
class TestTemples:
    def test_there_are_four_of_them(self):
        assert len(C.TEMPLES) == 4
        for imp in C.TEMPLES:
            assert R.imp_names[imp].endswith("temple")

    @pytest.mark.parametrize("imp,terrain,tech,cost", [
        (C.IMP_TEMPLE, C.T_FIELD, "free_spirit", 20),
        (C.IMP_MOUNTAIN_TEMPLE, C.T_MOUNTAIN, "meditation", 20),
        (C.IMP_FOREST_TEMPLE, C.T_FOREST, "spiritualism", 15),
        (C.IMP_WATER_TEMPLE, C.T_WATER, "aquatism", 20),
    ])
    def test_each_temple_has_its_terrain_tech_and_cost(self, imp, terrain, tech,
                                                       cost):
        """"All types of temples cost 20 stars, except for the Forest Temple,
        which costs only 15 stars."""
        assert int(R.imp_cost[imp]) == cost
        _, s = blank()
        s.tech[0] = 0
        s.owner[MID] = 0
        s.terrain[MID] = terrain
        s.city_level[MID + 3] = 1
        s.city_owner[MID + 3] = 0
        assert not can_build(s, MID, imp, 0, R), "buildable without the tech"
        s.grant_tech(0, R.tech_id[tech])
        assert can_build(s, MID, imp, 0, R)

    def test_a_temple_will_not_go_on_the_wrong_terrain(self):
        _, s = blank()
        s.owner[MID] = 0
        s.terrain[MID] = C.T_MOUNTAIN
        s.city_level[MID + 3] = 1
        s.city_owner[MID + 3] = 0
        assert not can_build(s, MID, C.IMP_TEMPLE, 0, R)
        assert can_build(s, MID, C.IMP_MOUNTAIN_TEMPLE, 0, R)

    def test_spiritualism_and_meditation_now_unlock_something(self):
        """Before the temples landed, Spiritualism unlocked nothing at all."""
        for tech in ("spiritualism", "meditation", "aquatism"):
            tid = R.tech_id[tech]
            assert any(int(R.imp_requires_tech[i]) == tid
                       for i in range(R.n_improvements)), tech

    def test_a_temple_levels_up_every_two_turns(self):
        """The wiki's table: level 1 at 0-2 turns, 2 at 3-5, 3 at 6-8, 4 at
        9-11, 5 at 12 or more."""
        _, s = blank()
        s.improvement[MID] = C.IMP_TEMPLE
        s.improvement_turn[MID] = 0
        expected = {0: 1, 2: 1, 3: 2, 5: 2, 6: 3, 8: 3, 9: 4, 11: 4, 12: 5,
                    30: 5}
        for turn, level in expected.items():
            s.turn = turn
            assert temple_level(s, MID) == level, f"turn {turn}"

    def test_score_is_a_hundred_plus_fifty_a_level(self):
        _, s = blank()
        s.improvement[MID] = C.IMP_TEMPLE
        s.improvement_turn[MID] = 0
        for turn, score in {0: 100, 3: 150, 6: 200, 9: 250, 12: 300,
                            40: 300}.items():
            s.turn = turn
            assert temple_score(s, MID) == score, f"turn {turn}"

    def test_the_game_score_counts_an_ageing_temple(self):
        game, s = blank()
        s.city_level[MID] = 1
        s.city_owner[MID] = 0
        s.owner[MID] = 0
        s.owner[MID + 1] = 0
        apply_action(s, Action(C.A_BUILD, MID + 1, -1, C.IMP_TEMPLE), R)
        s.turn = 0
        young = game.score(0)
        s.turn = 12
        assert game.score(0) == young + 200, "four levels of growth"

    def test_a_temple_is_not_double_counted(self):
        """It scores dynamically, so it must be excluded from the flat table."""
        game, s = blank()
        s.city_level[MID] = 1
        s.city_owner[MID] = 0
        s.owner[MID] = 0
        s.owner[MID + 1] = 0
        before = game.score(0)
        apply_action(s, Action(C.A_BUILD, MID + 1, -1, C.IMP_TEMPLE), R)
        s.turn = 0
        gained = game.score(0) - before
        # 100 for the temple itself, plus whatever the population bought.
        assert 100 <= gained <= 260, gained


# --- the Explorer ---------------------------------------------------------------------
class TestExplorer:
    def test_it_walks_and_reveals_far_more_than_it_stands_on(self):
        from polyengine.core.economy import _explore
        _, s = blank()
        s.fog_explored[0][:] = 0
        path = _explore(s, MID, 0)
        assert len(path) > 1, "an Explorer that never moves is not an Explorer"
        assert int(s.fog_explored[0].sum()) > 9 * 2

    def test_it_takes_fifteen_steps_at_most(self):
        from polyengine.core.economy import EXPLORER_MOVES, _explore
        _, s = blank()
        s.fog_explored[0][:] = 0
        path = _explore(s, MID, 0)
        assert len(path) <= EXPLORER_MOVES + 1

    def test_it_will_not_cross_terrain_it_lacks_the_tech_for(self):
        """"Explorers can only move onto mountain, shallow water, or ocean tiles
        if the appropriate technology has been researched."""
        from polyengine.core.economy import _explore
        _, s = blank()
        s.tech[0] = 0                                   # no Climbing, no Sailing
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_MOUNTAIN
        s.terrain[MID] = C.T_FIELD
        s.fog_explored[0][:] = 0
        path = _explore(s, MID, 0)
        assert path == [MID], "it should be stuck on its one legal tile"

    def test_the_reward_sends_one_out(self):
        from polyengine.core.economy import apply_reward
        _, s = blank()
        city = MID
        s.city_level[city] = 2
        s.city_owner[city] = 0
        s.owner[city] = 0
        s.city_pending_reward[city] = 2
        s.fog_explored[0][:] = 0
        rw = next(i for i in range(R.n_rewards)
                  if R.reward_effect[i] == "explore")
        apply_reward(s, city, rw, 0, R)
        assert int(s.fog_explored[0].sum()) > 9
