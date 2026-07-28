"""Healing, Diplomacy, Disband, naval upgrades and the Cloak.

Wiki sources quoted inline. The Cloak block is the one that touches invariant 1:
a hidden Cloak must be absent from the enemy's observation, and *present* again
the moment one of their units stands next to it.
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    can_disband,
    upgrade_options,
)
from polyengine.core.combat import heal_amount, recover_idle_units
from polyengine.core.economy import can_build, connected_cities, player_income
from polyengine.core.fog import recompute_fog
from polyengine.core.game import Game
from polyengine.core.movement import infiltrate_targets
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11


def blank(seed=7, owner=0):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.unit_hidden[t] = 0
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.city_conn_pop[t] = 0
        s.terrain[t] = C.T_FIELD
        s.owner[t] = owner
    s.current_player = 0
    return game, s


def put(s, tile, utype, owner=0, hp=None):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = owner
    s.unit_hp[tile] = int(R.unit_max_hp[utype]) if hp is None else hp


# --- healing ------------------------------------------------------------------
class TestHealing:
    """"Heals a unit by up to 4 HP in friendly territory and 2 HP elsewhere."""

    def test_heals_more_in_friendly_territory(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_WARRIOR, hp=3)
        s.owner[tile] = 0
        assert heal_amount(s, tile, R) == 4
        s.owner[tile] = 1
        assert heal_amount(s, tile, R) == 2

    def test_never_heals_past_maximum(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_WARRIOR, hp=int(R.unit_max_hp[C.U_WARRIOR]) - 1)
        assert heal_amount(s, tile, R) == 1

    def test_a_full_health_unit_heals_nothing(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_WARRIOR)
        assert heal_amount(s, tile, R) == 0

    def test_idle_units_recover_but_busy_ones_do_not(self):
        """"If a unit does not take any action ... it will automatically heal"."""
        _, s = blank()
        idle, busy = 5 * N + 5, 5 * N + 7
        put(s, idle, C.U_WARRIOR, hp=3)
        put(s, busy, C.U_WARRIOR, hp=3)
        s.unit_moved[busy] = 1
        recover_idle_units(s, 0, R)
        assert int(s.unit_hp[idle]) == 7
        assert int(s.unit_hp[busy]) == 3

    def test_a_hull_heals_towards_its_passengers_maximum(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_RAFT, hp=2)
        s.unit_carried[tile] = C.U_DEFENDER
        s.owner[tile] = 0
        assert heal_amount(s, tile, R) == 4


# --- disband ------------------------------------------------------------------
class TestDisband:
    """"Removes a unit and refunds half its cost, rounded down.  Available ...
    once Free Spirit has been researched."""

    def test_requires_free_spirit(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_WARRIOR)
        assert not can_disband(s, tile, 0, R)
        s.grant_tech(0, R.tech_id["free_spirit"])
        assert can_disband(s, tile, 0, R)

    def test_refunds_half_the_cost_rounded_down(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_WARRIOR)
        s.grant_tech(0, R.tech_id["free_spirit"])
        s.stars[0] = 0
        apply_action(s, Action(C.A_DISBAND, tile, tile), R)
        assert int(s.stars[0]) == int(R.unit_cost[C.U_WARRIOR]) // 2
        assert s.unit_type[tile] == C.U_NONE

    def test_a_hull_refunds_its_passenger_not_the_boat(self):
        _, s = blank()
        tile = 5 * N + 5
        put(s, tile, C.U_RAFT)
        s.unit_carried[tile] = C.U_DEFENDER
        s.grant_tech(0, R.tech_id["free_spirit"])
        s.stars[0] = 0
        apply_action(s, Action(C.A_DISBAND, tile, tile), R)
        assert int(s.stars[0]) == int(R.unit_cost[C.U_DEFENDER]) // 2


# --- naval upgrades -----------------------------------------------------------
class TestNavalUpgrades:
    """"A Raft can be upgraded into a Scout for five stars, once Sailing is
    researched. Upgrading a naval unit does not heal it."""

    def _raft(self):
        _, s = blank()
        tile = 5 * N + 5
        s.terrain[tile] = C.T_WATER
        put(s, tile, C.U_RAFT, hp=4)
        s.unit_carried[tile] = C.U_WARRIOR
        s.stars[0] = 50
        return s, tile

    def test_upgrade_needs_the_technology(self):
        s, tile = self._raft()
        assert C.U_SCOUT not in upgrade_options(s, tile, 0, R)
        s.grant_tech(0, R.tech_id["sailing"])
        assert C.U_SCOUT in upgrade_options(s, tile, 0, R)

    def test_upgrade_needs_the_stars(self):
        s, tile = self._raft()
        s.grant_tech(0, R.tech_id["sailing"])
        s.stars[0] = 0
        assert C.U_SCOUT not in upgrade_options(s, tile, 0, R)

    def test_upgrading_does_not_heal(self):
        s, tile = self._raft()
        s.grant_tech(0, R.tech_id["sailing"])
        apply_action(s, Action(C.A_UPGRADE, tile, -1, C.U_SCOUT), R)
        assert int(s.unit_type[tile]) == C.U_SCOUT
        assert int(s.unit_hp[tile]) == 4, "HP is the passenger's and must not change"

    def test_only_a_raft_upgrades(self):
        s, tile = self._raft()
        s.grant_tech(0, R.tech_id["sailing"])
        s.unit_type[tile] = C.U_SCOUT
        assert upgrade_options(s, tile, 0, R) == []

    def test_landing_drops_the_upgrade(self):
        """"Moving an upgraded naval unit onto land causes the unit to
        permanently lose its upgrade."""
        s, tile = self._raft()
        s.grant_tech(0, R.tech_id["sailing"])
        apply_action(s, Action(C.A_UPGRADE, tile, -1, C.U_SCOUT), R)
        shore = tile + 1
        s.terrain[shore] = C.T_FIELD
        apply_action(s, Action(C.A_MOVE, tile, shore), R)
        assert int(s.unit_type[shore]) == C.U_WARRIOR
        assert int(s.unit_carried[shore]) == C.U_NONE


# --- diplomacy ----------------------------------------------------------------
class TestDiplomacy:
    def _capitals(self):
        _, s = blank()
        mine, theirs = 5 * N + 2, 5 * N + 8
        for tile, owner in ((mine, 0), (theirs, 1)):
            s.city_owner[tile] = owner
            s.city_level[tile] = 1
            s.city_capital[tile] = 1
            s.owner[tile] = owner
        s.stars[0] = 50
        return s, mine, theirs

    def test_embassy_goes_only_on_a_rival_capital(self):
        s, mine, theirs = self._capitals()
        s.grant_tech(0, R.tech_id["diplomacy"])
        s.fog_explored[0][theirs] = 1
        assert can_build(s, theirs, C.IMP_EMBASSY, 0, R)
        assert not can_build(s, mine, C.IMP_EMBASSY, 0, R), "not our own capital"

    def test_embassy_needs_diplomacy(self):
        s, _, theirs = self._capitals()
        s.fog_explored[0][theirs] = 1
        assert not can_build(s, theirs, C.IMP_EMBASSY, 0, R)

    def test_embassy_pays_both_tribes(self):
        """"Provides income to both the tribe that builds and the tribe that
        hosts the Embassy."""
        s, mine, theirs = self._capitals()
        before_builder = player_income(s, 0, R)
        before_host = player_income(s, 1, R)
        s.improvement[theirs] = C.IMP_EMBASSY
        s.improvement_owner[theirs] = 0
        flat = int(R.imp_income_flat[C.IMP_EMBASSY])
        assert player_income(s, 0, R) == before_builder + flat
        assert player_income(s, 1, R) == before_host + flat

    def test_capital_vision_reveals_rival_capitals(self):
        """"allows the player to see the capitals of tribes they have discovered"."""
        s, mine, theirs = self._capitals()
        s.mem_city_owner[0][theirs] = C.NEUTRAL
        s.fog_explored[0][theirs] = 0
        recompute_fog(s, 0)
        assert not s.fog_explored[0][theirs]
        s.grant_tech(0, R.tech_id["diplomacy"])
        recompute_fog(s, 0)
        assert s.fog_explored[0][theirs]
        assert int(s.mem_city_owner[0][theirs]) == 1

    def test_capital_vision_does_not_reveal_units(self):
        """Knowing where a capital is must not become sight of what stands on it."""
        s, mine, theirs = self._capitals()
        put(s, theirs, C.U_WARRIOR, owner=1)
        s.grant_tech(0, R.tech_id["diplomacy"])
        recompute_fog(s, 0)
        assert int(s.mem_unit_type[0][theirs]) == C.U_NONE


# --- the cloak ----------------------------------------------------------------
class TestCloak:
    def _scene(self, city_level=3, defender=True):
        _, s = blank(owner=1)
        city, cloak = 5 * N + 5, 5 * N + 4
        s.city_owner[city] = 1
        s.city_level[city] = city_level
        put(s, cloak, C.U_CLOAK, owner=0)
        if defender:
            put(s, city, C.U_WARRIOR, owner=1)
        return s, city, cloak

    def test_a_cloak_can_only_target_adjacent_enemy_cities(self):
        """"it can only target adjacent enemy cities with its attack"."""
        s, city, cloak = self._scene()
        assert infiltrate_targets(s, cloak, R) == [city]
        s.city_owner[city] = 0                        # now ours
        assert infiltrate_targets(s, cloak, R) == []

    def test_a_besieged_city_cannot_be_infiltrated(self):
        """"Cloaks cannot infiltrate ... a city under siege."""
        s, city, cloak = self._scene(defender=False)
        put(s, city, C.U_WARRIOR, owner=0)            # our unit besieges it
        assert infiltrate_targets(s, cloak, R) == []

    def test_infiltration_consumes_the_cloak(self):
        s, city, cloak = self._scene()
        apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert s.unit_type[cloak] == C.U_NONE

    def test_infiltration_damages_the_garrison_by_the_cloaks_attack(self):
        """"damage ... equivalent to its attack statistic (2) at full health
        even if the cloak is damaged."""
        s, city, cloak = self._scene()
        s.unit_hp[cloak] = 1                          # wounded: damage unchanged
        before = int(s.unit_hp[city])
        apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert int(s.unit_hp[city]) == before - int(R.unit_attack[C.U_CLOAK])

    def test_infiltration_steals_the_citys_income(self):
        from polyengine.core.economy import city_income
        s, city, cloak = self._scene()
        expected = city_income(s, city)
        s.stars[0] = 0
        apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert int(s.stars[0]) == expected

    @pytest.mark.parametrize("level,expected", [(1, 1), (3, 3), (7, 5)])
    def test_dagger_count_is_city_level_capped_at_five(self, level, expected):
        s, city, cloak = self._scene(city_level=level, defender=False)
        info = apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert len(info["daggers"]) == expected

    def test_daggers_cannot_act_on_the_turn_they_spawn(self):
        """"They will not be able to perform any actions until their owner's
        next turn."""
        s, city, cloak = self._scene(defender=False)
        info = apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        for tile in info["daggers"]:
            assert s.unit_moved[tile] and s.unit_attacked[tile]
            assert int(s.unit_owner[tile]) == 0

    def test_daggers_are_independent_of_any_city(self):
        s, city, cloak = self._scene(defender=False)
        info = apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert all(int(s.unit_home[t]) == -1 for t in info["daggers"])

    def test_daggers_prefer_the_city_tile(self):
        s, city, cloak = self._scene(defender=False)
        info = apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert city in info["daggers"]

    # --- hiding, and the fog contract ---------------------------------------
    def test_a_moved_cloak_is_hidden_from_enemies(self):
        _, s = blank()
        start, dst = 5 * N + 3, 5 * N + 4
        put(s, start, C.U_CLOAK, owner=0)
        apply_action(s, Action(C.A_MOVE, start, dst), R)
        assert s.unit_hidden[dst]
        recompute_fog(s, 1)
        assert int(s.mem_unit_type[1][dst]) == C.U_NONE, "invariant 1"

    def test_an_adjacent_enemy_detects_a_hidden_cloak(self):
        """"all units can detect Cloaks on adjacent tiles"."""
        _, s = blank()
        start, dst = 5 * N + 3, 5 * N + 4
        put(s, start, C.U_CLOAK, owner=0)
        put(s, dst + 1, C.U_WARRIOR, owner=1)
        apply_action(s, Action(C.A_MOVE, start, dst), R)
        recompute_fog(s, 1)
        assert int(s.mem_unit_type[1][dst]) == C.U_CLOAK

    def test_hiding_does_not_blind_the_cloaks_owner(self):
        _, s = blank()
        start, dst = 5 * N + 3, 5 * N + 4
        put(s, start, C.U_CLOAK, owner=0)
        apply_action(s, Action(C.A_MOVE, start, dst), R)
        recompute_fog(s, 0)
        assert int(s.mem_unit_type[0][dst]) == C.U_CLOAK

    def test_creep_lets_a_cloak_ignore_zone_of_control(self):
        from polyengine.core.movement import reachable
        _, s = blank()
        lane = [5 * N + c for c in range(1, 8)]
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_OCEAN
        for t in lane:
            s.terrain[t] = C.T_FIELD
        put(s, lane[0], C.U_CLOAK, owner=0)
        flank = lane[1] + N
        s.terrain[flank] = C.T_FIELD
        put(s, flank, C.U_WARRIOR, owner=1)
        assert len([t for t in reachable(s, lane[0], R) if t in lane]) >= 2


# --- the empty-tile indexing trap ----------------------------------------------
def test_an_empty_tile_earns_nothing():
    """IMP_NONE is -1, and -1 indexes the LAST improvement in every rules array.

    That briefly paid out the Embassy's income on every empty tile on the map,
    which is exactly the kind of bug that looks like "the economy is a bit
    generous" rather than like an error.
    """
    from polyengine.core.economy import market_income
    _, s = blank()
    empty = 5 * N + 5
    assert int(s.improvement[empty]) == C.IMP_NONE
    assert market_income(s, empty, R) == 0

    lone = 5 * N + 2
    s.city_owner[lone] = 0
    s.city_level[lone] = 1
    s.city_capital[lone] = 1
    # One level-1 capital: 1 for the level, 1 for being the capital. Nothing else
    # on an otherwise bare board.
    assert player_income(s, 0, R) == 2
