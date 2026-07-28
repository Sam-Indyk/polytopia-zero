"""Tasks, monuments, lighthouses and Pirates.

Task conditions are quoted from the wiki's monument table:

    Altar of Peace   Pacifist:   Make no attacks for five turns.
    Emperor's Tomb   Wealth:     Gather 100 stars.
    Eye of God       Explorer:   Find all lighthouses.
    Gate of Power    Killer:     Kill 10 enemy units.
    Grand Bazaar     Trade:      Connect five cities to the capital.
    Park of Fortune  Metropolis: Have a level 5+ city.
    Tower of Wisdom  Genius:     Research all technologies.

"Each monument produces 3 population and is worth 400 points."
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import Action, apply_action
from polyengine.core.economy import (
    can_build,
    lighthouse_tiles,
    monument_available,
    task_satisfied,
    update_tasks,
)
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11


def blank(seed=7):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.city_conn_pop[t] = 0
        s.terrain[t] = C.T_FIELD
        s.owner[t] = 0
    for p in range(s.n_players):
        s.task_done[p] = 0
        s.monument_built[p] = 0
        s.kills[p] = 0
        s.calm_turns[p] = 0
    s.current_player = 0
    return game, s


# --- lighthouses --------------------------------------------------------------
def test_every_map_has_four_corner_lighthouses():
    game = Game.new(size=N, seed=3)
    assert len(lighthouse_tiles(game.state)) == 4


def test_lighthouses_sit_in_the_corners():
    game = Game.new(size=N, seed=3)
    assert set(lighthouse_tiles(game.state)) == {0, N - 1, N * (N - 1), N * N - 1}


def test_a_lighthouse_cannot_be_built_by_a_player():
    """It is a map feature, not an improvement anyone places."""
    _, s = blank()
    tile = 5 * N + 5
    s.terrain[tile] = C.T_WATER
    s.stars[0] = 99
    assert not can_build(s, tile, C.IMP_LIGHTHOUSE, 0, R)


# --- task conditions ----------------------------------------------------------
class TestTaskConditions:
    def test_wealth_needs_a_hundred_stars(self):
        _, s = blank()
        s.stars[0] = C.WEALTH_STARS - 1
        assert not task_satisfied(s, 0, C.TASK_WEALTH, R)
        s.stars[0] = C.WEALTH_STARS
        assert task_satisfied(s, 0, C.TASK_WEALTH, R)

    def test_killer_needs_ten_kills(self):
        _, s = blank()
        s.kills[0] = C.KILLER_KILLS - 1
        assert not task_satisfied(s, 0, C.TASK_KILLER, R)
        s.kills[0] = C.KILLER_KILLS
        assert task_satisfied(s, 0, C.TASK_KILLER, R)

    def test_metropolis_needs_a_level_five_city(self):
        _, s = blank()
        tile = 5 * N + 5
        s.city_owner[tile] = 0
        s.city_level[tile] = C.METROPOLIS_LEVEL - 1
        assert not task_satisfied(s, 0, C.TASK_METROPOLIS, R)
        s.city_level[tile] = C.METROPOLIS_LEVEL
        assert task_satisfied(s, 0, C.TASK_METROPOLIS, R)

    def test_genius_needs_every_technology(self):
        _, s = blank()
        for t in range(R.n_techs - 1):
            s.grant_tech(0, t)
        assert not task_satisfied(s, 0, C.TASK_GENIUS, R)
        s.grant_tech(0, R.n_techs - 1)
        assert task_satisfied(s, 0, C.TASK_GENIUS, R)

    def test_pacifist_needs_five_quiet_turns(self):
        _, s = blank()
        s.calm_turns[0] = C.PACIFIST_TURNS - 1
        assert not task_satisfied(s, 0, C.TASK_PACIFIST, R)
        s.calm_turns[0] = C.PACIFIST_TURNS
        assert task_satisfied(s, 0, C.TASK_PACIFIST, R)

    def test_explorer_needs_all_four_lighthouses(self):
        game = Game.new(size=N, seed=3)
        s = game.state
        houses = lighthouse_tiles(s)
        for t in houses:
            s.fog_explored[0][t] = 0
        assert not task_satisfied(s, 0, C.TASK_EXPLORER, R)
        for t in houses[:-1]:
            s.fog_explored[0][t] = 1
        assert not task_satisfied(s, 0, C.TASK_EXPLORER, R), "three of four is not all"
        s.fog_explored[0][houses[-1]] = 1
        assert task_satisfied(s, 0, C.TASK_EXPLORER, R)

    def test_network_needs_five_connected_cities(self):
        _, s = blank()
        cap = 4 * N + 1
        s.city_owner[cap] = 0
        s.city_level[cap] = 1
        s.city_capital[cap] = 1
        assert not task_satisfied(s, 0, C.TASK_NETWORK, R)
        # Five cities in a row, joined by roads.
        for k in range(1, 6):
            tile = cap + 2 * k
            s.city_owner[tile] = 0
            s.city_level[tile] = 1
            for road in range(cap + 2 * (k - 1) + 1, tile):
                s.improvement[road] = C.IMP_ROAD
        assert task_satisfied(s, 0, C.TASK_NETWORK, R)


# --- latching -----------------------------------------------------------------
class TestTaskLatching:
    def test_update_tasks_reports_only_newly_earned(self):
        _, s = blank()
        s.stars[0] = C.WEALTH_STARS
        assert update_tasks(s, 0, R) == [C.TASK_WEALTH]
        assert update_tasks(s, 0, R) == [], "already latched"

    def test_a_completed_task_is_not_taken_back(self):
        _, s = blank()
        s.stars[0] = C.WEALTH_STARS
        update_tasks(s, 0, R)
        s.stars[0] = 0
        assert monument_available(s, 0, C.IMP_EMPERORS_TOMB, R)

    def test_tasks_are_per_player(self):
        _, s = blank()
        s.stars[0] = C.WEALTH_STARS
        update_tasks(s, 0, R)
        update_tasks(s, 1, R)
        assert monument_available(s, 0, C.IMP_EMPERORS_TOMB, R)
        assert not monument_available(s, 1, C.IMP_EMPERORS_TOMB, R)


# --- monuments ----------------------------------------------------------------
class TestMonuments:
    def _earned(self):
        _, s = blank()
        s.stars[0] = C.WEALTH_STARS
        update_tasks(s, 0, R)
        s.grant_tech(0, R.tech_id["trade"])
        return s, 5 * N + 5

    def test_a_monument_needs_its_task(self):
        _, s = blank()
        s.grant_tech(0, R.tech_id["trade"])
        s.stars[0] = 99
        assert not can_build(s, 5 * N + 5, C.IMP_EMPERORS_TOMB, 0, R)

    def test_a_monument_is_free(self):
        s, tile = self._earned()
        s.stars[0] = 0
        assert can_build(s, tile, C.IMP_EMPERORS_TOMB, 0, R)

    def test_a_monument_yields_three_population(self):
        assert int(R.imp_population[C.IMP_EMPERORS_TOMB]) == 3

    def test_a_monument_is_worth_four_hundred_points(self):
        assert int(R.imp_score[C.IMP_EMPERORS_TOMB]) == C.MONUMENT_SCORE

    def test_only_one_monument_per_task(self):
        s, tile = self._earned()
        apply_action(s, Action(C.A_BUILD, tile, -1, C.IMP_EMPERORS_TOMB), R)
        assert not monument_available(s, 0, C.IMP_EMPERORS_TOMB, R)
        assert not can_build(s, tile + 2, C.IMP_EMPERORS_TOMB, 0, R)

    def test_monuments_score_for_their_builder(self):
        game, s = blank()
        s.stars[0] = C.WEALTH_STARS
        update_tasks(s, 0, R)
        s.grant_tech(0, R.tech_id["trade"])
        before = game.score(0)
        apply_action(s, Action(C.A_BUILD, 5 * N + 5, -1, C.IMP_EMPERORS_TOMB), R)
        assert game.score(0) - before >= C.MONUMENT_SCORE

    @pytest.mark.parametrize("terrain,ok", [
        (C.T_FIELD, True), (C.T_WATER, True),
        (C.T_FOREST, False), (C.T_MOUNTAIN, False),
    ])
    def test_monuments_go_on_field_or_shallow_water(self, terrain, ok):
        """"Monuments can be built on field and shallow water terrain."""
        s, tile = self._earned()
        s.terrain[tile] = terrain
        s.grant_tech(0, R.tech_id["fishing"])
        s.grant_tech(0, R.tech_id["climbing"])
        assert can_build(s, tile, C.IMP_EMPERORS_TOMB, 0, R) is ok


# --- pirates ------------------------------------------------------------------
class TestPirates:
    def test_a_dagger_moving_onto_a_port_becomes_a_pirate(self):
        """"It is created when a Dagger moves onto or is spawned on a Port"."""
        _, s = blank()
        land, port = 5 * N + 4, 5 * N + 5
        s.terrain[port] = C.T_WATER
        s.improvement[port] = C.IMP_PORT
        s.unit_type[land] = C.U_DAGGER
        s.unit_owner[land] = 0
        s.unit_hp[land] = 10
        s.grant_tech(0, R.tech_id["fishing"])
        apply_action(s, Action(C.A_MOVE, land, port), R)
        assert int(s.unit_type[port]) == C.U_PIRATE
        assert int(s.unit_carried[port]) == C.U_NONE, "a Pirate is not a hull"

    def test_a_pirate_reverts_to_a_dagger_on_land(self):
        """"will revert to a Dagger if it chooses to land"."""
        _, s = blank()
        sea, shore = 5 * N + 5, 5 * N + 6
        s.terrain[sea] = C.T_WATER
        s.unit_type[sea] = C.U_PIRATE
        s.unit_owner[sea] = 0
        s.unit_hp[sea] = 10
        s.grant_tech(0, R.tech_id["fishing"])
        apply_action(s, Action(C.A_MOVE, sea, shore), R)
        assert int(s.unit_type[shore]) == C.U_DAGGER

    def test_a_pirate_moves_one_further_than_a_dagger(self):
        """"they can move 2 tiles instead of the usual 1"."""
        assert int(R.unit_movement[C.U_PIRATE]) == int(R.unit_movement[C.U_DAGGER]) + 1

    def test_a_pirate_keeps_the_daggers_combat_stats(self):
        for stat in ("unit_attack", "unit_defense", "unit_max_hp"):
            assert getattr(R, stat)[C.U_PIRATE] == getattr(R, stat)[C.U_DAGGER]

    def test_a_pirate_keeps_surprise(self):
        assert "surprise" in R.unit_skills[C.U_PIRATE]

    def test_a_pirate_cannot_be_upgraded(self):
        """"Unlike a Raft, the Pirate cannot be upgraded into other naval units."""
        from polyengine.core.actions import upgrade_options
        _, s = blank()
        tile = 5 * N + 5
        s.terrain[tile] = C.T_WATER
        s.unit_type[tile] = C.U_PIRATE
        s.unit_owner[tile] = 0
        s.unit_hp[tile] = 10
        s.stars[0] = 99
        s.grant_tech(0, R.tech_id["sailing"])
        assert upgrade_options(s, tile, 0, R) == []

    def test_daggers_spawn_on_land_before_water(self):
        """"A Dagger can spawn on water if there are no empty land tiles left."""
        _, s = blank()
        city, cloak = 5 * N + 5, 5 * N + 4
        s.city_owner[city] = 1
        s.city_level[city] = 2
        for t in range(s.n_tiles):
            s.owner[t] = 1
        s.unit_type[cloak] = C.U_CLOAK
        s.unit_owner[cloak] = 0
        s.unit_hp[cloak] = 10
        s.grant_tech(0, R.tech_id["fishing"])
        info = apply_action(s, Action(C.A_ATTACK, cloak, city), R)
        assert all(int(s.unit_type[t]) == C.U_DAGGER for t in info["daggers"]), \
            "land was available, so no Pirates"


# --- the market correction ----------------------------------------------------
def test_market_pays_one_star_per_adjacent_building_level():
    """Resources says 2, Buildings says 1. 1 is correct."""
    from polyengine.core.economy import market_income
    _, s = blank()
    centre = 5 * N + 5
    s.improvement[centre] = C.IMP_MARKET
    nbs = list(s.neighbors(centre))
    s.improvement[nbs[0]] = C.IMP_SAWMILL
    assert market_income(s, centre, R) == 1
    s.improvement[nbs[1]] = C.IMP_WINDMILL
    assert market_income(s, centre, R) == 2
