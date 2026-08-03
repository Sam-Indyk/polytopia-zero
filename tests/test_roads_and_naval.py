"""Roads, zone of control, city connections, markets and naval embarking.

The engine had none of these. Each assertion below quotes the wiki rule it
encodes, because every one of them changes what the agent is being taught:

* Movement:    "Moving from one tile to an adjacent tile incurs a cost of 1, but
                moving between tiles with roads has a cost of 0.5."
* Movement:    "Rough terrain, including mountains and forests (without roads)
                block movement... they can move onto them, but no further."
* Movement:    "A unit that moves to a tile adjacent to an enemy unit cannot
                move any further, even if the unit had excess movement points."
* Cities:      "each city connection adds one population to both the capital and
                the connected city"
* Resources:   "Market | 2 per each level of adjacent Windmill, Sawmill, or
                Forge, max. level 8"
* Movement:    "Most land units moving into a Port will be turned into Rafts and
                will be unable to perform actions for the rest of that turn"
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import Action, apply_action
from polyengine.core.combat import effective_max_hp
from polyengine.core.economy import (
    can_build,
    connected_cities,
    market_income,
    recompute_connections,
    remove_population,
)
from polyengine.core.game import Game
from polyengine.core.movement import reachable
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11


def blank(seed=7):
    """An 11x11 board wiped to plain owned field, so tests control everything."""
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.improvement[t] = C.IMP_NONE
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.city_pop[t] = 0
        s.city_conn_pop[t] = 0
        s.terrain[t] = C.T_FIELD
        s.owner[t] = 0
    s.current_player = 0
    return game, s


def corridor(s, row=5, cols=range(1, 10)):
    """Make one row the only passable path, by drowning everything else."""
    lane = [row * N + c for c in cols]
    for t in range(s.n_tiles):
        s.terrain[t] = C.T_OCEAN            # impassable without sailing
    for t in lane:
        s.terrain[t] = C.T_FIELD
        s.owner[t] = 0
    return lane


def put(s, tile, utype, owner=0, hp=10):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = owner
    s.unit_hp[tile] = hp


def steps(s, lane, src):
    return sorted(t - src for t in reachable(s, src, R) if t in lane)


# --- roads --------------------------------------------------------------------
class TestRoads:
    def test_roads_double_a_riders_range(self):
        """Rider (movement 2) reaches 4 tiles on roads -- the wiki's 4/6 figure."""
        _, s = blank()
        lane = corridor(s)
        put(s, lane[0], C.U_RIDER)
        assert steps(s, lane, lane[0]) == [1, 2]
        for t in lane:
            s.improvement[t] = C.IMP_ROAD
        assert steps(s, lane, lane[0]) == [1, 2, 3, 4]

    def test_a_city_counts_as_a_road_node(self):
        """"between tiles with Roads (or a Road and a city)"."""
        _, s = blank()
        lane = corridor(s)
        put(s, lane[0], C.U_RIDER)
        s.city_level[lane[0]] = 1
        s.city_owner[lane[0]] = 0
        for t in lane[1:]:
            s.improvement[t] = C.IMP_ROAD
        assert steps(s, lane, lane[0]) == [1, 2, 3, 4]

    def test_roads_cannot_be_built_on_mountains(self):
        _, s = blank()
        s.grant_tech(0, R.tech_id["roads"])
        s.grant_tech(0, R.tech_id["climbing"])
        s.stars[0] = 50
        tile = 5 * N + 5
        s.terrain[tile] = C.T_MOUNTAIN
        assert not can_build(s, tile, C.IMP_ROAD, 0, R)
        s.terrain[tile] = C.T_FOREST
        assert can_build(s, tile, C.IMP_ROAD, 0, R)

    def test_roads_may_be_built_on_neutral_land(self):
        """"Can be built on neutral land" -- unique to roads and bridges."""
        _, s = blank()
        s.grant_tech(0, R.tech_id["roads"])
        s.stars[0] = 50
        tile = 5 * N + 5
        s.owner[tile] = C.NEUTRAL
        s.fog_explored[0][tile] = 1
        assert can_build(s, tile, C.IMP_ROAD, 0, R)
        assert not can_build(s, tile, C.IMP_LUMBER_HUT, 0, R)

    def test_neutral_land_must_have_been_explored_first(self):
        """Invariant 1: never act on information the observation cannot contain.

        Roads on neutral ground are the only build that can reach outside our
        own territory, so it is the only one that could road a tile the player
        has never seen.
        """
        _, s = blank()
        s.grant_tech(0, R.tech_id["roads"])
        s.stars[0] = 50
        tile = 5 * N + 5
        s.owner[tile] = C.NEUTRAL
        s.fog_explored[0][tile] = 0
        assert not can_build(s, tile, C.IMP_ROAD, 0, R)
        s.fog_explored[0][tile] = 1
        assert can_build(s, tile, C.IMP_ROAD, 0, R)

    def test_roads_may_not_be_built_on_enemy_land(self):
        _, s = blank()
        s.grant_tech(0, R.tech_id["roads"])
        s.stars[0] = 50
        tile = 5 * N + 5
        s.owner[tile] = 1
        assert not can_build(s, tile, C.IMP_ROAD, 0, R)


# --- rough terrain -------------------------------------------------------------
class TestRoughTerrain:
    def test_forest_halts_movement(self):
        _, s = blank()
        lane = corridor(s)
        put(s, lane[0], C.U_RIDER)
        s.terrain[lane[1]] = C.T_FOREST
        assert steps(s, lane, lane[0]) == [1], "may enter a forest, not pass through"

    def test_a_road_neutralises_the_forest_penalty(self):
        """"Road on Forest tiles will neutralize movement restriction."""
        _, s = blank()
        lane = corridor(s)
        put(s, lane[0], C.U_RIDER)
        s.terrain[lane[1]] = C.T_FOREST
        for t in lane:
            s.improvement[t] = C.IMP_ROAD
        assert steps(s, lane, lane[0]) == [1, 2, 3, 4]

    def test_mountains_halt_movement(self):
        _, s = blank()
        lane = corridor(s)
        put(s, lane[0], C.U_RIDER)
        s.grant_tech(0, R.tech_id["climbing"])
        s.terrain[lane[1]] = C.T_MOUNTAIN
        assert steps(s, lane, lane[0]) == [1]


# --- zone of control -----------------------------------------------------------
class TestZoneOfControl:
    def _setup(self):
        _, s = blank()
        lane = corridor(s)
        for t in lane:
            s.improvement[t] = C.IMP_ROAD    # long budget so ZoC actually binds
        put(s, lane[0], C.U_RIDER)
        return s, lane

    def test_enemy_adjacency_stops_a_road_run(self):
        s, lane = self._setup()
        assert steps(s, lane, lane[0]) == [1, 2, 3, 4]
        flank = lane[2] + N
        s.terrain[flank] = C.T_FIELD
        put(s, flank, C.U_WARRIOR, owner=1)
        assert steps(s, lane, lane[0]) == [1], "must stop on entering the ZoC"

    def test_creep_ignores_zone_of_control(self):
        """"Units with the creep skill completely bypass zone of control"."""
        s, lane = self._setup()
        flank = lane[2] + N
        s.terrain[flank] = C.T_FIELD
        put(s, flank, C.U_WARRIOR, owner=1)
        put(s, lane[0], C.U_CLOAK)           # cloak has creep
        assert steps(s, lane, lane[0]) == [1, 2, 3, 4]

    def test_a_unit_may_leave_a_zone_of_control_it_started_in(self):
        """"A unit that starts its turn within Zone of Control may move out"."""
        s, lane = self._setup()
        flank = lane[0] + N
        s.terrain[flank] = C.T_FIELD
        put(s, flank, C.U_WARRIOR, owner=1)
        assert steps(s, lane, lane[0]), "starting in ZoC must not freeze the unit"


# --- city connections ----------------------------------------------------------
class TestCityConnections:
    def _two_cities(self):
        _, s = blank()
        lane = [4 * N + c for c in range(1, 6)]
        cap, other = lane[0], lane[4]
        for tile in (cap, other):
            s.city_owner[tile] = 0
            s.city_level[tile] = 1
        s.city_capital[cap] = 1
        return s, lane, cap, other

    def test_a_road_chain_connects_a_city(self):
        s, lane, cap, other = self._two_cities()
        assert connected_cities(s, 0, R) == set()
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        assert connected_cities(s, 0, R) == {other}

    def test_connection_adds_population_to_both_cities(self):
        """"adds one population to both the capital and the connected city"."""
        s, lane, cap, other = self._two_cities()
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        recompute_connections(s, 0, R)
        assert int(s.city_pop[cap]) == 1
        assert int(s.city_pop[other]) == 1

    def test_breaking_the_road_takes_the_population_back(self):
        s, lane, cap, other = self._two_cities()
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        recompute_connections(s, 0, R)
        s.improvement[lane[2]] = C.IMP_NONE
        recompute_connections(s, 0, R)
        assert connected_cities(s, 0, R) == set()
        assert int(s.city_pop[cap]) == 0
        assert int(s.city_pop[other]) == 0

    def test_recompute_is_idempotent(self):
        s, lane, cap, other = self._two_cities()
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        for _ in range(4):
            recompute_connections(s, 0, R)
        assert int(s.city_pop[cap]) == 1, "population must not accumulate per call"

    def test_a_port_conducts_a_connection(self):
        """The Port's own description: "...and forms city connections"."""
        s, lane, cap, other = self._two_cities()
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        s.terrain[lane[2]] = C.T_WATER
        s.improvement[lane[2]] = C.IMP_PORT
        assert connected_cities(s, 0, R) == {other}

    def test_capital_gains_one_population_per_connection(self):
        s, lane, cap, other = self._two_cities()
        third = 3 * N + 1                      # directly above the capital
        s.city_owner[third] = 0
        s.city_level[third] = 1
        for t in lane[1:4]:
            s.improvement[t] = C.IMP_ROAD
        recompute_connections(s, 0, R)
        assert connected_cities(s, 0, R) == {other, third}
        # Two population is exactly the level-2 threshold, so it is spent on the
        # level rather than left sitting in city_pop -- assert on what was
        # granted, not on the residue.
        assert int(s.city_conn_pop[cap]) == 2
        assert int(s.city_level[cap]) == 2, "the granted population levelled it"


# --- population removal --------------------------------------------------------
def test_remove_population_can_de_level_a_city():
    _, s = blank()
    tile = 5 * N + 5
    s.city_owner[tile] = 0
    s.city_level[tile] = 3
    s.city_pop[tile] = 0
    lost = remove_population(s, tile, 1, R)
    assert lost == [3] and int(s.city_level[tile]) == 2


def test_a_level_one_city_cannot_shrink_below_zero():
    _, s = blank()
    tile = 5 * N + 5
    s.city_owner[tile] = 0
    s.city_level[tile] = 1
    s.city_pop[tile] = 0
    remove_population(s, tile, 5, R)
    assert int(s.city_level[tile]) == 1 and int(s.city_pop[tile]) == 0


# --- market --------------------------------------------------------------------
class TestMarket:
    def _market(self):
        _, s = blank()
        s.grant_tech(0, R.tech_id["trade"])
        s.stars[0] = 50
        centre = 5 * N + 5
        return s, centre, list(s.neighbors(centre))

    def test_market_needs_a_partner_building(self):
        s, centre, nbs = self._market()
        assert not can_build(s, centre, C.IMP_MARKET, 0, R)
        s.improvement[nbs[0]] = C.IMP_SAWMILL
        assert can_build(s, centre, C.IMP_MARKET, 0, R)

    @pytest.mark.parametrize("partner", [C.IMP_SAWMILL, C.IMP_WINDMILL, C.IMP_FORGE])
    def test_any_of_the_three_partners_counts(self, partner):
        s, centre, nbs = self._market()
        s.improvement[nbs[0]] = partner
        assert can_build(s, centre, C.IMP_MARKET, 0, R)

    def test_income_scales_with_partner_count(self):
        s, centre, nbs = self._market()
        s.improvement[centre] = C.IMP_MARKET
        s.improvement[nbs[0]] = C.IMP_SAWMILL
        one = market_income(s, centre, R)
        s.improvement[nbs[1]] = C.IMP_WINDMILL
        assert market_income(s, centre, R) == 2 * one

    def test_income_is_capped(self):
        s, centre, nbs = self._market()
        s.improvement[centre] = C.IMP_MARKET
        for nb in nbs:
            s.improvement[nb] = C.IMP_SAWMILL
        cap = int(R.imp_income_max_levels[C.IMP_MARKET])
        per = int(R.imp_income_per_level[C.IMP_MARKET])
        assert market_income(s, centre, R) == per * min(len(nbs), cap)

    def test_a_lumber_hut_is_not_a_market_partner(self):
        s, centre, nbs = self._market()
        s.improvement[nbs[0]] = C.IMP_LUMBER_HUT
        assert not can_build(s, centre, C.IMP_MARKET, 0, R)


# --- naval ---------------------------------------------------------------------
class TestNavalEmbarking:
    def _shore(self, utype=C.U_WARRIOR):
        game, s = blank()
        land, port, sea, shore = 5 * N + 3, 5 * N + 4, 5 * N + 5, 5 * N + 6
        s.terrain[port] = C.T_WATER
        s.improvement[port] = C.IMP_PORT
        s.terrain[sea] = C.T_WATER
        put(s, land, utype)
        s.unit_hp[land] = int(R.unit_max_hp[utype])
        s.grant_tech(0, R.tech_id["fishing"])
        s.grant_tech(0, R.tech_id["sailing"])
        return s, land, port, sea, shore

    def test_a_land_unit_cannot_walk_on_water(self):
        s, land, port, sea, _ = self._shore()
        assert sea not in reachable(s, land, R)

    def test_a_land_unit_may_step_onto_a_port(self):
        s, land, port, _, _ = self._shore()
        assert port in reachable(s, land, R)

    def test_stepping_onto_a_port_boards_a_raft(self):
        s, land, port, _, _ = self._shore()
        apply_action(s, Action(C.A_MOVE, land, port), R)
        assert int(s.unit_type[port]) == C.U_RAFT
        assert int(s.unit_carried[port]) == C.U_WARRIOR

    def test_boarding_consumes_the_rest_of_the_turn(self):
        """"unable to perform actions for the rest of that turn"."""
        s, land, port, _, _ = self._shore()
        apply_action(s, Action(C.A_MOVE, land, port), R)
        assert s.unit_moved[port] and s.unit_attacked[port]

    def test_a_giant_boards_a_juggernaut(self):
        """The Port "turns land units into Rafts, turns Giants into Juggernauts"."""
        s, land, port, _, _ = self._shore(utype=C.U_GIANT)
        apply_action(s, Action(C.A_MOVE, land, port), R)
        assert int(s.unit_type[port]) == C.U_JUGGERNAUT
        assert int(s.unit_carried[port]) == C.U_GIANT

    def test_the_hull_keeps_the_passengers_health(self):
        """Naval health is "Varies" because the hull has none of its own."""
        s, land, port, _, _ = self._shore(utype=C.U_DEFENDER)
        expected = int(R.unit_max_hp[C.U_DEFENDER])
        apply_action(s, Action(C.A_MOVE, land, port), R)
        assert int(s.unit_hp[port]) == expected
        assert effective_max_hp(s, port, R) == expected

    def test_a_raft_may_sail(self):
        s, land, port, sea, _ = self._shore()
        apply_action(s, Action(C.A_MOVE, land, port), R)
        s.unit_moved[port] = 0
        s.unit_attacked[port] = 0
        assert sea in reachable(s, port, R)

    def test_landing_restores_the_passenger(self):
        s, land, port, sea, shore = self._shore()
        apply_action(s, Action(C.A_MOVE, land, port), R)
        s.unit_moved[port] = 0
        s.unit_attacked[port] = 0
        apply_action(s, Action(C.A_MOVE, port, shore), R)
        assert int(s.unit_type[shore]) == C.U_WARRIOR
        assert int(s.unit_carried[shore]) == C.U_NONE

    def test_a_dead_hull_leaves_no_passenger_behind(self):
        from polyengine.core.actions import _clear_unit
        s, land, port, _, _ = self._shore()
        apply_action(s, Action(C.A_MOVE, land, port), R)
        _clear_unit(s, port)
        assert int(s.unit_carried[port]) == C.U_NONE


# --- bridges -------------------------------------------------------------------
def test_a_bridge_needs_land_on_both_sides():
    _, s = blank()
    s.grant_tech(0, R.tech_id["roads"])
    s.stars[0] = 50
    tile = 5 * N + 5
    s.terrain[tile] = C.T_WATER
    # Water everywhere around: nothing to span.
    for nb in s.neighbors(tile):
        s.terrain[nb] = C.T_WATER
    assert not can_build(s, tile, C.IMP_BRIDGE, 0, R)
    # Land directly east and west: a legal span.
    s.terrain[tile - 1] = C.T_FIELD
    s.terrain[tile + 1] = C.T_FIELD
    assert can_build(s, tile, C.IMP_BRIDGE, 0, R)
