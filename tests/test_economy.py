"""The population economy: unit support caps, harvests, buildings, level-ups.

These pin the rule whose absence let a trained agent end an 11x11 game with 75
units and 344 unspent stars: **a city supports only `level + 1` units.** Without
that ceiling, star income converts directly into army and every other decision
in the game stops mattering.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyengine.core import constants as C
from polyengine.core.actions import Action, enumerate_actions
from polyengine.core.economy import (
    add_population,
    can_build,
    can_train_at,
    city_income,
    free_support,
    improvement_population,
    level_cost,
    pending_reward_cities,
    player_income,
    rewards_for_level,
)
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()


def a_game(seed=3, size=11):
    game = Game.new(size=size, seed=seed)
    return game, game.state


def _clear_tile(state, tile):
    """Remove any unit standing on a tile (cities start garrisoned)."""
    state.unit_type[tile] = C.U_NONE
    state.unit_owner[tile] = C.NEUTRAL
    state.unit_hp[tile] = 0
    state.unit_home[tile] = -1


def capital_of(state, player=0):
    return next(t for t in range(state.n_tiles)
                if state.city_owner[t] == player and state.city_level[t] > 0)


# --- unit support ---------------------------------------------------------------
def test_capacity_is_level_plus_one():
    _, s = a_game()
    city = capital_of(s)
    for level in (1, 2, 3, 5):
        s.city_level[city] = level
        assert s.unit_capacity(city) == level + 1


def test_training_fills_support_then_stops():
    """The rule that was missing: a full city cannot train, however rich you are."""
    game, s = a_game()
    city = capital_of(s)
    s.stars[0] = 500
    s.city_level[city] = 1              # capacity 2
    _clear_tile(s, city)                # a city starts with a unit standing on it
    assert free_support(s, city) == 2

    trained = 0
    for _ in range(10):
        trains = [a for a in enumerate_actions(s, 0, R) if a.atype == C.A_TRAIN]
        if not trains:
            break
        game.step(trains[0])
        trained += 1
        # move the new unit off the city tile so occupancy isn't the limiter
        unit_tile = city
        for nb in s.neighbors(city):
            if s.unit_type[nb] == C.U_NONE and s.terrain[nb] == C.T_FIELD:
                for arr in ("unit_type", "unit_owner", "unit_hp", "unit_home"):
                    getattr(s, arr)[nb] = getattr(s, arr)[unit_tile]
                s.unit_type[unit_tile] = C.U_NONE
                s.unit_owner[unit_tile] = C.NEUTRAL
                s.unit_home[unit_tile] = -1
                break
    assert trained == 2, f"trained {trained} units from a capacity-2 city"
    assert s.stars[0] > 400, "stars were not the limiting factor"


def test_a_full_city_offers_no_train_action():
    _, s = a_game()
    city = capital_of(s)
    s.stars[0] = 500
    s.city_level[city] = 1
    s.unit_home[:] = -1
    free = [t for t in s.neighbors(city) if s.unit_type[t] == C.U_NONE][:2]
    for t in free:                      # fill both support slots
        s.unit_type[t] = C.U_WARRIOR
        s.unit_owner[t] = 0
        s.unit_hp[t] = 10
        s.unit_home[t] = city
    assert free_support(s, city) == 0
    assert not can_train_at(s, city, 0)
    assert not [a for a in enumerate_actions(s, 0, R) if a.atype == C.A_TRAIN]


def test_levelling_a_city_raises_its_unit_cap():
    _, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 1
    before = s.unit_capacity(city)
    s.city_level[city] = 2
    assert s.unit_capacity(city) == before + 1


def test_a_dead_unit_frees_its_slot():
    from polyengine.core.combat import _clear_unit
    _, s = a_game()
    city = capital_of(s)
    tile = next(t for t in s.neighbors(city) if s.unit_type[t] == C.U_NONE)
    s.unit_type[tile] = C.U_WARRIOR
    s.unit_owner[tile] = 0
    s.unit_home[tile] = city
    used = s.supported_units(city)
    _clear_unit(s, tile)
    assert s.supported_units(city) == used - 1


def test_a_moving_unit_keeps_its_home_city():
    game, s = a_game(seed=5)
    city = capital_of(s)
    s.stars[0] = 50
    _clear_tile(s, city)
    trains = [a for a in enumerate_actions(s, 0, R) if a.atype == C.A_TRAIN]
    game.step(trains[0])
    assert s.unit_home[city] == city
    s.unit_moved[city] = 0
    moves = [a for a in enumerate_actions(s, 0, R)
             if a.atype == C.A_MOVE and a.src == city]
    if moves:
        game.step(moves[0])
        assert s.unit_home[moves[0].dst] == city, "home city lost on move"
        assert s.unit_home[city] == -1


def test_capturing_rehomes_the_unit_to_the_new_city():
    """Wiki: a unit that captures a city is supported by that city afterwards."""
    from tests.test_capture import refresh, setup_unit_next_to_city
    game, city, adj = setup_unit_next_to_city()
    game.step(Action(C.A_MOVE, adj, city))
    refresh(game.state, city)
    game.step(Action(C.A_CAPTURE, city, city))
    assert int(game.state.unit_home[city]) == city


# --- population and levels --------------------------------------------------------
def test_level_n_costs_n_population():
    assert [level_cost(n) for n in (2, 3, 4, 5)] == [2, 3, 4, 5]


def test_population_accumulates_and_levels_the_city():
    _, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 1
    s.city_pop[city] = 0
    assert add_population(s, city, 1, R) == []          # not enough for level 2
    assert int(s.city_level[city]) == 1
    assert add_population(s, city, 1, R) == [2]         # 2 population -> level 2
    assert int(s.city_level[city]) == 2
    assert int(s.city_pop[city]) == 0


def test_one_build_can_cross_several_levels():
    _, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 1
    s.city_pop[city] = 0
    reached = add_population(s, city, 5, R)             # 2 then 3 = 5
    assert reached == [2, 3]
    assert int(s.city_level[city]) == 3


def test_cities_do_not_grow_on_their_own():
    """Population must be bought; free growth was the old placeholder economy."""
    game, s = a_game(seed=11)
    city = capital_of(s)
    pop_before = int(s.city_pop[city])
    level_before = int(s.city_level[city])
    for _ in range(12):
        game.step(Action(C.A_END_TURN))
    assert int(s.city_pop[city]) == pop_before
    assert int(s.city_level[city]) == level_before


# --- income -----------------------------------------------------------------------
def test_income_is_level_plus_capital_and_workshop():
    _, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 3
    s.city_capital[city] = 0
    s.city_workshop[city] = 0
    assert city_income(s, city) == 3
    s.city_capital[city] = 1
    assert city_income(s, city) == 4
    s.city_workshop[city] = 1
    assert city_income(s, city) == 5


def test_player_income_sums_owned_cities():
    _, s = a_game()
    total = sum(city_income(s, t) for t in range(s.n_tiles)
                if s.city_owner[t] == 0 and s.city_level[t] > 0)
    assert player_income(s, 0) == total


# --- improvements -------------------------------------------------------------------
def test_build_costs_and_yields_match_the_wiki_table():
    expected = {"harvest_fruit": (2, 1), "hunt_game": (2, 1), "fishing": (2, 1),
                "lumber_hut": (3, 1), "farm": (5, 2), "mine": (5, 2),
                "port": (7, 1), "temple": (20, 1)}
    for name, (cost, pop) in expected.items():
        i = R.imp_id[name]
        assert int(R.imp_cost[i]) == cost, f"{name} cost"
        assert int(R.imp_population[i]) == pop, f"{name} population"


def test_building_requires_territory_tech_and_stars():
    _, s = a_game(seed=2)
    imp = R.imp_id["harvest_fruit"]
    tile = next((t for t in range(s.n_tiles)
                 if s.resource[t] == C.R_FRUIT and s.city_level[t] == 0), None)
    if tile is None:
        pytest.skip("no fruit on this map")
    s.owner[tile] = 0
    s.stars[0] = 0
    assert not can_build(s, tile, imp, 0, R), "should be unaffordable"
    s.stars[0] = 50
    s.grant_tech(0, R.tech_id["organization"])
    assert can_build(s, tile, imp, 0, R)
    s.owner[tile] = 1
    assert not can_build(s, tile, imp, 0, R), "not our territory"


def test_harvest_consumes_the_resource_and_adds_population():
    game, s = a_game(seed=2)
    imp = R.imp_id["harvest_fruit"]
    tile = next((t for t in range(s.n_tiles)
                 if s.resource[t] == C.R_FRUIT and s.city_level[t] == 0), None)
    if tile is None:
        pytest.skip("no fruit on this map")
    s.owner[tile] = 0
    s.stars[0] = 50
    s.grant_tech(0, R.tech_id["organization"])
    city = capital_of(s)
    s.city_pop[city] = 0
    s.city_level[city] = 5              # far from a level-up threshold
    game.step(Action(C.A_BUILD, tile, -1, imp))
    assert int(s.resource[tile]) == C.R_NONE, "fruit should be consumed"
    assert int(s.improvement[tile]) == C.IMP_NONE, "harvest leaves no building"
    assert int(s.stars[0]) == 48
    assert int(s.city_pop[city]) == 1


def test_building_persists_on_the_tile():
    game, s = a_game(seed=4)
    imp = R.imp_id["lumber_hut"]
    tile = next((t for t in range(s.n_tiles)
                 if s.terrain[t] == C.T_FOREST and s.city_level[t] == 0), None)
    if tile is None:
        pytest.skip("no forest on this map")
    s.owner[tile] = 0
    s.stars[0] = 50
    s.grant_tech(0, R.tech_id["forestry"])
    game.step(Action(C.A_BUILD, tile, -1, imp))
    assert int(s.improvement[tile]) == imp


def test_cannot_build_twice_on_one_tile():
    _, s = a_game(seed=4)
    imp = R.imp_id["lumber_hut"]
    tile = next((t for t in range(s.n_tiles)
                 if s.terrain[t] == C.T_FOREST and s.city_level[t] == 0), None)
    if tile is None:
        pytest.skip("no forest on this map")
    s.owner[tile] = 0
    s.stars[0] = 50
    s.grant_tech(0, R.tech_id["forestry"])
    s.improvement[tile] = imp
    assert not can_build(s, tile, imp, 0, R)


def test_adjacency_buildings_scale_with_neighbours():
    _, s = a_game(seed=6)
    sawmill = R.imp_id["sawmill"]
    hut = R.imp_id["lumber_hut"]
    centre = next(t for t in range(s.n_tiles)
                  if s.terrain[t] == C.T_FIELD and s.city_level[t] == 0
                  and len(list(s.neighbors(t))) == 8)
    assert improvement_population(s, centre, sawmill, R) == 0
    neighbours = list(s.neighbors(centre))[:3]
    for nb in neighbours:
        s.improvement[nb] = hut
    assert improvement_population(s, centre, sawmill, R) == 3


def test_worthless_adjacency_building_is_not_offered():
    _, s = a_game(seed=6)
    sawmill = R.imp_id["sawmill"]
    tile = next(t for t in range(s.n_tiles)
                if s.terrain[t] == C.T_FIELD and s.city_level[t] == 0)
    s.owner[tile] = 0
    s.stars[0] = 50
    s.grant_tech(0, R.tech_id["mathematics"])
    assert not can_build(s, tile, sawmill, 0, R), "no adjacent lumber huts"


# --- level-up rewards ----------------------------------------------------------------
def test_each_level_offers_exactly_two_rewards():
    for level in (2, 3, 4, 5):
        assert len(rewards_for_level(level, R)) == 2


def test_levels_above_five_reuse_the_top_pair():
    assert rewards_for_level(7, R) == rewards_for_level(5, R)


def test_a_pending_reward_blocks_every_other_action():
    """The real game shows a forced popup; the action space should match."""
    _, s = a_game()
    city = capital_of(s)
    s.city_pending_reward[city] = 2
    actions = enumerate_actions(s, 0, R)
    assert actions, "must offer the choice itself"
    assert {a.atype for a in actions} == {C.A_CHOOSE_REWARD}
    assert pending_reward_cities(s, 0) == [city]


def test_choosing_workshop_raises_income_and_clears_the_flag():
    game, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 2
    s.city_pending_reward[city] = 2
    before = city_income(s, city)
    game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_WORKSHOP))
    assert s.city_workshop[city]
    assert city_income(s, city) == before + 1
    assert int(s.city_pending_reward[city]) == 0


def test_choosing_resources_grants_stars():
    game, s = a_game()
    city = capital_of(s)
    s.city_pending_reward[city] = 3
    s.stars[0] = 10
    game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_RESOURCES))
    assert int(s.stars[0]) == 15


def test_choosing_population_growth_adds_three_population():
    game, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 3
    s.city_pop[city] = 0
    s.city_pending_reward[city] = 4
    game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_POPULATION_GROWTH))
    # Level 4 costs 4 population, so +3 alone is not enough to level again.
    assert int(s.city_pop[city]) == 3
    assert int(s.city_level[city]) == 3


def test_population_growth_levels_the_city_when_it_completes_a_threshold():
    game, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 3
    s.city_pop[city] = 1                # 1 + 3 = 4, exactly level 4's cost
    s.city_pending_reward[city] = 4
    game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_POPULATION_GROWTH))
    assert int(s.city_level[city]) == 4


def test_super_unit_reward_spawns_a_giant_that_costs_no_support():
    game, s = a_game()
    city = capital_of(s)
    s.city_level[city] = 5
    s.city_pending_reward[city] = 5
    s.unit_type[city] = C.U_NONE
    game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_SUPER_UNIT))
    assert int(s.unit_type[city]) == R.unit_id["giant"]
    assert int(s.unit_home[city]) == -1, "super units are not supported"


def test_an_illegal_reward_choice_raises():
    game, s = a_game()
    city = capital_of(s)
    s.city_pending_reward[city] = 2
    with pytest.raises(ValueError, match="illegal reward"):
        game.step(Action(C.A_CHOOSE_REWARD, city, -1, C.RW_PARK))


# --- the behaviour this whole module exists to prevent --------------------------------
def test_armies_stay_bounded_by_city_capacity():
    from polyengine.agents.random_agent import RandomAgent
    for seed in range(3):
        game = Game.new(size=11, seed=seed)
        agents = [RandomAgent(seed=seed * 7 + p) for p in range(2)]
        steps = 0
        while not game.is_terminal() and steps < 1500:
            game.step(agents[game.current_player].act(game))
            steps += 1
        s = game.state
        for p in range(2):
            units = int(((s.unit_owner == p) & (s.unit_type != C.U_NONE)).sum())
            owned = np.flatnonzero((s.city_owner == p) & (s.city_level > 0))
            capacity = sum(int(s.city_level[t]) + 1 for t in owned)
            # Super units and captured units can exceed a strict cap briefly;
            # the point is that army size tracks capacity, not spare stars.
            assert units <= capacity + 4, (
                f"seed {seed} P{p}: {units} units vs capacity {capacity}")


# --- income rules the engine used to get wrong ---------------------------------
# Both were silently wrong: a Park is a level-up reward the agent can pick, and
# it granted score but no stars, so the agent was being taught a good reward was
# a bad one. Siege did not exist at all, so occupying an enemy city cost them
# nothing and the agent never learned to apply or break one.
class TestIncomeCorrections:
    def _city(self, seed=3):
        from polyengine.core.game import Game
        game = Game.new(size=9, seed=seed)
        s = game.state
        tile = next(t for t in range(s.n_tiles)
                    if s.city_level[t] > 0 and s.city_owner[t] != C.NEUTRAL)
        s.unit_type[tile] = C.U_NONE
        s.unit_owner[tile] = C.NEUTRAL
        s.city_workshop[tile] = 0
        s.city_park[tile] = 0
        s.city_capital[tile] = 0
        return s, tile, int(s.city_owner[tile])

    def test_park_adds_one_star(self):
        """Wiki: "Workshops AND Parks increase a city's income by one extra star"."""
        from polyengine.core.economy import city_income
        s, tile, _ = self._city()
        before = city_income(s, tile)
        s.city_park[tile] = 1
        assert city_income(s, tile) == before + 1

    def test_park_and_workshop_stack(self):
        from polyengine.core.economy import city_income
        s, tile, _ = self._city()
        base = city_income(s, tile)
        s.city_park[tile] = 1
        s.city_workshop[tile] = 1
        assert city_income(s, tile) == base + 2

    def test_enemy_unit_on_the_city_besieges_it(self):
        from polyengine.core.economy import city_income, is_under_siege
        s, tile, owner = self._city()
        assert city_income(s, tile) > 0
        s.unit_type[tile] = R.unit_id["warrior"]
        s.unit_owner[tile] = 1 - owner
        assert is_under_siege(s, tile)
        assert city_income(s, tile) == 0

    def test_own_garrison_does_not_besiege(self):
        from polyengine.core.economy import city_income, is_under_siege
        s, tile, owner = self._city()
        expected = city_income(s, tile)
        s.unit_type[tile] = R.unit_id["warrior"]
        s.unit_owner[tile] = owner
        assert not is_under_siege(s, tile)
        assert city_income(s, tile) == expected

    def test_siege_removes_the_city_from_player_income(self):
        from polyengine.core.economy import city_income, player_income
        s, tile, owner = self._city()
        before = player_income(s, owner)
        lost = city_income(s, tile)
        s.unit_type[tile] = R.unit_id["warrior"]
        s.unit_owner[tile] = 1 - owner
        assert player_income(s, owner) == before - lost

    def test_neutral_village_is_never_under_siege(self):
        from polyengine.core.economy import is_under_siege
        s, tile, _ = self._city()
        s.city_owner[tile] = C.NEUTRAL
        s.unit_type[tile] = R.unit_id["warrior"]
        s.unit_owner[tile] = 0
        assert not is_under_siege(s, tile)
