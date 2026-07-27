"""The population economy: unit support, harvesting, buildings, city level-ups.

This module exists because of a rule the earlier engine was missing entirely:
**a city supports only `level + 1` units.** Without it, star income had nothing
to compete for and the optimal strategy degenerated into flooding the map with
warriors — which a trained agent duly discovered, ending one 11x11 game with 75
units and 344 unspent stars.

The real chain of scarcity is:

    resources + buildings  ->  population  ->  city level  ->  unit capacity
                                            \\-> +1 star of income per level

So expansion has to be *bought*, and the interesting Polytopia decisions
(harvest now or save for a Farm? level a city for capacity or for income?)
become real decisions the agent has to make.

Every cost and yield here is from the wiki's Population and Cities tables; see
`polyengine/data/improvements.toml` for the provenance note.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset


# --- unit support --------------------------------------------------------------
def city_of_unit(state, tile: int) -> int:
    """Tile of the city supporting the unit on `tile` (-1 if unsupported)."""
    return int(state.unit_home[tile])


def free_support(state, city_tile: int) -> int:
    """Remaining unit slots in this city."""
    return state.unit_capacity(city_tile) - state.supported_units(city_tile)


def can_train_at(state, city_tile: int, player: int) -> bool:
    """A city can train if it's ours, empty, and not at its support cap."""
    if state.city_owner[city_tile] != player or state.city_level[city_tile] <= 0:
        return False
    if state.unit_type[city_tile] != C.U_NONE:
        return False                      # a unit is standing on the tile
    return free_support(state, city_tile) > 0


def rehome_unit(state, tile: int, city_tile: int) -> None:
    """Point a unit at the city that supports it (used on train and capture)."""
    state.unit_home[tile] = city_tile


def release_support(state, tile: int) -> None:
    """Free the slot a unit occupied — on death or disband."""
    state.unit_home[tile] = -1


# --- improvements ---------------------------------------------------------------
def _adjacent_count(state, tile: int, imp: int) -> int:
    return sum(1 for nb in state.neighbors(tile) if state.improvement[nb] == imp)


def improvement_population(state, tile: int, imp_id: int, rules: Ruleset) -> int:
    """Population an improvement yields if placed on `tile` right now.

    Adjacency buildings (Sawmill / Windmill / Forge) pay per neighbouring
    building of their partner type, which is why this is positional.
    """
    per = int(rules.imp_adjacency[imp_id])
    if per > 0:
        partner = int(rules.imp_adjacent_to[imp_id])
        return per * _adjacent_count(state, tile, partner)
    return int(rules.imp_population[imp_id])


def can_build(state, tile: int, imp_id: int, player: int, rules: Ruleset) -> bool:
    """Is placing improvement `imp_id` on `tile` legal for `player` right now?"""
    if state.owner[tile] != player:
        return False                                  # must be our territory
    if state.city_level[tile] > 0:
        return False                                  # not on the city tile
    if state.improvement[tile] != C.IMP_NONE:
        return False                                  # already improved
    if not state.has_tech(player, int(rules.imp_requires_tech[imp_id])):
        return False
    if int(rules.imp_cost[imp_id]) > int(state.stars[player]):
        return False

    need_terrain = int(rules.imp_terrain[imp_id])
    if need_terrain >= 0 and int(state.terrain[tile]) != need_terrain:
        return False
    need_resource = int(rules.imp_resource[imp_id])
    if need_resource >= 0 and int(state.resource[tile]) != need_resource:
        return False
    # Adjacency buildings are legal but worthless with no partner; requiring at
    # least one keeps them out of the action mask as pure noise.
    if int(rules.imp_adjacency[imp_id]) > 0:
        if improvement_population(state, tile, imp_id, rules) <= 0:
            return False
    return True


def nearest_owned_city(state, tile: int, player: int) -> int:
    """The player's city that a tile's population should feed (closest wins)."""
    best, best_d = -1, 1 << 30
    for t in range(state.n_tiles):
        if state.city_owner[t] == player and state.city_level[t] > 0:
            d = state.chebyshev(tile, t)
            if d < best_d:
                best_d, best = d, t
    return best


def apply_build(state, tile: int, imp_id: int, player: int, rules: Ruleset) -> dict:
    """Pay for an improvement, place it, and credit its population to a city."""
    if not can_build(state, tile, imp_id, player, rules):
        raise ValueError(f"illegal build {rules.imp_names[imp_id]} at {tile}")
    cost = int(rules.imp_cost[imp_id])
    pop = improvement_population(state, tile, imp_id, rules)
    state.stars[player] -= cost

    if rules.imp_kind[imp_id] == "harvest":
        state.resource[tile] = C.R_NONE       # consumed, no building left behind
    else:
        state.improvement[tile] = imp_id

    city = nearest_owned_city(state, tile, player)
    levelled = []
    if city >= 0 and pop > 0:
        levelled = add_population(state, city, pop, rules)
    return {"type": "build", "tile": tile, "improvement": imp_id,
            "cost": cost, "population": pop, "city": city, "levelled": levelled}


# --- population and city levels --------------------------------------------------
def level_cost(level: int) -> int:
    """Population needed to reach `level` from the one below (wiki: level n costs n)."""
    return level


def add_population(state, city_tile: int, amount: int, rules: Ruleset) -> list[int]:
    """Add population, levelling the city up as thresholds are crossed.

    Returns the list of new levels reached, each of which owes the player a
    reward choice.
    """
    state.city_pop[city_tile] += amount
    reached = []
    while int(state.city_level[city_tile]) < C.CITY_MAX_LEVEL:
        nxt = int(state.city_level[city_tile]) + 1
        need = level_cost(nxt)
        if int(state.city_pop[city_tile]) < need:
            break
        state.city_pop[city_tile] -= need
        state.city_level[city_tile] = nxt
        reached.append(nxt)
    if reached:
        # Only one choice is pending at a time; the highest level wins if a
        # single build crossed several thresholds at once.
        state.city_pending_reward[city_tile] = reached[-1]
    return reached


def city_income(state, city_tile: int) -> int:
    """Stars per turn: one per level, +1 for the capital, +1 for a Workshop."""
    income = int(state.city_level[city_tile])
    if state.city_capital[city_tile]:
        income += 1
    if state.city_workshop[city_tile]:
        income += 1
    return income


def player_income(state, player: int) -> int:
    total = 0
    for t in range(state.n_tiles):
        if state.city_owner[t] == player and state.city_level[t] > 0:
            total += city_income(state, t)
    return total


# --- level-up rewards -------------------------------------------------------------
def pending_reward_cities(state, player: int) -> list[int]:
    """Cities of `player` waiting on a level-up reward choice."""
    return [t for t in range(state.n_tiles)
            if state.city_pending_reward[t] > 0 and state.city_owner[t] == player]


def rewards_for_level(level: int, rules: Ruleset) -> list[int]:
    """The two reward ids offered at `level` (levels above 5 reuse the 5+ pair)."""
    tier = min(int(level), 5)
    return [r for r in range(rules.n_rewards) if int(rules.reward_level[r]) == tier]


def can_choose_reward(state, city_tile: int, reward_id: int, player: int,
                      rules: Ruleset) -> bool:
    if state.city_owner[city_tile] != player:
        return False
    pending = int(state.city_pending_reward[city_tile])
    if pending <= 0:
        return False
    return reward_id in rewards_for_level(pending, rules)


def apply_reward(state, city_tile: int, reward_id: int, player: int,
                 rules: Ruleset) -> dict:
    """Grant a chosen level-up reward and clear the pending flag."""
    if not can_choose_reward(state, city_tile, reward_id, player, rules):
        raise ValueError(f"illegal reward {reward_id} at city {city_tile}")
    effect = rules.reward_effect[reward_id]
    value = int(rules.reward_value[reward_id])
    state.city_pending_reward[city_tile] = 0

    if effect == "income":
        state.city_workshop[city_tile] = 1
    elif effect == "walls":
        state.city_walls[city_tile] = 1
    elif effect == "stars":
        state.stars[player] += value
    elif effect == "population":
        add_population(state, city_tile, value, rules)
    elif effect == "border":
        _grow_border(state, city_tile, player)
    elif effect == "score":
        state.city_park[city_tile] = 1
    elif effect == "super_unit":
        _spawn_super_unit(state, city_tile, player, rules)
    elif effect == "explore":
        _explore(state, city_tile, player)
    return {"type": "reward", "city": city_tile, "reward": reward_id,
            "effect": effect}


def _grow_border(state, city_tile: int, player: int) -> None:
    """Claim the ring of tiles two steps out from the city."""
    for t in range(state.n_tiles):
        if state.chebyshev(city_tile, t) <= 2 and state.owner[t] == C.NEUTRAL:
            state.owner[t] = player


def _explore(state, city_tile: int, player: int) -> None:
    """Reveal a patch of map, as the Explorer reward does."""
    for t in range(state.n_tiles):
        if state.chebyshev(city_tile, t) <= 4:
            state.fog_explored[player][t] = 1


def _spawn_super_unit(state, city_tile: int, player: int, rules: Ruleset) -> None:
    """Place a Giant in the city if the tile is free. Super units cost no support."""
    if state.unit_type[city_tile] != C.U_NONE:
        return
    giant = rules.unit_id.get("giant", -1)
    if giant < 0:
        return
    state.unit_type[city_tile] = giant
    state.unit_owner[city_tile] = player
    state.unit_hp[city_tile] = rules.unit_max_hp[giant]
    state.unit_moved[city_tile] = 1
    state.unit_attacked[city_tile] = 1
    state.unit_home[city_tile] = -1     # super units are not supported
