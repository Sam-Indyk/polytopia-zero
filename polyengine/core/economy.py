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


def bridge_span_ok(state, tile: int) -> bool:
    """A bridge needs "water in between 2 non-water tiles" — opposite sides.

    Checked as a straight span rather than any two land neighbours, so a bridge
    cannot be dropped into a bay it does not actually cross.
    """
    if not state.rules.terrain_is_land[state.terrain[tile]]:
        x, y = tile % state.size, tile // state.size
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            ax, ay = x - dx, y - dy
            bx, by = x + dx, y + dy
            if not (0 <= ax < state.size and 0 <= ay < state.size
                    and 0 <= bx < state.size and 0 <= by < state.size):
                continue
            a, b = ay * state.size + ax, by * state.size + bx
            if (state.rules.terrain_is_land[state.terrain[a]]
                    and state.rules.terrain_is_land[state.terrain[b]]):
                return True
    return False


def can_build(state, tile: int, imp_id: int, player: int, rules: Ruleset) -> bool:
    """Is placing improvement `imp_id` on `tile` legal for `player` right now?"""
    # The Embassy inverts every placement rule: it goes on ANOTHER tribe's
    # capital, which is the one tile every other improvement forbids.
    if rules.imp_on_enemy_capital[imp_id]:
        if not state.city_capital[tile] or state.city_level[tile] <= 0:
            return False
        owner = int(state.city_owner[tile])
        if owner == player or owner == C.NEUTRAL:
            return False
        if state.improvement[tile] != C.IMP_NONE:
            return False
        if not state.has_tech(player, int(rules.imp_requires_tech[imp_id])):
            return False
        if int(rules.imp_cost[imp_id]) > int(state.stars[player]):
            return False
        return bool(state.fog_explored[player][tile])

    # Roads and bridges are the exception to "own territory only": the wiki says
    # both may be built on neutral land. Enemy ground is still off limits, and
    # the tile must have been *seen* -- letting the agent road a tile it has
    # never explored would be acting on information the observation does not
    # contain, which breaks invariant 1 in CLAUDE.md.
    if rules.imp_allow_neutral[imp_id]:
        if state.owner[tile] not in (player, C.NEUTRAL):
            return False
        if state.owner[tile] != player and not state.fog_explored[player][tile]:
            return False
    elif state.owner[tile] != player:
        return False
    if state.city_level[tile] > 0:
        return False                                  # not on the city tile
    if state.improvement[tile] != C.IMP_NONE:
        return False                                  # already improved
    if not state.has_tech(player, int(rules.imp_requires_tech[imp_id])):
        return False
    if int(rules.imp_cost[imp_id]) > int(state.stars[player]):
        return False

    allowed = rules.imp_terrains[imp_id]
    if allowed and int(state.terrain[tile]) not in allowed:
        return False
    need_terrain = int(rules.imp_terrain[imp_id])
    if not allowed and need_terrain >= 0 and int(state.terrain[tile]) != need_terrain:
        return False
    if rules.imp_needs_bridge_span[imp_id] and not bridge_span_ok(state, tile):
        return False
    # The Market must touch at least one of its partner buildings.
    partners = rules.imp_adjacent_any[imp_id]
    if partners and int(rules.imp_adjacency[imp_id]) == 0:
        if not any(int(state.improvement[nb]) in partners
                   for nb in state.neighbors(tile)):
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
        state.improvement_owner[tile] = player

    city = nearest_owned_city(state, tile, player)
    levelled = []
    if city >= 0 and pop > 0:
        levelled = add_population(state, city, pop, rules)
    # A road, bridge or port may have just joined a city to the capital.
    conn = {"connected": set(), "levelled": []}
    if rules.imp_connector[imp_id] or imp_id == C.IMP_PORT:
        conn = recompute_connections(state, player, rules)
    return {"type": "build", "tile": tile, "improvement": imp_id,
            "cost": cost, "population": pop, "city": city, "levelled": levelled,
            "connected": sorted(conn["connected"])}


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


def remove_population(state, city_tile: int, amount: int, rules: Ruleset) -> list[int]:
    """Take population back, de-levelling if that drops below a threshold.

    The inverse of `add_population`, needed because a city connection can be
    *broken* — a road razed, a linking city captured — and the population it
    granted has to go back. Returns the levels lost.
    """
    state.city_pop[city_tile] -= amount
    lost = []
    while int(state.city_pop[city_tile]) < 0 and int(state.city_level[city_tile]) > 1:
        level = int(state.city_level[city_tile])
        state.city_pop[city_tile] += level_cost(level)
        state.city_level[city_tile] = level - 1
        lost.append(level)
    if int(state.city_pop[city_tile]) < 0:
        state.city_pop[city_tile] = 0          # a level-1 city cannot shrink further
    if lost:
        # A city that fell back below a threshold no longer owes a choice for it.
        if int(state.city_pending_reward[city_tile]) >= lost[-1]:
            state.city_pending_reward[city_tile] = 0
    return lost


# --- city connections -------------------------------------------------------------
def _conducts(state, tile: int, player: int, rules: Ruleset) -> bool:
    """Can a connection run through this tile for `player`?

    Roads, bridges and Ports carry connections (the wiki names Ports explicitly),
    and the player's own cities are endpoints that also relay. Roads may sit on
    neutral land, so ownership is not required for the road itself — only enemy
    *cities* break a chain.
    """
    if state.city_level[tile] > 0:
        return int(state.city_owner[tile]) == player
    return bool(rules.imp_connector[state.improvement[tile]]) or \
        int(state.improvement[tile]) == C.IMP_PORT


def connected_cities(state, player: int, rules: Ruleset) -> set[int]:
    """Cities of `player` joined to their capital by roads/bridges/ports.

    Flood-fill outward from the capital across conducting tiles. Cities found
    on the way are connected and keep relaying, so a chain of cities linked by
    short road stubs works the way it does in game.

    Ports additionally reach each other across open water: two Ports on the same
    body of water are connected however far apart they sit, which is how island
    empires join up without a bridge. Water conducts only *between* Ports, so a
    stretch of sea is not itself a road.
    """
    capital = -1
    for t in range(state.n_tiles):
        if state.city_capital[t] and int(state.city_owner[t]) == player:
            capital = t
            break
    if capital < 0:
        return set()

    seen = {capital}
    stack = [capital]
    found: set[int] = set()
    while stack:
        cur = stack.pop()
        neighbours = list(state.neighbors(cur))
        if int(state.improvement[cur]) == C.IMP_PORT:
            neighbours += _ports_reachable_by_sea(state, cur)
        for nb in neighbours:
            if nb in seen or not _conducts(state, nb, player, rules):
                continue
            seen.add(nb)
            stack.append(nb)
            if state.city_level[nb] > 0 and int(state.city_owner[nb]) == player:
                found.add(nb)
    found.discard(capital)
    return found


def _ports_reachable_by_sea(state, port: int) -> list[int]:
    """Other Ports on the same body of water as `port`.

    Flood-fills across water tiles only. The sea is a medium between Ports, not
    a connector in its own right, so nothing but Ports is returned.
    """
    seen = {port}
    stack = [port]
    out = []
    while stack:
        cur = stack.pop()
        for nb in state.neighbors(cur):
            if nb in seen:
                continue
            if state.rules.terrain_is_land[state.terrain[nb]]:
                continue
            seen.add(nb)
            stack.append(nb)
            if int(state.improvement[nb]) == C.IMP_PORT:
                out.append(nb)
    return out


def recompute_connections(state, player: int, rules: Ruleset) -> dict:
    """Bring connection population in line with the current road network.

    Wiki: "each city connection adds one population to both the capital and the
    connected city". The capital therefore holds one population per connection,
    each other city at most one. Population already granted is tracked in
    `city_conn_pop`, so this is idempotent and safe to call after any build,
    capture or destruction.
    """
    capital = -1
    for t in range(state.n_tiles):
        if state.city_capital[t] and int(state.city_owner[t]) == player:
            capital = t
            break
    if capital < 0:
        return {"connected": set(), "levelled": [], "lost": []}

    linked = connected_cities(state, player, rules)
    want = {capital: len(linked)}
    for t in range(state.n_tiles):
        if state.city_owner[t] == player and state.city_level[t] > 0 and t != capital:
            want[t] = 1 if t in linked else 0

    levelled, lost = [], []
    for tile, target in want.items():
        delta = target - int(state.city_conn_pop[tile])
        if delta > 0:
            levelled += [(tile, lv) for lv in add_population(state, tile, delta, rules)]
        elif delta < 0:
            lost += [(tile, lv) for lv in remove_population(state, tile, -delta, rules)]
        state.city_conn_pop[tile] = target
    return {"connected": linked, "levelled": levelled, "lost": lost}


# --- markets ----------------------------------------------------------------------
def market_income(state, tile: int, rules: Ruleset) -> int:
    """Stars per turn from the Market on `tile`.

    Wiki (Resources): "2 per each level of adjacent Windmill, Sawmill, or Forge,
    max. level 8". Those partner buildings do not carry their own level in this
    engine, so each contributes 1 level — revisit if building levels are added.
    """
    imp = int(state.improvement[tile])
    if imp < 0:
        return 0              # IMP_NONE is -1, which would index the LAST row
    per = int(rules.imp_income_per_level[imp])
    if per <= 0:
        return 0
    partners = rules.imp_adjacent_any[imp]
    levels = sum(1 for nb in state.neighbors(tile)
                 if int(state.improvement[nb]) in partners)
    cap = int(rules.imp_income_max_levels[imp])
    if cap > 0:
        levels = min(levels, cap)
    return per * levels


def is_under_siege(state, city_tile: int) -> bool:
    """True if an enemy unit is standing on the city tile.

    Sieging is just occupation: a unit that moves onto an enemy city holds it
    without capturing (capture is its own action, and needs the unit to *start*
    a turn there). The wiki: "Cities that are under siege by an enemy unit do
    not produce any income."
    """
    utype = int(state.unit_type[city_tile])
    if utype == C.U_NONE:
        return False
    owner = int(state.city_owner[city_tile])
    return owner != C.NEUTRAL and int(state.unit_owner[city_tile]) != owner


def city_income(state, city_tile: int) -> int:
    """Stars per turn: one per level, +1 each for capital, Workshop and Park.

    A besieged city produces nothing, which is what makes parking a unit on an
    enemy city a real strategic threat rather than a slow way to lose a unit.
    """
    if is_under_siege(state, city_tile):
        return 0
    income = int(state.city_level[city_tile])
    if state.city_capital[city_tile]:
        income += 1
    if state.city_workshop[city_tile]:
        income += 1
    if state.city_park[city_tile]:
        income += 1
    return income


def player_income(state, player: int, rules: Ruleset | None = None) -> int:
    """Stars per turn across the empire: cities plus Markets.

    Market income is deliberately *not* part of `city_income`. It belongs to the
    tile, not a city, and the wiki notes it survives effects that zero a city's
    output ("This does not affect other methods of star production: the Market,
    Ice Bank or Sanctuary") — so a besieged city still earns from its Markets.
    """
    total = 0
    for t in range(state.n_tiles):
        if state.city_owner[t] == player and state.city_level[t] > 0:
            total += city_income(state, t)
        if rules is None:
            continue
        if state.owner[t] == player:
            total += market_income(state, t, rules)
        # An Embassy "provides income to both the tribe that builds and the
        # tribe that hosts", so it pays out on the host's capital *and* to
        # whoever placed it, who is a different player by construction.
        imp = int(state.improvement[t])
        if imp < 0:
            continue          # IMP_NONE is -1, and -1 indexes the LAST row
        flat = int(rules.imp_income_flat[imp])
        if flat and (int(state.improvement_owner[t]) == player
                     or int(state.city_owner[t]) == player):
            total += flat
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
