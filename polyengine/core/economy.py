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
    """Point a unit at the city that supports it (used on train and capture).

    "Independent: units with this skill do not take up a population slot in or
    belong to any city." The Daggers a Cloak's infiltration spawns are the point
    -- a revolt that also cost the raider army capacity would not be a revolt.
    """
    rules = state.rules
    utype = int(state.unit_type[tile])
    if utype != C.U_NONE and "independent" in rules.unit_skills[utype]:
        state.unit_home[tile] = -1
        return
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


def build_candidates(state, player: int, rules: Ruleset):
    """The (tiles, improvements) worth asking `can_build` about this turn.

    A **conservative pre-filter and nothing more**: every improvement it drops
    is one `can_build` would reject on *every* tile, and every tile it drops is
    one no surviving improvement could go on. `can_build` remains the only
    authority on any particular pair, so this cannot change what is legal — only
    how long it takes to find out.

    It exists because both `enumerate_actions` and `masks.legal_sources` used to
    brute-force the whole cross product, 121 tiles x 27 improvements, at every
    single decision. Profiled: **9.5M `can_build` calls per 3,200 decisions**,
    2,972 per decision, 44% of rollout time and the single reason the engine sat
    at 1,340 micro-steps/s against PLAN.md's 5,000 target.

    Both callers share it deliberately. They had *identical* copies of the same
    cross product, which is two chances to optimise one of them and drift —
    exactly the failure invariant 2 exists to prevent.
    """
    stars = int(state.stars[player])
    imps = []
    for imp in range(rules.n_improvements):
        # Cost and tech are checked on *both* branches of `can_build`, so a
        # failure here is a failure on every tile.
        if int(rules.imp_cost[imp]) > stars:
            continue
        if not state.has_tech(player, int(rules.imp_requires_tech[imp])):
            continue
        # The Embassy returns down the other branch and never reaches these two,
        # so they may only be applied to everything else.
        if not rules.imp_on_enemy_capital[imp]:
            if rules.imp_map_feature[imp]:
                continue                      # ruins and lighthouses are placed by mapgen
            if not monument_available(state, player, imp, rules):
                continue
        imps.append(imp)

    if not imps:
        return np.empty(0, dtype=np.int64), ()

    # A tile that already carries a building takes nothing more — that holds on
    # both branches, the Embassy included.
    free = state.improvement == C.IMP_NONE
    # Own territory always. Neutral ground only if something that may go there
    # survived (roads and bridges), and rival capitals only if the Embassy did —
    # without those, most of the board cannot host anything at all.
    ground = state.owner == player
    if any(rules.imp_allow_neutral[i] for i in imps):
        ground = ground | (state.owner == C.NEUTRAL)
    # Everything except the Embassy is refused on a city tile; the Embassy is
    # refused anywhere else.
    mask = ground & free & (state.city_level == 0)
    if any(rules.imp_on_enemy_capital[i] for i in imps):
        mask = mask | ((state.city_capital > 0) & free)
    return np.flatnonzero(mask), tuple(imps)


def can_build(state, tile: int, imp_id: int, player: int, rules: Ruleset) -> bool:
    """Is placing improvement `imp_id` on `tile` legal for `player` right now?"""
    # The Embassy inverts every placement rule: it goes on ANOTHER tribe's
    # capital, which is the one tile every other improvement forbids.
    if rules.imp_on_enemy_capital[imp_id]:
        from polyengine.core.diplomacy import at_war
        if not state.city_capital[tile] or state.city_level[tile] <= 0:
            return False
        owner = int(state.city_owner[tile])
        if owner == player or owner == C.NEUTRAL:
            return False
        if state.improvement[tile] != C.IMP_NONE:
            return False
        # "Embassies may only be built in capital cities controlled by its
        # original owner, not in capitals captured by other tribes." The tile's
        # `tile_tribe` is the biome, so the test is that the capital still
        # belongs to the player who started on it -- which is exactly what
        # `unit_home`-style bookkeeping cannot tell us, so compare tribes.
        if int(state.tribe[owner]) != int(state.tile_tribe[tile]):
            return False
        # "Diplomatic buildings which can only be built in other tribes'
        # capitals **when players are not at war with them**."
        if at_war(state, player, owner):
            return False
        # "You may only build one embassy in each tribe."
        for t in range(state.n_tiles):
            if (int(state.improvement[t]) == C.IMP_EMBASSY
                    and int(state.improvement_owner[t]) == player
                    and int(state.city_owner[t]) == owner):
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
    if rules.imp_map_feature[imp_id]:
        return False                                  # placed by mapgen only
    if state.city_level[tile] > 0:
        return False                                  # not on the city tile
    if state.improvement[tile] != C.IMP_NONE:
        return False                                  # already improved
    if not state.has_tech(player, int(rules.imp_requires_tech[imp_id])):
        return False
    if int(rules.imp_cost[imp_id]) > int(state.stars[player]):
        return False
    # Monuments are awarded, not bought: gated on the task, and one apiece.
    if not monument_available(state, player, imp_id, rules):
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


def temple_level(state, tile: int) -> int:
    """"Temples level up based on how many turns have passed since their
    construction, including the turn of their construction. All Temples level up
    every two turns", to a maximum of five.

    The wiki's own table: level 1 at 0-2 turns, 2 at 3-5, 3 at 6-8, 4 at 9-11,
    5 at 12 or more.
    """
    built = int(state.improvement_turn[tile])
    if built < 0:
        return 1
    elapsed = max(0, int(state.turn) - built)
    return min(C.TEMPLE_MAX_LEVEL, 1 + elapsed // C.TEMPLE_TURNS_PER_LEVEL)


def temple_score(state, tile: int) -> int:
    """"Temples ... are worth 100 points at time of construction ... With each
    level, its point value is increased by 50 points"."""
    return (C.TEMPLE_BASE_SCORE
            + C.TEMPLE_LEVEL_SCORE * (temple_level(state, tile) - 1))


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
        state.improvement_turn[tile] = int(state.turn)
        if imp_id == C.IMP_EMBASSY:
            # "Building an Embassy will additionally reveal a 3x3 area centered
            # on the host capital city." Permanent knowledge, like Capital
            # Vision: the tiles become explored, not visible, so it grants a map
            # rather than a live view of their army.
            for nb in list(state.neighbors(tile)) + [tile]:
                state.fog_explored[player][int(nb)] = 1
        task = int(rules.imp_requires_task[imp_id])
        if task >= 0:
            state.monument_built[player] |= (1 << task)   # one monument per task

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


# --- tasks and monuments ----------------------------------------------------------
def lighthouse_tiles(state) -> list[int]:
    """The four corner lighthouses, if the map has them."""
    s = state.size
    return [t for t in (0, s - 1, s * (s - 1), s * s - 1)
            if int(state.improvement[t]) == C.IMP_LIGHTHOUSE]


def task_satisfied(state, player: int, task: int, rules: Ruleset) -> bool:
    """Is `player` currently meeting the condition for `task`?

    Conditions verbatim from the wiki's monument table. Note these are *current*
    conditions -- `update_tasks` latches them, because a monument once earned is
    not taken back when the condition lapses.
    """
    if task == C.TASK_PACIFIST:
        return int(state.calm_turns[player]) >= C.PACIFIST_TURNS
    if task == C.TASK_WEALTH:
        return int(state.stars[player]) >= C.WEALTH_STARS
    if task == C.TASK_KILLER:
        return int(state.kills[player]) >= C.KILLER_KILLS
    if task == C.TASK_METROPOLIS:
        owned = (state.city_owner == player)
        return bool((state.city_level[owned] >= C.METROPOLIS_LEVEL).any())
    if task == C.TASK_GENIUS:
        return all(state.has_tech(player, t) for t in range(rules.n_techs))
    if task == C.TASK_NETWORK:
        return len(connected_cities(state, player, rules)) >= C.NETWORK_CITIES
    if task == C.TASK_EXPLORER:
        houses = lighthouse_tiles(state)
        return bool(houses) and all(state.fog_explored[player][t] for t in houses)
    return False


def update_tasks(state, player: int, rules: Ruleset) -> list[int]:
    """Latch any newly completed tasks. Returns the ones earned just now."""
    earned = []
    for task in range(C.N_TASKS):
        if int(state.task_done[player]) >> task & 1:
            continue
        if task_satisfied(state, player, task, rules):
            state.task_done[player] |= (1 << task)
            earned.append(task)
    return earned


def monument_available(state, player: int, imp_id: int, rules: Ruleset) -> bool:
    """A monument may be placed once its task is done and it is not yet built."""
    task = int(rules.imp_requires_task[imp_id])
    if task < 0:
        return True
    if not (int(state.task_done[player]) >> task & 1):
        return False
    return not (int(state.monument_built[player]) >> task & 1)


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
    total = int(state.income_bonus[player])
    for t in range(state.n_tiles):
        if state.city_owner[t] == player and state.city_level[t] > 0:
            total += city_income(state, t)
        if rules is None:
            continue
        if state.owner[t] == player:
            total += market_income(state, t, rules)
        imp = int(state.improvement[t])
        if imp < 0:
            continue          # IMP_NONE is -1, and -1 indexes the LAST row
        if imp == C.IMP_EMBASSY:
            # Pays "both the tribe that builds and the tribe that hosts", and
            # the rate doubles under a treaty -- see diplomacy.embassy_income.
            from polyengine.core.diplomacy import embassy_income
            total += embassy_income(state, t, player, rules)
            continue
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


EXPLORER_MOVES = 15


def _explore(state, start: int, player: int) -> list[int]:
    """Send out an Explorer, which walks the map revealing it.

    "The Explorer is a figure that moves automatically to clear fog tiles and
    reveal terrain ... Each Explorer moves 15 times", and it "can only move onto
    mountain, shallow water, or ocean tiles if the appropriate technology has
    been researched (Climbing, Sailing, or Navigation)". It walks over other
    units, friend or enemy, so occupancy is not a constraint.

    Modelled as the walk itself rather than as a radius-N blob, because the two
    are not the same shape: an Explorer follows the fog, so it strings out along
    the unexplored frontier and reveals far more on an open map than a cramped
    one. It heads for "the closest fog tile to uncover" each step.

    Returns the path, for the replay log.
    """
    rules = state.rules
    explored = state.fog_explored[player]
    ok = np.ones(rules.n_terrain, dtype=bool)
    for t in range(rules.n_terrain):
        ok[t] = state.has_tech(player, int(rules.terrain_requires_tech[t]))

    def reveal(tile: int) -> None:
        for nb in list(state.neighbors(tile)) + [tile]:
            explored[int(nb)] = 1

    here = start
    reveal(here)
    path = [here]
    for _ in range(EXPLORER_MOVES):
        # Step towards the nearest tile that is still dark, preferring the step
        # that uncovers the most.
        dark = [t for t in range(state.n_tiles)
                if explored[t] == 0 and ok[int(state.terrain[t])]]
        if not dark:
            break
        target = min(dark, key=lambda t: (state.chebyshev(here, t), t))
        options = [int(nb) for nb in state.neighbors(here)
                   if ok[int(state.terrain[int(nb)])]]
        if not options:
            break
        here = min(options,
                   key=lambda t: (state.chebyshev(t, target),
                                  -sum(1 for nb in state.neighbors(t)
                                       if explored[int(nb)] == 0), t))
        reveal(here)
        path.append(here)
    return path


def _spawn_super_unit(state, city_tile: int, player: int, rules: Ruleset) -> None:
    """Place a Giant in the city, shoving whatever is standing there.

    "The forced spawn of super units are commonly used in besieged cities to push
    out the invader" — so the occupant being an *enemy* is not an obstacle, it is
    the point. A besieging unit walked onto the city, so its recorded direction
    sends it straight back out the way it came, and the siege is broken without a
    shot fired.

    If the besieger has nowhere to go it is destroyed outright: surround the city
    on all eight sides, level up, and the enemy Giant standing on it vanishes.
    The community calls that a Houdini. See `push_unit`.

    Super units cost no support.
    """
    from polyengine.core.actions import push_unit
    if int(state.unit_type[city_tile]) != C.U_NONE:
        push_unit(state, city_tile, rules, spawner=player)
    giant = rules.unit_id.get("giant", -1)
    if giant < 0:
        return
    state.unit_type[city_tile] = giant
    state.unit_owner[city_tile] = player
    state.unit_hp[city_tile] = rules.unit_max_hp[giant]
    state.unit_moved[city_tile] = 1
    state.unit_attacked[city_tile] = 1
    state.unit_home[city_tile] = -1     # super units are not supported
