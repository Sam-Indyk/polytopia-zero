"""Movement and attack-range queries.

Dijkstra over the 8-connected grid within a unit's movement budget. Units cannot
stack (Polytopia allows at most one unit per tile) and cannot pass through
occupied tiles. Terrain gated by tech (mountains need climbing; water is naval,
deferred to M4) is impassable to units whose owner lacks the tech.
"""
from __future__ import annotations

import heapq

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset


def _can_enter_terrain(state, player: int, tile: int, rules: Ruleset) -> bool:
    tid = state.terrain[tile]
    req = rules.terrain_requires_tech[tid]
    return state.has_tech(player, int(req))


def reachable(state, src: int, rules: Ruleset) -> list[int]:
    """Tiles the unit on `src` can legally move to this turn (excludes src)."""
    utype = state.unit_type[src]
    if utype == C.U_NONE:
        return []
    if state.unit_moved[src]:
        return []
    player = int(state.unit_owner[src])
    budget = int(rules.unit_movement[utype])

    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        d, cur = heapq.heappop(pq)
        if d > dist.get(cur, 1 << 30):
            continue
        for nb in state.neighbors(cur):
            # Cannot traverse or land on an occupied tile.
            if state.unit_type[nb] != C.U_NONE:
                continue
            if not _can_enter_terrain(state, player, nb, rules):
                continue
            step = int(rules.terrain_move_cost[state.terrain[nb]])
            nd = d + step
            if nd <= budget and nd < dist.get(nb, 1 << 30):
                dist[nb] = nd
                heapq.heappush(pq, (nd, nb))
    # Deterministic tile order; exclude the origin.
    return sorted(set(dist) - {src})


def attack_targets(state, src: int, rules: Ruleset) -> list[int]:
    """Enemy-occupied tiles the unit on `src` can attack this turn."""
    utype = state.unit_type[src]
    if utype == C.U_NONE or state.unit_attacked[src]:
        return []
    player = int(state.unit_owner[src])
    rng = int(rules.unit_range[utype])
    out = []
    for tile in range(state.n_tiles):
        if state.unit_type[tile] == C.U_NONE:
            continue
        if state.unit_owner[tile] == player:
            continue
        if state.chebyshev(src, tile) <= rng:
            out.append(tile)
    return out
