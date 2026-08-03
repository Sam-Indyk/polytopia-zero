"""Numba kernels for the per-step hot loops.

These operate directly on the flat GameState arrays (or 1-D per-player views of
them) so the O(N-tiles) scans run in compiled code. They are pure functions of
their array arguments — no engine imports — so they JIT-compile fast and stay
easy to test in isolation. The pure-Python paths in fog.py / movement.py remain
as the reference behaviour these must match.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_INF = np.int16(32000)


@njit(cache=True)
def fog_kernel(size, player, unit_sight, sight_city, terrain, terrain_sight,
               unit_owner, unit_type, city_owner, city_level, owner,
               resource, improvement,
               visible, explored, stale,
               mem_unit_type, mem_unit_owner, mem_city_owner, mem_city_level,
               mem_owner, mem_terrain, mem_resource, mem_improvement):
    """`unit_sight` is indexed by unit type and `terrain_sight` by terrain, and a
    unit gets the better of the two.

    "Scout: allows a unit to explore a 5x5 area instead of a 3x3 area", and of
    mountains: "all units on this terrain have a sight radius of two tiles" --
    which is why taking the high ground is worth a turn.
    """
    n = size * size
    for i in range(n):
        visible[i] = 0
    # Owned territory is always visible.
    for i in range(n):
        if owner[i] == player:
            visible[i] = 1
    # Units and owned cities reveal a Chebyshev radius.
    for i in range(n):
        reveal = -1
        if unit_owner[i] == player and unit_type[i] != -1:
            reveal = unit_sight[unit_type[i]]
            if terrain_sight[terrain[i]] > reveal:
                reveal = terrain_sight[terrain[i]]
        if city_owner[i] == player and city_level[i] > 0 and sight_city > reveal:
            reveal = sight_city
        if reveal >= 0:
            cx = i % size
            cy = i // size
            y0 = cy - reveal if cy - reveal > 0 else 0
            y1 = cy + reveal + 1 if cy + reveal + 1 < size else size
            x0 = cx - reveal if cx - reveal > 0 else 0
            x1 = cx + reveal + 1 if cx + reveal + 1 < size else size
            for y in range(y0, y1):
                for x in range(x0, x1):
                    visible[y * size + x] = 1
    # Update explored / staleness / memory.
    for i in range(n):
        if visible[i]:
            explored[i] = 1
            stale[i] = 0
            mem_unit_type[i] = unit_type[i]
            mem_unit_owner[i] = unit_owner[i]
            mem_city_owner[i] = city_owner[i]
            mem_city_level[i] = city_level[i]
            mem_owner[i] = owner[i]
            # Terrain looks static but is not — Clear, Grow and Burn Forest all
            # rewrite it — and a building or a harvested resource plainly is
            # not. All three are remembered rather than observed.
            mem_terrain[i] = terrain[i]
            mem_resource[i] = resource[i]
            mem_improvement[i] = improvement[i]
        elif explored[i]:
            stale[i] += 1


@njit(cache=True)
def reachable_kernel(size, src, budget, terrain, occupied, passable, move_cost,
                     rough, road_like, zoc, dist):
    """Fill `dist` (int16[N]) with movement cost from src; _INF if unreachable.

    Everything here is in HALF-moves, because roads cost 0.5: a plain step is 2,
    a road-to-road (or road-to-city) step is 1, and `budget` is movement * 2.

    Two rules make a tile *enterable but not traversable* — you may finish there
    but not continue through it in the same turn:

      * rough terrain (mountain, forest) that has no road on it
      * zone of control: any tile adjacent to an enemy unit

    Both are modelled by refusing to *expand* such a node rather than by
    refusing to enter it. The source tile is always expandable: a unit that
    starts its turn inside an enemy's zone of control is free to leave it.

    Passability is per TILE, not per terrain: a land unit may step onto a water
    tile if (and only if) a Port is there to carry it.

    `occupied` is what the MOVING PLAYER believes blocks a tile, not ground
    truth. A hidden Cloak must not appear here: if it did, the hole it carved in
    the legal-move set would announce its position through the action mask.
    """
    n = size * size
    for i in range(n):
        dist[i] = _INF
    dist[src] = 0
    changed = True
    while changed:
        changed = False
        for cur in range(n):
            d = dist[cur]
            if d >= budget or d == _INF:
                continue
            if cur != src:
                if zoc[cur]:
                    continue
                if rough[terrain[cur]] and not road_like[cur]:
                    continue
            cx = cur % size
            cy = cur // size
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or nx >= size or ny < 0 or ny >= size:
                        continue
                    nb = ny * size + nx
                    if occupied[nb]:             # cannot enter/traverse occupied
                        continue
                    if not passable[nb]:
                        continue
                    t = terrain[nb]
                    # A road ignores the terrain penalty entirely, but only
                    # between two road-like tiles (a city counts as one).
                    if road_like[cur] and road_like[nb]:
                        step = np.int16(1)
                    else:
                        step = move_cost[t]
                    # "When calculating movement cost of a tile, the game rounds
                    # up - meaning even if you have 0.5 movement points
                    # remaining, you can still move onto a tile costing 1 or
                    # even 3." Any movement left is enough to enter; the unit
                    # simply arrives spent, which `nd = budget` expresses (a
                    # node at exactly `budget` is never expanded from).
                    nd = d + step
                    if nd > budget:
                        nd = np.int16(budget)
                    if nd < dist[nb]:
                        dist[nb] = nd
                        changed = True
    return dist


@njit(cache=True)
def attack_targets_kernel(size, unit_type, unit_owner, player, src, rng_range, out):
    """Write enemy-occupied tiles within `rng_range` of src into `out`; return count."""
    n = size * size
    sx = src % size
    sy = src // size
    c = 0
    for i in range(n):
        if unit_type[i] == -1 or unit_owner[i] == player:
            continue
        ix = i % size
        iy = i // size
        dxx = ix - sx if ix > sx else sx - ix
        dyy = iy - sy if iy > sy else sy - iy
        d = dxx if dxx > dyy else dyy
        if d <= rng_range:
            out[c] = i
            c += 1
    return c
