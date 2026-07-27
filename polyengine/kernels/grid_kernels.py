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
def fog_kernel(size, player, sight_unit, sight_city,
               unit_owner, unit_type, city_owner, city_level, owner,
               visible, explored, stale,
               mem_unit_type, mem_unit_owner, mem_city_owner, mem_city_level):
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
            reveal = sight_unit
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
        elif explored[i]:
            stale[i] += 1


@njit(cache=True)
def reachable_kernel(size, src, budget, terrain, unit_type, terr_ok, move_cost, dist):
    """Fill `dist` (int16[N]) with movement cost from src; _INF if unreachable."""
    n = size * size
    for i in range(n):
        dist[i] = _INF
    dist[src] = 0
    changed = True
    while changed:
        changed = False
        for cur in range(n):
            d = dist[cur]
            if d >= budget:
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
                    if unit_type[nb] != -1:      # cannot enter/traverse occupied
                        continue
                    t = terrain[nb]
                    if not terr_ok[t]:
                        continue
                    nd = d + move_cost[t]
                    if nd <= budget and nd < dist[nb]:
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
