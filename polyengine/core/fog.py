"""Fog of war with per-player memory.

The engine does the remembering: each player has a `fog_explored` mask, a
`fog_staleness` counter, and `mem_*` snapshots of the last-seen unit/city on
every explored tile. The Phase-2 observation encoder renders unit/city planes
from these `mem_*` arrays, NOT ground truth — so the network sees exactly what
the real client shows, and the same contract holds for the Phase-4 screen parser.
"""
from __future__ import annotations

from polyengine.core import constants as C


def _mark_radius(state, center: int, radius: int, visible) -> None:
    cx, cy = state.xy(center)
    s = state.size
    for y in range(max(0, cy - radius), min(s, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(s, cx + radius + 1)):
            visible[y * s + x] = 1


def recompute_fog(state, player: int) -> None:
    """Recompute `player`'s visibility and refresh their memory snapshots."""
    visible = state.fog_visible[player]
    visible[:] = 0

    # Owned territory is always visible.
    owned = state.owner == player
    visible[owned] = 1

    # Units and cities reveal a Chebyshev radius around themselves.
    for tile in range(state.n_tiles):
        if state.unit_owner[tile] == player and state.unit_type[tile] != C.U_NONE:
            _mark_radius(state, tile, C.SIGHT_RADIUS_UNIT, visible)
        if state.city_owner[tile] == player:
            _mark_radius(state, tile, C.SIGHT_RADIUS_CITY, visible)

    explored = state.fog_explored[player]
    explored |= visible

    # Staleness: reset where visible now, increment where explored-but-hidden.
    stale = state.fog_staleness[player]
    stale[visible == 1] = 0
    hidden_explored = (visible == 0) & (explored == 1)
    stale[hidden_explored] += 1

    # Refresh memory on visible tiles; leave stale memory untouched elsewhere.
    vis_mask = visible == 1
    state.mem_unit_type[player][vis_mask] = state.unit_type[vis_mask]
    state.mem_unit_owner[player][vis_mask] = state.unit_owner[vis_mask]
    state.mem_city_owner[player][vis_mask] = state.city_owner[vis_mask]
    state.mem_city_level[player][vis_mask] = state.city_level[vis_mask]


def recompute_all(state) -> None:
    for p in range(state.n_players):
        recompute_fog(state, p)
