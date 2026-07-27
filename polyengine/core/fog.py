"""Fog of war with per-player memory.

The engine does the remembering: each player has a `fog_explored` mask, a
`fog_staleness` counter, and `mem_*` snapshots of the last-seen unit/city on
every explored tile. The Phase-2 observation encoder renders unit/city planes
from these `mem_*` arrays, NOT ground truth — so the network sees exactly what
the real client shows, and the same contract holds for the Phase-4 screen parser.

Visibility is computed by a Numba kernel (grid_kernels.fog_kernel) over the flat
per-player array views; this is the single most-called hot loop in a game.
"""
from __future__ import annotations

from polyengine.core import constants as C
from polyengine.kernels.grid_kernels import fog_kernel


def recompute_fog(state, player: int) -> None:
    """Recompute `player`'s visibility and refresh their memory snapshots."""
    fog_kernel(
        state.size, player, C.SIGHT_RADIUS_UNIT, C.SIGHT_RADIUS_CITY,
        state.unit_owner, state.unit_type, state.city_owner, state.city_level,
        state.owner,
        state.fog_visible[player], state.fog_explored[player],
        state.fog_staleness[player],
        state.mem_unit_type[player], state.mem_unit_owner[player],
        state.mem_city_owner[player], state.mem_city_level[player],
    )


def recompute_all(state) -> None:
    for p in range(state.n_players):
        recompute_fog(state, p)
