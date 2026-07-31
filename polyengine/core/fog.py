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

import numpy as np

from polyengine.core import constants as C
from polyengine.kernels.grid_kernels import fog_kernel


def recompute_fog(state, player: int) -> None:
    """Recompute `player`'s visibility and refresh their memory snapshots."""
    fog_kernel(
        state.size, player, state.rules.unit_sight, C.SIGHT_RADIUS_CITY,
        state.terrain, state.rules.terrain_sight,
        state.unit_owner, state.unit_type, state.city_owner, state.city_level,
        state.owner,
        state.fog_visible[player], state.fog_explored[player],
        state.fog_staleness[player],
        state.mem_unit_type[player], state.mem_unit_owner[player],
        state.mem_city_owner[player], state.mem_city_level[player],
    )
    _hide_cloaks_from(state, player)
    _apply_capital_vision(state, player)


def _hide_cloaks_from(state, player: int) -> None:
    """Erase hidden enemy Cloaks from `player`'s memory of the board.

    Unconditional. Standing next to a hidden Cloak does NOT tell you where it
    is: the wiki's own illustration is "a Cloak is hiding in the empty green
    field tile, and three neighbouring units have vibrating eye symbols
    indicating that there is a Cloak on an adjacent tile" -- the eye sits on
    *your* unit, and you triangulate from several of them. Writing the Cloak
    into memory would hand over the answer for free.

    See `cloak_adjacent` for the flag that detection actually grants.

    Runs after the fog kernel, which has no concept of stealth and will happily
    have written the Cloak in.
    """
    mem_type = state.mem_unit_type[player]
    mem_owner = state.mem_unit_owner[player]
    for tile in range(state.n_tiles):
        if not state.unit_hidden[tile]:
            continue
        if int(state.unit_owner[tile]) == player:
            continue
        mem_type[tile] = C.U_NONE
        mem_owner[tile] = C.NEUTRAL


def cloak_adjacent(state, player: int) -> np.ndarray:
    """The eye symbol: which of `player`'s units have a hidden Cloak beside them.

    "All units can detect Cloaks on adjacent tiles; a small eye symbol will
    appear on the unit's icon if an enemy cloak is next to the unit."

    Returned per tile, flagged on OUR unit rather than on the Cloak, which is
    the whole point -- it narrows the Cloak down to eight tiles, and only a
    second flagged unit narrows it further. That is information the player
    genuinely has, so the observation may carry it.
    """
    flags = np.zeros(state.n_tiles, dtype=np.bool_)
    for tile in range(state.n_tiles):
        if not state.unit_hidden[tile] or int(state.unit_owner[tile]) == player:
            continue
        for nb in state.neighbors(tile):
            if (int(state.unit_owner[nb]) == player
                    and state.unit_type[nb] != C.U_NONE):
                flags[nb] = True
    return flags


def _apply_capital_vision(state, player: int) -> None:
    """Diplomacy reveals rival capitals.

    "Capital Vision, which allows the player to see the capitals of tribes they
    have discovered (including capitals that have been captured)."

    Only the *city* memory is filled in — this grants knowledge of where a
    capital is, not sight of the units standing on it, so it cannot become a
    back door around the fog contract in invariant 1.
    """
    diplomacy = state.rules.tech_id.get("diplomacy", -1)
    if diplomacy < 0 or not state.has_tech(player, diplomacy):
        return
    for tile in range(state.n_tiles):
        if not state.city_capital[tile] or state.city_level[tile] <= 0:
            continue
        owner = int(state.city_owner[tile])
        if owner == player or owner == C.NEUTRAL:
            continue
        state.fog_explored[player][tile] = 1
        state.mem_city_owner[player][tile] = owner
        state.mem_city_level[player][tile] = state.city_level[tile]


def recompute_all(state) -> None:
    for p in range(state.n_players):
        recompute_fog(state, p)
