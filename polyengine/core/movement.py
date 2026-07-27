"""Movement and attack-range queries.

Dijkstra over the 8-connected grid within a unit's movement budget, and an
enemy-in-range scan — both delegated to Numba kernels over the flat arrays.
Units cannot stack (Polytopia allows at most one unit per tile) and cannot pass
through occupied tiles. Terrain gated by tech (mountains need climbing; water is
naval, deferred to M4) is impassable to units whose owner lacks the tech.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset
from polyengine.kernels.grid_kernels import attack_targets_kernel, reachable_kernel


def _terr_ok_mask(state, player: int, rules: Ruleset) -> np.ndarray:
    """Per-terrain passability for `player` given their techs (length n_terrain)."""
    ok = np.ones(rules.n_terrain, dtype=np.bool_)
    for t in range(rules.n_terrain):
        ok[t] = state.has_tech(player, int(rules.terrain_requires_tech[t]))
    return ok


def reachable(state, src: int, rules: Ruleset) -> list[int]:
    """Tiles the unit on `src` can legally move to this turn (excludes src)."""
    utype = state.unit_type[src]
    if utype == C.U_NONE or state.unit_moved[src]:
        return []
    player = int(state.unit_owner[src])
    budget = int(rules.unit_movement[utype])
    terr_ok = _terr_ok_mask(state, player, rules)
    dist = np.empty(state.n_tiles, dtype=np.int16)
    reachable_kernel(state.size, src, budget, state.terrain, state.unit_type,
                     terr_ok, rules.terrain_move_cost.astype(np.int16), dist)
    out = [i for i in range(state.n_tiles) if i != src and 0 < dist[i] <= budget]
    return out


def attack_targets(state, src: int, rules: Ruleset) -> list[int]:
    """Enemy-occupied tiles the unit on `src` can attack this turn."""
    utype = state.unit_type[src]
    if utype == C.U_NONE or state.unit_attacked[src]:
        return []
    player = int(state.unit_owner[src])
    rng = int(rules.unit_range[utype])
    out = np.empty(state.n_tiles, dtype=np.int32)
    n = attack_targets_kernel(state.size, state.unit_type, state.unit_owner,
                              player, src, rng, out)
    return out[:n].tolist()
