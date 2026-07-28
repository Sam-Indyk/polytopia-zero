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


def passable_mask(state, player: int, utype: int, rules: Ruleset) -> np.ndarray:
    """Per-TILE passability, which terrain alone cannot express.

    A land unit may not walk on water however much technology it has — it needs
    a boat, and the only place it can board one is a Port ("Most land units
    moving into a Port will be turned into Rafts"). Naval hulls conversely move
    on water and step onto land only to disembark.
    """
    terr_ok = _terr_ok_mask(state, player, rules)
    ok = terr_ok[state.terrain]
    is_water = ~rules.terrain_is_land[state.terrain]
    if not rules.unit_naval[utype]:
        ok &= (~is_water) | (state.improvement == C.IMP_PORT)
    return ok


def road_like_mask(state, rules: Ruleset) -> np.ndarray:
    """Tiles a road-speed step can run between.

    The wiki: "Movement between two tiles with Roads (or a Road and a city)
    requires only 0.5 moves" — so cities act as road nodes, and bridges are
    roads that happen to sit on water.
    """
    mask = np.zeros(state.n_tiles, dtype=np.bool_)
    for imp in (C.IMP_ROAD, C.IMP_BRIDGE):
        mask |= state.improvement == imp
    mask |= state.city_level > 0
    return mask


def zoc_mask(state, player: int) -> np.ndarray:
    """Tiles adjacent to an enemy unit, which a mover may enter but not leave.

    "A unit that moves to a tile adjacent to an enemy unit cannot move any
    further, even if the unit had excess movement points." This takes priority
    over the road bonus, so it is applied as a hard stop in the search rather
    than as an added cost.
    """
    blocked = np.zeros(state.n_tiles, dtype=np.bool_)
    enemies = np.where((state.unit_type != C.U_NONE)
                       & (state.unit_owner != player)
                       & (state.unit_owner != C.NEUTRAL))[0]
    for e in enemies:
        for nb in state.neighbors(int(e)):
            blocked[nb] = True
    return blocked


def reachable(state, src: int, rules: Ruleset) -> list[int]:
    """Tiles the unit on `src` can legally move to this turn (excludes src)."""
    utype = state.unit_type[src]
    if utype == C.U_NONE or state.unit_moved[src]:
        return []
    player = int(state.unit_owner[src])
    # Half-moves throughout, so a road's 0.5 is an integer. See reachable_kernel.
    budget = int(rules.unit_movement[utype]) * 2
    passable = passable_mask(state, player, int(utype), rules)
    road_like = road_like_mask(state, rules)
    zoc = zoc_mask(state, player)
    if "creep" in rules.unit_skills[utype]:
        zoc = np.zeros_like(zoc)          # creep ignores zone of control entirely
    dist = np.empty(state.n_tiles, dtype=np.int16)
    reachable_kernel(state.size, src, budget, state.terrain, state.unit_type,
                     passable, (rules.terrain_move_cost * 2).astype(np.int16),
                     rules.terrain_rough, road_like, zoc, dist)
    out = [i for i in range(state.n_tiles) if i != src and 0 < dist[i] <= budget]
    return out


def infiltrate_targets(state, src: int, rules: Ruleset) -> list[int]:
    """Adjacent enemy cities the Cloak on `src` can infiltrate.

    "The Cloak cannot fight; because of its Infiltrate unit skill, it can only
    target adjacent enemy cities with its attack." Infiltration is therefore not
    a new action type -- it *is* the Cloak's attack.

    "Cloaks cannot infiltrate ... a city under siege."
    """
    from polyengine.core.economy import is_under_siege
    player = int(state.unit_owner[src])
    out = []
    for nb in state.neighbors(src):
        if state.city_level[nb] <= 0:
            continue
        owner = int(state.city_owner[nb])
        if owner == player or owner == C.NEUTRAL:
            continue
        if is_under_siege(state, nb):
            continue
        out.append(nb)
    return out


def attack_targets(state, src: int, rules: Ruleset) -> list[int]:
    """Enemy-occupied tiles the unit on `src` can attack this turn."""
    utype = state.unit_type[src]
    if utype == C.U_NONE or state.unit_attacked[src]:
        return []
    if "infiltrate" in rules.unit_skills[int(utype)]:
        return infiltrate_targets(state, src, rules)
    player = int(state.unit_owner[src])
    rng = int(rules.unit_range[utype])
    out = np.empty(state.n_tiles, dtype=np.int32)
    n = attack_targets_kernel(state.size, state.unit_type, state.unit_owner,
                              player, src, rng, out)
    return out[:n].tolist()
