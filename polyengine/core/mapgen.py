"""Deterministic map generation.

Everything derives from the single per-game PCG64 stream on the state, so
(map_seed, ruleset_version, tribe_config) fully reproduces a game. M0 keeps the
generator intentionally simple (land-only, no oceans — naval is M4) but routes
terrain/resource weighting through the tribe data so the shape is already right.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset
from polyengine.core.state import GameState

# Global land-terrain distribution for M0 (oceans deferred to the naval rework).
_BASE_TERRAIN = {C.T_FIELD: 0.60, C.T_FOREST: 0.25, C.T_MOUNTAIN: 0.15}
_RESOURCE_BY_TERRAIN = {
    C.T_FIELD: [(C.R_NONE, 0.75), (C.R_FRUIT, 0.15), (C.R_CROP, 0.10)],
    C.T_FOREST: [(C.R_NONE, 0.75), (C.R_GAME, 0.25)],
    C.T_MOUNTAIN: [(C.R_NONE, 0.70), (C.R_METAL, 0.30)],
}
STARTING_STARS = 5


def _weighted_choice(rng, pairs):
    vals, weights = zip(*pairs)
    return vals[rng.choice(len(vals), p=np.array(weights) / sum(weights))]


def _pick_spawns(state: GameState, n: int, margin: int = 1) -> list[int]:
    """Greedily pick n well-separated spawn tiles (deterministic via rng)."""
    s = state.size
    candidates = [y * s + x for y in range(margin, s - margin)
                  for x in range(margin, s - margin)]
    if len(candidates) < n:
        candidates = list(range(state.n_tiles))
    first = int(state.rng.choice(candidates))
    chosen = [first]
    while len(chosen) < n:
        best, best_d = None, -1
        for t in candidates:
            if t in chosen:
                continue
            d = min(state.chebyshev(t, c) for c in chosen)
            if d > best_d:
                best_d, best = d, t
        chosen.append(best)
    return chosen


def new_game(rules: Ruleset, size: int = 11, tribes: list[str] | None = None,
             seed: int = 0) -> GameState:
    """Create a fresh Domination game. `tribes` names base tribes per player."""
    tribes = tribes or ["imperius", "bardur"]
    n_players = len(tribes)
    state = GameState(rules, size, n_players, seed)

    # Terrain + resources.
    terr_ids = list(_BASE_TERRAIN)
    terr_p = np.array([_BASE_TERRAIN[t] for t in terr_ids], dtype=float)
    terr_p /= terr_p.sum()
    for tile in range(state.n_tiles):
        t = int(state.rng.choice(terr_ids, p=terr_p))
        state.terrain[tile] = t
        state.resource[tile] = _weighted_choice(state.rng, _RESOURCE_BY_TERRAIN[t])

    # Player capitals + starting units + starting tech.
    spawns = _pick_spawns(state, n_players)
    for pid, (tribe_name, cap) in enumerate(zip(tribes, spawns)):
        tribe = rules.tribes[tribe_name]
        state.tribe[pid] = tribe["id"]
        state.stars[pid] = STARTING_STARS
        state.grant_tech(pid, rules.tech_id[tribe["starting_tech"]])

        # Clear the capital tile to field so the starting unit isn't stranded.
        state.terrain[cap] = C.T_FIELD
        state.resource[cap] = C.R_NONE
        state.city_owner[cap] = pid
        state.city_level[cap] = 1
        state.city_capital[cap] = 1
        state.owner[cap] = pid
        for nb in state.neighbors(cap):
            if state.owner[nb] == C.NEUTRAL:
                state.owner[nb] = pid

        # Starting Warrior on the capital.
        state.unit_type[cap] = C.U_WARRIOR
        state.unit_owner[cap] = pid
        state.unit_hp[cap] = rules.unit_max_hp[C.U_WARRIOR]

    # Seed a couple of neutral villages for early expansion targets.
    _place_villages(state, count=max(0, size // 4))
    _place_lighthouses(state)

    from polyengine.core.fog import recompute_all
    recompute_all(state)
    return state


def _place_lighthouses(state: GameState) -> None:
    """Put a lighthouse in each of the four corners.

    The one piece of Polytopia's map generation implemented so far (the rest is
    still deferred, see PLAN.md). It is here because the Explorer task -- "find
    all lighthouses" -- is unreachable without them, so the Eye of God monument
    would be dead data.

    Corners are forced to ocean, which is also what the real generator does.
    """
    s = state.size
    for tile in (0, s - 1, s * (s - 1), s * s - 1):
        state.terrain[tile] = C.T_OCEAN
        state.resource[tile] = C.R_NONE
        state.improvement[tile] = C.IMP_LIGHTHOUSE
        state.city_owner[tile] = C.NEUTRAL
        state.city_level[tile] = 0
        state.unit_type[tile] = C.U_NONE


def _place_villages(state: GameState, count: int) -> None:
    empty = [t for t in range(state.n_tiles)
             if state.city_owner[t] == C.NEUTRAL and state.owner[t] == C.NEUTRAL
             and state.unit_type[t] == C.U_NONE
             and state.terrain[t] != C.T_MOUNTAIN]
    if not empty:
        return
    state.rng.shuffle(empty)
    for t in empty[:count]:
        state.terrain[t] = C.T_FIELD
        state.city_owner[t] = C.NEUTRAL   # neutral village, capturable
        state.city_level[t] = 1
