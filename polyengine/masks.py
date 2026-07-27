"""Action masks for the auto-regressive policy heads.

The network factorises one micro-action into four conditional decisions, the
AlphaStar pattern:

    action-type  ->  source tile  ->  target tile  ->  sub-type

Each head samples over a boolean mask supplied here, so the policy can never
emit an illegal action and the log-probs are exact over the legal set.

Not every head applies to every action type:

    type       source   target   sub-type
    END_TURN     -        -         -
    MOVE        unit     tile       -
    ATTACK      unit     tile       -
    CAPTURE     unit      =src      -
    TRAIN       city      -       unit id
    RESEARCH     -        -       tech id

`NEEDS_*` below is the contract the sampler follows. These masks are built from
the same primitives as `enumerate_actions`, and `tests/test_masks.py` asserts
the two agree exactly on random states — the masks are the training-time view of
the legality contract, so any drift is a bug in one of them.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import can_capture, research_cost
from polyengine.core.movement import attack_targets, reachable
from polyengine.core.rules import Ruleset

NEEDS_SOURCE = frozenset({C.A_MOVE, C.A_ATTACK, C.A_CAPTURE, C.A_TRAIN})
NEEDS_TARGET = frozenset({C.A_MOVE, C.A_ATTACK})
NEEDS_SUBTYPE = frozenset({C.A_TRAIN, C.A_RESEARCH})


def subtype_dim(rules: Ruleset) -> int:
    """Width of the sub-type head: unit ids for TRAIN, tech ids for RESEARCH."""
    return max(rules.n_units, rules.n_techs)


# --- individual heads ---------------------------------------------------------
def legal_sources(state, player: int, rules: Ruleset, atype: int) -> np.ndarray:
    """Boolean mask over tiles that can be the source of `atype`."""
    mask = np.zeros(state.n_tiles, dtype=bool)
    if atype == C.A_TRAIN:
        if not legal_subtypes(state, player, rules, C.A_TRAIN).any():
            return mask                    # nothing affordable/unlocked anywhere
        np.logical_and.reduce([
            state.city_owner == player,
            state.city_level > 0,
            state.unit_type == C.U_NONE,
        ], out=mask)
        return mask
    if atype not in NEEDS_SOURCE:
        return mask
    # Only the player's own occupied tiles can be a source; the per-unit checks
    # below run Numba kernels, so narrowing the candidate set first matters.
    own = np.flatnonzero((state.unit_owner == player)
                         & (state.unit_type != C.U_NONE))
    if atype == C.A_MOVE:
        fresh = own[state.unit_moved[own] == 0]
        for t in fresh:
            if reachable(state, int(t), rules):
                mask[t] = True
    elif atype == C.A_ATTACK:
        fresh = own[state.unit_attacked[own] == 0]
        for t in fresh:
            if attack_targets(state, int(t), rules):
                mask[t] = True
    elif atype == C.A_CAPTURE:
        # Capture needs a city on the tile — cheap filter before the full check.
        cand = own[(state.city_level[own] > 0)
                   & (state.city_owner[own] != player)]
        for t in cand:
            if can_capture(state, int(t), rules):
                mask[t] = True
    return mask


def legal_targets(state, player: int, rules: Ruleset, atype: int,
                  src: int) -> np.ndarray:
    """Boolean mask over tiles that can be the target, given type and source."""
    mask = np.zeros(state.n_tiles, dtype=bool)
    if atype == C.A_MOVE:
        for t in reachable(state, src, rules):
            mask[t] = True
    elif atype == C.A_ATTACK:
        for t in attack_targets(state, src, rules):
            mask[t] = True
    elif atype == C.A_CAPTURE:
        mask[src] = True
    return mask


def legal_subtypes(state, player: int, rules: Ruleset, atype: int,
                   src: int = -1) -> np.ndarray:
    """Boolean mask over sub-types (unit id for TRAIN, tech id for RESEARCH).

    Fully vectorised: the scalar version called `has_tech` once per unit/tech and
    `research_cost` once per tech (each of which rescanned the board to count
    cities), which together dominated rollout cost.
    """
    mask = np.zeros(subtype_dim(rules), dtype=bool)
    stars = int(state.stars[player])
    known = state.known_techs(player, rules.n_techs)

    if atype == C.A_TRAIN:
        req = rules.unit_requires_tech
        have_req = np.where(req < 0, True, known[np.clip(req, 0, None)])
        ok = rules.unit_trainable & (rules.unit_cost <= stars) & have_req
        mask[:rules.n_units] = ok
    elif atype == C.A_RESEARCH:
        req = rules.tech_requires
        have_req = np.where(req < 0, True, known[np.clip(req, 0, None)])
        costs = tech_costs(state, player, rules)
        mask[:rules.n_techs] = ~known & have_req & (costs <= stars)
    return mask


def tech_costs(state, player: int, rules: Ruleset) -> np.ndarray:
    """Research cost of every tech for `player` (city count counted once)."""
    n_cities = int(((state.city_owner == player) & (state.city_level > 0)).sum())
    costs = rules.tech_tier.astype(np.int32) * n_cities + 4
    override = rules.tech_cost_override
    costs = np.where(override >= 0, override, costs)
    phil = rules.tech_id.get("philosophy", -1)
    if phil >= 0 and state.has_tech(player, phil):
        costs = -(-costs * 2 // 3)          # Literacy: ceil(cost * 2/3)
    return costs


def legal_types(state, player: int, rules: Ruleset) -> np.ndarray:
    """Boolean mask over action types. END_TURN is always legal."""
    return ActionMasks(state, player, rules).types()


# --- bundled view for the rollout worker --------------------------------------
class ActionMasks:
    """Lazily-built masks for one decision point, cached per head."""

    __slots__ = ("state", "player", "rules", "_types", "_sources", "_targets",
                 "_subtypes")

    def __init__(self, state, player: int, rules: Ruleset):
        self.state = state
        self.player = player
        self.rules = rules
        self._types = None
        self._sources: dict[int, np.ndarray] = {}
        self._targets: dict[tuple[int, int], np.ndarray] = {}
        self._subtypes: dict[int, np.ndarray] = {}

    def types(self) -> np.ndarray:
        """Legal action types, reusing (and populating) the per-head caches.

        The free-function `legal_types` recomputes source masks internally; going
        through the cache means `act()` doesn't pay for them a second time.
        """
        if self._types is None:
            mask = np.zeros(C.N_ACTION_TYPES, dtype=bool)
            mask[C.A_END_TURN] = True
            for atype in (C.A_MOVE, C.A_ATTACK, C.A_CAPTURE):
                if self.sources(atype).any():
                    mask[atype] = True
            if self.sources(C.A_TRAIN).any() and self.subtypes(C.A_TRAIN).any():
                mask[C.A_TRAIN] = True
            if self.subtypes(C.A_RESEARCH).any():
                mask[C.A_RESEARCH] = True
            self._types = mask
        return self._types

    def sources(self, atype: int) -> np.ndarray:
        if atype not in self._sources:
            self._sources[atype] = legal_sources(self.state, self.player,
                                                 self.rules, atype)
        return self._sources[atype]

    def targets(self, atype: int, src: int) -> np.ndarray:
        key = (atype, src)
        if key not in self._targets:
            self._targets[key] = legal_targets(self.state, self.player,
                                               self.rules, atype, src)
        return self._targets[key]

    def subtypes(self, atype: int) -> np.ndarray:
        if atype not in self._subtypes:
            self._subtypes[atype] = legal_subtypes(self.state, self.player,
                                                   self.rules, atype)
        return self._subtypes[atype]
