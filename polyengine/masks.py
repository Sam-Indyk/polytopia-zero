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
        stars = int(state.stars[player])
        cheapest = _cheapest_trainable(state, player, rules)
        if cheapest is None or cheapest > stars:
            return mask
        for t in range(state.n_tiles):
            if (state.city_owner[t] == player and state.city_level[t] > 0
                    and state.unit_type[t] == C.U_NONE):
                mask[t] = True
        return mask
    if atype not in NEEDS_SOURCE:
        return mask
    for t in range(state.n_tiles):
        if state.unit_owner[t] != player or state.unit_type[t] == C.U_NONE:
            continue
        if atype == C.A_MOVE and reachable(state, t, rules):
            mask[t] = True
        elif atype == C.A_ATTACK and attack_targets(state, t, rules):
            mask[t] = True
        elif atype == C.A_CAPTURE and can_capture(state, t, rules):
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
    """Boolean mask over sub-types (unit id for TRAIN, tech id for RESEARCH)."""
    mask = np.zeros(subtype_dim(rules), dtype=bool)
    stars = int(state.stars[player])
    if atype == C.A_TRAIN:
        for uid in range(rules.n_units):
            if not rules.unit_trainable[uid]:
                continue
            if int(rules.unit_cost[uid]) > stars:
                continue
            if not state.has_tech(player, int(rules.unit_requires_tech[uid])):
                continue
            mask[uid] = True
    elif atype == C.A_RESEARCH:
        for tid in range(rules.n_techs):
            if state.has_tech(player, tid):
                continue
            if not state.has_tech(player, int(rules.tech_requires[tid])):
                continue
            if research_cost(state, player, tid, rules) <= stars:
                mask[tid] = True
    return mask


def legal_types(state, player: int, rules: Ruleset) -> np.ndarray:
    """Boolean mask over action types. END_TURN is always legal."""
    mask = np.zeros(C.N_ACTION_TYPES, dtype=bool)
    mask[C.A_END_TURN] = True
    for atype in (C.A_MOVE, C.A_ATTACK, C.A_CAPTURE):
        if legal_sources(state, player, rules, atype).any():
            mask[atype] = True
    if (legal_sources(state, player, rules, C.A_TRAIN).any()
            and legal_subtypes(state, player, rules, C.A_TRAIN).any()):
        mask[C.A_TRAIN] = True
    if legal_subtypes(state, player, rules, C.A_RESEARCH).any():
        mask[C.A_RESEARCH] = True
    return mask


def _cheapest_trainable(state, player: int, rules: Ruleset) -> int | None:
    best = None
    for uid in range(rules.n_units):
        if not rules.unit_trainable[uid]:
            continue
        if not state.has_tech(player, int(rules.unit_requires_tech[uid])):
            continue
        cost = int(rules.unit_cost[uid])
        if best is None or cost < best:
            best = cost
    return best


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
        if self._types is None:
            self._types = legal_types(self.state, self.player, self.rules)
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
