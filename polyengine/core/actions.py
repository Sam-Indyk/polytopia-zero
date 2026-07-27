"""Action representation, legality enumeration, and application.

One micro-action per step; a turn is a sequence of actions ending in END_TURN.
The four fields (atype, src, dst, subtype) map directly onto the Phase-2
auto-regressive heads: action-type -> source tile -> target tile -> sub-type.
The legality helpers here are what the network's action masks will be built from.

M0 scope note: the economy is deliberately simplified (see game.py) so
random-vs-random games progress and terminate; the action *contract* is the
part being frozen, and it already covers move/attack/capture/train/research.
"""
from __future__ import annotations

from typing import NamedTuple

from polyengine.core import constants as C
from polyengine.core.combat import resolve_attack
from polyengine.core.movement import attack_targets, reachable
from polyengine.core.rules import Ruleset


class Action(NamedTuple):
    atype: int
    src: int = -1
    dst: int = -1
    subtype: int = -1

    def describe(self, rules: Ruleset | None = None) -> str:
        name = C.ACTION_NAMES[self.atype]
        if self.atype == C.A_END_TURN:
            return name
        if self.atype == C.A_RESEARCH:
            tech = rules.tech_names[self.subtype] if rules else self.subtype
            return f"{name}({tech})"
        if self.atype == C.A_TRAIN:
            unit = rules.unit_names[self.subtype] if rules else self.subtype
            return f"{name}(city={self.src}, {unit})"
        return f"{name}({self.src}->{self.dst})"


# --- research pricing (current formula: tier * cities + 4) --------------------
def research_cost(state, player: int, tech_id: int, rules: Ruleset) -> int:
    override = int(rules.tech_cost_override[tech_id])
    if override >= 0:
        base = override
    else:
        n_cities = int(_count_cities(state, player))
        tier = int(rules.tech_tier[tech_id])
        base = tier * max(1, n_cities) + 4
    # Literacy (from Philosophy) discounts research by 33%, rounded up.
    if _has_literacy(state, player, rules):
        base = -(-base * 2 // 3)  # ceil(base * 2/3)
    return base


def _has_literacy(state, player: int, rules: Ruleset) -> bool:
    phil = rules.tech_id.get("philosophy", -1)
    return phil >= 0 and state.has_tech(player, phil)


def _count_cities(state, player: int) -> int:
    return int(((state.city_owner == player) & (state.city_level > 0)).sum())


def _owned_city_tiles(state, player: int) -> list[int]:
    return [t for t in range(state.n_tiles)
            if state.city_owner[t] == player and state.city_level[t] > 0]


def _player_units(state, player: int) -> list[int]:
    return [t for t in range(state.n_tiles)
            if state.unit_owner[t] == player and state.unit_type[t] != C.U_NONE]


# --- enumeration (basis for both the random agent and the action masks) -------
def enumerate_actions(state, player: int, rules: Ruleset) -> list[Action]:
    actions: list[Action] = [Action(C.A_END_TURN)]

    for src in _player_units(state, player):
        for dst in attack_targets(state, src, rules):
            actions.append(Action(C.A_ATTACK, src, dst))
        for dst in reachable(state, src, rules):
            actions.append(Action(C.A_MOVE, src, dst))

    stars = int(state.stars[player])
    for city in _owned_city_tiles(state, player):
        if state.unit_type[city] != C.U_NONE:
            continue  # occupied: nothing to place onto
        for uid in range(rules.n_units):
            if not rules.unit_trainable[uid]:
                continue
            if int(rules.unit_cost[uid]) > stars:
                continue
            if not state.has_tech(player, int(rules.unit_requires_tech[uid])):
                continue
            actions.append(Action(C.A_TRAIN, city, -1, uid))

    for tid in range(rules.n_techs):
        if state.has_tech(player, tid):
            continue
        if not state.has_tech(player, int(rules.tech_requires[tid])):
            continue
        if research_cost(state, player, tid, rules) <= stars:
            actions.append(Action(C.A_RESEARCH, -1, -1, tid))

    return actions


# --- head-wise legality (used later for auto-regressive masking) --------------
def legal_action_types(state, player: int, rules: Ruleset) -> set[int]:
    return {a.atype for a in enumerate_actions(state, player, rules)}


# --- application --------------------------------------------------------------
def apply_action(state, action: Action, rules: Ruleset) -> dict:
    """Apply `action` for state.current_player. Mutates state; returns an info dict."""
    player = state.current_player
    at = action.atype
    if at == C.A_END_TURN:
        return {"type": "end_turn"}
    if at == C.A_MOVE:
        return _apply_move(state, player, action, rules)
    if at == C.A_ATTACK:
        res = resolve_attack(state, action.src, action.dst, rules)
        return {"type": "attack", "result": res}
    if at == C.A_TRAIN:
        return _apply_train(state, player, action, rules)
    if at == C.A_RESEARCH:
        cost = research_cost(state, player, action.subtype, rules)
        state.stars[player] -= cost
        state.grant_tech(player, action.subtype)
        return {"type": "research", "tech": action.subtype, "cost": cost}
    raise ValueError(f"unknown action type {at}")


def _apply_move(state, player: int, action: Action, rules: Ruleset) -> dict:
    src, dst = action.src, action.dst
    # Relocate the unit.
    for arr in ("unit_type", "unit_owner", "unit_hp", "unit_veteran",
                "unit_kills", "unit_attacked"):
        getattr(state, arr)[dst] = getattr(state, arr)[src]
    state.unit_moved[dst] = 1
    _clear_unit(state, src)

    captured = False
    if state.city_level[dst] > 0 and state.city_owner[dst] != player:
        _capture_city(state, player, dst)
        captured = True
    return {"type": "move", "src": src, "dst": dst, "captured": captured}


def _apply_train(state, player: int, action: Action, rules: Ruleset) -> dict:
    city, uid = action.src, action.subtype
    state.stars[player] -= int(rules.unit_cost[uid])
    state.unit_type[city] = uid
    state.unit_owner[city] = player
    state.unit_hp[city] = rules.unit_max_hp[uid]
    state.unit_veteran[city] = 0
    state.unit_kills[city] = 0
    state.unit_moved[city] = 1     # freshly trained: cannot act this turn
    state.unit_attacked[city] = 1
    return {"type": "train", "city": city, "unit": uid}


def _capture_city(state, player: int, tile: int) -> None:
    state.city_owner[tile] = player
    state.owner[tile] = player
    for nb in state.neighbors(tile):
        if state.owner[nb] == C.NEUTRAL:
            state.owner[nb] = player


def _clear_unit(state, tile: int) -> None:
    state.unit_type[tile] = C.U_NONE
    state.unit_owner[tile] = C.NEUTRAL
    state.unit_hp[tile] = 0
    state.unit_moved[tile] = 0
    state.unit_attacked[tile] = 0
    state.unit_veteran[tile] = 0
    state.unit_kills[tile] = 0
