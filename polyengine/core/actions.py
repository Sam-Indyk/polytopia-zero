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
from polyengine.core.economy import (
    apply_build,
    recompute_connections,
    apply_reward,
    can_build,
    can_choose_reward,
    can_train_at,
    pending_reward_cities,
    rehome_unit,
    rewards_for_level,
)
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
        if self.atype == C.A_CAPTURE:
            return f"{name}(tile={self.src})"
        if self.atype == C.A_BUILD:
            imp = rules.imp_names[self.subtype] if rules else self.subtype
            return f"{name}({imp} @ {self.src})"
        if self.atype == C.A_CHOOSE_REWARD:
            rw = rules.reward_names[self.subtype] if rules else self.subtype
            return f"{name}(city={self.src}, {rw})"
        return f"{name}({self.src}->{self.dst})"


# --- research pricing (current formula: tier * cities + 4) --------------------
def research_cost(state, player: int, tech_id: int, rules: Ruleset) -> int:
    override = int(rules.tech_cost_override[tech_id])
    if override >= 0:
        base = override
    else:
        # Wiki "Technology Costs" table: tier * cities + 4, with no clamp — at
        # zero cities every tier costs 4.
        n_cities = int(_count_cities(state, player))
        tier = int(rules.tech_tier[tech_id])
        base = tier * n_cities + 4
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


def can_capture(state, tile: int, rules: Ruleset) -> bool:
    """Can the unit standing on `tile` capture the city there right now?

    Capture is a distinct action: the unit must be on an enemy/neutral city and
    (by default) must not have already moved or attacked this turn, so moving
    onto a village and taking it are two separate turns.
    """
    player = int(state.unit_owner[tile])
    if state.unit_type[tile] == C.U_NONE:
        return False
    if state.city_level[tile] <= 0 or state.city_owner[tile] == player:
        return False
    if rules.capture_requires_fresh_unit:
        if state.unit_moved[tile] or state.unit_attacked[tile]:
            return False
    return True


def capture_targets(state, player: int, rules: Ruleset) -> list[int]:
    """Tiles where one of `player`'s units can capture a city this turn."""
    return [t for t in _player_units(state, player) if can_capture(state, t, rules)]


# --- enumeration (basis for both the random agent and the action masks) -------
def enumerate_actions(state, player: int, rules: Ruleset) -> list[Action]:
    # A city that just levelled up owes a reward choice, and the real game
    # blocks on that popup. Offering only those keeps the agent's action space
    # honest instead of letting it bank an unclaimed reward indefinitely.
    pending = pending_reward_cities(state, player)
    if pending:
        out: list[Action] = []
        for city in pending:
            level = int(state.city_pending_reward[city])
            for rw in rewards_for_level(level, rules):
                out.append(Action(C.A_CHOOSE_REWARD, city, -1, rw))
        return out

    actions: list[Action] = [Action(C.A_END_TURN)]

    for src in _player_units(state, player):
        if can_capture(state, src, rules):
            actions.append(Action(C.A_CAPTURE, src, src))
        for dst in attack_targets(state, src, rules):
            actions.append(Action(C.A_ATTACK, src, dst))
        for dst in reachable(state, src, rules):
            actions.append(Action(C.A_MOVE, src, dst))

    stars = int(state.stars[player])
    for city in _owned_city_tiles(state, player):
        # A city supports only `level + 1` units; at the cap it cannot train.
        if not can_train_at(state, city, player):
            continue
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

    for tile in range(state.n_tiles):
        # Roads and bridges may be placed on neutral ground, so this cannot
        # pre-filter to owned territory — `can_build` makes the finer decision.
        if state.owner[tile] != player and state.owner[tile] != C.NEUTRAL:
            continue
        for imp in range(rules.n_improvements):
            if can_build(state, tile, imp, player, rules):
                actions.append(Action(C.A_BUILD, tile, -1, imp))

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
    if at == C.A_CAPTURE:
        return _apply_capture(state, player, action, rules)
    if at == C.A_BUILD:
        return apply_build(state, action.src, action.subtype, player, rules)
    if at == C.A_CHOOSE_REWARD:
        return apply_reward(state, action.src, action.subtype, player, rules)
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
                "unit_kills", "unit_attacked", "unit_home", "unit_carried"):
        getattr(state, arr)[dst] = getattr(state, arr)[src]
    state.unit_moved[dst] = 1
    _clear_unit(state, src)

    boarded = _resolve_boarding(state, dst, rules)

    # NB: moving onto a city does NOT capture it. Capture is its own action
    # (see A_CAPTURE / capture_targets), which is why taking a village costs a
    # turn of standing on it.
    return {"type": "move", "src": src, "dst": dst, **boarded}


def _resolve_boarding(state, tile: int, rules: Ruleset) -> dict:
    """Turn a land unit into a hull on a Port, and back again on land.

    The hull is a vehicle, not a replacement: the passenger's type is parked in
    `unit_carried` and its HP is left untouched, which is what the wiki means by
    listing naval health as "Varies". Boarding and landing both consume the rest
    of the turn -- "unable to perform actions for the rest of that turn, even if
    they had excess movement points or had not attacked".
    """
    utype = int(state.unit_type[tile])
    carried = int(state.unit_carried[tile])
    on_land = bool(rules.terrain_is_land[state.terrain[tile]])

    if carried == C.U_NONE and not rules.unit_naval[utype] and not on_land:
        # Stepping onto a Port: board a hull. A Giant is too big for a Raft and
        # gets a Juggernaut, per the Port's description.
        hull = C.U_JUGGERNAUT if utype == C.U_GIANT else C.U_RAFT
        state.unit_carried[tile] = utype
        state.unit_type[tile] = hull
        state.unit_moved[tile] = 1
        state.unit_attacked[tile] = 1
        return {"embarked": rules.unit_names[hull]}

    if carried != C.U_NONE and on_land:
        state.unit_type[tile] = carried
        state.unit_carried[tile] = C.U_NONE
        state.unit_moved[tile] = 1
        state.unit_attacked[tile] = 1
        return {"disembarked": rules.unit_names[carried]}
    return {}


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
    rehome_unit(state, city, city)   # this city now supports it
    return {"type": "train", "city": city, "unit": uid}


def _apply_capture(state, player: int, action: Action, rules: Ruleset) -> dict:
    tile = action.src
    if not can_capture(state, tile, rules):
        raise ValueError(f"illegal capture at tile {tile}")
    previous = int(state.city_owner[tile])
    _capture_city(state, player, tile)
    # Wiki: a unit that captures a city is supported by that city from then on.
    rehome_unit(state, tile, tile)
    if rules.capture_ends_unit_turn:
        state.unit_moved[tile] = 1
        state.unit_attacked[tile] = 1
    # Taking a city can join it to the captor's road network and can sever the
    # previous owner's, so both sides are re-settled.
    conn = recompute_connections(state, player, rules)
    if previous != C.NEUTRAL:
        recompute_connections(state, previous, rules)
    return {"type": "capture", "tile": tile, "from": previous,
            "connected": sorted(conn["connected"])}


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
    # Free the supporting city's slot — a dead unit stops taking up space.
    state.unit_home[tile] = -1
    state.unit_veteran[tile] = 0
    state.unit_kills[tile] = 0
    state.unit_carried[tile] = C.U_NONE
