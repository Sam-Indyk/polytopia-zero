"""Combat resolution: turns the kernel's damage numbers into state changes.

Handles the full melee exchange (hit + retaliation), ranged (no retaliation),
death, and moving the attacker into a vacated tile on a killing melee blow.
"""
from __future__ import annotations

from dataclasses import dataclass

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset
from polyengine.kernels.combat_kernel import damage_numbers


@dataclass
class CombatResult:
    attacker_tile: int
    defender_tile: int
    damage_to_defender: int
    retaliation_to_attacker: int
    defender_died: bool
    attacker_died: bool
    attacker_advanced: bool  # melee kill on an adjacent tile -> attacker moved in


def defense_bonus_for(state, tile: int, rules: Ruleset) -> float:
    """Defense multiplier a unit gets standing on `tile`.

    A garrisoned city (walls, or simply owning the city tile) fortifies; else the
    terrain's own bonus applies. City fortification takes precedence.
    """
    if state.city_owner[tile] != C.NEUTRAL and state.city_walls[tile]:
        return rules.defense_bonus_map["fortified"]
    return float(rules.terrain_defense_bonus[state.terrain[tile]])


def resolve_attack(state, src: int, dst: int, rules: Ruleset) -> CombatResult:
    """Apply an attack from the unit on `src` against the unit on `dst`.

    Mutates `state` in place. Assumes legality was already checked (there is an
    attacker on src, an enemy on dst, and dst is within the attacker's range).
    """
    a_type = state.unit_type[src]
    d_type = state.unit_type[dst]
    a_atk = float(rules.unit_attack[a_type])
    d_def = float(rules.unit_defense[d_type])
    a_maxhp = int(rules.unit_max_hp[a_type])
    d_maxhp = int(rules.unit_max_hp[d_type])
    a_range = int(rules.unit_range[a_type])
    dbonus = defense_bonus_for(state, dst, rules)

    atk_res, def_res = damage_numbers(
        a_atk, int(state.unit_hp[src]), a_maxhp,
        d_def, int(state.unit_hp[dst]), d_maxhp,
        dbonus, rules.accuracy,
    )

    defender_died = False
    attacker_died = False
    attacker_advanced = False

    state.unit_hp[dst] -= atk_res
    if state.unit_hp[dst] <= 0:
        defender_died = True
        state.unit_kills[src] += 1
        _clear_unit(state, dst)
        # Melee units advance into the vacated tile; ranged do not.
        if a_range == 1:
            _move_unit(state, src, dst)
            attacker_advanced = True
    else:
        # Retaliation only for melee attacks against a surviving defender.
        if a_range == 1:
            state.unit_hp[src] -= def_res
            if state.unit_hp[src] <= 0:
                attacker_died = True
                _clear_unit(state, src)

    # An attack always consumes the attacker's turn.
    if not attacker_died and not attacker_advanced:
        state.unit_moved[src] = 1
        state.unit_attacked[src] = 1
    elif attacker_advanced:
        state.unit_moved[dst] = 1
        state.unit_attacked[dst] = 1

    return CombatResult(src, dst, atk_res, def_res if not defender_died else 0,
                        defender_died, attacker_died, attacker_advanced)


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


def _move_unit(state, src: int, dst: int) -> None:
    state.unit_type[dst] = state.unit_type[src]
    state.unit_owner[dst] = state.unit_owner[src]
    state.unit_hp[dst] = state.unit_hp[src]
    state.unit_veteran[dst] = state.unit_veteran[src]
    state.unit_kills[dst] = state.unit_kills[src]
    state.unit_moved[dst] = state.unit_moved[src]
    state.unit_attacked[dst] = state.unit_attacked[src]
    _clear_unit(state, src)
