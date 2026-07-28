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
    """Defense multiplier for the unit standing on `tile` (it is the defender).

    Cities are pickier than they look, and the previous version got all three
    conditions wrong:

    * A city fortifies **only its owner's** units. "Invading units besieging a
      city do not receive any defence bonus" — so occupying an enemy city must
      not hand the invader that city's walls, which is what used to happen and
      made a captured city harder to take back than an open field.
    * A city fortifies **only units with the `fortify` skill**. A Giant standing
      in a city gets nothing (it may still get a terrain bonus elsewhere).
    * A city **without** walls still fortifies (1.5x); walls make it 4x. We
      previously gave unwalled cities nothing at all.

    Uncaptured neutral villages never fortify anyone, which falls out of the
    ownership test since `city_owner` is NEUTRAL for them.
    """
    utype = int(state.unit_type[tile])
    terrain_bonus = float(rules.terrain_defense_bonus[state.terrain[tile]])
    if utype == C.U_NONE:
        return terrain_bonus

    city_owner = int(state.city_owner[tile])
    if state.city_level[tile] > 0 and city_owner != C.NEUTRAL:
        garrisoned = (city_owner == int(state.unit_owner[tile])
                      and "fortify" in rules.unit_skills[utype])
        if not garrisoned:
            return rules.defense_bonus_map["none"]
        key = "city_walls" if state.city_walls[tile] else "city"
        return rules.defense_bonus_map[key]
    return terrain_bonus


def effective_max_hp(state, tile: int, rules: Ruleset) -> int:
    """Max HP of the unit on `tile`, following a naval hull to its passenger.

    The wiki lists naval health as "Varies" because a hull does not have health
    of its own -- it carries the passenger's. Reading `unit_max_hp[hull]` would
    silently use the placeholder value in units.toml.
    """
    utype = int(state.unit_type[tile])
    carried = int(state.unit_carried[tile])
    if carried != C.U_NONE and rules.unit_hull_hp_varies[utype]:
        return int(rules.unit_max_hp[carried])
    return int(rules.unit_max_hp[utype])


def resolve_attack(state, src: int, dst: int, rules: Ruleset) -> CombatResult:
    """Apply an attack from the unit on `src` against the unit on `dst`.

    Mutates `state` in place. Assumes legality was already checked (there is an
    attacker on src, an enemy on dst, and dst is within the attacker's range).
    """
    a_type = state.unit_type[src]
    d_type = state.unit_type[dst]
    a_atk = float(rules.unit_attack[a_type])
    d_def = float(rules.unit_defense[d_type])
    a_maxhp = effective_max_hp(state, src, rules)
    d_maxhp = effective_max_hp(state, dst, rules)
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
