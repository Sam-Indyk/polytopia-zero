"""Hot-loop combat math as a Numba kernel.

Kept deliberately tiny and dependency-free so it JIT-compiles instantly and can
be called directly from golden tests (an njit function is also callable from
plain Python). The exact formula and its provenance live in data/combat.toml.
"""
from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def _round_half_up(x: float) -> int:
    # Math.round semantics (0.5 rounds up), matching the game client — NOT
    # Python's banker's rounding.
    return int(math.floor(x + 0.5))


@njit(cache=True)
def damage_numbers(att_attack: float, att_hp: int, att_max_hp: int,
                   def_defense: float, def_hp: int, def_max_hp: int,
                   defense_bonus: float, accuracy: float):
    """Return (attack_result, defense_result) for one melee/ranged exchange.

    attack_result  = damage dealt to the defender.
    defense_result = retaliation damage the attacker would take IF the defender
                     survives and the attack is melee (caller enforces that).
    """
    attack_force = att_attack * (att_hp / att_max_hp)
    defense_force = def_defense * (def_hp / def_max_hp) * defense_bonus
    total = attack_force + defense_force
    if total <= 0.0:
        return 0, 0
    attack_result = _round_half_up((attack_force / total) * att_attack * accuracy)
    defense_result = _round_half_up((defense_force / total) * def_defense * accuracy)
    return attack_result, defense_result
