"""Golden tests for the combat damage formula.

Expected numbers are hand-computed from the wiki formula (see data/combat.toml)
with accuracy=4.5 and round-half-up. These pin the single most bug-prone piece
of the engine; if the formula or rounding ever drifts, these fail first.
"""
import math

import pytest

from polyengine.core import constants as C
from polyengine.core.combat import resolve_attack
from polyengine.core.rules import load_ruleset
from polyengine.core.state import GameState
from polyengine.kernels.combat_kernel import _round_half_up, damage_numbers

RULES = load_ruleset()
ACC = RULES.accuracy


@pytest.mark.parametrize("x,expected", [
    (4.5, 5), (2.5, 3), (0.5, 1), (0.49, 0), (3.0, 3), (-0.5, 0), (9.99, 10),
])
def test_round_half_up(x, expected):
    assert _round_half_up(x) == expected


# (att_atk, att_hp, att_max, def_def, def_hp, def_max, bonus) -> (atk_res, def_res)
GOLDEN = [
    # Warrior vs Warrior, full HP, open ground.
    ((2.0, 10, 10, 2.0, 10, 10, 1.0), (5, 5)),
    # Warrior vs Defender, full HP, open ground.
    ((2.0, 10, 10, 3.0, 15, 15, 1.0), (4, 8)),
    # Warrior vs Defender in a friendly city with no walls (standard 1.5x).
    ((2.0, 10, 10, 3.0, 15, 15, 1.5), (3, 9)),
    # Warrior vs Defender behind a city wall (4x). The wall bonus was 1.5 here
    # until it was reconciled against the wiki's damage formula, which makes
    # walls nearly three times weaker than they should be.
    # attack 2*(10/10)=2, defence 3*(15/15)*4=12, total 14
    #   attack_result  = round_half_up(2/14  * 2 * 4.5) = round(1.29)  =  1
    #   defence_result = round_half_up(12/14 * 3 * 4.5) = round(11.57) = 12
    ((2.0, 10, 10, 3.0, 15, 15, 4.0), (1, 12)),
    # Wounded Warrior (5/10) vs full Warrior.
    ((2.0, 5, 10, 2.0, 10, 10, 1.0), (3, 6)),
    # Catapult (att 4) vs full Warrior.
    ((4.0, 10, 10, 2.0, 10, 10, 1.0), (12, 3)),
    # Knight (att 3.5) vs full Warrior.
    ((3.5, 10, 10, 2.0, 10, 10, 1.0), (10, 3)),
]


@pytest.mark.parametrize("args,expected", GOLDEN)
def test_damage_numbers(args, expected):
    a_atk, a_hp, a_max, d_def, d_hp, d_max, bonus = args
    assert damage_numbers(a_atk, a_hp, a_max, d_def, d_hp, d_max, bonus, ACC) == expected


def _two_unit_state(atk_type, def_type, atk_hp=None, def_hp=None):
    s = GameState(RULES, size=3, n_players=2, map_seed=0)
    s.unit_type[0] = atk_type
    s.unit_owner[0] = 0
    s.unit_hp[0] = atk_hp if atk_hp is not None else RULES.unit_max_hp[atk_type]
    s.unit_type[1] = def_type
    s.unit_owner[1] = 1
    s.unit_hp[1] = def_hp if def_hp is not None else RULES.unit_max_hp[def_type]
    return s


def test_melee_exchange_both_survive():
    s = _two_unit_state(C.U_WARRIOR, C.U_WARRIOR)
    res = resolve_attack(s, 0, 1, RULES)
    assert (res.damage_to_defender, res.retaliation_to_attacker) == (5, 5)
    assert s.unit_hp[1] == 5 and s.unit_hp[0] == 5
    assert not res.defender_died and not res.attacker_died
    assert not res.attacker_advanced
    assert s.unit_attacked[0] == 1


def test_melee_kill_advances_into_tile():
    s = _two_unit_state(C.U_WARRIOR, C.U_WARRIOR, def_hp=3)
    res = resolve_attack(s, 0, 1, RULES)
    assert res.defender_died and res.attacker_advanced
    assert s.unit_type[0] == C.U_NONE            # vacated origin
    assert s.unit_type[1] == C.U_WARRIOR and s.unit_owner[1] == 0
    assert s.unit_kills[1] == 1


def test_ranged_takes_no_retaliation():
    # Catapult vs a Giant that survives: ranged attacker keeps full HP.
    s = _two_unit_state(C.U_CATAPULT, C.U_GIANT)
    res = resolve_attack(s, 0, 1, RULES)
    assert not res.defender_died
    assert s.unit_hp[0] == RULES.unit_max_hp[C.U_CATAPULT]  # no retaliation
    assert not res.attacker_advanced                        # ranged never advances
