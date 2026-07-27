"""Guards on the current-version ruleset data.

These pin the *shape* of the current rules (naval roster present, Diplomacy
present, tech tree well-formed, current cost formula) so a data edit that drops
or mis-tiers content fails loudly. Exact combat/stat fidelity is the job of the
Steam replay fixtures via polyengine.testing.differential.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyengine.core.actions import research_cost
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()


# --- tech tree ----------------------------------------------------------------
def test_tech_tree_is_well_formed():
    for t in range(R.n_techs):
        tier = int(R.tech_tier[t])
        req = int(R.tech_requires[t])
        assert 1 <= tier <= 3
        if tier == 1:
            assert req == -1, f"{R.tech_names[t]} is T1 but has a prerequisite"
        else:
            assert req >= 0, f"{R.tech_names[t]} is T{tier} with no prerequisite"
            assert int(R.tech_tier[req]) == tier - 1, \
                f"{R.tech_names[t]} (T{tier}) requires a non-T{tier - 1} tech"


def test_tech_tree_has_no_cycles():
    for t in range(R.n_techs):
        seen, cur = set(), t
        while cur != -1:
            assert cur not in seen, f"cycle through {R.tech_names[t]}"
            seen.add(cur)
            cur = int(R.tech_requires[cur])


def test_current_naval_and_diplomacy_techs_present():
    for name in ("sailing", "navigation", "aquatism", "diplomacy", "philosophy"):
        assert name in R.tech_id, f"missing current tech: {name}"
    assert R.tech_names[int(R.tech_requires[R.tech_id["navigation"]])] == "sailing"
    assert R.tech_names[int(R.tech_requires[R.tech_id["diplomacy"]])] == "strategy"


# --- research pricing (current formula: tier * cities + 4) ---------------------
# Rows transcribed from the wiki's "Technology Costs" table.
@pytest.mark.parametrize("tier,cities,plain,literate", [
    (1, 0, 4, 3), (2, 0, 4, 3), (3, 0, 4, 3),      # no cities: flat 4
    (1, 1, 5, 4), (2, 1, 6, 4), (3, 1, 7, 5),
    (1, 2, 6, 4), (2, 2, 8, 6), (3, 2, 10, 7),
])
def test_research_cost_table(tier, cities, plain, literate):
    assert tier * cities + 4 == plain
    assert -(-plain * 2 // 3) == literate      # Literacy: ceil(cost * 2/3)


def test_research_cost_matches_formula_in_engine():
    game = Game.new(size=9, seed=1)
    s = game.state
    n_cities = int(((s.city_owner == 0) & (s.city_level > 0)).sum())
    for t in range(R.n_techs):
        want = int(R.tech_tier[t]) * n_cities + 4
        assert research_cost(s, 0, t, R) == want


def test_literacy_discounts_research_by_a_third():
    game = Game.new(size=9, seed=1)
    s = game.state
    phil = R.tech_id["philosophy"]
    target = R.tech_id["chivalry"]
    full = research_cost(s, 0, target, R)
    s.grant_tech(0, phil)
    discounted = research_cost(s, 0, target, R)
    assert discounted == -(-full * 2 // 3)      # ceil(full * 2/3)
    assert discounted < full


# --- unit roster ---------------------------------------------------------------
def test_current_naval_roster_present():
    for name in ("raft", "scout", "rammer", "bomber", "juggernaut"):
        assert name in R.unit_id, f"missing naval unit: {name}"
        assert R.unit_naval[R.unit_id[name]], f"{name} should be naval"


def test_diplomacy_units_present():
    for name in ("cloak", "dagger"):
        assert name in R.unit_id
    cloak = R.unit_id["cloak"]
    assert R.tech_names[int(R.unit_requires_tech[cloak])] == "diplomacy"
    assert "infiltrate" in R.unit_skills[cloak]
    # Verified against the wiki: the Cloak is a 5 HP, 0.5-defence infiltrator
    # that does have an attack of 2 (it is not a pure non-combat unit).
    assert float(R.unit_attack[cloak]) == 2.0
    assert int(R.unit_max_hp[cloak]) == 5
    assert float(R.unit_defense[cloak]) == 0.5


def test_naval_branch_matches_the_current_tree():
    """Fishing -> {Sailing, Ramming}; Sailing -> Navigation; Ramming -> Aquatism."""
    req = lambda name: R.tech_names[int(R.tech_requires[R.tech_id[name]])]
    assert req("sailing") == "fishing"
    assert req("ramming") == "fishing"
    assert req("navigation") == "sailing"
    assert req("aquatism") == "ramming"
    assert int(R.tech_tier[R.tech_id["aquatism"]]) == 3
    # Naval hulls are gated by the tech that unlocks them.
    unlock = lambda u: R.tech_names[int(R.unit_requires_tech[R.unit_id[u]])]
    assert unlock("raft") == "fishing"
    assert unlock("scout") == "sailing"
    assert unlock("rammer") == "ramming"
    assert unlock("bomber") == "navigation"


def test_tech_count_matches_the_current_game():
    """25 techs: 5 tier-one roots, 10 tier-two, 10 tier-three."""
    assert R.n_techs == 25
    tiers = [int(t) for t in R.tech_tier]
    assert tiers.count(1) == 5 and tiers.count(2) == 10 and tiers.count(3) == 10
    for removed in ("whaling", "engineering"):
        assert removed not in R.tech_id, f"{removed} was removed from the game"
    for name in ("ramming", "spiritualism"):
        assert name in R.tech_id


def test_naval_hulls_take_hp_from_their_passenger():
    """Wiki lists naval health as 'Varies'; those rows are flagged as placeholders."""
    for name in ("raft", "scout", "rammer", "bomber"):
        assert R.unit_hull_hp_varies[R.unit_id[name]], f"{name} should be flagged"
    assert not R.unit_hull_hp_varies[R.unit_id["warrior"]]


def test_defender_comes_from_strategy():
    """Current rules moved Defender onto Strategy (was Shields)."""
    assert R.tech_names[int(R.unit_requires_tech[R.unit_id["defender"]])] == "strategy"


def test_every_unit_tech_requirement_resolves():
    for u in range(R.n_units):
        req = int(R.unit_requires_tech[u])
        assert -1 <= req < R.n_techs


def test_trainable_units_are_land_and_affordable():
    trainable = [u for u in range(R.n_units) if R.unit_trainable[u]]
    assert trainable, "some units must be trainable"
    for u in trainable:
        assert not R.unit_naval[u], f"{R.unit_names[u]} trainable but naval"
        assert int(R.unit_cost[u]) > 0
        assert int(R.unit_max_hp[u]) > 0


def test_ranged_units_outrange_melee():
    assert int(R.unit_range[R.unit_id["archer"]]) == 2
    assert int(R.unit_range[R.unit_id["catapult"]]) == 3
    assert int(R.unit_range[R.unit_id["warrior"]]) == 1


def test_ruleset_versions_agree_across_data_files():
    assert R.version == "current-2026.07"
