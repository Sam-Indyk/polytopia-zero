"""Every unit skill, quoted from `.wikicache/Unit Skills.wiki`.

Eleven of these were present in `units.toml` and read by no code at all, which
is a failure mode with no symptom: the game runs, the agent trains, and the
Catapult is quietly a better unit than it should be. So each skill gets a test
that fails if the mechanic disappears, and each test names the unit whose
identity depends on it.

    Dash        "allows a unit to attack or break ice after moving in the same
                 turn"
    Escape      "allows a unit to move after attacking in the same turn. The
                 unit cannot attack again in the same turn if they have dash"
    Persist     "allows a unit to attack again immediately after killing an
                 enemy unit. There is no limit on the number of kills"
    Stiff       "prevents a unit from being able to do retaliation attacks when
                 attacked by an enemy unit"
    Surprise    "prevents a unit from triggering retaliation attacks when
                 attacking an enemy unit"
    Splash      "allows a unit to damage or poison enemy units adjacent to the
                 targeted unit"
    Stomp       "causes a unit to deal damage to all adjacent enemy units when
                 it moves"
    Convert     "allows a unit to convert an enemy unit into a friendly unit by
                 attacking it"
    Heal        "heals all adjacent friendly units by up to 4 HP"
    Scout       "allows a unit to explore a 5x5 area instead of a 3x3 area"
    Independent "units with this skill do not take up a population slot in or
                 belong to any city"
    Static      "prevents a unit from becoming a veteran"
    Creep       "allows a unit to ignore movement barriers imposed by terrain
                 **except mountains**"
    Hide        "allows a unit to hide itself ... such units also ignore zone of
                 control and can move through enemy units"
    Fortify     "allows a unit to receive a defence bonus in a city"
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    can_heal_others,
    enumerate_actions,
)
from polyengine.core.combat import effective_max_hp, resolve_attack
from polyengine.core.fog import recompute_all
from polyengine.core.game import Game
from polyengine.core.movement import attack_targets, reachable
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11
MID = 5 * N + 5


def blank(seed=1, all_tech=True):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.unit_hidden[t] = 0
        s.unit_facing[t] = -1
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.resource[t] = C.R_NONE
        s.terrain[t] = C.T_FIELD
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.owner[t] = C.NEUTRAL
    if all_tech:
        for k in range(R.n_techs):
            s.grant_tech(0, k)
            s.grant_tech(1, k)
    s.current_player = 0
    return game, s


def put(s, tile, utype=C.U_WARRIOR, player=0, hp=None, kills=0):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = hp if hp is not None else int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    s.unit_veteran[tile] = 0
    s.unit_kills[tile] = kills
    return tile


def skilled(skill):
    return [u for u in range(R.n_units) if skill in R.unit_skills[u]]


# --- dash / escape / persist: who may act twice ----------------------------------
class TestActionEconomy:
    def test_a_unit_without_dash_may_not_attack_after_moving(self):
        """The Catapult must choose between repositioning and firing."""
        _, s = blank()
        cat = put(s, MID, C.U_CATAPULT)
        put(s, MID + 3, C.U_WARRIOR, player=1)
        assert attack_targets(s, cat, R), "in range before moving"
        apply_action(s, Action(C.A_MOVE, cat, MID + 1), R)
        assert attack_targets(s, MID + 1, R) == []

    def test_a_unit_with_dash_may_attack_after_moving(self):
        _, s = blank()
        w = put(s, MID, C.U_WARRIOR)
        put(s, MID + 2, C.U_WARRIOR, player=1)
        apply_action(s, Action(C.A_MOVE, w, MID + 1), R)
        assert attack_targets(s, MID + 1, R)

    def test_escape_lets_the_rider_withdraw_after_attacking(self):
        _, s = blank()
        r = put(s, MID, C.U_RIDER)
        put(s, MID + 1, C.U_WARRIOR, player=1, hp=99)
        apply_action(s, Action(C.A_ATTACK, r, MID + 1), R)
        assert reachable(s, r, R), "a Rider must be able to hit and run"

    def test_escape_plus_dash_may_not_attack_twice(self):
        """"The unit cannot attack again in the same turn if they have dash."""
        _, s = blank()
        r = put(s, MID, C.U_RIDER)
        put(s, MID + 1, C.U_WARRIOR, player=1, hp=99)
        assert "dash" in R.unit_skills[C.U_RIDER]
        apply_action(s, Action(C.A_ATTACK, r, MID + 1), R)
        assert attack_targets(s, r, R) == []

    def test_a_unit_without_escape_is_spent_by_attacking(self):
        _, s = blank()
        w = put(s, MID, C.U_WARRIOR)
        put(s, MID + 1, C.U_WARRIOR, player=1, hp=99)
        apply_action(s, Action(C.A_ATTACK, w, MID + 1), R)
        assert reachable(s, w, R) == []

    def test_persist_lets_the_knight_keep_killing(self):
        """"Allows a unit to attack again immediately after killing an enemy
        unit. There is no limit on the number of kills in a single turn."""
        _, s = blank()
        kn = put(s, MID, C.U_KNIGHT)
        for tile in (MID + 1, MID + 1 + N, MID + N):
            put(s, tile, C.U_WARRIOR, player=1, hp=1)
        killed = 0
        for _ in range(3):
            targets = attack_targets(s, kn, R)
            if not targets:
                break
            apply_action(s, Action(C.A_ATTACK, kn, targets[0]), R)
            kn = next(t for t in range(s.n_tiles)
                      if int(s.unit_type[t]) == C.U_KNIGHT)
            killed += 1
        assert killed == 3, f"the Knight stopped after {killed} kills"

    def test_persist_does_not_apply_when_the_target_survives(self):
        _, s = blank()
        kn = put(s, MID, C.U_KNIGHT)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        apply_action(s, Action(C.A_ATTACK, kn, MID + 1), R)
        assert attack_targets(s, kn, R) == []


# --- retaliation ------------------------------------------------------------------
class TestRetaliation:
    def test_a_stiff_defender_does_not_hit_back(self):
        """Mind Bender has defence 1 and the stiff skill -- the Catapult's
        defence is 0, so testing with it proves nothing."""
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR)
        put(s, MID + 1, C.U_MIND_BENDER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 1, R)
        assert int(s.unit_hp[a]) == before

    def test_a_normal_defender_does_hit_back(self):
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 1, R)
        assert int(s.unit_hp[a]) < before

    def test_surprise_means_the_dagger_takes_nothing_back(self):
        _, s = blank()
        a = put(s, MID, C.U_DAGGER)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 1, R)
        assert int(s.unit_hp[a]) == before

    def test_a_defender_out_of_its_own_range_cannot_hit_back(self):
        """"No retaliation occurs if the unit being attacked ... cannot reach
        ... the attacker (the attacker is outside the unit's range)."

        This is the case that made the old rule look right: a melee Defender
        shot from two tiles genuinely cannot answer. It was written down as
        "ranged attacks never draw retaliation", which is the usual consequence
        mistaken for the rule -- see the two tests below for where they part.
        """
        _, s = blank()
        a = put(s, MID, C.U_ARCHER)
        put(s, MID + 2, C.U_DEFENDER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 2, R)
        assert int(s.unit_hp[a]) == before

    def test_a_ranged_defender_does_hit_back_at_a_ranged_attacker(self):
        """An Archer shot by an Archer at two tiles is inside its own range, so
        it answers. Gating on the attacker's range made ranged-versus-ranged
        risk-free for whoever shot first."""
        _, s = blank()
        a = put(s, MID, C.U_ARCHER)
        put(s, MID + 2, C.U_ARCHER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 2, R)
        assert int(s.unit_hp[a]) < before, (
            "a defender that can reach the attacker retaliates, whatever the "
            "attacker's own range is")

    def test_an_adjacent_melee_defender_hits_back_at_an_archer(self):
        """The other half: range 2 does not mean "safe", it means "may stand
        off". An Archer that shoots from an adjacent tile is well inside a
        Warrior's range and takes the reply -- which is the whole reason to
        keep a bowman out of melee."""
        _, s = blank()
        a = put(s, MID, C.U_ARCHER)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 1, R)
        assert int(s.unit_hp[a]) < before

    def test_a_catapult_still_takes_nothing_back_from_anything(self):
        """Defence 0 deals 0 retaliation damage, and range 3 outruns every
        defender's reach -- so the siege unit's safety survives the fix."""
        _, s = blank()
        a = put(s, MID, C.U_CATAPULT)
        put(s, MID + 3, C.U_ARCHER, player=1, hp=99)
        before = int(s.unit_hp[a])
        resolve_attack(s, a, MID + 3, R)
        assert int(s.unit_hp[a]) == before

    def test_a_retaliation_kill_counts_toward_the_defenders_promotion(self):
        """"after it kills three enemy units (through attacks or retaliation)"."""
        _, s = blank()
        d = put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        for src in (MID, MID + 2, MID + N):
            a = put(s, src, C.U_WARRIOR, hp=1)
            resolve_attack(s, a, d, R)
        assert int(s.unit_kills[d]) == 3


# --- area effects -----------------------------------------------------------------
class TestAreaEffects:
    def test_the_bomber_splashes_neighbours_of_its_target(self):
        _, s = blank()
        s.terrain[MID] = C.T_WATER
        b = put(s, MID, C.U_BOMBER)
        tgt = put(s, MID + 2, C.U_WARRIOR, player=1, hp=99)
        near = put(s, MID + 2 + N, C.U_WARRIOR, player=1, hp=99)
        far = put(s, MID + 5, C.U_WARRIOR, player=1, hp=99)
        resolve_attack(s, b, tgt, R)
        assert int(s.unit_hp[near]) < 99, "adjacent to the target: splashed"
        assert int(s.unit_hp[far]) == 99, "three tiles away: untouched"

    def test_splash_does_not_hit_our_own_units(self):
        _, s = blank()
        s.terrain[MID] = C.T_WATER
        b = put(s, MID, C.U_BOMBER)
        tgt = put(s, MID + 2, C.U_WARRIOR, player=1, hp=99)
        friend = put(s, MID + 2 + N, C.U_WARRIOR, player=0, hp=99)
        resolve_attack(s, b, tgt, R)
        assert int(s.unit_hp[friend]) == 99

    def test_the_juggernaut_stomps_what_it_sails_past(self):
        _, s = blank()
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_WATER
        j = put(s, MID, C.U_JUGGERNAUT)
        victim = put(s, MID + 1 + N, C.U_RAFT, player=1, hp=99)
        friend = put(s, MID + 1 - N, C.U_RAFT, player=0, hp=99)
        apply_action(s, Action(C.A_MOVE, j, MID + 1), R)
        assert int(s.unit_hp[victim]) < 99
        assert int(s.unit_hp[friend]) == 99, "stomp is enemies only"

    def test_only_the_juggernaut_stomps(self):
        _, s = blank()
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_WATER
        put(s, MID, C.U_RAMMER)
        victim = put(s, MID + 1 + N, C.U_RAFT, player=1, hp=99)
        apply_action(s, Action(C.A_MOVE, MID, MID + 1), R)
        assert int(s.unit_hp[victim]) == 99


# --- the Mind Bender ---------------------------------------------------------------
class TestMindBender:
    def test_attacking_converts_instead_of_damaging(self):
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        enemy = put(s, MID + 1, C.U_WARRIOR, player=1, hp=7)
        resolve_attack(s, mb, enemy, R)
        assert int(s.unit_owner[enemy]) == 0
        assert int(s.unit_type[enemy]) == C.U_WARRIOR
        assert int(s.unit_hp[enemy]) == 7, "conversion is not damage"

    def test_a_converted_unit_is_spent_for_the_turn(self):
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        enemy = put(s, MID + 1, C.U_WARRIOR, player=1)
        resolve_attack(s, mb, enemy, R)
        assert s.unit_moved[enemy] and s.unit_attacked[enemy]

    def test_a_converted_unit_is_supported_by_the_capital(self):
        """"Converted units take up population in the mind bender's capital."""
        _, s = blank()
        cap = MID + 3
        s.city_level[cap] = 3
        s.city_owner[cap] = 0
        s.city_capital[cap] = 1
        mb = put(s, MID, C.U_MIND_BENDER)
        enemy = put(s, MID + 1, C.U_WARRIOR, player=1)
        resolve_attack(s, mb, enemy, R)
        assert int(s.unit_home[enemy]) == cap

    def test_an_independent_unit_is_converted_but_costs_no_slot(self):
        """"(except units with Independent skill)"."""
        _, s = blank()
        cap = MID + 3
        s.city_level[cap] = 3
        s.city_owner[cap] = 0
        s.city_capital[cap] = 1
        mb = put(s, MID, C.U_MIND_BENDER)
        enemy = put(s, MID + 1, C.U_DAGGER, player=1)
        resolve_attack(s, mb, enemy, R)
        assert int(s.unit_owner[enemy]) == 0
        assert int(s.unit_home[enemy]) == -1

    def test_heal_others_heals_every_wounded_neighbour(self):
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        hurt = put(s, MID + 1, C.U_WARRIOR, hp=2)
        healthy = put(s, MID - 1, C.U_WARRIOR)
        enemy = put(s, MID + N, C.U_WARRIOR, player=1, hp=2)
        assert can_heal_others(s, mb, 0, R)
        apply_action(s, Action(C.A_HEAL_OTHERS, mb, mb), R)
        assert int(s.unit_hp[hurt]) == 6, "4 HP, capped at maximum"
        assert int(s.unit_hp[healthy]) == int(R.unit_max_hp[C.U_WARRIOR])
        assert int(s.unit_hp[enemy]) == 2, "enemies are not friends"

    def test_heal_others_costs_the_turn(self):
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        put(s, MID + 1, C.U_WARRIOR, hp=2)
        apply_action(s, Action(C.A_HEAL_OTHERS, mb, mb), R)
        assert s.unit_moved[mb] and s.unit_attacked[mb]

    def test_heal_others_is_offered_by_the_action_space_and_the_masks(self):
        """Same reasoning as the promotion: a Mind Bender standing beside a
        wounded friend is rare enough that no integration run reaches it."""
        from polyengine.masks import ActionMasks
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        put(s, MID + 1, C.U_WARRIOR, hp=2)
        assert any(a.atype == C.A_HEAL_OTHERS and a.src == mb
                   for a in enumerate_actions(s, 0, R))
        m = ActionMasks(s, 0, R)
        assert m.types()[C.A_HEAL_OTHERS]
        assert m.sources(C.A_HEAL_OTHERS)[mb]

    def test_heal_others_is_not_offered_with_nobody_to_heal(self):
        _, s = blank()
        mb = put(s, MID, C.U_MIND_BENDER)
        put(s, MID + 1, C.U_WARRIOR)          # at full health
        assert not can_heal_others(s, mb, 0, R)
        assert not any(a.atype == C.A_HEAL_OTHERS
                       for a in enumerate_actions(s, 0, R))

    def test_only_the_mind_bender_can_heal_others(self):
        assert skilled("heal") == [C.U_MIND_BENDER]
        _, s = blank()
        w = put(s, MID, C.U_WARRIOR)
        put(s, MID + 1, C.U_WARRIOR, hp=2)
        assert not can_heal_others(s, w, 0, R)


# --- vision and support --------------------------------------------------------------
class TestScoutAndIndependent:
    def test_a_scout_sees_a_five_by_five(self):
        _, s = blank()
        put(s, MID, C.U_SCOUT)
        s.terrain[:] = C.T_WATER
        s.fog_explored[0][:] = 0
        recompute_all(s)
        assert int(s.fog_explored[0].sum()) == 25

    def test_an_ordinary_unit_sees_a_three_by_three(self):
        _, s = blank()
        put(s, MID, C.U_WARRIOR)
        s.fog_explored[0][:] = 0
        recompute_all(s)
        assert int(s.fog_explored[0].sum()) == 9

    def test_the_cloak_scouts_too(self):
        assert "scout" in R.unit_skills[C.U_CLOAK]

    def test_an_independent_unit_takes_no_city_slot(self):
        """A Dagger from a Cloak's revolt must not cost the raider army space."""
        from polyengine.core.economy import free_support, rehome_unit
        _, s = blank()
        city = MID
        s.city_level[city] = 1
        s.city_owner[city] = 0
        s.owner[city] = 0
        before = free_support(s, city)
        d = put(s, MID + 2, C.U_DAGGER)
        rehome_unit(s, d, city)
        assert int(s.unit_home[d]) == -1
        assert free_support(s, city) == before

    def test_an_ordinary_unit_does_take_a_slot(self):
        from polyengine.core.economy import free_support, rehome_unit
        _, s = blank()
        city = MID
        s.city_level[city] = 1
        s.city_owner[city] = 0
        s.owner[city] = 0
        before = free_support(s, city)
        w = put(s, MID + 2, C.U_WARRIOR)
        rehome_unit(s, w, city)
        assert free_support(s, city) == before - 1


# --- movement skills ------------------------------------------------------------------
class TestMovementSkills:
    def test_creep_ignores_forest_but_not_mountains(self):
        """"Allows a unit to ignore movement barriers imposed by terrain except
        mountains." The two wiki pages disagree; the Unit Skills page wins --
        zone of control is the *hide*/*sneak* skill's business, not creep's."""
        _, s = blank()
        cloak = put(s, MID, C.U_CLOAK)
        assert "creep" in R.unit_skills[C.U_CLOAK]
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_FOREST
        through_forest = reachable(s, cloak, R)
        for t in range(s.n_tiles):
            s.terrain[t] = C.T_MOUNTAIN
        s.terrain[MID] = C.T_FIELD
        through_mountain = reachable(s, cloak, R)
        assert len(through_forest) > len(through_mountain), \
            "a mountain still stops a creeping unit"

    def test_a_normal_unit_is_stopped_by_forest(self):
        _, s = blank()
        rider = put(s, MID, C.U_RIDER)
        open_ground = len(reachable(s, rider, R))
        for t in range(s.n_tiles):
            if t != MID:
                s.terrain[t] = C.T_FOREST
        assert len(reachable(s, rider, R)) < open_ground

    def test_hide_walks_through_enemy_units_but_not_onto_them(self):
        """"Such units also ignore zone of control and can move through enemy
        units."""
        _, s = blank()
        cloak = put(s, MID, C.U_CLOAK)
        wall = MID + 1
        put(s, wall, C.U_WARRIOR, player=1)
        dests = reachable(s, cloak, R)
        assert wall not in dests, "it may pass through, not stop on top"
        assert MID + 2 in dests, "the tile beyond the enemy is reachable"

    def test_an_ordinary_unit_is_blocked_by_an_enemy(self):
        _, s = blank()
        rider = put(s, MID, C.U_RIDER)
        put(s, MID + 1, C.U_WARRIOR, player=1)
        assert MID + 2 not in reachable(s, rider, R)


# --- static / fortify ------------------------------------------------------------------
class TestStaticAndFortify:
    def test_static_units_never_promote(self):
        from polyengine.core.actions import can_promote
        _, s = blank()
        for utype in skilled("static"):
            tile = put(s, MID, utype, kills=9)
            assert not can_promote(s, tile, 0, R), R.unit_names[utype]

    def test_a_non_static_unit_does_promote(self):
        from polyengine.core.actions import can_promote
        _, s = blank()
        tile = put(s, MID, C.U_WARRIOR, kills=3)
        assert can_promote(s, tile, 0, R)

    def test_fortify_is_what_a_city_defends(self):
        """The Swordsman has no fortify skill, so a city does nothing for it."""
        from polyengine.core.combat import defense_bonus_for
        _, s = blank()
        s.city_level[MID] = 1
        s.city_owner[MID] = 0
        s.owner[MID] = 0
        put(s, MID, C.U_DEFENDER)
        with_skill = defense_bonus_for(s, MID, R)
        put(s, MID, C.U_SWORDSMAN)
        without = defense_bonus_for(s, MID, R)
        assert "fortify" in R.unit_skills[C.U_DEFENDER]
        assert "fortify" not in R.unit_skills[C.U_SWORDSMAN]
        assert with_skill > without


# --- the data and the code agree --------------------------------------------------------
def test_every_skill_in_the_ruleset_is_read_by_the_engine():
    """The trap this whole file exists for: a skill in `units.toml` that no code
    consults is a rule the engine silently does not have."""
    import pathlib

    src = "".join(
        p.read_text(encoding="utf-8")
        for p in pathlib.Path("polyengine").rglob("*.py"))
    used = set()
    for utype in range(R.n_units):
        used |= set(R.unit_skills[utype])
    unread = sorted(s for s in used if f'"{s}"' not in src)
    assert not unread, f"skills present in data but read by no code: {unread}"
