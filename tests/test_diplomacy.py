"""Peace treaties, tribe relations, and the Embassy.

Quoted from `.wikicache/Strategy.wiki`, `.wikicache/Tribe Relations.wiki` and
`.wikicache/Embassy.wiki`:

    "The Peace Treaty offers another player the opportunity to become friends
     indefinitely. The Strategy technology is required to send, but not respond
     to, a peace treaty. While two tribes are at peace, they cannot perform any
     hostile actions, such as attacking, occupying cities, etc. Their units can
     also freely move past each other and use each other's roads. And income
     from Embassies is doubled. A peace treaty can be unilaterally terminated,
     but the player that breaks the peace treaty cannot attack the other player
     on that turn. Upon breaking the treaty, all of the aggressor's active units
     will be frozen for the rest of the turn, and any of their units within the
     enemy's borders will be disbanded without any star refund."

    "To be at war, either side must have attacked the other side at most a turn
     ago. After that, the relation goes back to neutral."

    "[The Embassy] produces two stars per turn for the creator ... as well as the
     tribe whose capital it was built in ... doubled to four stars per turn to
     each party if they have a Peace Treaty active ... Embassies may only be
     built in capital cities controlled by its original owner ... You may only
     build one embassy in each tribe. Attacking a tribe (a unit or a city) will
     destroy the attacker's embassy in the defender's capital ... as well as the
     defender's embassy in the attacker's capital."
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    can_capture,
    enumerate_actions,
)
from polyengine.core.combat import resolve_attack
from polyengine.core.diplomacy import (
    at_peace,
    at_war,
    can_offer_peace,
    embassy_income,
    offered,
    relation,
    roll_turn,
)
from polyengine.core.economy import can_build, player_income
from polyengine.core.game import Game
from polyengine.core.movement import attack_targets, reachable
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11
MID = 5 * N + 5


def blank(seed=1, players=2, all_tech=True):
    tribes = ["imperius", "bardur", "xin_xi", "oumaji"][:players]
    game = Game.new(size=N, seed=seed, tribes=tribes)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.resource[t] = C.R_NONE
        s.terrain[t] = C.T_FIELD
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.owner[t] = C.NEUTRAL
    for p in range(players):
        s.tech[p] = 0
        if all_tech:
            for k in range(R.n_techs):
                s.grant_tech(p, k)
        s.stars[p] = 50
    s.current_player = 0
    return game, s


def put(s, tile, utype=C.U_WARRIOR, player=0, hp=None):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = hp if hp is not None else int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    return tile


def make_peace(s):
    apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 1), R)
    s.current_player = 1
    apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 0), R)
    s.current_player = 0
    assert at_peace(s, 0, 1)


# --- forming a treaty --------------------------------------------------------------
class TestFormingATreaty:
    def test_offering_alone_is_not_peace(self):
        _, s = blank()
        apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 1), R)
        assert offered(s, 0, 1)
        assert not at_peace(s, 0, 1)

    def test_both_sides_offering_concludes_it(self):
        _, s = blank()
        make_peace(s)
        assert at_peace(s, 1, 0), "the treaty is symmetric"

    def test_offering_needs_strategy(self):
        _, s = blank(all_tech=False)
        assert not can_offer_peace(s, 0, 1, R)
        s.grant_tech(0, R.tech_id["strategy"])
        assert can_offer_peace(s, 0, 1, R)

    def test_responding_does_not_need_strategy(self):
        """"required to send, but not respond to, a peace treaty"."""
        _, s = blank(all_tech=False)
        s.grant_tech(0, R.tech_id["strategy"])
        apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 1), R)
        assert not s.has_tech(1, R.tech_id["strategy"])
        assert can_offer_peace(s, 1, 0, R), "an answer needs no technology"
        s.current_player = 1
        apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 0), R)
        assert at_peace(s, 0, 1)

    def test_an_offer_is_not_repeated(self):
        _, s = blank()
        apply_action(s, Action(C.A_PEACE_OFFER, -1, -1, 1), R)
        assert not can_offer_peace(s, 0, 1, R)

    def test_nobody_makes_peace_with_themselves(self):
        _, s = blank()
        assert not can_offer_peace(s, 0, 0, R)
        assert not at_peace(s, 0, 0)

    def test_the_actions_show_up_in_the_action_space(self):
        _, s = blank()
        assert any(a.atype == C.A_PEACE_OFFER and a.subtype == 1
                   for a in enumerate_actions(s, 0, R))
        make_peace(s)
        assert any(a.atype == C.A_PEACE_BREAK and a.subtype == 1
                   for a in enumerate_actions(s, 0, R))


# --- war is a memory, not a state -------------------------------------------------
class TestWar:
    def test_attacking_starts_a_war(self):
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR, player=0)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        assert relation(s, 0, 1) == "neutral"
        resolve_attack(s, a, MID + 1, R)
        assert relation(s, 0, 1) == "war"

    def test_it_is_symmetric(self):
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR, player=0)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        resolve_attack(s, a, MID + 1, R)
        assert at_war(s, 1, 0)

    def test_war_lapses_after_a_turn(self):
        """"To be at war, either side must have attacked the other side at most
        a turn ago. After that, the relation goes back to neutral."""
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR, player=0)
        put(s, MID + 1, C.U_DEFENDER, player=1, hp=99)
        resolve_attack(s, a, MID + 1, R)
        assert at_war(s, 0, 1)
        roll_turn(s, 0)
        assert at_war(s, 0, 1), "still remembered one turn later"
        roll_turn(s, 0)
        assert not at_war(s, 0, 1)
        assert relation(s, 0, 1) == "neutral"


# --- what a treaty forbids ------------------------------------------------------------
class TestPeaceForbidsHostility:
    def test_an_ally_cannot_be_attacked(self):
        _, s = blank()
        a = put(s, MID, C.U_WARRIOR, player=0)
        put(s, MID + 1, C.U_WARRIOR, player=1)
        assert attack_targets(s, a, R), "attackable before the treaty"
        make_peace(s)
        assert attack_targets(s, a, R) == []

    def test_an_allys_city_cannot_be_captured(self):
        """"they cannot perform any hostile actions, such as attacking,
        occupying cities, etc."."""
        _, s = blank()
        s.city_level[MID] = 1
        s.city_owner[MID] = 1
        put(s, MID, C.U_WARRIOR, player=0)
        assert can_capture(s, MID, R)
        make_peace(s)
        assert not can_capture(s, MID, R)

    def test_an_ally_cannot_be_infiltrated(self):
        from polyengine.core.movement import infiltrate_targets
        _, s = blank()
        s.city_level[MID + 1] = 1
        s.city_owner[MID + 1] = 1
        cloak = put(s, MID, C.U_CLOAK, player=0)
        assert infiltrate_targets(s, cloak, R)
        make_peace(s)
        assert infiltrate_targets(s, cloak, R) == []

    def test_a_neutral_village_is_still_fair_game(self):
        _, s = blank()
        make_peace(s)
        s.city_level[MID] = 1
        s.city_owner[MID] = C.NEUTRAL
        put(s, MID, C.U_WARRIOR, player=0)
        assert can_capture(s, MID, R)


# --- what a treaty allows ---------------------------------------------------------------
class TestPeaceAllowsPassage:
    def test_allied_units_do_not_exert_zone_of_control(self):
        _, s = blank()
        rider = put(s, MID, C.U_RIDER, player=0)
        put(s, MID + 1 + N, C.U_WARRIOR, player=1)
        before = len(reachable(s, rider, R))
        make_peace(s)
        assert len(reachable(s, rider, R)) > before

    def test_units_move_past_each_other(self):
        """"Their units can also freely move past each other" -- through, but
        never onto: one unit per tile is still the rule."""
        _, s = blank()
        rider = put(s, MID, C.U_RIDER, player=0)
        wall = put(s, MID + 1, C.U_WARRIOR, player=1)
        make_peace(s)
        dests = reachable(s, rider, R)
        assert MID + 2 in dests, "the tile beyond the ally is reachable"
        assert wall not in dests, "but not the ally's own tile"

    def test_allied_roads_are_ours_to_use(self):
        """"...and use each other's roads"."""
        def furthest(peace):
            _, s = blank()
            for t in (MID, MID + 1, MID + 2, MID + 3, MID + 4):
                s.improvement[t] = C.IMP_ROAD
            s.owner[MID] = 0
            for t in (MID + 1, MID + 2, MID + 3, MID + 4):
                s.owner[t] = 1
            src = put(s, MID, C.U_RIDER, player=0)
            if peace:
                make_peace(s)
            row = [t for t in reachable(s, src, R)
                   if t // N == MID // N and t % N > MID % N]
            return max((t % N for t in row), default=MID % N)

        assert furthest(peace=True) > furthest(peace=False)

    def test_an_allys_cloak_is_visible(self):
        """"Cloaks that belong to players with which the Cloak's owner has made
        peace are also always visible and do not create the eye."""
        from polyengine.core.fog import cloak_adjacent, recompute_all
        _, s = blank()
        cloak = put(s, MID + 1, C.U_CLOAK, player=1)
        s.unit_hidden[cloak] = 1
        put(s, MID, C.U_WARRIOR, player=0)
        recompute_all(s)
        assert int(s.mem_unit_type[0][cloak]) == C.U_NONE, "hidden while neutral"
        assert cloak_adjacent(s, 0)[MID], "and it raises an eye"
        make_peace(s)
        recompute_all(s)
        assert int(s.mem_unit_type[0][cloak]) == C.U_CLOAK
        assert not cloak_adjacent(s, 0)[MID], "an ally's Cloak raises no eye"


# --- breaking it ---------------------------------------------------------------------
class TestBreakingATreaty:
    def setup_border(self, s):
        for t in (MID + 3, MID + 4):
            s.owner[t] = 1
        inside = put(s, MID + 3, C.U_WARRIOR, player=0)
        outside = put(s, MID, C.U_WARRIOR, player=0)
        return inside, outside

    def test_it_takes_two_to_keep_peace_but_one_to_end_it(self):
        _, s = blank()
        make_peace(s)
        apply_action(s, Action(C.A_PEACE_BREAK, -1, -1, 1), R)
        assert not at_peace(s, 0, 1)
        assert not at_peace(s, 1, 0)

    def test_every_unit_is_frozen_for_the_rest_of_the_turn(self):
        _, s = blank()
        make_peace(s)
        _, outside = self.setup_border(s)
        apply_action(s, Action(C.A_PEACE_BREAK, -1, -1, 1), R)
        assert s.unit_moved[outside] and s.unit_attacked[outside]
        assert reachable(s, outside, R) == []

    def test_the_aggressor_cannot_attack_that_turn(self):
        """A frozen army cannot strike, which is the rule's whole point."""
        _, s = blank()
        make_peace(s)
        a = put(s, MID, C.U_WARRIOR, player=0)
        put(s, MID + 1, C.U_WARRIOR, player=1)
        apply_action(s, Action(C.A_PEACE_BREAK, -1, -1, 1), R)
        assert attack_targets(s, a, R) == []

    def test_units_inside_their_borders_are_destroyed(self):
        """"any of their units within the enemy's borders will be disbanded
        without any star refund"."""
        _, s = blank()
        make_peace(s)
        inside, outside = self.setup_border(s)
        stars = int(s.stars[0])
        info = apply_action(s, Action(C.A_PEACE_BREAK, -1, -1, 1), R)
        assert int(s.unit_type[inside]) == C.U_NONE
        assert int(s.unit_type[outside]) != C.U_NONE
        assert inside in info["disbanded"]
        assert int(s.stars[0]) == stars, "without any star refund"

    def test_peace_can_be_made_again_afterwards(self):
        _, s = blank()
        make_peace(s)
        apply_action(s, Action(C.A_PEACE_BREAK, -1, -1, 1), R)
        assert can_offer_peace(s, 0, 1, R)


# --- the Embassy -------------------------------------------------------------------------
class TestEmbassy:
    def capitals(self, s):
        cap0, cap1 = MID, MID + 4
        for tile, owner in ((cap0, 0), (cap1, 1)):
            s.city_level[tile] = 1
            s.city_owner[tile] = owner
            s.city_capital[tile] = 1
            s.owner[tile] = owner
            s.tile_tribe[tile] = int(s.tribe[owner])
            s.fog_explored[0][tile] = 1
            s.fog_explored[1][tile] = 1
        return cap0, cap1

    def test_it_costs_five_stars(self):
        assert int(R.imp_cost[C.IMP_EMBASSY]) == 5

    def test_it_pays_two_to_each_side(self):
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        assert embassy_income(s, cap1, 0, R) == 2, "the builder"
        assert embassy_income(s, cap1, 1, R) == 2, "the host"
        assert embassy_income(s, cap1, 0, R) == embassy_income(s, cap1, 1, R)

    def test_a_treaty_doubles_it(self):
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        make_peace(s)
        assert embassy_income(s, cap1, 0, R) == 4
        assert embassy_income(s, cap1, 1, R) == 4

    def test_the_doubling_reaches_the_income_total(self):
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        before = player_income(s, 0, R)
        make_peace(s)
        assert player_income(s, 0, R) == before + 2

    def test_it_reveals_the_host_capitals_surroundings(self):
        """"Building an Embassy will additionally reveal a 3x3 area centered on
        the host capital city."""
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        s.fog_explored[0][:] = 0
        s.fog_explored[0][cap1] = 1
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        assert int(s.fog_explored[0].sum()) >= 9

    def test_only_one_per_tribe(self):
        """"You may only build one embassy in each tribe."

        A tribe has exactly one capital, so in an ordinary game the rule is
        implied by there being nowhere else to put a second. The second capital
        here is artificial, and exists to prove the guard is real rather than
        incidental.
        """
        _, s = blank(players=3)
        cap0, cap1 = self.capitals(s)
        second = MID + 4 * N
        for field, value in (("city_level", 1), ("city_owner", 1),
                             ("city_capital", 1), ("owner", 1)):
            getattr(s, field)[second] = value
        s.tile_tribe[second] = int(s.tribe[1])
        s.fog_explored[0][second] = 1

        assert can_build(s, second, C.IMP_EMBASSY, 0, R)
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        assert not can_build(s, second, C.IMP_EMBASSY, 0, R), \
            "a second embassy in the same tribe"

    def test_not_in_a_captured_capital(self):
        """"Embassies may only be built in capital cities controlled by its
        original owner, not in capitals captured by other tribes."""
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        assert can_build(s, cap1, C.IMP_EMBASSY, 0, R)
        s.city_owner[cap1] = 1                       # still theirs...
        s.tile_tribe[cap1] = int(s.tribe[0])         # ...but originally ours
        assert not can_build(s, cap1, C.IMP_EMBASSY, 0, R)

    def test_not_while_at_war(self):
        """"...can only be built in other tribes' capitals when players are not
        at war with them."""
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        assert can_build(s, cap1, C.IMP_EMBASSY, 0, R)
        a = put(s, MID + 1, C.U_WARRIOR, player=0)
        put(s, MID + 2, C.U_DEFENDER, player=1, hp=99)
        resolve_attack(s, a, MID + 2, R)
        assert at_war(s, 0, 1)
        assert not can_build(s, cap1, C.IMP_EMBASSY, 0, R)

    def test_attacking_razes_both_embassies(self):
        """"Attacking a tribe (a unit or a city) will destroy the attacker's
        embassy in the defender's capital ... as well as the defender's embassy
        in the attacker's capital."""
        _, s = blank()
        cap0, cap1 = self.capitals(s)
        apply_action(s, Action(C.A_BUILD, cap1, -1, C.IMP_EMBASSY), R)
        s.current_player = 1
        apply_action(s, Action(C.A_BUILD, cap0, -1, C.IMP_EMBASSY), R)
        s.current_player = 0
        assert int(s.improvement[cap0]) == C.IMP_EMBASSY
        assert int(s.improvement[cap1]) == C.IMP_EMBASSY

        a = put(s, MID + 1, C.U_WARRIOR, player=0)
        put(s, MID + 2, C.U_DEFENDER, player=1, hp=99)
        resolve_attack(s, a, MID + 2, R)
        assert int(s.improvement[cap0]) == C.IMP_NONE
        assert int(s.improvement[cap1]) == C.IMP_NONE

    def test_a_third_partys_embassy_survives_someone_elses_war(self):
        _, s = blank(players=3)
        cap0, cap1 = self.capitals(s)
        cap2 = MID + 4 * N
        s.city_level[cap2] = 1
        s.city_owner[cap2] = 2
        s.city_capital[cap2] = 1
        s.owner[cap2] = 2
        s.tile_tribe[cap2] = int(s.tribe[2])
        s.fog_explored[0][cap2] = 1
        apply_action(s, Action(C.A_BUILD, cap2, -1, C.IMP_EMBASSY), R)
        a = put(s, MID + 1, C.U_WARRIOR, player=0)
        put(s, MID + 2, C.U_DEFENDER, player=1, hp=99)
        resolve_attack(s, a, MID + 2, R)
        assert int(s.improvement[cap2]) == C.IMP_EMBASSY


# --- observation ------------------------------------------------------------------------
def test_relations_are_in_the_global_vector():
    """Your own treaties are always on your own screen, so the policy may see
    them — unlike the map type, which it has to infer."""
    from polyengine import obs as O
    _, s = blank()
    stride = 4 + R.n_techs
    before = O.encode_global(s, 0, R)
    assert before[2 * stride + 2] == 0.0
    make_peace(s)
    after = O.encode_global(s, 0, R)
    assert after[2 * stride + 2] == 1.0, "at peace with our only opponent"


def test_relations_survive_a_serialization_round_trip():
    _, s = blank()
    make_peace(s)
    clone = type(s).from_dict(s.to_dict(), R)
    assert at_peace(clone, 0, 1)
