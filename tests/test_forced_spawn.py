"""Forced unit spawns: when something must appear on an occupied tile.

Two places in the game do this, and both used to drop the reward on the floor
instead:

* a ruin whose reward is a unit -- the examiner is *always* standing on the ruin,
  since it had to start its turn there;
* a city awarded a super unit -- garrisoning the city you are levelling up is the
  normal thing to do, so the tile is usually taken.

Quoted from `.wikicache/Forced Unit Spawns.wiki`, which specifies the whole
mechanic:

    "A forced unit spawn occurs when a unit is spawned in a tile that already has
     a unit on it. The unit that is already on the tile is pushed in a specific
     direction ... The forced spawn of super units are commonly used in besieged
     cities to push out the invader."

    "Friendly units that previously moved will be pushed in the same direction of
     their movement. Enemy units that previously moved will be pushed in the
     opposite direction of their movement. Ranged units get pushed in the
     direction of their last move or last attack. Units that were not previously
     moved will be pushed toward the center of the map. If the city where the
     unit spawns is on the exact center of the map, the unit will be pushed
     south. If the tile where the unit is supposed to go is occupied or
     impassable, it will try counterclockwise and then clockwise one tile at a
     time until it finds a free tile, or if there is no free tile, it will be
     removed from the game entirely. This does not count as a kill for the Gate
     of Power, nor does it count as an attack for the Altar of Peace."

That last sentence is the "Houdini": ring a besieged city with eight units, take
the super-unit reward, and the enemy Giant on your city simply vanishes.
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    facing_toward,
    push_order,
    push_unit,
)
from polyengine.core.economy import apply_reward
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11
MID = (N // 2) * N + (N // 2)          # the exact centre of an 11x11 board


def blank(seed=3):
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.unit_facing[t] = -1
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.resource[t] = C.R_NONE
        s.terrain[t] = C.T_FIELD
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.city_pending_reward[t] = 0
        s.owner[t] = C.NEUTRAL
    s.current_player = 0
    return game, s


def put_unit(s, tile, utype=C.U_WARRIOR, player=0, facing=-1):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    s.unit_facing[tile] = facing
    return tile


def step(s, frm, to):
    """The (dx, dy) between two tiles, for readable assertions."""
    fx, fy = s.xy(frm)
    tx, ty = s.xy(to)
    return (tx - fx, ty - fy)


# --- facing -------------------------------------------------------------------
class TestFacing:
    @pytest.mark.parametrize("idx", range(8))
    def test_every_direction_round_trips(self, idx):
        _, s = blank()
        dx, dy = C.PUSH_COMPASS[idx]
        x, y = s.xy(MID)
        dst = (y + dy) * N + (x + dx)
        assert facing_toward(s, MID, dst) == idx

    def test_a_move_records_the_direction_it_went(self):
        _, s = blank()
        src = put_unit(s, MID)
        dst = MID + 1                       # one step east
        apply_action(s, Action(C.A_MOVE, src, dst), R)
        assert int(s.unit_facing[dst]) == C.PUSH_COMPASS.index((1, 0))

    def test_a_multi_tile_move_records_net_displacement(self):
        """The engine keeps no path, so a two-tile dash records where it ended
        up relative to where it started."""
        _, s = blank()
        s.grant_tech(0, R.tech_id["riding"])
        src = put_unit(s, MID, utype=C.U_RIDER)
        dst = MID + 2 * N                   # two steps south
        apply_action(s, Action(C.A_MOVE, src, dst), R)
        assert int(s.unit_facing[dst]) == C.PUSH_COMPASS.index((0, 1))

    def test_a_fresh_unit_has_never_moved(self):
        _, s = blank()
        tile = put_unit(s, MID)
        assert int(s.unit_facing[tile]) == -1

    def test_facing_survives_a_serialization_round_trip(self):
        _, s = blank()
        tile = put_unit(s, MID, facing=5)
        clone = type(s).from_dict(s.to_dict(), R)
        assert int(clone.unit_facing[tile]) == 5


# --- push order ---------------------------------------------------------------
class TestPushOrder:
    def test_the_facing_direction_comes_first(self):
        _, s = blank()
        for idx in range(8):
            put_unit(s, MID, facing=idx)
            assert step(s, MID, push_order(s, MID)[0]) == C.PUSH_COMPASS[idx]

    def test_it_tries_counterclockwise_then_clockwise(self):
        """"it will try counterclockwise and then clockwise one tile at a time"
        -- so facing, -1, +1, -2, +2, ... and the way it came from comes last."""
        _, s = blank()
        facing = 2                                     # east
        put_unit(s, MID, facing=facing)
        order = [C.PUSH_COMPASS.index(step(s, MID, t)) for t in push_order(s, MID)]
        assert order == [2, 1, 3, 0, 4, 7, 5, 6]
        assert order[-1] == (facing + 4) % 8, "backwards should be the last resort"

    def test_all_eight_neighbours_are_offered_exactly_once(self):
        _, s = blank()
        put_unit(s, MID, facing=0)
        order = push_order(s, MID)
        assert len(order) == 8 and len(set(order)) == 8

    def test_off_board_directions_are_skipped_at_the_edge(self):
        _, s = blank()
        corner = 0
        put_unit(s, corner, facing=0)                  # facing north, off-board
        order = push_order(s, corner)
        assert len(order) == 3                         # E, SE, S
        assert all(0 <= t < s.n_tiles for t in order)

    def test_a_unit_that_never_moved_is_pushed_toward_the_centre(self):
        _, s = blank()
        for tile in (0, N - 1, N * (N - 1), s.n_tiles - 1, 3, N * 4):
            put_unit(s, tile)                          # facing == -1
            first = push_order(s, tile)[0]
            assert s.chebyshev(first, MID) < s.chebyshev(tile, MID), \
                f"tile {tile} pushed away from the middle"

    def test_a_unit_on_the_exact_centre_is_pushed_south(self):
        """"If the city where the unit spawns is on the exact center of the map,
        the unit will be pushed south."""
        _, s = blank()
        put_unit(s, MID)
        assert step(s, MID, push_order(s, MID)[0]) == (0, 1)

    def test_an_enemy_is_pushed_back_the_way_it_came(self):
        """"Enemy units that previously moved will be pushed in the opposite
        direction of their movement." This is the whole un-siege tactic: the
        besieger walked onto the city, so it goes straight back out."""
        _, s = blank()
        north = C.PUSH_COMPASS.index((0, -1))
        put_unit(s, MID, player=1, facing=north)       # marched north onto us
        first = push_order(s, MID, spawner=0)[0]
        assert step(s, MID, first) == (0, 1), "should be shoved south"
        # ... while our own unit facing north is pushed north.
        put_unit(s, MID, player=0, facing=north)
        assert step(s, MID, push_order(s, MID, spawner=0)[0]) == (0, -1)

    def test_ownership_only_matters_when_a_spawner_is_named(self):
        _, s = blank()
        north = C.PUSH_COMPASS.index((0, -1))
        put_unit(s, MID, player=1, facing=north)
        assert step(s, MID, push_order(s, MID)[0]) == (0, -1)

    def test_a_ranged_attack_sets_the_push_direction(self):
        """"Ranged units get pushed in the direction of their last move or last
        attack."""
        from polyengine.core.combat import resolve_attack
        _, s = blank()
        archer = put_unit(s, MID, utype=C.U_ARCHER)
        target = MID + 2                               # two tiles east, in range
        put_unit(s, target, utype=C.U_WARRIOR, player=1)
        s.unit_hp[target] = 99                         # survive the volley
        resolve_attack(s, archer, target, R)
        assert int(s.unit_facing[archer]) == C.PUSH_COMPASS.index((1, 0))


# --- the push itself -----------------------------------------------------------
class TestPushUnit:
    def test_it_goes_the_way_it_was_heading(self):
        _, s = blank()
        put_unit(s, MID, facing=C.PUSH_COMPASS.index((1, 0)))
        dst = push_unit(s, MID, R)
        assert step(s, MID, dst) == (1, 0)
        assert int(s.unit_type[MID]) == C.U_NONE
        assert int(s.unit_type[dst]) == C.U_WARRIOR

    def test_a_blocked_direction_backs_up_one_notch(self):
        _, s = blank()
        east = C.PUSH_COMPASS.index((1, 0))
        put_unit(s, MID, facing=east)
        put_unit(s, MID + 1, utype=C.U_ARCHER)         # block due east
        dst = push_unit(s, MID, R)
        assert step(s, MID, dst) == (1, -1), "counterclockwise first: NE"

    def test_the_pushed_unit_keeps_its_state(self):
        _, s = blank()
        tile = put_unit(s, MID, facing=4)
        s.unit_hp[tile] = 3
        s.unit_veteran[tile] = 1
        s.unit_moved[tile] = 1
        dst = push_unit(s, MID, R)
        assert int(s.unit_hp[dst]) == 3
        assert s.unit_veteran[dst] and s.unit_moved[dst]
        assert int(s.unit_facing[dst]) == 4, "being shoved is not a move it chose"

    def test_a_land_unit_is_not_pushed_into_the_sea(self):
        """Impassable counts the same as occupied -- and with all eight
        neighbours water, a land unit has nowhere to go at all."""
        _, s = blank()
        put_unit(s, MID, facing=C.PUSH_COMPASS.index((1, 0)))
        for nb in s.neighbors(MID):
            s.terrain[nb] = C.T_WATER
        assert push_unit(s, MID, R) == -1
        assert int(s.unit_type[MID]) == C.U_NONE, "nowhere to go means removed"


# --- the Houdini ------------------------------------------------------------------
class TestHoudini:
    """"If there is no free tile, it will be removed from the game entirely."

    Ring a besieged city with eight units, take the super-unit reward, and the
    enemy Giant standing on your city vanishes. It is not scored as combat:
    "this does not count as a kill for the Gate of Power, nor does it count as
    an attack for the Altar of Peace"."""

    def boxed_in(self, seed=3, victim_player=1):
        _, s = blank(seed=seed)
        put_unit(s, MID, utype=C.U_GIANT, player=victim_player, facing=0)
        for nb in s.neighbors(MID):
            put_unit(s, int(nb), utype=C.U_ARCHER, player=0)
        return s

    def test_a_boxed_in_unit_is_removed_from_the_game(self):
        s = self.boxed_in()
        assert push_unit(s, MID, R, spawner=0) == -1
        assert int(s.unit_type[MID]) == C.U_NONE

    def test_it_does_not_count_as_a_kill_or_a_loss(self):
        """The wiki is explicit, and both counters feed real mechanics: `kills`
        drives the Gate of Power, `losses` the heuristic's battle skill."""
        s = self.boxed_in()
        kills, losses = list(s.kills), list(s.losses)
        push_unit(s, MID, R, spawner=0)
        assert list(s.kills) == kills
        assert list(s.losses) == losses

    def test_a_pacifist_streak_survives_it(self):
        """"nor does it count as an attack for the Altar of Peace"."""
        s = self.boxed_in()
        s.calm_turns[0] = 4
        push_unit(s, MID, R, spawner=0)
        assert int(s.calm_turns[0]) == 4

    def test_one_gap_is_enough_to_escape(self):
        s = self.boxed_in()
        escape = int(next(iter(s.neighbors(MID))))
        s.unit_type[escape] = C.U_NONE
        s.unit_owner[escape] = C.NEUTRAL
        assert push_unit(s, MID, R, spawner=0) == escape
        assert int(s.unit_type[escape]) == C.U_GIANT


# --- the two call sites --------------------------------------------------------
class TestSuperUnitReward:
    def _level_five_city(self, s, tile, player=0):
        s.city_level[tile] = 5
        s.city_owner[tile] = player
        s.city_capital[tile] = 1
        s.owner[tile] = player
        s.city_pending_reward[tile] = 5
        return tile

    def _super_unit_id(self):
        for rw, name in enumerate(R.reward_names):
            if R.reward_effect[rw] == "super_unit":
                return rw
        pytest.skip("no super-unit reward in the ruleset")

    def test_the_garrison_is_pushed_off_and_the_giant_takes_the_tile(self):
        """Picking the strongest unit in the game and getting nothing was a trap:
        the reward silently did nothing whenever the city was garrisoned."""
        _, s = blank()
        city = self._level_five_city(s, MID)
        put_unit(s, city, utype=C.U_WARRIOR,
                 facing=C.PUSH_COMPASS.index((1, 0)))
        apply_reward(s, city, self._super_unit_id(), 0, R)
        assert int(s.unit_type[city]) == C.U_GIANT
        assert int(s.unit_type[MID + 1]) == C.U_WARRIOR, "pushed east, as it faced"

    def test_an_empty_city_still_just_gets_its_giant(self):
        _, s = blank()
        city = self._level_five_city(s, MID)
        apply_reward(s, city, self._super_unit_id(), 0, R)
        assert int(s.unit_type[city]) == C.U_GIANT

    def test_the_giant_breaks_a_siege_by_shoving_the_invader_out(self):
        """"The forced spawn of super units are commonly used in besieged cities
        to push out the invader." The besieger marched north onto the city, so
        the enemy-reversal rule sends it straight back south."""
        _, s = blank()
        city = self._level_five_city(s, MID)
        north = C.PUSH_COMPASS.index((0, -1))
        put_unit(s, city, utype=C.U_WARRIOR, player=1, facing=north)
        apply_reward(s, city, self._super_unit_id(), 0, R)
        assert int(s.unit_type[city]) == C.U_GIANT
        assert int(s.unit_owner[city]) == 0
        assert int(s.unit_type[MID + N]) == C.U_WARRIOR, "shoved back south"
        assert int(s.unit_owner[MID + N]) == 1

    def test_a_houdini_destroys_the_besieger_outright(self):
        """The city is walled in on all eight sides, so the invader vanishes."""
        _, s = blank()
        city = self._level_five_city(s, MID)
        put_unit(s, city, utype=C.U_GIANT, player=1, facing=0)
        for nb in s.neighbors(city):
            put_unit(s, int(nb), utype=C.U_ARCHER, player=0)
        kills = list(s.kills)
        apply_reward(s, city, self._super_unit_id(), 0, R)
        assert int(s.unit_type[city]) == C.U_GIANT
        assert int(s.unit_owner[city]) == 0, "our Giant, not theirs"
        assert sum(1 for t in range(s.n_tiles)
                   if int(s.unit_owner[t]) == 1
                   and s.unit_type[t] != C.U_NONE) == 0
        assert list(s.kills) == kills, "a Houdini is not a Gate of Power kill"


class TestRuinUnitReward:
    def test_the_examiner_is_pushed_the_way_it_came(self):
        for seed in range(80):
            _, s = blank(seed=seed)
            east = C.PUSH_COMPASS.index((1, 0))
            put_unit(s, MID, facing=east)
            s.improvement[MID] = C.IMP_RUIN
            out = apply_action(s, Action(C.A_EXAMINE, MID, MID), R)
            if out["reward"] != "unit":
                continue
            assert int(s.unit_type[MID]) == C.U_SWORDSMAN
            assert int(s.unit_type[MID + 1]) == C.U_WARRIOR
            return
        raise AssertionError("the unit reward never came up")
