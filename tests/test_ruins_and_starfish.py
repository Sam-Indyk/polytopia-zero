"""Ruins, starfish, and the EXAMINE action that serves both.

Quoted from `.wikicache/Ruins.wiki`, `.wikicache/Starfish.wiki` and
`.wikicache/Map Generation.wiki`:

    "Examining a ruin requires a unit to start the turn on the ruin and consumes
     the unit's actions for that turn (like capturing a village)."
    "The rewards for ruins are chosen randomly, with no special weighting
     accorded to any one option."
    "Ruins can spawn on all types of terrain except shallow water. However,
     ruins cannot spawn next to a capital city or another ruin."
    "Starfish ... can be harvested by a naval unit much like if it were examining
     a ruin, which removes the Starfish from the tile, and gives the player 8
     stars. The ability is unlocked by the Navigation technology."
    "There is approximately one starfish for every 25 water tiles."

The one that matters for training: EXAMINE is a real action type with its own
mask, so invariant 2 applies to it as much as to MOVE.
"""
from __future__ import annotations

import collections

from polyengine.core import constants as C
from polyengine.core.actions import (
    Action,
    apply_action,
    can_examine,
    enumerate_actions,
)
from polyengine.core.game import Game
from polyengine.core.mapgen import new_game
from polyengine.core.rules import load_ruleset

R = load_ruleset()
N = 11


def blank(seed=3, map_type="drylands"):
    """An empty field map with the units and features stripped out."""
    game = Game.new(size=N, seed=seed, map_type=map_type)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.resource[t] = C.R_NONE
        s.terrain[t] = C.T_FIELD
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.owner[t] = C.NEUTRAL
    s.current_player = 0
    return game, s


def put_unit(s, tile, utype=C.U_WARRIOR, player=0):
    s.unit_type[tile] = utype
    s.unit_owner[tile] = player
    s.unit_hp[tile] = int(R.unit_max_hp[utype])
    s.unit_moved[tile] = 0
    s.unit_attacked[tile] = 0
    s.unit_home[tile] = -1
    return tile


# --- the shared mechanic -------------------------------------------------------
class TestExamineMechanic:
    def test_a_unit_standing_on_a_ruin_may_examine_it(self):
        game, s = blank()
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        assert can_examine(s, tile, 0, R)
        assert any(a.atype == C.A_EXAMINE and a.src == tile
                   for a in enumerate_actions(s, 0, R))

    def test_examining_consumes_the_turn(self):
        """"consumes the unit's actions for that turn (like capturing)"."""
        game, s = blank()
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
        assert s.unit_moved[tile] and s.unit_attacked[tile]
        assert not can_examine(s, tile, 0, R)

    def test_a_unit_that_already_moved_cannot_examine(self):
        """It must *start* the turn on the ruin, exactly like capture."""
        game, s = blank()
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        s.unit_moved[tile] = 1
        assert not can_examine(s, tile, 0, R)

    def test_an_empty_tile_offers_nothing_to_examine(self):
        game, s = blank()
        tile = put_unit(s, 60)
        assert not can_examine(s, tile, 0, R)

    def test_an_enemy_unit_on_the_ruin_is_not_ours_to_examine(self):
        game, s = blank()
        tile = put_unit(s, 60, player=1)
        s.improvement[tile] = C.IMP_RUIN
        assert not can_examine(s, tile, 0, R)

    def test_examining_removes_the_ruin(self):
        game, s = blank()
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
        assert int(s.improvement[tile]) != C.IMP_RUIN


# --- ruin rewards --------------------------------------------------------------
class TestRuinRewards:
    def _rewards(self, seeds=40):
        got = collections.Counter()
        for seed in range(seeds):
            game, s = blank(seed=seed)
            cap = 5
            s.city_level[cap] = 1
            s.city_owner[cap] = 0
            s.city_capital[cap] = 1
            s.owner[cap] = 0
            tile = put_unit(s, 60)
            s.improvement[tile] = C.IMP_RUIN
            got[apply_action(s, Action(C.A_EXAMINE, tile, tile),
                             R)["reward"]] += 1
        return got

    def test_several_different_rewards_appear(self):
        """"chosen randomly, with no special weighting" -- so not one outcome."""
        got = self._rewards()
        assert len(got) >= 3, got

    def test_stars_are_always_in_the_pool(self):
        assert self._rewards()["stars"] > 0

    def test_the_stars_reward_pays_ten(self):
        game, s = blank()
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        before = int(s.stars[0])
        for _ in range(60):                 # draw until "stars" comes up
            s.improvement[tile] = C.IMP_RUIN
            s.unit_moved[tile] = 0
            s.unit_attacked[tile] = 0
            s.stars[0] = before
            out = apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
            if out["reward"] == "stars":
                assert int(s.stars[0]) == before + C.RUIN_STARS
                return
        raise AssertionError("the stars reward never came up in 60 draws")

    def test_a_technology_reward_actually_grants_a_technology(self):
        game, s = blank()
        tile = put_unit(s, 60)
        for _ in range(80):
            s.improvement[tile] = C.IMP_RUIN
            s.unit_moved[tile] = 0
            s.unit_attacked[tile] = 0
            known = sum(1 for k in range(R.n_techs) if s.has_tech(0, k))
            out = apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
            if out["reward"] == "technology":
                now = sum(1 for k in range(R.n_techs) if s.has_tech(0, k))
                assert now == known + 1
                return
        raise AssertionError("the technology reward never came up")

    def _first_unit_reward(self, setup, seeds=80):
        """Draw until the unit reward comes up; return (state, tile, info)."""
        for seed in range(seeds):
            game, s = blank(seed=seed)
            tile = setup(s)
            out = apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
            if out["reward"] == "unit":
                return s, tile, out
        raise AssertionError("the unit reward never came up")

    def test_the_unit_reward_pushes_the_examiner_aside(self):
        """"the examining unit is pushed in a forced unit spawn".

        Without the push the examiner still occupies the ruin, so this reward
        can never once be paid -- the pool would be silently one option short
        every single time, since a unit MUST be standing there to examine.
        """
        def setup(s):
            tile = put_unit(s, 60, utype=C.U_WARRIOR)
            s.improvement[tile] = C.IMP_RUIN
            return tile
        s, tile, out = self._first_unit_reward(setup)
        assert out["unit"] == R.unit_names[C.U_SWORDSMAN]
        assert int(s.unit_type[tile]) == C.U_SWORDSMAN
        assert s.unit_veteran[tile], "\"Spawns a Veteran Swordsman\""
        pushed = [int(nb) for nb in s.neighbors(tile)
                  if int(s.unit_type[nb]) == C.U_WARRIOR]
        assert len(pushed) == 1, "the examiner should be standing next door"

    def test_a_land_ruin_spawns_a_swordsman(self):
        """"New Friends ... Ruin is NOT on Shallow Water or Ocean tile"."""
        def setup(s):
            tile = put_unit(s, 60)
            s.improvement[tile] = C.IMP_RUIN
            return tile
        _, _, out = self._first_unit_reward(setup)
        assert out["unit"] == R.unit_names[C.U_SWORDSMAN]

    def test_a_water_ruin_spawns_a_rammer_carrying_a_warrior(self):
        """"Rammer | Spawns a Veteran Rammer (Carries a Warrior)"."""
        def setup(s):
            tile = 60
            for t in range(s.n_tiles):
                s.terrain[t] = C.T_WATER
            s.grant_tech(0, R.tech_id["fishing"])   # or the push has nowhere legal
            s.improvement[tile] = C.IMP_RUIN
            put_unit(s, tile, utype=C.U_RAFT)
            return tile
        s, tile, out = self._first_unit_reward(setup)
        assert out["unit"] == R.unit_names[C.U_RAMMER]
        assert int(s.unit_carried[tile]) == C.U_WARRIOR

    def test_the_explorer_reward_reveals_map_and_needs_somewhere_dark(self):
        """"Grants a free Explorer | Any tile within a 5x5 area centered on the
        ruin is not explored"."""
        got = collections.Counter()
        for seed in range(40):
            game, s = blank(seed=seed)
            tile = put_unit(s, 60)
            s.improvement[tile] = C.IMP_RUIN
            s.fog_explored[0][:] = 0
            before = int(s.fog_explored[0].sum())
            out = apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
            got[out["reward"]] += 1
            if out["reward"] == "explorer":
                assert int(s.fog_explored[0].sum()) > before
        assert got["explorer"] > 0, got

        # ... and on a fully-mapped board it is not offered at all.
        game, s = blank(seed=1)
        tile = put_unit(s, 60)
        s.improvement[tile] = C.IMP_RUIN
        s.fog_explored[0][:] = 1
        offered = collections.Counter()
        for _ in range(60):
            s.improvement[tile] = C.IMP_RUIN
            s.unit_moved[tile] = 0
            s.unit_attacked[tile] = 0
            for nb in s.neighbors(tile):        # undo any forced spawn
                if int(nb) != tile:
                    s.unit_type[int(nb)] = C.U_NONE
            put_unit(s, tile)
            offered[apply_action(s, Action(C.A_EXAMINE, tile, tile),
                                 R)["reward"]] += 1
        assert offered["explorer"] == 0, offered


# --- starfish -----------------------------------------------------------------
class TestStarfish:
    def _fish_state(self, nav=True):
        game, s = blank()
        tile = 60
        s.terrain[tile] = C.T_WATER
        s.resource[tile] = C.R_STARFISH
        put_unit(s, tile, utype=C.U_RAFT)
        if nav:
            s.grant_tech(0, R.tech_id["navigation"])
        return game, s, tile

    def test_navigation_is_required(self):
        """"The ability is unlocked by the Navigation technology."""
        _, s, tile = self._fish_state(nav=False)
        assert not can_examine(s, tile, 0, R)
        _, s, tile = self._fish_state(nav=True)
        assert can_examine(s, tile, 0, R)

    def test_harvesting_pays_eight_stars_and_clears_the_tile(self):
        _, s, tile = self._fish_state()
        before = int(s.stars[0])
        out = apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
        assert out["reward"] == "starfish"
        assert int(s.stars[0]) == before + C.STARFISH_STARS
        assert int(s.resource[tile]) == C.R_NONE

    def test_harvesting_consumes_the_turn(self):
        _, s, tile = self._fish_state()
        apply_action(s, Action(C.A_EXAMINE, tile, tile), R)
        assert s.unit_moved[tile] and s.unit_attacked[tile]

    def test_starfish_survives_a_serialization_round_trip(self):
        _, s, tile = self._fish_state()
        clone = type(s).from_dict(s.to_dict(), R)
        assert int(clone.resource[tile]) == C.R_STARFISH


# --- generation ---------------------------------------------------------------
class TestRuinPlacement:
    def test_ruin_count_follows_the_size_table(self):
        for size, expected in C.RUINS_BY_SIZE.items():
            s = new_game(R, size=size, tribes=["imperius", "bardur"], seed=1)
            n = int((s.improvement == C.IMP_RUIN).sum())
            assert n == expected, f"size {size}: {n} ruins, expected {expected}"

    def test_no_ruin_sits_on_shallow_water(self):
        """"Ruins can spawn on all types of terrain except shallow water."""
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "kickoo"], seed=seed,
                         map_type="continents")
            for t in range(s.n_tiles):
                if s.improvement[t] == C.IMP_RUIN:
                    assert s.terrain[t] != C.T_WATER

    def test_no_ruin_is_adjacent_to_a_capital_or_another_ruin(self):
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            ruins = [t for t in range(s.n_tiles) if s.improvement[t] == C.IMP_RUIN]
            caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
            for i, a in enumerate(ruins):
                for c in caps:
                    assert s.chebyshev(a, c) > 1
                for b in ruins[i + 1:]:
                    assert s.chebyshev(a, b) > 1

    def test_ruins_do_not_land_on_a_city(self):
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            for t in range(s.n_tiles):
                if s.improvement[t] == C.IMP_RUIN:
                    assert s.city_level[t] == 0


class TestStarfishPlacement:
    def test_a_dry_map_has_no_starfish_and_a_wet_one_has_several(self):
        dry = new_game(R, size=16, tribes=["imperius", "bardur"], seed=2,
                       map_type="drylands")
        wet = new_game(R, size=16, tribes=["imperius", "bardur"], seed=2,
                       map_type="archipelago")
        assert int((wet.resource == C.R_STARFISH).sum()) > \
            int((dry.resource == C.R_STARFISH).sum())

    def test_roughly_one_starfish_per_twenty_five_water_tiles(self):
        s = new_game(R, size=20, tribes=["imperius", "bardur"], seed=4,
                     map_type="archipelago")
        water = int((~R.terrain_is_land[s.terrain]).sum())
        n = int((s.resource == C.R_STARFISH).sum())
        assert n == water // C.STARFISH_PER_WATER_TILES

    def test_starfish_only_sit_on_water(self):
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="archipelago")
            for t in range(s.n_tiles):
                if s.resource[t] == C.R_STARFISH:
                    assert not R.terrain_is_land[s.terrain[t]]

    def test_no_starfish_beside_a_city_lighthouse_or_another_starfish(self):
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "kickoo"], seed=seed,
                         map_type="archipelago")
            fish = [t for t in range(s.n_tiles) if s.resource[t] == C.R_STARFISH]
            avoid = [t for t in range(s.n_tiles)
                     if s.city_level[t] > 0
                     or s.improvement[t] == C.IMP_LIGHTHOUSE]
            for i, a in enumerate(fish):
                for b in avoid:
                    assert s.chebyshev(a, b) > 1
                for b in fish[i + 1:]:
                    assert s.chebyshev(a, b) > 1
