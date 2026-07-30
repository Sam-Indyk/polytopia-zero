"""Map generation against the wiki's documented algorithm.

Sources are `.wikicache/Map Generation.wiki` and `.wikicache/Tribes.wiki`. The
rules with real strategic weight, and which the previous generator had none of:

* terrain is generated per tribe, "distributed roughly equally between all
  tribes in the game"
* "All resources spawn only within two tiles of cities or villages", and tiles
  adjacent to one are about three times richer than tiles two away
* water exists at all, in six documented wetness bands
* capitals go in separate quadrants; villages keep two tiles from the edge and
  two from each other
"""
from __future__ import annotations

import collections

import pytest

from polyengine.core import constants as C
from polyengine.core.mapgen import (
    BASE_TERRAIN,
    MAP_SIZES,
    MAP_TYPES,
    new_game,
    size_for_opponents,
    terrain_rates,
)
from polyengine.core.rules import load_ruleset

R = load_ruleset()


def biome_owner(state, tile):
    caps = {int(state.city_owner[t]): t for t in range(state.n_tiles)
            if state.city_capital[t]}
    return min(caps, key=lambda p: state.chebyshev(tile, caps[p]))


# --- sizes --------------------------------------------------------------------
class TestMapSize:
    def test_the_six_documented_sizes(self):
        assert MAP_SIZES == {"tiny": 11, "small": 14, "normal": 16,
                             "large": 18, "huge": 20, "massive": 30}

    @pytest.mark.parametrize("opponents,expected", [
        (1, 11), (2, 14), (3, 16), (4, 18), (7, 18),
    ])
    def test_domination_picks_size_by_opponent_count(self, opponents, expected):
        """"one opponent ... tiny, two ... 196-tile, three ... normal, four or
        more on a large" map."""
        assert size_for_opponents(opponents) == expected

    @pytest.mark.parametrize("size", sorted(MAP_SIZES.values()))
    def test_every_size_generates(self, size):
        s = new_game(R, size=size, tribes=["imperius", "bardur"], seed=1)
        assert s.size == size and s.n_tiles == size * size


# --- tribe terrain ------------------------------------------------------------
class TestTribeTerrain:
    def test_luxidoor_is_the_base_rate(self):
        """"Luxidoor: same as base rate", and the base is 48/38/14."""
        rates = terrain_rates(R.tribes["luxidoor"])
        assert rates[C.T_FIELD] == pytest.approx(BASE_TERRAIN[C.T_FIELD])
        assert rates[C.T_FOREST] == pytest.approx(BASE_TERRAIN[C.T_FOREST])
        assert rates[C.T_MOUNTAIN] == pytest.approx(BASE_TERRAIN[C.T_MOUNTAIN])

    def test_rates_always_sum_to_one(self):
        for name in R.tribes:
            rates = terrain_rates(R.tribes[name])
            assert sum(rates.values()) == pytest.approx(1.0), name

    def test_the_mountain_multiplier_is_applied(self):
        """"Xin-xi: 1.5x mountain"."""
        base = terrain_rates(R.tribes["luxidoor"])[C.T_MOUNTAIN]
        assert terrain_rates(R.tribes["xin_xi"])[C.T_MOUNTAIN] == \
            pytest.approx(base * 1.5)

    def test_a_mountain_multiplier_rescales_the_rest(self):
        """"An increase of mountains will thus proportionally change the forest
        rate and field rate."""
        lux = terrain_rates(R.tribes["luxidoor"])
        xin = terrain_rates(R.tribes["xin_xi"])
        assert xin[C.T_FOREST] < lux[C.T_FOREST]
        assert xin[C.T_FIELD] < lux[C.T_FIELD]

    def test_hoodrick_is_the_forest_tribe(self):
        """"Hoodrick: 0.5x mountain, 1.5x forest"."""
        rates = terrain_rates(R.tribes["hoodrick"])
        assert rates[C.T_FOREST] > rates[C.T_FIELD]

    def test_oumaji_is_nearly_treeless(self):
        """"Oumaji: 0.2x forest ..."."""
        rates = terrain_rates(R.tribes["oumaji"])
        assert rates[C.T_FOREST] < 0.12
        assert rates[C.T_FIELD] > 0.75

    def test_each_tribes_own_land_reflects_its_rates(self):
        """The land is partitioned between tribes, not generated uniformly."""
        forest = {"hoodrick": 0, "oumaji": 0}
        total = {"hoodrick": 0, "oumaji": 0}
        for seed in range(12):
            s = new_game(R, size=14, tribes=["hoodrick", "oumaji"], seed=seed)
            names = ["hoodrick", "oumaji"]
            for t in range(s.n_tiles):
                name = names[biome_owner(s, t)]
                total[name] += 1
                if s.terrain[t] == C.T_FOREST:
                    forest[name] += 1
        assert forest["hoodrick"] / total["hoodrick"] > \
            forest["oumaji"] / total["oumaji"] * 3


# --- tribe resources ----------------------------------------------------------
class TestTribeResources:
    def _counts(self, pair, seeds=25, size=14):
        tot = [collections.Counter(), collections.Counter()]
        for seed in range(seeds):
            s = new_game(R, size=size, tribes=list(pair), seed=seed)
            for t in range(s.n_tiles):
                if s.resource[t] == C.R_NONE:
                    continue
                tot[biome_owner(s, t)][int(s.resource[t])] += 1
        return tot

    def test_bardur_land_grows_no_crop_at_all(self):
        """"Bardur: 0.8x forest, 0x crop" -- a hard zero, not merely rare."""
        bardur, _ = self._counts(("bardur", "oumaji"))
        assert bardur[C.R_CROP] == 0

    def test_vengir_is_metal_rich_and_fruit_poor(self):
        """"Vengir: 2.0x metal, 0.1x wild animal, 0.1x fruit, 0.1x fish"."""
        vengir, imperius = self._counts(("vengir", "imperius"))
        assert vengir[C.R_METAL] > vengir[C.R_FRUIT] * 3
        assert vengir[C.R_FRUIT] < imperius[C.R_FRUIT] / 3

    def test_imperius_is_the_fruit_tribe(self):
        """"Imperius: 0.5x wild animal, 2.0x fruit"."""
        imperius, bardur = self._counts(("imperius", "bardur"))
        assert imperius[C.R_FRUIT] > imperius[C.R_GAME] * 2

    def test_xin_xi_metal_compounds_mountain_and_metal(self):
        """1.5x mountain AND 1.5x metal multiply together."""
        xin, zebasi = self._counts(("xin_xi", "zebasi"))
        assert xin[C.R_METAL] > zebasi[C.R_METAL] * 2


# --- the two-tile rule --------------------------------------------------------
class TestResourcePlacement:
    def test_resources_only_appear_near_a_city(self):
        """"All resources spawn only within two tiles of cities or villages."""
        for seed in range(8):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            cities = [t for t in range(s.n_tiles) if s.city_level[t] > 0]
            for t in range(s.n_tiles):
                if s.resource[t] == C.R_NONE:
                    continue
                assert min(s.chebyshev(t, c) for c in cities) <= 2, \
                    f"resource at {t} is more than two tiles from any city"

    def test_tiles_beside_a_city_are_richer_than_tiles_two_away(self):
        """Inner rates are roughly 3x outer (18% vs 6% fruit, and so on)."""
        near = far = near_n = far_n = 0
        for seed in range(15):
            s = new_game(R, size=16, tribes=["luxidoor", "luxidoor"], seed=seed)
            cities = [t for t in range(s.n_tiles) if s.city_level[t] > 0]
            for t in range(s.n_tiles):
                if s.city_level[t] > 0 or not R.terrain_is_land[s.terrain[t]]:
                    continue
                d = min(s.chebyshev(t, c) for c in cities)
                has = s.resource[t] != C.R_NONE
                if d == 1:
                    near_n += 1
                    near += has
                elif d == 2:
                    far_n += 1
                    far += has
        assert near_n and far_n
        assert (near / near_n) > (far / far_n) * 1.8


# --- water --------------------------------------------------------------------
class TestWater:
    def test_drylands_is_nearly_dry(self):
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=3,
                     map_type="drylands")
        water = sum(1 for t in range(s.n_tiles)
                    if not R.terrain_is_land[s.terrain[t]])
        assert water / s.n_tiles < 0.15

    def test_continents_is_substantially_wet(self):
        """Without this, Ports, Bridges, embarking and Aquatism are dead code."""
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=3,
                     map_type="continents")
        water = sum(1 for t in range(s.n_tiles)
                    if not R.terrain_is_land[s.terrain[t]])
        assert 0.25 < water / s.n_tiles < 0.85

    def test_wetter_map_types_really_are_wetter(self):
        def wetness(kind):
            total = 0
            for seed in range(6):
                s = new_game(R, size=14, tribes=["imperius", "bardur"],
                             seed=seed, map_type=kind)
                total += sum(1 for t in range(s.n_tiles)
                             if not R.terrain_is_land[s.terrain[t]]) / s.n_tiles
            return total / 6
        assert wetness("drylands") < wetness("lakes") < wetness("archipelago")

    def test_an_unknown_map_type_is_rejected(self):
        with pytest.raises(ValueError):
            new_game(R, size=11, tribes=["imperius", "bardur"], seed=0,
                     map_type="swamp")

    def test_every_documented_map_type_generates(self):
        for kind in MAP_TYPES:
            s = new_game(R, size=14, tribes=["imperius", "bardur"], seed=2,
                         map_type=kind)
            assert s.n_tiles == 196

    def test_deep_ocean_is_water_away_from_land(self):
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=4,
                     map_type="continents")
        for t in range(s.n_tiles):
            if s.terrain[t] != C.T_OCEAN or s.improvement[t] == C.IMP_LIGHTHOUSE:
                continue
            assert not any(R.terrain_is_land[s.terrain[nb]]
                           for nb in s.neighbors(t))


# --- placement ----------------------------------------------------------------
class TestPlacement:
    def test_capitals_are_never_on_the_border(self):
        """The deferred-mapgen note in PLAN.md called this out specifically."""
        for seed in range(15):
            s = new_game(R, size=14, tribes=["imperius", "bardur"], seed=seed)
            for t in range(s.n_tiles):
                if not s.city_capital[t]:
                    continue
                x, y = s.xy(t)
                assert 2 <= x < s.size - 2 and 2 <= y < s.size - 2

    def test_capitals_land_in_different_quadrants(self):
        for seed in range(15):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
            quads = {(s.xy(t)[0] // (s.size // 2), s.xy(t)[1] // (s.size // 2))
                     for t in caps}
            assert len(quads) == len(caps), "two capitals shared a quadrant"

    def test_villages_keep_their_distance(self):
        """"not within two tiles of other villages", nor of the map edge."""
        for seed in range(10):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            cities = [t for t in range(s.n_tiles) if s.city_level[t] > 0]
            for i, a in enumerate(cities):
                x, y = s.xy(a)
                assert 2 <= x < s.size - 2 and 2 <= y < s.size - 2
                for b in cities[i + 1:]:
                    assert s.chebyshev(a, b) > 2

    def test_lighthouses_still_occupy_the_corners(self):
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=1)
        corners = (0, s.size - 1, s.size * (s.size - 1), s.n_tiles - 1)
        assert all(s.improvement[t] == C.IMP_LIGHTHOUSE for t in corners)


# --- per-map-type feature passes ------------------------------------------------
class TestMapTypeFeatures:
    """The wiki's feature table: which village passes each map type runs.

    "Suburbs: Lakes, Archipelago / Pre-terrain villages: Lakes, Archipelago,
    Water World / Post-terrain villages: all / Tiny island villages: Continents,
    Pangea / Quadrants: all except Continents and Pangea."
    """

    def villages(self, state):
        return [t for t in range(state.n_tiles)
                if state.city_level[t] > 0 and not state.city_capital[t]]

    def test_every_map_type_still_produces_villages_to_capture(self):
        """Water World generates almost all water; "land is manually forced for
        cities", without which the map has nothing to expand into at all."""
        for kind in MAP_TYPES:
            for seed in range(3):
                s = new_game(R, size=14, tribes=["imperius", "bardur"],
                             seed=seed, map_type=kind)
                assert self.villages(s), f"{kind} seed {seed}: no villages"

    def test_lakes_puts_suburbs_near_the_capitals(self):
        """"After the capital city is placed, the algorithm puts up to two
        villages nearby" -- so a Lakes capital has close neighbours a Drylands
        one usually does not."""
        def close_villages(kind):
            n = 0
            for seed in range(8):
                s = new_game(R, size=16, tribes=["imperius", "bardur"],
                             seed=seed, map_type=kind)
                caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
                n += sum(1 for v in self.villages(s)
                         if min(s.chebyshev(v, c) for c in caps) <= 4)
            return n
        assert close_villages("lakes") > close_villages("drylands")

    def test_continents_and_pangea_place_island_villages(self):
        """"Continents and Pangea have a fixed amount of villages that spawn in
        the water without any land connections."""
        for kind in ("continents", "pangea"):
            found = 0
            for seed in range(8):
                s = new_game(R, size=18, tribes=["imperius", "kickoo"],
                             seed=seed, map_type=kind)
                for v in self.villages(s):
                    if all(not R.terrain_is_land[s.terrain[nb]]
                           for nb in s.neighbors(v)):
                        found += 1
            assert found > 0, f"{kind} produced no island villages"

    def test_drylands_has_no_island_villages(self):
        """The pass does not run there -- and there is barely any water anyway."""
        for seed in range(6):
            s = new_game(R, size=18, tribes=["imperius", "bardur"], seed=seed,
                         map_type="drylands")
            for v in self.villages(s):
                assert any(R.terrain_is_land[s.terrain[nb]]
                           for nb in s.neighbors(v))

    def test_continents_keeps_capitals_apart_without_quadrants(self):
        """"Continents and Pangea are the only two map types that don't use
        quadrants for capital placement."

        With two players the two placement rules are behaviourally
        indistinguishable -- any pair of capitals six or more tiles apart is
        already in different halves of a 16-wide map -- so what this pins is the
        separation guarantee that survives dropping quadrants.
        """
        for seed in range(20):
            for kind in ("continents", "pangea"):
                s = new_game(R, size=16, tribes=["imperius", "bardur"],
                             seed=seed, map_type=kind)
                caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
                assert s.chebyshev(caps[0], caps[1]) >= 6, f"{kind} seed {seed}"

    def test_a_crowded_continents_map_can_share_a_domain(self):
        """Where the two rules DO diverge: six players on a massive map.

        The quadrant path would deal one capital per 3x3 domain; scattered
        placement only enforces spacing, so some seed should double up. If this
        ever stops happening, `_scattered_capitals` has quietly become the
        quadrant rule again.
        """
        from polyengine.core.mapgen import _scattered_capitals
        from polyengine.core.state import GameState

        shared = 0
        for seed in range(30):
            s = GameState(R, 30, 6, seed)
            caps = _scattered_capitals(s, 6)
            domains = [(s.xy(t)[0] // 10, s.xy(t)[1] // 10) for t in caps]
            if len(set(domains)) < len(caps):
                shared += 1
            for i, a in enumerate(caps):
                for b in caps[i + 1:]:
                    assert s.chebyshev(a, b) >= 6
        assert shared > 0, "scattered placement is behaving like quadrants"

    def test_the_pre_terrain_village_formula(self):
        """"(([map width]/3)^2 - [capitals and suburbs]) * (density coefficient)",
        "rounded down before being squared"."""
        from polyengine.core.mapgen import _pre_terrain_village_count

        class Fake:
            size = 16
        assert _pre_terrain_village_count(Fake(), 0, 0.3) == int((16 // 3) ** 2 * 0.3)
        assert _pre_terrain_village_count(Fake(), 4, 0.3) == int(((16 // 3) ** 2 - 4) * 0.3)
        assert _pre_terrain_village_count(Fake(), 1000, 0.3) == 0


# --- starting position ---------------------------------------------------------
class TestStartingPosition:
    def test_starting_stars_come_from_the_tribe(self):
        """Xin-xi start on 7, Luxidoor on 2 -- it is not a constant."""
        s = new_game(R, size=14, tribes=["xin_xi", "luxidoor"], seed=1)
        assert int(s.stars[0]) == 7
        assert int(s.stars[1]) == 2

    def test_starting_unit_comes_from_the_tribe(self):
        s = new_game(R, size=14, tribes=["oumaji", "hoodrick"], seed=1)
        caps = {int(s.city_owner[t]): t for t in range(s.n_tiles)
                if s.city_capital[t]}
        assert int(s.unit_type[caps[0]]) == C.U_RIDER
        assert int(s.unit_type[caps[1]]) == C.U_ARCHER

    def test_luxidoor_starts_with_a_level_three_capital_and_no_tech(self):
        """"None" technology, and a level three city instead."""
        s = new_game(R, size=14, tribes=["luxidoor", "bardur"], seed=1)
        caps = {int(s.city_owner[t]): t for t in range(s.n_tiles)
                if s.city_capital[t]}
        assert int(s.city_level[caps[0]]) == 3
        assert int(s.tech[0]) == 0

    def test_every_regular_tribe_can_start_a_game(self):
        names = sorted(R.tribes)
        assert len(names) == 12
        for name in names:
            s = new_game(R, size=14, tribes=[name, "luxidoor"], seed=0)
            assert int(((s.city_capital > 0)).sum()) == 2


# --- determinism ---------------------------------------------------------------
def test_generation_is_reproducible():
    a = new_game(R, size=16, tribes=["xin_xi", "vengir"], seed=99,
                 map_type="continents")
    b = new_game(R, size=16, tribes=["xin_xi", "vengir"], seed=99,
                 map_type="continents")
    assert a.state_hash() == b.state_hash()


def test_a_different_seed_gives_a_different_map():
    a = new_game(R, size=16, tribes=["xin_xi", "vengir"], seed=1)
    b = new_game(R, size=16, tribes=["xin_xi", "vengir"], seed=2)
    assert a.state_hash() != b.state_hash()
