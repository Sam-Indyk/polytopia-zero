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
    CONTINENT_MIN,
    MAP_SIZES,
    MAP_TYPES,
    _land_components,
    new_game,
    size_for_opponents,
    terrain_rates,
)
from polyengine.core.rules import load_ruleset

R = load_ruleset()


def biome_owner(state, tile):
    """Which player's biome a tile belongs to.

    Read straight off `tile_tribe`. Guessing by nearest capital was fine while
    biomes were a Voronoi partition; the generator now grows them by randomised
    flood fill, exactly as the game does, and the borders are ragged enough that
    the guess is wrong often enough to hide real spawn-rate bugs.
    """
    by_id = {int(state.tribe[p]): p for p in range(state.n_players)}
    return by_id[int(state.tile_tribe[tile])]


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
        """"All resources spawn only within two tiles of cities or villages."

        Starfish are the documented exception -- "starfish can spawn on any
        water tiles, and are scattered across the map" -- which is what makes
        them worth sailing out for.
        """
        for seed in range(8):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            cities = [t for t in range(s.n_tiles) if s.city_level[t] > 0]
            for t in range(s.n_tiles):
                if s.resource[t] in (C.R_NONE, C.R_STARFISH):
                    continue
                assert min(s.chebyshev(t, c) for c in cities) <= 2, \
                    f"resource at {t} is more than two tiles from any city"

    def test_starfish_are_not_bound_by_the_two_tile_rule(self):
        """The converse of the above: they really do scatter."""
        far = 0
        for seed in range(10):
            s = new_game(R, size=16, tribes=["imperius", "kickoo"], seed=seed,
                         map_type="archipelago")
            cities = [t for t in range(s.n_tiles) if s.city_level[t] > 0]
            far += sum(1 for t in range(s.n_tiles)
                       if s.resource[t] == C.R_STARFISH
                       and min(s.chebyshev(t, c) for c in cities) > 2)
        assert far > 0, "every starfish sits next to a city, which is not scattering"

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

    def test_deep_ocean_shares_no_edge_with_land(self):
        """"Ocean ... does not share any edges with land tiles" -- edges, not
        corners, so a tile diagonally off a headland is still ocean."""
        def orthogonal(s, t):
            x, y = s.xy(t)
            for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < s.size and 0 <= ny < s.size:
                    yield ny * s.size + nx

        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="continents")
            for t in range(s.n_tiles):
                if s.terrain[t] != C.T_OCEAN or s.improvement[t] == C.IMP_LIGHTHOUSE:
                    continue
                assert not any(R.terrain_is_land[s.terrain[nb]]
                               for nb in orthogonal(s, t))

    def test_shallow_water_hugs_the_coast(self):
        """The converse: every shallow-water tile does touch land by an edge."""
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=4,
                     map_type="archipelago")
        found = 0
        for t in range(s.n_tiles):
            if s.terrain[t] != C.T_WATER:
                continue
            x, y = s.xy(t)
            nbs = [(y + dy) * s.size + (x + dx)
                   for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1))
                   if 0 <= x + dx < s.size and 0 <= y + dy < s.size]
            assert any(R.terrain_is_land[s.terrain[nb]] for nb in nbs)
            found += 1
        assert found, "an archipelago with no coastline is not an archipelago"


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

    def test_the_suburb_pass_puts_up_to_two_villages_beside_a_capital(self):
        """"After the capital city is placed, the algorithm puts up to two
        villages nearby ... you usually get two."

        Tested on the pass itself. On a finished map the post-terrain pass fills
        in around every capital anyway, so counting neighbours cannot tell a
        Lakes map from a Drylands one — the difference is real but invisible by
        the time generation ends.
        """
        from polyengine.core.mapgen import _place_suburbs
        from polyengine.core.state import GameState

        for seed in range(10):
            s = GameState(R, 16, 2, seed)
            caps = [3 * 16 + 3, 12 * 16 + 12]
            taken = list(caps)
            subs = _place_suburbs(s, caps, taken)
            assert 2 <= len(subs) <= 4, len(subs)      # one or two per capital
            for v in subs:
                near = min(s.chebyshev(v, c) for c in caps)
                assert 2 < near <= 4, "a suburb should be a couple of tiles out"

    def test_the_feature_table_is_transcribed_exactly(self):
        """The wiki's per-type table, as three lookups. Every one of these was
        wrong at some point, and each wrong entry is invisible on the board."""
        from polyengine.core.mapgen import (
            ISLAND_VILLAGE_TYPES,
            PRE_TERRAIN_COEFFICIENT,
            QUADRANT_TYPES,
            SUBURB_TYPES,
        )
        assert QUADRANT_TYPES == {"drylands", "lakes", "archipelago",
                                  "waterworld"}
        assert SUBURB_TYPES == {"lakes", "archipelago"}
        assert PRE_TERRAIN_COEFFICIENT == {"lakes": 0.3, "archipelago": 0.3,
                                           "waterworld": 0.1}
        assert ISLAND_VILLAGE_TYPES == {"waterworld", "continents", "pangea"}
        assert set(MAP_TYPES) - QUADRANT_TYPES == {"continents", "pangea"}

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

    def test_continents_capitals_land_on_different_continents(self):
        """"Then, like on Pangea, one village per player is converted to a
        capital. Capitals are placed on different land masses, space
        permitting."""
        for seed in range(10):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="continents")
            caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
            mass = {}
            for i, comp in enumerate(_land_components(s)):
                for t in comp:
                    mass[t] = i
            assert mass[caps[0]] != mass[caps[1]], f"seed {seed}: same continent"

    def test_capitals_are_converted_villages_and_prefer_the_coast(self):
        """"The algorithm prefers to have capitals next to water tiles. That
        means you almost always get capitals spawns on the coast."""
        coastal = total = 0
        for kind in ("continents", "pangea"):
            for seed in range(10):
                s = new_game(R, size=16, tribes=["imperius", "bardur"],
                             seed=seed, map_type=kind)
                for cap in [t for t in range(s.n_tiles) if s.city_capital[t]]:
                    total += 1
                    coastal += any(not R.terrain_is_land[s.terrain[nb]]
                                   for nb in s.neighbors(cap))
        assert coastal >= 0.8 * total, f"only {coastal}/{total} capitals coastal"

    def test_the_pre_terrain_village_formula(self):
        """"(([map width]/3)^2 - [capitals and suburbs]) * (density coefficient)",
        "rounded down before being squared"."""
        from polyengine.core.mapgen import _pre_terrain_village_count

        class Fake:
            size = 16
        assert _pre_terrain_village_count(Fake(), 0, 0.3) == int((16 // 3) ** 2 * 0.3)
        assert _pre_terrain_village_count(Fake(), 4, 0.3) == int(((16 // 3) ** 2 - 4) * 0.3)
        assert _pre_terrain_village_count(Fake(), 1000, 0.3) == 0


# --- landmass shape --------------------------------------------------------------
class TestLandmassShape:
    """The map type has to mean something on the board, not just in the water
    percentage. "Continents: large but discrete landmasses" / "Pangea: a large
    landmass in the middle of the map surrounded by ocean" / "Archipelago:
    misshapen and discontinuous landmasses" / "Water World: small landmasses of
    one or two villages"."""

    def masses(self, s, min_size=2):
        """Landmasses, ignoring the one-tile islands that island villages make."""
        return [c for c in _land_components(s) if len(c) >= min_size]

    def test_continents_are_discrete_and_large(self):
        for seed in range(8):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="continents")
            masses = self.masses(s)
            assert len(masses) >= 2, f"seed {seed}: continents merged into one"
            for m in masses:
                assert len(m) >= CONTINENT_MIN / 2, \
                    f"seed {seed}: a {len(m)}-tile scrap is not a continent"

    def test_continents_are_separated_by_ocean(self):
        """"[Continents] must be at least one tile apart" -- and the gap must be
        real water, not a diagonal handshake."""
        for seed in range(8):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="continents")
            masses = self.masses(s)
            for i, a in enumerate(masses):
                for b in masses[i + 1:]:
                    gap = min(s.chebyshev(x, y) for x in a for y in b)
                    assert gap >= 2, "two continents are touching"

    def test_pangea_is_one_landmass_around_the_middle(self):
        for seed in range(8):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="pangea")
            masses = self.masses(s)
            assert len(masses) == 1, f"seed {seed}: pangea is not one landmass"
            biggest = masses[0]
            mid = (s.size // 2) * s.size + (s.size // 2)
            assert mid in biggest, "the landmass should cover the centre"

    def test_pangea_edges_are_ocean(self):
        """"...surrounded by ocean on the outskirts"."""
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="pangea")
            border = [t for t in range(s.n_tiles)
                      if s.xy(t)[0] in (0, s.size - 1) or s.xy(t)[1] in (0, s.size - 1)]
            land = sum(1 for t in border if R.terrain_is_land[s.terrain[t]])
            assert land == 0, f"seed {seed}: {land} land tiles on the border"

    def test_archipelago_is_many_small_islands(self):
        """No one island may hold most of the land. (Villages are forced onto
        land, so two islands a tile apart can end up bridged by one -- which is
        why this is a share of the total rather than an absolute cap.)"""
        s = new_game(R, size=18, tribes=["imperius", "kickoo"], seed=2,
                     map_type="archipelago")
        masses = self.masses(s)
        land = sum(len(m) for m in masses)
        assert len(masses) >= 4
        assert max(len(m) for m in masses) <= 0.5 * land

    def test_lakes_keeps_a_land_border(self):
        """"A border of land bridges will always generate on the edges of the
        map." Without it a "lakes" map is just a wet drylands."""
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="lakes")
            for t in range(s.n_tiles):
                x, y = s.xy(t)
                on_border = x in (0, s.size - 1) or y in (0, s.size - 1)
                if on_border and s.improvement[t] != C.IMP_LIGHTHOUSE:
                    assert R.terrain_is_land[s.terrain[t]], f"water at {x},{y}"

    def test_a_continents_map_actually_needs_boats(self):
        """The point of the shape: a capital cannot walk to the other capital.

        This is what the old smoothed-noise generator quietly failed at -- it
        made a lace of one-tile isthmuses, so Ports and boats had nothing to do
        on the one map type built around them.
        """
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed,
                         map_type="continents")
            caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
            reach, stack = {caps[0]}, [caps[0]]
            while stack:
                for nb in s.neighbors(stack.pop()):
                    nb = int(nb)
                    if nb not in reach and R.terrain_is_land[s.terrain[nb]]:
                        reach.add(nb)
                        stack.append(nb)
            assert caps[1] not in reach, f"seed {seed}: you can walk there"


# --- tile biomes ------------------------------------------------------------------
class TestTileBiomes:
    def test_every_tile_belongs_to_a_tribe_in_the_game(self):
        s = new_game(R, size=16, tribes=["xin_xi", "vengir"], seed=1)
        ids = {int(s.tribe[p]) for p in range(s.n_players)}
        assert set(int(v) for v in s.tile_tribe) <= ids

    def test_the_land_is_split_roughly_equally(self):
        """"During the initial map generation, the land is distributed roughly
        equally between all tribes in the game." The randomised flood fill is
        self-balancing because the tribes take turns."""
        for seed in range(6):
            s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=seed)
            counts = [int((s.tile_tribe == int(s.tribe[p])).sum())
                      for p in range(s.n_players)]
            assert min(counts) > 0.30 * s.n_tiles, counts

    def test_a_capital_sits_in_its_own_biome(self):
        s = new_game(R, size=16, tribes=["imperius", "bardur"], seed=3)
        for t in range(s.n_tiles):
            if s.city_capital[t]:
                p = int(s.city_owner[t])
                assert int(s.tile_tribe[t]) == int(s.tribe[p])

    def test_biomes_survive_a_serialization_round_trip(self):
        s = new_game(R, size=14, tribes=["hoodrick", "oumaji"], seed=4)
        clone = type(s).from_dict(s.to_dict(), R)
        assert (clone.tile_tribe == s.tile_tribe).all()


# --- guaranteed starts -------------------------------------------------------------
class TestGuaranteedCapitalResources:
    def ring(self, s, tile):
        return [int(nb) for nb in s.neighbors(tile)]

    def capital_of(self, s, player):
        return next(t for t in range(s.n_tiles)
                    if s.city_capital[t] and int(s.city_owner[t]) == player)

    def test_bardur_always_starts_with_game_to_hunt(self):
        for seed in range(8):
            s = new_game(R, size=14, tribes=["bardur", "oumaji"], seed=seed)
            cap = self.capital_of(s, 0)
            n = sum(1 for t in self.ring(s, cap) if s.resource[t] == C.R_GAME)
            assert n >= 2, f"seed {seed}: Bardur has {n} game beside its capital"

    def test_imperius_always_starts_with_fruit(self):
        for seed in range(8):
            s = new_game(R, size=14, tribes=["imperius", "oumaji"], seed=seed)
            cap = self.capital_of(s, 0)
            n = sum(1 for t in self.ring(s, cap) if s.resource[t] == C.R_FRUIT)
            assert n >= 2, f"seed {seed}: Imperius has {n} fruit"

    def test_kickoo_gets_water_with_fish_even_on_drylands(self):
        """"Drylands: entirely land. (Kickoo and Aquarion capitals will include
        two water tiles with a fish in each.)"."""
        for seed in range(8):
            s = new_game(R, size=14, tribes=["kickoo", "oumaji"], seed=seed,
                         map_type="drylands")
            cap = self.capital_of(s, 0)
            fish = [t for t in self.ring(s, cap) if s.resource[t] == C.R_FISH]
            assert len(fish) >= 2, f"seed {seed}: {len(fish)} fish"
            for t in fish:
                assert not R.terrain_is_land[s.terrain[t]]

    def test_zebasi_gets_a_crop(self):
        for seed in range(8):
            s = new_game(R, size=14, tribes=["zebasi", "oumaji"], seed=seed)
            cap = self.capital_of(s, 0)
            n = sum(1 for t in self.ring(s, cap) if s.resource[t] == C.R_CROP)
            assert n >= 1, f"seed {seed}: Zebasi has no crop"

    def test_a_tribe_without_a_promise_is_left_to_the_dice(self):
        """The fish is Kickoo's promise specifically, not a general handout: an
        Oumaji start on Drylands sometimes has no fish at all."""
        dry_starts = 0
        for seed in range(10):
            s = new_game(R, size=14, tribes=["oumaji", "luxidoor"], seed=seed,
                         map_type="drylands")
            for player in (0, 1):
                cap = self.capital_of(s, player)
                if not any(s.resource[t] == C.R_FISH for t in self.ring(s, cap)):
                    dry_starts += 1
        assert dry_starts > 0, "everyone is being handed Kickoo's guarantee"


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


# --- everyone gets a capital ------------------------------------------------------
class TestEveryPlayerIsSeated:
    """A player without a capital has no city and no unit, so it is dead before
    the first move and the game is terminal at turn 0.

    That is not a hypothetical: village promotion has nothing to promote when a
    cramped map holds fewer villages than players, and the quadrant route's
    "no land in this quadrant, look outside it" fallback can send two quadrants
    to the same shore. It showed up as a training smoke run whose rollouts
    collapsed from 1024 steps to 64 — the games were ending instantly.
    """

    @pytest.mark.parametrize("size", [6, 8, 11, 14, 16])
    @pytest.mark.parametrize("kind", sorted(MAP_TYPES))
    def test_two_players_both_get_a_capital(self, size, kind):
        for seed in range(6):
            s = new_game(R, size=size, tribes=["imperius", "bardur"],
                         seed=seed, map_type=kind)
            caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
            assert len(caps) == 2, f"{kind} {size} seed {seed}: {len(caps)} capitals"
            assert {int(s.city_owner[t]) for t in caps} == {0, 1}
            assert all(s.unit_type[t] != C.U_NONE for t in caps)

    def test_a_crowded_map_still_seats_everyone(self):
        tribes = ["imperius", "bardur", "xin_xi", "oumaji"]
        for kind in ("continents", "waterworld"):
            for seed in range(6):
                s = new_game(R, size=11, tribes=tribes, seed=seed,
                             map_type=kind)
                caps = [t for t in range(s.n_tiles) if s.city_capital[t]]
                assert len(caps) == 4, f"{kind} seed {seed}"
                assert len({int(s.city_owner[t]) for t in caps}) == 4

    def test_a_game_is_never_over_before_it_starts(self):
        from polyengine.core.game import Game
        for kind in sorted(MAP_TYPES):
            for size in (6, 11):
                for seed in range(8):
                    g = Game.new(size=size, seed=seed, map_type=kind)
                    assert not g.is_terminal(), f"{kind} {size} seed {seed}"


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
