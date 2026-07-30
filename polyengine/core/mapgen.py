"""Deterministic map generation, following Polytopia's own algorithm.

Everything derives from the single per-game PCG64 stream on the state, so
`(map_seed, ruleset_version, tribe_config)` fully reproduces a game.

The wiki's Map Generation page documents this in enough detail to implement
rather than approximate, and the order below is its order:

    capitals -> villages -> terrain -> resources -> lighthouses

Three rules carry most of the strategic weight, and the previous generator had
none of them:

* **Terrain is per-tribe.** "During the initial map generation, the land is
  distributed roughly equally between all tribes in the game", each biome
  generated at that tribe's own rates. Bardur's quarter really is forest; a
  Vengir neighbour really is sitting on metal. We partition tiles to the nearest
  capital and generate each region with its owner's modifiers.
* **Resources only spawn near cities.** "All resources spawn only within two
  tiles of cities or villages", and tiles *adjacent* to one are roughly three
  times as rich as tiles two away. Expansion is therefore how you reach
  resources at all, rather than resources being smeared evenly over the map.
* **Water exists.** Map types run from Drylands (0-10% water) to Water World
  (90-100%). The old generator produced no water whatsoever, which quietly made
  Ports, embarking, Bridges, Pirates and Aquatism unreachable.

Each map type runs a different combination of village passes, per the wiki's
feature table: suburbs on Lakes and Archipelago, pre-terrain villages on those
plus Water World, post-terrain villages everywhere, and tiny island villages on
Continents and Pangea.

Still simplified, and deliberately: Continents uses a smoothed-noise landmass
rather than the real seeded growth, Pangea is not seeded from the centre, and
capitals are not preferentially coastal.
"""
from __future__ import annotations

import numpy as np

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset
from polyengine.core.state import GameState

# --- map sizes ----------------------------------------------------------------
# "Tiny: 121 tiles, 11x11 / Small: 196, 14x14 / Normal: 256, 16x16 /
#  Large: 324, 18x18 / Huge: 400, 20x20 / Massive: 900, 30x30"
MAP_SIZES = {"tiny": 11, "small": 14, "normal": 16,
             "large": 18, "huge": 20, "massive": 30}

# "In Domination, the map size changes depending on the number of opponents:
#  Games with one opponent are on a tiny map, two opponents on a 196-tile
#  (14x14) map, three opponents on a normal map, and four or more on a large."
def size_for_opponents(n_opponents: int) -> int:
    if n_opponents <= 1:
        return MAP_SIZES["tiny"]
    if n_opponents == 2:
        return MAP_SIZES["small"]
    if n_opponents == 3:
        return MAP_SIZES["normal"]
    return MAP_SIZES["large"]


# --- water ---------------------------------------------------------------------
# Wetness bands from the map-type table. The value is the fraction of tiles that
# end up water or ocean.
MAP_TYPES = {
    "drylands": (0.00, 0.10),
    "lakes": (0.25, 0.30),
    "continents": (0.40, 0.70),
    "pangea": (0.40, 0.60),
    "archipelago": (0.60, 0.80),
    "waterworld": (0.90, 1.00),
}

# --- base spawn rates (Luxidoor) ------------------------------------------------
# "Field (Total) 48% / Forest (Total) 38% / Mountain (Total) 14%"
BASE_TERRAIN = {C.T_FIELD: 0.48, C.T_FOREST: 0.38, C.T_MOUNTAIN: 0.14}

# Chance of a resource ON its terrain, split by distance to the nearest city.
# "Inner city refers to tiles that are adjacent to a city or village, while
# outer city refers to those that are not."
#   inner: fruit 18 / crop 18 of 48 field; animal 19 of 38 forest; metal 11 of 14
#   outer: fruit  6 / crop  6 of 48 field; animal  6 of 38 forest; metal  3 of 14
INNER_RESOURCE = {
    C.T_FIELD: {C.R_FRUIT: 18 / 48, C.R_CROP: 18 / 48},
    C.T_FOREST: {C.R_GAME: 19 / 38},
    C.T_MOUNTAIN: {C.R_METAL: 11 / 14},
}
OUTER_RESOURCE = {
    C.T_FIELD: {C.R_FRUIT: 6 / 48, C.R_CROP: 6 / 48},
    C.T_FOREST: {C.R_GAME: 6 / 38},
    C.T_MOUNTAIN: {C.R_METAL: 3 / 14},
}
FISH_RATE = 0.50                 # "the standard spawn rate for fish is 50%"
RESOURCE_RADIUS = 2              # "only within two tiles of cities or villages"

# "Lakes and Archipelago use a density coefficient of 0.3; waterworld uses 0.1."
PRE_TERRAIN_COEFFICIENT = {"lakes": 0.3, "archipelago": 0.3, "waterworld": 0.1}
# "Continents and Pangea ... don't use quadrants for capital placement."
QUADRANT_TYPES = frozenset({"drylands", "lakes", "archipelago", "waterworld"})

VILLAGE_EDGE_MARGIN = 2          # "not within two tiles of the edge of the map"
VILLAGE_SPACING = 2              # "not within two tiles of other villages"


def terrain_rates(tribe: dict) -> dict[int, float]:
    """Terrain shares for one tribe, applying multipliers in the wiki's order.

    "First the mountain modifier is checked. If this is not the base rate, it
    will affect both the forest and field rate ... After this was changed, the
    forest multiplier is checked. Since this is checked after the mountain
    modifier, it will no longer impact the mountain rate, but only impact the
    fields. Finally the field rate ... is thus just calculated via the simple
    100% - Mountain% - Forest%."
    """
    mult = tribe.get("terrain_mult", {}) or {}
    mountain = BASE_TERRAIN[C.T_MOUNTAIN] * float(mult.get("mountain", 1.0))
    mountain = min(mountain, 0.90)
    # Mountains take their share out of the remaining land proportionally.
    remaining = 1.0 - mountain
    base_rest = BASE_TERRAIN[C.T_FOREST] + BASE_TERRAIN[C.T_FIELD]
    forest = remaining * (BASE_TERRAIN[C.T_FOREST] / base_rest)
    forest *= float(mult.get("forest", 1.0))
    forest = min(forest, remaining)
    field = max(0.0, 1.0 - mountain - forest)
    return {C.T_FIELD: field, C.T_FOREST: forest, C.T_MOUNTAIN: mountain}


def resource_rate(tribe: dict, terrain: int, resource: int, inner: bool) -> float:
    table = INNER_RESOURCE if inner else OUTER_RESOURCE
    base = table.get(terrain, {}).get(resource, 0.0)
    mult = (tribe.get("resource_mult", {}) or {})
    key = {C.R_FRUIT: "fruit", C.R_CROP: "crop", C.R_GAME: "game",
           C.R_METAL: "metal", C.R_FISH: "fish"}[resource]
    return base * float(mult.get(key, 1.0))


# --- placement ------------------------------------------------------------------
def _quadrant_capitals(state: GameState, n: int) -> list[int]:
    """One capital per quadrant, so players never start on top of each other.

    "Most maps use quadrants to place capitals so players don't start too close
    to one another ... 1-4 players creates 4 domains, 5-9 players makes 9
    domains, and 10-16 players makes 16 domains."
    """
    grid = 2 if n <= 4 else (3 if n <= 9 else 4)
    size = state.size
    step = size / grid
    quadrants = [(qx, qy) for qy in range(grid) for qx in range(grid)]
    order = state.rng.permutation(len(quadrants))[:n]

    out = []
    margin = VILLAGE_EDGE_MARGIN
    for qi in order:
        qx, qy = quadrants[int(qi)]
        x0, x1 = int(qx * step), int((qx + 1) * step) - 1
        y0, y1 = int(qy * step), int((qy + 1) * step) - 1
        # Keep capitals off the border: cities never spawn on the map edge.
        x0, x1 = max(x0, margin), min(x1, size - 1 - margin)
        y0, y1 = max(y0, margin), min(y1, size - 1 - margin)
        if x1 < x0:
            x0 = x1 = min(max(margin, size // 2), size - 1 - margin)
        if y1 < y0:
            y0 = y1 = min(max(margin, size // 2), size - 1 - margin)

        # Quadrants alone are not enough: two capitals either side of a shared
        # border can land two tiles apart, which is exactly the "players start
        # too close to one another" the quadrants exist to prevent. Within the
        # quadrant, take the tile furthest from every capital placed so far.
        tiles = [y * size + x
                 for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        if not out:
            out.append(int(state.rng.choice(tiles)))
            continue
        best, best_d = None, -1
        for tile in tiles:
            d = min(state.chebyshev(tile, c) for c in out)
            # Jitter breaks ties without unpicking determinism.
            score = d * 100 + int(state.rng.integers(100))
            if score > best_d:
                best_d, best = score, tile
        out.append(int(best))
    return out


def _scattered_capitals(state: GameState, n: int) -> list[int]:
    """Capitals without quadrants, for Continents and Pangea.

    "Continents and Pangea are the only two map types that don't use quadrants
    for capital placement." Capitals may therefore land anywhere, including two
    in the same corner of the map, as long as they are not crowded: each is drawn
    uniformly from the tiles at least `size // 2` from every capital already
    down. Taking the single furthest tile instead would reintroduce quadrants by
    the back door -- with two players it always returns opposite corners.

    Six is the floor because that is what the quadrant path guarantees, and two
    capitals closer than that is the "players start too close to one another"
    problem quadrants exist to prevent in the first place.
    """
    size = state.size
    margin = VILLAGE_EDGE_MARGIN
    tiles = [y * size + x
             for y in range(margin, size - margin)
             for x in range(margin, size - margin)]
    spacing = max(6, size // 4)
    out = [int(state.rng.choice(tiles))]
    while len(out) < n:
        far = [t for t in tiles
               if min(state.chebyshev(t, c) for c in out) >= spacing]
        if far:
            out.append(int(state.rng.choice(far)))
            continue
        # Crowded map: fall back to the furthest tile there is.
        out.append(max((t for t in tiles if t not in out),
                       key=lambda t: min(state.chebyshev(t, c) for c in out)))
    return out


def _place_villages(state: GameState, taken: list[int],
                    limit: int | None = None,
                    force_land: bool = False) -> list[int]:
    """Scatter villages until no legal tile remains, or `limit` are placed.

    "It just randomly throws in villages following post-terrain rules until
    there is no tile left where it can place one." The rules are: at least two
    tiles from the map edge, and at least two tiles from any other village.

    `taken` is shared and appended to, so the suburb, pre-terrain and
    post-terrain passes all respect one another's spacing.

    `force_land` is for Water World, where "almost all tiles become water or
    ocean; land is manually forced for cities" -- otherwise nowhere qualifies
    and the map generates with no villages at all.
    """
    size = state.size
    candidates = [y * size + x
                  for y in range(VILLAGE_EDGE_MARGIN, size - VILLAGE_EDGE_MARGIN)
                  for x in range(VILLAGE_EDGE_MARGIN, size - VILLAGE_EDGE_MARGIN)]
    state.rng.shuffle(candidates)

    villages = []
    for tile in candidates:
        if limit is not None and len(villages) >= limit:
            break
        if state.terrain[tile] in (C.T_OCEAN, C.T_WATER) and not force_land:
            continue
        if any(state.chebyshev(tile, t) <= VILLAGE_SPACING for t in taken):
            continue
        taken.append(tile)
        villages.append(tile)
    return villages


def _assign_biomes(state: GameState, capitals: list[int]) -> np.ndarray:
    """Which tribe's land each tile is, by nearest capital.

    "During the initial map generation, the land is distributed roughly equally
    between all tribes" -- a nearest-capital partition is the simplest thing
    that produces that, and it is what makes a neighbour's terrain legible.
    """
    owner = np.zeros(state.n_tiles, dtype=np.int8)
    for tile in range(state.n_tiles):
        best, best_d = 0, 1 << 30
        for pid, cap in enumerate(capitals):
            d = state.chebyshev(tile, cap)
            if d < best_d:
                best_d, best = d, pid
        owner[tile] = best
    return owner


def _carve_water(state: GameState, wetness: float, keep: set[int]) -> None:
    """Turn a `wetness` fraction of the map into water, in blobs rather than dots.

    Smoothed noise stands in for the real seeded landmass growth: it gives
    coastlines and the occasional one-tile channel ("rivers, which can be great
    for bridges") instead of salt-and-pepper, which is what matters for whether
    Ports and Bridges are ever worth building.
    """
    if wetness <= 0:
        return
    size = state.size
    noise = state.rng.random((size, size))
    # Two box-blur passes: cheap, deterministic, and enough to clump.
    for _ in range(2):
        padded = np.pad(noise, 1, mode="edge")
        noise = sum(padded[dy:dy + size, dx:dx + size]
                    for dy in range(3) for dx in range(3)) / 9.0

    flat = noise.reshape(-1)
    protected = np.array(sorted(keep), dtype=int)
    order = np.argsort(flat)                      # wettest (lowest) first
    budget = int(round(wetness * state.n_tiles))
    made = 0
    for tile in order:
        if made >= budget:
            break
        tile = int(tile)
        if tile in keep:
            continue
        if any(state.chebyshev(tile, int(p)) <= 1 for p in protected):
            continue                              # leave a shore around cities
        state.terrain[tile] = C.T_WATER
        made += 1

    # Deep ocean is water with no land within one tile.
    for tile in range(state.n_tiles):
        if state.terrain[tile] != C.T_WATER:
            continue
        if not any(state.rules.terrain_is_land[state.terrain[nb]]
                   for nb in state.neighbors(tile)):
            state.terrain[tile] = C.T_OCEAN


def _place_suburbs(state: GameState, capitals: list[int],
                   taken: list[int]) -> list[int]:
    """Up to two villages near each capital. Lakes and Archipelago only.

    "After the capital city is placed, the algorithm puts up to two villages
    nearby ... because suburbs are placed before terrain, they are usually a
    couple tiles away."
    """
    out = []
    size = state.size
    for cap in capitals:
        wanted = int(state.rng.integers(1, 3))          # usually two
        ring = [t for t in range(state.n_tiles)
                if VILLAGE_SPACING < state.chebyshev(t, cap) <= VILLAGE_SPACING + 2]
        state.rng.shuffle(ring)
        placed = 0
        for tile in ring:
            if placed >= wanted:
                break
            x, y = state.xy(tile)
            if not (VILLAGE_EDGE_MARGIN <= x < size - VILLAGE_EDGE_MARGIN
                    and VILLAGE_EDGE_MARGIN <= y < size - VILLAGE_EDGE_MARGIN):
                continue
            if any(state.chebyshev(tile, t) <= VILLAGE_SPACING for t in taken):
                continue
            taken.append(tile)
            out.append(tile)
            placed += 1
    return out


def _pre_terrain_village_count(state: GameState, placed: int,
                               coefficient: float) -> int:
    """"(([map width]/3)^2 - [capitals and suburbs]) * (map density coefficient)".

    "The map width/3 number is rounded down before being squared."
    """
    base = (state.size // 3) ** 2
    return max(0, int((base - placed) * coefficient))


def _place_island_villages(state: GameState, taken: list[int],
                           count: int) -> list[int]:
    """Villages "that spawn in the water without any land connections".

    Continents and Pangea only, a fixed number per map size.
    """
    out = []
    candidates = [t for t in range(state.n_tiles)
                  if not state.rules.terrain_is_land[state.terrain[t]]
                  and state.improvement[t] == C.IMP_NONE]
    state.rng.shuffle(candidates)
    size = state.size
    for tile in candidates:
        if len(out) >= count:
            break
        x, y = state.xy(tile)
        if not (VILLAGE_EDGE_MARGIN <= x < size - VILLAGE_EDGE_MARGIN
                and VILLAGE_EDGE_MARGIN <= y < size - VILLAGE_EDGE_MARGIN):
            continue
        if any(state.chebyshev(tile, t) <= VILLAGE_SPACING for t in taken):
            continue
        # "without any land connections": every neighbour must be water.
        if any(state.rules.terrain_is_land[state.terrain[nb]]
               for nb in state.neighbors(tile)):
            continue
        state.terrain[tile] = C.T_FIELD          # the island itself
        # An island has a coast: deep ocean touching it becomes shallow water,
        # which is also what keeps "ocean means no land within one tile" true.
        for nb in state.neighbors(tile):
            if state.terrain[nb] == C.T_OCEAN:
                state.terrain[nb] = C.T_WATER
        taken.append(tile)
        out.append(tile)
    return out


def _place_ruins(state: GameState, cities: list[int]) -> list[int]:
    """Ruins, "placed randomly after villages, resources, and lighthouses".

    "Ruins can spawn on all types of terrain except shallow water. However,
    ruins cannot spawn next to a capital city or another ruin."
    """
    count = C.RUINS_BY_SIZE.get(state.size)
    if count is None:                              # a size the table omits
        count = max(1, round(4 * state.n_tiles / 121))
    capitals = [t for t in range(state.n_tiles) if state.city_capital[t]]
    candidates = [t for t in range(state.n_tiles)
                  if state.terrain[t] != C.T_WATER
                  and state.improvement[t] == C.IMP_NONE
                  and state.resource[t] == C.R_NONE
                  and state.city_level[t] == 0
                  and state.unit_type[t] == C.U_NONE]
    state.rng.shuffle(candidates)

    placed: list[int] = []
    for tile in candidates:
        if len(placed) >= count:
            break
        if any(state.chebyshev(tile, c) <= 1 for c in capitals):
            continue                                # "not next to a capital"
        if any(state.chebyshev(tile, r) <= 1 for r in placed):
            continue                                # "nor another ruin"
        state.improvement[tile] = C.IMP_RUIN
        placed.append(tile)
    return placed


def _place_starfish(state: GameState) -> list[int]:
    """"approximately one starfish for every 25 water tiles".

    "Starfish cannot be next to another starfish, lighthouse, or city."
    """
    water = [t for t in range(state.n_tiles)
             if not state.rules.terrain_is_land[state.terrain[t]]]
    count = len(water) // C.STARFISH_PER_WATER_TILES
    if count <= 0:
        return []
    avoid = [t for t in range(state.n_tiles)
             if state.city_level[t] > 0 or state.improvement[t] == C.IMP_LIGHTHOUSE]
    state.rng.shuffle(water)

    placed: list[int] = []
    for tile in water:
        if len(placed) >= count:
            break
        if state.resource[tile] != C.R_NONE or state.improvement[tile] != C.IMP_NONE:
            continue
        if any(state.chebyshev(tile, a) <= 1 for a in avoid):
            continue
        if any(state.chebyshev(tile, s) <= 1 for s in placed):
            continue
        state.resource[tile] = C.R_STARFISH
        placed.append(tile)
    return placed


def _place_lighthouses(state: GameState) -> None:
    """A lighthouse in each corner; the Explorer task needs all four."""
    s = state.size
    for tile in (0, s - 1, s * (s - 1), s * s - 1):
        state.terrain[tile] = C.T_OCEAN
        state.resource[tile] = C.R_NONE
        state.improvement[tile] = C.IMP_LIGHTHOUSE
        state.city_owner[tile] = C.NEUTRAL
        state.city_level[tile] = 0
        state.unit_type[tile] = C.U_NONE


def _sprinkle_resources(state: GameState, rules: Ruleset, biome: np.ndarray,
                        tribes: list[str], cities: list[int]) -> None:
    """Resources, but only near a city, and richer the closer they are."""
    if not cities:
        return
    dist = np.full(state.n_tiles, 1 << 20, dtype=np.int32)
    for tile in range(state.n_tiles):
        dist[tile] = min(state.chebyshev(tile, c) for c in cities)

    for tile in range(state.n_tiles):
        if state.improvement[tile] != C.IMP_NONE or state.city_level[tile] > 0:
            continue
        d = int(dist[tile])
        if d > RESOURCE_RADIUS:
            continue                              # the two-tile rule
        tribe = rules.tribes[tribes[int(biome[tile])]]
        terrain = int(state.terrain[tile])
        inner = d <= 1

        if terrain in (C.T_WATER, C.T_OCEAN):
            if terrain == C.T_WATER:
                rate = FISH_RATE * float(
                    (tribe.get("resource_mult", {}) or {}).get("fish", 1.0))
                if state.rng.random() < rate:
                    state.resource[tile] = C.R_FISH
            continue

        for resource in (C.R_FRUIT, C.R_CROP, C.R_GAME, C.R_METAL):
            rate = resource_rate(tribe, terrain, resource, inner)
            if rate > 0 and state.rng.random() < rate:
                state.resource[tile] = resource
                break


def new_game(rules: Ruleset, size: int | None = None,
             tribes: list[str] | None = None, seed: int = 0,
             map_type: str = "drylands") -> GameState:
    """Create a fresh Domination game.

    `size` defaults to what Domination would pick for this many opponents.
    `map_type` selects a wetness band from the wiki's table.
    """
    tribes = tribes or ["imperius", "bardur"]
    n_players = len(tribes)
    if size is None:
        size = size_for_opponents(n_players - 1)
    if map_type not in MAP_TYPES:
        raise ValueError(f"unknown map type {map_type!r}; "
                         f"choose from {sorted(MAP_TYPES)}")
    state = GameState(rules, size, n_players, seed)

    # 1. Capitals: one per quadrant, except on Continents and Pangea.
    if map_type in QUADRANT_TYPES:
        capitals = _quadrant_capitals(state, n_players)
    else:
        capitals = _scattered_capitals(state, n_players)
    biome = _assign_biomes(state, capitals)

    # 2. Terrain, at each region's own tribe rates.
    rate_by_tribe = {name: terrain_rates(rules.tribes[name]) for name in tribes}
    for tile in range(state.n_tiles):
        rates = rate_by_tribe[tribes[int(biome[tile])]]
        ids = list(rates)
        p = np.array([rates[t] for t in ids], dtype=float)
        p = p / p.sum()
        state.terrain[tile] = int(state.rng.choice(ids, p=p))

    # 3. Water. Kickoo land is wetter ("2.0x water"), so the band is nudged by
    #    the average multiplier of the tribes present.
    lo, hi = MAP_TYPES[map_type]
    wetness = float(state.rng.uniform(lo, hi))
    water_mult = float(np.mean([rules.tribes[t].get("water_mult", 1.0)
                                for t in tribes]))
    wetness = min(0.95, wetness * water_mult)
    _carve_water(state, wetness, keep=set(capitals))

    # 4. Villages. Which passes run depends on the map type, per the wiki's
    #    feature table: suburbs on Lakes and Archipelago, pre-terrain villages
    #    on those plus Waterworld, post-terrain villages everywhere, and tiny
    #    island villages on Continents and Pangea.
    taken = list(capitals)
    villages: list[int] = []
    if map_type in ("lakes", "archipelago"):
        villages += _place_suburbs(state, capitals, taken)
    force_land = map_type == "waterworld"
    coefficient = PRE_TERRAIN_COEFFICIENT.get(map_type, 0.0)
    if coefficient:
        wanted = _pre_terrain_village_count(state, len(taken), coefficient)
        villages += _place_villages(state, taken, limit=wanted,
                                    force_land=force_land)
    # post-terrain: "throws in villages ... until there is no tile left"
    villages += _place_villages(state, taken, force_land=force_land)
    if map_type in ("continents", "pangea"):
        n_islands = C.ISLAND_VILLAGES_BY_SIZE.get(state.size, 0)
        villages += _place_island_villages(state, taken, n_islands)

    for tile in villages:
        if state.rules.terrain_is_land[state.terrain[tile]]:
            state.terrain[tile] = C.T_FIELD
        state.city_owner[tile] = C.NEUTRAL          # neutral, capturable
        state.city_level[tile] = 1

    for pid, (tribe_name, cap) in enumerate(zip(tribes, capitals)):
        tribe = rules.tribes[tribe_name]
        state.tribe[pid] = tribe["id"]
        state.stars[pid] = int(tribe.get("starting_stars", 5))
        starting_tech = tribe.get("starting_tech", "")
        if starting_tech:
            state.grant_tech(pid, rules.tech_id[starting_tech])

        state.terrain[cap] = C.T_FIELD          # forced, even on Water World
        state.resource[cap] = C.R_NONE
        state.improvement[cap] = C.IMP_NONE
        state.city_owner[cap] = pid
        # Luxidoor buys a level-3 capital with its starting technology.
        state.city_level[cap] = int(tribe.get("starting_city_level", 1))
        state.city_capital[cap] = 1
        state.owner[cap] = pid
        for nb in state.neighbors(cap):
            if state.owner[nb] == C.NEUTRAL:
                state.owner[nb] = pid

        unit = rules.unit_id[tribe.get("starting_unit", "warrior")]
        state.unit_type[cap] = unit
        state.unit_owner[cap] = pid
        state.unit_hp[cap] = rules.unit_max_hp[unit]
        state.unit_home[cap] = cap

    # 5. Resources, near cities only.
    _sprinkle_resources(state, rules, biome, tribes, capitals + villages)
    _place_lighthouses(state)
    # "Ruins are placed randomly after villages, resources, and lighthouses."
    _place_ruins(state, capitals + villages)
    _place_starfish(state)

    from polyengine.core.fog import recompute_all
    recompute_all(state)
    return state
