"""Deterministic map generation, following Polytopia's own algorithm.

Everything derives from the single per-game PCG64 stream on the state, so
`(map_seed, ruleset_version, tribe_config)` fully reproduces a game.

Sources, in order of authority:

* `.wikicache/Map Generation.wiki` — the algorithm, its order, the per-type
  feature table, the spawn-rate tables and the size/wetness tables.
* `.wikicache/Terrain.wiki` — in particular "Ocean ... does not share any *edges*
  with land tiles", which is what separates shallow water from deep ocean.
* QuasiStellar/Polytopia-Map-Generator (`app.py`) — a community port of the
  pre-Moonrise generator. It is not the shipping code, but it agrees with the
  wiki everywhere the two overlap and it fills in mechanics the wiki only
  describes in prose: the village ranking that also classifies inner/outer
  resource tiles, the exact `1/3` outer-city falloff, the randomised flood fill
  that hands each tribe its share of the land, and the guaranteed resources
  beside certain capitals.

**Two pipelines, because the game has two.** The wiki's feature table splits the
map types down the middle:

    Drylands / Lakes / Archipelago / Water World
        capitals (quadrants) -> suburbs -> pre-terrain villages -> terrain and
        water -> post-terrain villages -> resources -> ruins, starfish

    Continents / Pangea
        land -> villages -> *convert* villages to capitals -> terrain ->
        island villages -> resources -> ruins, starfish

That is why those two types have no quadrants and no pre-terrain villages:
"Villages are added after all the land is made, that's why Continents and Pangea
don't use quadrants for capital placement." Capitals are chosen out of the
villages already on the ground — as far apart as possible, preferring the coast.

**Landmasses are grown, not thresholded.** A blob-growth primitive
(`_grow_blobs`) seeds N points and accretes tiles onto them one at a time, never
letting two blobs touch. Every map type is then a choice of how many blobs, how
big, and whether the blobs are land in an ocean or lakes in a continent:

| type | shape |
|---|---|
| drylands | land everywhere, a couple of small lakes |
| lakes | land everywhere, several inland lakes, land border forced |
| continents | one large landmass per player, 30–200 tiles, discrete |
| pangea | a single mass grown from the centre |
| archipelago | many small misshapen masses |
| waterworld | islets of a few tiles each |

The earlier implementation thresholded smoothed noise, which produced *speckle*
rather than continents — on a "continents" map the land came out as a lace of
one-tile isthmuses with no ocean worth crossing, so Ports, Bridges and boats had
nothing to do. Growing blobs and keeping them apart is what makes the map type
mean something strategically.
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

# Types where the water is *inside* the land rather than the land inside the
# water. "Lakes: moderate wetness results in scattered inland lakes."
LAKE_TYPES = frozenset({"drylands", "lakes"})
# "Continents and Pangea are the only two map types that don't use quadrants."
QUADRANT_TYPES = frozenset({"drylands", "lakes", "archipelago", "waterworld"})
# Types whose capitals are converted from villages after the land exists.
CONVERTED_CAPITAL_TYPES = frozenset({"continents", "pangea"})

# "Continents have between 30 and 200 tiles of land and must be at least one
# tile apart." The wiki works the example itself: "For a 196 tiles map and two
# players, half the map being water, then each continent would have about 50
# tiles" -- so the mass count tracks the player count.
CONTINENT_MIN, CONTINENT_MAX = 30, 200
# Rough tiles per island for the types that are made of many small pieces.
# "Archipelago: misshapen and discontinuous landmasses. Almost every city is
# coastal." / "Water World: small landmasses of one or two villages."
ARCHIPELAGO_ISLAND = 14
WATERWORLD_ISLAND = 4
LAKE_SIZE = 10                   # tiles per inland lake

# --- base spawn rates (Luxidoor) ------------------------------------------------
# "Field (Total) 48% / Forest (Total) 38% / Mountain (Total) 14%"
BASE_TERRAIN = {C.T_FIELD: 0.48, C.T_FOREST: 0.38, C.T_MOUNTAIN: 0.14}

# Chance of a resource ON its terrain, split by distance to the nearest city.
# "Inner city refers to tiles that are adjacent to a city or village, while
# outer city refers to those that are not."
#   inner: fruit 18 / crop 18 of 48 field; animal 19 of 38 forest; metal 11 of 14
#   outer: fruit  6 / crop  6 of 48 field; animal  6 of 38 forest; metal  3 of 14
# The outer column is the inner one over three, which the community generator
# spells out as a single `BORDER_EXPANSION = 1/3` coefficient.
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

# The feature table, as three lookups. Suburbs go to Lakes and Archipelago;
# pre-terrain villages to those two plus Water World, at the documented density
# coefficients; tiny island villages to Water World, Continents and Pangea.
SUBURB_TYPES = frozenset({"lakes", "archipelago"})
# "Lakes and Archipelago use a density coefficient of 0.3; waterworld uses 0.1."
PRE_TERRAIN_COEFFICIENT = {"lakes": 0.3, "archipelago": 0.3, "waterworld": 0.1}
ISLAND_VILLAGE_TYPES = frozenset({"waterworld", "continents", "pangea"})

VILLAGE_EDGE_MARGIN = 2          # "not within two tiles of the edge of the map"
PRE_TERRAIN_EDGE_MARGIN = 1      # "at least one tile from the edge of the map"
VILLAGE_SPACING = 2              # "not within two tiles of other villages"

# "Ruins ... cannot spawn right next to another ruin or a village" (Map
# Generation) versus "cannot spawn next to a capital city or another ruin"
# (Ruins). The Ruins page is the newer of the two -- its own trivia notes that
# ruins could not spawn next to villages only *before* update 2.0.58 -- so the
# capital rule is the one implemented.
# "On Lakes, a maximum of one third of these ruins are allowed to spawn on
# water." The community generator applies the same one-third cap on every map.
WATER_RUIN_FRACTION = 1 / 3

# "Kickoo and Aquarion capitals will include two water tiles with a fish in
# each", and the community generator forces the same for three land tribes.
# Aquarion and Elyrion are special tribes and excluded with the rest.
GUARANTEED_CAPITAL_RESOURCES = {
    "kickoo": (C.R_FISH, C.T_WATER, 2),
    "imperius": (C.R_FRUIT, C.T_FIELD, 2),
    "bardur": (C.R_GAME, C.T_FOREST, 2),
    "zebasi": (C.R_CROP, C.T_FIELD, 1),
}


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


# --- geometry helpers -----------------------------------------------------------
def _in_bounds(state: GameState, tile: int, margin: int) -> bool:
    x, y = state.xy(tile)
    return (margin <= x < state.size - margin
            and margin <= y < state.size - margin)


def _orthogonal(state: GameState, tile: int):
    """The four edge-sharing neighbours. Ocean depth is defined on edges, not
    corners: "Ocean ... does not share any edges with land tiles"."""
    x, y = state.xy(tile)
    for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < state.size and 0 <= ny < state.size:
            yield ny * state.size + nx


# --- landmass -------------------------------------------------------------------
def _grow_blobs(state: GameState, n_blobs: int, budget: int,
                seeds: list[int] | None = None,
                margin: int = 0, separated: bool = True,
                cap: int | None = None) -> list[set[int]]:
    """Grow `n_blobs` connected blobs covering about `budget` tiles in total.

    Each blob accretes one tile at a time from its own frontier, so the result
    is a compact, organically-shaped mass rather than the speckle a noise
    threshold gives. With `separated`, a tile touching a *different* blob is
    never taken, which is how "continents ... must be at least one tile apart"
    holds by construction rather than by a later repair pass.

    Blobs take turns, so they finish at roughly equal sizes — the same property
    the game's own randomised flood fill relies on to distribute land "roughly
    equally between all tribes".
    """
    if n_blobs <= 0 or budget <= 0:
        return []
    owner = np.full(state.n_tiles, -1, dtype=np.int32)
    blobs: list[set[int]] = []
    frontiers: list[list[int]] = []

    if seeds is None:
        seeds = _spread_seeds(state, n_blobs, margin)
    for i, seed in enumerate(seeds[:n_blobs]):
        owner[seed] = i
        blobs.append({seed})
        frontiers.append([seed])

    placed = len(blobs)
    per_blob = cap if cap is not None else budget
    live = list(range(len(blobs)))
    while placed < budget and live:
        for i in list(live):
            if placed >= budget:
                break
            if len(blobs[i]) >= per_blob:
                live.remove(i)
                continue
            tile = _accrete(state, owner, blobs[i], frontiers[i], i, margin,
                            separated)
            if tile < 0:
                live.remove(i)
                continue
            placed += 1
    return blobs


def _accrete(state: GameState, owner: np.ndarray, blob: set[int],
             frontier: list[int], idx: int, margin: int,
             separated: bool) -> int:
    """Take one legal tile onto `blob`, or return -1 if it is walled in."""
    while frontier:
        pick = int(state.rng.integers(len(frontier)))
        base = frontier[pick]
        options = []
        for nb in state.neighbors(base):
            nb = int(nb)
            if owner[nb] >= 0 or not _in_bounds(state, nb, margin):
                continue
            if separated and any(owner[o] >= 0 and owner[o] != idx
                                 for o in state.neighbors(nb)):
                continue                     # would touch a different blob
            options.append(nb)
        if not options:
            frontier.pop(pick)               # exhausted; stop revisiting it
            continue
        tile = options[int(state.rng.integers(len(options)))]
        owner[tile] = idx
        blob.add(tile)
        frontier.append(tile)
        return tile
    return -1


def _spread_seeds(state: GameState, n: int, margin: int) -> list[int]:
    """`n` tiles spread over the board: each new seed is the one furthest from
    those already chosen, ties broken uniformly."""
    tiles = [t for t in range(state.n_tiles) if _in_bounds(state, t, margin)]
    if not tiles:
        tiles = list(range(state.n_tiles))
    out = [int(state.rng.choice(tiles))]
    while len(out) < n:
        best_d, best = -1, []
        for tile in tiles:
            if tile in out:
                continue
            d = min(state.chebyshev(tile, c) for c in out)
            if d > best_d:
                best_d, best = d, [tile]
            elif d == best_d:
                best.append(tile)
        if not best:
            break
        out.append(int(state.rng.choice(best)))
    return out


def _landmass(state: GameState, map_type: str, wetness: float,
              n_players: int) -> None:
    """Lay down land and water for one map type. Everything else builds on this.

    The wetness band decides *how much* water; the map type decides its shape.
    """
    n = state.n_tiles
    water_budget = int(round(wetness * n))

    if map_type in LAKE_TYPES:
        # Land everywhere, with lakes punched into it. "Lakes: a border of land
        # bridges will always generate on the edges of the map", so the lakes
        # are kept off the border ring entirely.
        state.terrain[:] = C.T_FIELD
        if water_budget <= 0:
            return
        n_lakes = max(1, water_budget // LAKE_SIZE)
        for blob in _grow_blobs(state, n_lakes, water_budget, margin=1):
            for tile in blob:
                state.terrain[tile] = C.T_WATER
        return

    # Ocean everywhere, with land grown into it.
    state.terrain[:] = C.T_OCEAN
    land_budget = max(n_players, n - water_budget)

    if map_type == "pangea":
        # "A single starting point in the center of the map is chosen to seed
        # the landmass."
        centre = (state.size // 2) * state.size + (state.size // 2)
        blobs = _grow_blobs(state, 1, land_budget, seeds=[centre], margin=1)
    elif map_type == "continents":
        # "The number of land masses is based on the number of players, wetness
        # level, and map size", and each is 30-200 tiles.
        n_masses = max(1, min(n_players, max(1, land_budget // CONTINENT_MIN)))
        while land_budget // max(1, n_masses) > CONTINENT_MAX:
            n_masses += 1
        blobs = _grow_blobs(state, n_masses, land_budget, margin=1,
                            cap=CONTINENT_MAX)
    elif map_type == "waterworld":
        blobs = _grow_blobs(state, max(n_players, land_budget // WATERWORLD_ISLAND),
                            land_budget, margin=1, cap=WATERWORLD_ISLAND)
    else:                                            # archipelago
        blobs = _grow_blobs(state, max(n_players, land_budget // ARCHIPELAGO_ISLAND),
                            land_budget, margin=1, cap=ARCHIPELAGO_ISLAND)

    for blob in blobs:
        for tile in blob:
            state.terrain[tile] = C.T_FIELD


def _classify_water(state: GameState) -> None:
    """Shallow where it touches land by an edge, deep ocean everywhere else.

    "Ocean ... does not share any edges with land tiles" — corners do not count,
    so a tile diagonally off a headland is still ocean.
    """
    is_land = state.rules.terrain_is_land[state.terrain]
    for tile in range(state.n_tiles):
        if is_land[tile]:
            continue
        touching = any(is_land[nb] for nb in _orthogonal(state, tile))
        state.terrain[tile] = C.T_WATER if touching else C.T_OCEAN


def _land_components(state: GameState) -> list[list[int]]:
    """The 8-connected landmasses, largest first."""
    is_land = state.rules.terrain_is_land[state.terrain]
    seen = np.zeros(state.n_tiles, dtype=bool)
    out = []
    for start in range(state.n_tiles):
        if seen[start] or not is_land[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            tile = stack.pop()
            comp.append(tile)
            for nb in state.neighbors(tile):
                nb = int(nb)
                if not seen[nb] and is_land[nb]:
                    seen[nb] = True
                    stack.append(nb)
        out.append(comp)
    out.sort(key=len, reverse=True)
    return out


# --- capitals -------------------------------------------------------------------
def _quadrant_capitals(state: GameState, n: int,
                       map_type: str = "drylands") -> list[int]:
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
        # Stand on land if the quadrant has any. Dropping a capital into open
        # ocean and forcing the tile to field gives that player a one-tile
        # island to start on, which is a different game from the one everyone
        # else is playing. Water World is the documented exception -- there
        # "land is manually forced for cities" because there is no alternative.
        dry = [t for t in tiles if state.rules.terrain_is_land[state.terrain[t]]]
        if not dry and map_type != "waterworld":
            # An all-ocean quadrant: reach outside it for the nearest land
            # rather than marooning a player on a tile of their own making.
            centre = tiles[len(tiles) // 2]
            elsewhere = [t for t in range(state.n_tiles)
                         if state.rules.terrain_is_land[state.terrain[t]]
                         and _in_bounds(state, t, margin)]
            if elsewhere:
                best = min(state.chebyshev(t, centre) for t in elsewhere)
                dry = [t for t in elsewhere
                       if state.chebyshev(t, centre) == best]
        tiles = dry or tiles
        if not out:
            out.append(int(state.rng.choice(tiles)))
            continue
        best, best_d = [], -1
        for tile in tiles:
            d = min(state.chebyshev(tile, c) for c in out)
            if d > best_d:
                best_d, best = d, [tile]
            elif d == best_d:
                best.append(tile)
        out.append(int(state.rng.choice(best)))
    return out


def _ensure_capitals(state: GameState, capitals: list[int], n: int) -> list[int]:
    """Every player gets exactly one capital, on its own tile. No exceptions.

    Both placement routes can come up short on a cramped map: village promotion
    has nothing to promote when the land holds fewer villages than players, and
    the quadrant route's "no land in this quadrant, reach outside it" fallback
    can send two quadrants to the same shore.

    A missing capital is not a cosmetic problem. The player has no city and no
    unit, so it is dead on arrival and the game is terminal before the first
    move — which is exactly how this was found, as a smoke run whose rollouts
    quietly collapsed from 1024 steps to 64. Rather than let generation limp on,
    top up from the best tiles left, relaxing the margin, then the spacing, then
    the requirement that it be land at all (the capital tile is forced to field
    anyway, which is what Water World does for every one of its cities).
    """
    out: list[int] = []
    for tile in capitals:                       # de-duplicate, keep order
        if int(tile) not in out:
            out.append(int(tile))
    if len(out) >= n:
        return out[:n]

    is_land = state.rules.terrain_is_land[state.terrain]
    relaxations = ((VILLAGE_EDGE_MARGIN, VILLAGE_SPACING, True),
                   (1, VILLAGE_SPACING, True),
                   (1, 1, True),
                   (0, 1, False))
    for margin, spacing, need_land in relaxations:
        while len(out) < n:
            pool = [t for t in range(state.n_tiles)
                    if t not in out
                    and _in_bounds(state, t, margin)
                    and (need_land is False or is_land[t])
                    and (not out
                         or min(state.chebyshev(t, c) for c in out) > spacing)]
            if not pool:
                break
            if out:
                far = max(min(state.chebyshev(t, c) for c in out) for t in pool)
                pool = [t for t in pool
                        if min(state.chebyshev(t, c) for c in out) == far]
            out.append(int(state.rng.choice(pool)))
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(
            f"cannot seat {n} capitals on a {state.size}x{state.size} map")
    return out


def _promote_villages_to_capitals(state: GameState, villages: list[int],
                                  n: int) -> list[int]:
    """Turn `n` of the placed villages into capitals. Continents and Pangea.

    "Next, some of those villages that are already generated are converted to
    capitals based on two factors. First, the game wants the capitals to be as
    far away from each other as possible. Second, the algorithm prefers to have
    capitals next to water tiles. That means you almost always get capital
    spawns on the coast." On Continents, "capitals are placed on different land
    masses, space permitting".
    """
    if not villages:
        return []
    coastal = {v for v in villages
               if any(not state.rules.terrain_is_land[state.terrain[nb]]
                      for nb in state.neighbors(v))}
    mass_of = {}
    for i, comp in enumerate(_land_components(state)):
        for tile in comp:
            mass_of[tile] = i

    # First capital: coastal if any village is, else anything.
    pool = sorted(coastal) or sorted(villages)
    out = [int(state.rng.choice(pool))]
    used_masses = {mass_of.get(out[0], -1)}

    while len(out) < n:
        remaining = [v for v in villages if v not in out]
        if not remaining:
            break
        fresh = [v for v in remaining if mass_of.get(v, -1) not in used_masses]
        candidates = fresh or remaining        # "space permitting"
        # Distance first, then the coast: the wiki lists them in that order.
        best_key, best = None, []
        for v in candidates:
            key = (min(state.chebyshev(v, c) for c in out), v in coastal)
            if best_key is None or key > best_key:
                best_key, best = key, [v]
            elif key == best_key:
                best.append(v)
        pick = int(state.rng.choice(best))
        out.append(pick)
        used_masses.add(mass_of.get(pick, -1))
    return out


# --- villages -------------------------------------------------------------------
def _place_villages(state: GameState, taken: list[int],
                    limit: int | None = None,
                    force_land: bool = False,
                    margin: int = VILLAGE_EDGE_MARGIN,
                    within: set[int] | None = None) -> list[int]:
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
                  for y in range(margin, size - margin)
                  for x in range(margin, size - margin)]
    if within is not None:
        candidates = [t for t in candidates if t in within]
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


def _place_mainland_villages(state: GameState, taken: list[int]) -> list[int]:
    """One landmass at a time, guaranteeing every mass gets at least one.

    "The algorithm places one village at a time on each continent. Villages must
    be two tiles away from each other. The game also makes sure to put at least
    one village on each continent."
    """
    out: list[int] = []
    for comp in _land_components(state):
        pool = set(comp)
        seeded = _place_villages(state, taken, limit=1, within=pool)
        if not seeded:
            # A mass with no legal interior tile still gets one if it can: the
            # wiki's guarantee is per continent, not per legal-margin tile.
            free = [t for t in comp
                    if all(state.chebyshev(t, o) > VILLAGE_SPACING
                           for o in taken)]
            if free:
                pick = int(state.rng.choice(sorted(free)))
                taken.append(pick)
                seeded = [pick]
        out += seeded
        out += _place_villages(state, taken, within=pool)
    return out


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

    Continents, Pangea and Water World, a fixed number per map size.
    """
    out = []
    candidates = [t for t in range(state.n_tiles)
                  if not state.rules.terrain_is_land[state.terrain[t]]
                  and state.improvement[t] == C.IMP_NONE]
    state.rng.shuffle(candidates)
    for tile in candidates:
        if len(out) >= count:
            break
        if not _in_bounds(state, tile, VILLAGE_EDGE_MARGIN):
            continue
        if any(state.chebyshev(tile, t) <= VILLAGE_SPACING for t in taken):
            continue
        # "without any land connections": every neighbour must be water.
        if any(state.rules.terrain_is_land[state.terrain[nb]]
               for nb in state.neighbors(tile)):
            continue
        state.terrain[tile] = C.T_FIELD          # the island itself
        taken.append(tile)
        out.append(tile)
    return out


# --- biomes and terrain ---------------------------------------------------------
def _flood_biomes(state: GameState, capitals: list[int]) -> np.ndarray:
    """Which tribe's land each tile is, by randomised flood fill from capitals.

    "During the initial map generation, the land is distributed roughly equally
    between all tribes in the game." The community generator does this by
    growing every tribe's region one random adjacent tile at a time, taking
    turns — which is self-balancing, and gives ragged natural borders instead of
    the straight bisectors a nearest-capital partition produces. Land is
    preferred over water, so a tribe's biome follows its own continent rather
    than jumping the sea.
    """
    owner = np.full(state.n_tiles, -1, dtype=np.int8)
    if not capitals:
        return np.zeros(state.n_tiles, dtype=np.int8)
    is_land = state.rules.terrain_is_land[state.terrain]
    frontiers: list[list[int]] = []
    for pid, cap in enumerate(capitals):
        owner[cap] = pid
        frontiers.append([cap])

    remaining = state.n_tiles - len(capitals)
    live = list(range(len(capitals)))
    while remaining > 0 and live:
        for pid in list(live):
            frontier = frontiers[pid]
            claimed = -1
            while frontier:
                pick = int(state.rng.integers(len(frontier)))
                base = frontier[pick]
                free = [int(nb) for nb in state.neighbors(base)
                        if owner[nb] < 0]
                if not free:
                    frontier.pop(pick)
                    continue
                land = [t for t in free if is_land[t]]
                pool = land or free
                claimed = pool[int(state.rng.integers(len(pool)))]
                owner[claimed] = pid
                frontier.append(claimed)
                break
            if claimed < 0:
                live.remove(pid)
            else:
                remaining -= 1

    # Anything unreachable (an island nobody's fill could touch) goes to the
    # nearest capital, so every tile has a biome.
    for tile in np.flatnonzero(owner < 0):
        owner[tile] = min(range(len(capitals)),
                          key=lambda p: state.chebyshev(int(tile), capitals[p]))
    return owner


def _fill_terrain(state: GameState, rules: Ruleset, biome: np.ndarray,
                  tribes: list[str], keep: set[int]) -> None:
    """Field / forest / mountain over the land, at each biome's own rates.

    "Since update 2.0.58 (Balance Pass 2), terrain will always be generated in
    accordance with the percentage rates — for example, Luxidoor will always
    have 48% of their biome (or as close as possible) as field tiles." So the
    counts are dealt out exactly rather than sampled per tile, which also drops
    a large chunk of map-to-map variance in how mountainous a start is.
    """
    is_land = rules.terrain_is_land[state.terrain]
    for pid, name in enumerate(tribes):
        tiles = [t for t in range(state.n_tiles)
                 if is_land[t] and biome[t] == pid and t not in keep]
        if not tiles:
            continue
        state.rng.shuffle(tiles)
        rates = terrain_rates(rules.tribes[name])
        n = len(tiles)
        n_mountain = int(round(rates[C.T_MOUNTAIN] * n))
        n_forest = int(round(rates[C.T_FOREST] * n))
        n_forest = min(n_forest, n - n_mountain)
        for i, tile in enumerate(tiles):
            if i < n_mountain:
                state.terrain[tile] = C.T_MOUNTAIN
            elif i < n_mountain + n_forest:
                state.terrain[tile] = C.T_FOREST
            else:
                state.terrain[tile] = C.T_FIELD


# --- features -------------------------------------------------------------------
def _place_ruins(state: GameState, cities: list[int]) -> list[int]:
    """Ruins, "placed randomly after villages, resources, and lighthouses".

    "Ruins can spawn on all types of terrain except shallow water. However,
    ruins cannot spawn next to a capital city or another ruin." At most a third
    of them go on deep ocean.
    """
    count = C.RUINS_BY_SIZE.get(state.size)
    if count is None:                              # a size the table omits
        count = max(1, round(4 * state.n_tiles / 121))
    water_allowance = int(round(count * WATER_RUIN_FRACTION))
    capitals = [t for t in range(state.n_tiles) if state.city_capital[t]]
    candidates = [t for t in range(state.n_tiles)
                  if state.terrain[t] != C.T_WATER
                  and state.improvement[t] == C.IMP_NONE
                  and state.resource[t] == C.R_NONE
                  and state.city_level[t] == 0
                  and state.unit_type[t] == C.U_NONE]
    state.rng.shuffle(candidates)

    placed: list[int] = []
    on_water = 0
    for tile in candidates:
        if len(placed) >= count:
            break
        if state.terrain[tile] == C.T_OCEAN and on_water >= water_allowance:
            continue
        if any(state.chebyshev(tile, c) <= 1 for c in capitals):
            continue                                # "not next to a capital"
        if any(state.chebyshev(tile, r) <= 1 for r in placed):
            continue                                # "nor another ruin"
        state.improvement[tile] = C.IMP_RUIN
        placed.append(tile)
        if state.terrain[tile] == C.T_OCEAN:
            on_water += 1
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


def _guarantee_capital_resources(state: GameState, rules: Ruleset,
                                 capital: int, tribe_name: str) -> None:
    """Some tribes are promised a resource or two beside their capital.

    "Drylands: entirely land (Kickoo and Aquarion capitals will include two
    water tiles with a fish in each.)" The community generator forces the same
    for Imperius (2 fruit), Bardur (2 game) and Zebasi (1 crop) — without which
    a Bardur start on an unlucky map has nothing to hunt, which is its whole
    early economy.
    """
    want = GUARANTEED_CAPITAL_RESOURCES.get(tribe_name)
    if want is None:
        return
    resource, terrain, quantity = want
    ring = [int(t) for t in state.neighbors(capital)]
    have = sum(1 for t in ring if state.resource[t] == resource)
    usable = [t for t in ring
              if state.improvement[t] == C.IMP_NONE and state.city_level[t] == 0
              and state.resource[t] != resource]
    # Empty tiles first, but the promise outranks whatever else the map rolled:
    # a ring that happens to be full of fruit must still yield the two game a
    # Bardur start is owed, or the guarantee is not one.
    empty = [t for t in usable if state.resource[t] == C.R_NONE]
    occupied = [t for t in usable if state.resource[t] != C.R_NONE]
    state.rng.shuffle(empty)
    state.rng.shuffle(occupied)
    for tile in empty + occupied:
        if have >= quantity:
            break
        state.terrain[tile] = terrain
        state.resource[tile] = resource
        have += 1


# --- the two pipelines ----------------------------------------------------------
def _settle_capital(state: GameState, rules: Ruleset, pid: int,
                    tribe_name: str, cap: int) -> None:
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

    lo, hi = MAP_TYPES[map_type]
    wetness = float(state.rng.uniform(lo, hi))
    # NB: no tribe water multiplier. Kickoo's "2.0x water" and Aquarion's
    # "1.5x water" are documented as bugged -- "this effect is bugged and does
    # not currently apply to map generation" -- and Oumaji's 0.5x is "found in
    # the source code but ... not translated into the world generation". The
    # data keeps them; generation deliberately does not read them.
    _landmass(state, map_type, wetness, n_players)

    taken: list[int] = []
    villages: list[int] = []

    if map_type in CONVERTED_CAPITAL_TYPES:
        # Land first, then villages, then promote some of them to capitals.
        villages = _place_mainland_villages(state, taken)
        capitals = _promote_villages_to_capitals(state, villages, n_players)
        capitals = _ensure_capitals(state, capitals, n_players)
        villages = [v for v in villages if v not in capitals]
        taken += [c for c in capitals if c not in taken]
    else:
        capitals = _ensure_capitals(
            state, _quadrant_capitals(state, n_players, map_type), n_players)
        taken += capitals
        if map_type in SUBURB_TYPES:
            villages += _place_suburbs(state, capitals, taken)
        force_land = map_type == "waterworld"
        coefficient = PRE_TERRAIN_COEFFICIENT.get(map_type, 0.0)
        if coefficient:
            wanted = _pre_terrain_village_count(state, len(taken), coefficient)
            villages += _place_villages(state, taken, limit=wanted,
                                        force_land=force_land,
                                        margin=PRE_TERRAIN_EDGE_MARGIN)
        # post-terrain: "throws in villages ... until there is no tile left"
        villages += _place_villages(state, taken, force_land=force_land)

    # Terrain detail, per tribe, over whatever land exists.
    biome = _flood_biomes(state, capitals)
    state.tile_tribe[:] = [rules.tribes[tribes[int(p)]]["id"] for p in biome]
    _fill_terrain(state, rules, biome, tribes, keep=set(capitals) | set(villages))

    # Tiny island villages come after the mainland ones, and only on the three
    # map types the feature table lists.
    if map_type in ISLAND_VILLAGE_TYPES:
        villages += _place_island_villages(
            state, taken, C.ISLAND_VILLAGES_BY_SIZE.get(state.size, 0))

    for tile in villages:
        # Cities stand on land. On Water World that is the whole point of the
        # rule -- "almost all tiles become water or ocean; land is manually
        # forced for cities" -- and elsewhere it just flattens the forest or
        # mountain the village was dropped on.
        state.terrain[tile] = C.T_FIELD
        state.city_owner[tile] = C.NEUTRAL          # neutral, capturable
        state.city_level[tile] = 1

    for pid, (tribe_name, cap) in enumerate(zip(tribes, capitals)):
        _settle_capital(state, rules, pid, tribe_name, cap)

    _classify_water(state)
    _sprinkle_resources(state, rules, biome, tribes, capitals + villages)
    for tribe_name, cap in zip(tribes, capitals):
        _guarantee_capital_resources(state, rules, cap, tribe_name)
    _classify_water(state)          # forced fish tiles can open new coastline
    _place_lighthouses(state)
    # "Ruins are placed randomly after villages, resources, and lighthouses."
    _place_ruins(state, capitals + villages)
    _place_starfish(state)

    from polyengine.core.fog import recompute_all
    recompute_all(state)
    return state
