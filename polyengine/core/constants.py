"""Stable integer enums and dtypes shared across the engine.

These ids MUST match the `id` fields in polyengine/data/*.toml. The loader in
rules.py asserts agreement so the two never silently drift.
"""
from __future__ import annotations

import numpy as np

# --- Terrain ids (see data/terrain.toml) ---
T_OCEAN = 0
T_WATER = 1
T_FIELD = 2
T_FOREST = 3
T_MOUNTAIN = 4
LAND_TERRAIN = frozenset({T_FIELD, T_FOREST, T_MOUNTAIN})

# --- Resource ids (see data/terrain.toml) ---
R_NONE = 0
R_FRUIT = 1
R_CROP = 2
R_GAME = 3
R_METAL = 4
R_FISH = 5
R_STARFISH = 6

# --- Unit ids (see data/units.toml) ---
U_NONE = -1
U_WARRIOR = 0
U_ARCHER = 1
U_RIDER = 2
U_DEFENDER = 3
U_SWORDSMAN = 4
U_CATAPULT = 5
U_KNIGHT = 6
U_MIND_BENDER = 7
U_GIANT = 8
U_CLOAK = 9
U_DAGGER = 10
U_RAFT = 11
U_SCOUT = 12
U_RAMMER = 13
U_BOMBER = 14
U_JUGGERNAUT = 15
U_PIRATE = 16

# Resources that are INVISIBLE until a technology reveals them, and the techs
# that do it. "Crop is hidden until Organization or Farming has been researched"
# (Terrain page); "Climbing ... reveals metal, a resource used by the Mine".
# This is an observation rule, not a legality one -- the buildings that consume
# them sit behind the same techs anyway -- but it is load-bearing for invariant
# 1: a player planning expansion cannot see crop they have not unlocked, so
# neither may the network.
HIDDEN_RESOURCE_TECHS = {
    R_CROP: ("organization", "farming"),
    R_METAL: ("climbing", "mining"),
}

# --- Ownership sentinels ---
NEUTRAL = -1  # unowned territory / neutral (village) city

# "Tiny maps support a maximum of nine players, while other maps support up to
# 16" (Map Generation). One definition, because this number is load-bearing in
# three unrelated places -- how many seats mapgen will deal, how wide the
# action head's sub-type axis is, and how many per-opponent slots the
# observation carries -- and three copies is how they drift apart.
MAX_PLAYERS = 16
MAX_PLAYERS_TINY = 9

# --- Improvement ids (see data/improvements.toml) ---
IMP_NONE = -1
IMP_HARVEST_FRUIT = 0
IMP_HUNT_GAME = 1
IMP_FISHING = 2
IMP_LUMBER_HUT = 3
IMP_FARM = 4
IMP_MINE = 5
IMP_PORT = 6
IMP_TEMPLE = 7
IMP_SAWMILL = 8
IMP_WINDMILL = 9
IMP_FORGE = 10
IMP_ROAD = 11
IMP_BRIDGE = 12
IMP_MARKET = 13
IMP_EMBASSY = 14
IMP_ALTAR_OF_PEACE = 15
IMP_EMPERORS_TOMB = 16
IMP_EYE_OF_GOD = 17
IMP_GATE_OF_POWER = 18
IMP_GRAND_BAZAAR = 19
IMP_PARK_OF_FORTUNE = 20
IMP_TOWER_OF_WISDOM = 21
IMP_LIGHTHOUSE = 22
IMP_RUIN = 23
IMP_MOUNTAIN_TEMPLE = 24
IMP_FOREST_TEMPLE = 25
IMP_WATER_TEMPLE = 26

# All four temples score alike: 100 points at construction, +50 per level, and
# "all Temples level up every two turns" to a maximum of level 5.
TEMPLES = frozenset({IMP_TEMPLE, IMP_MOUNTAIN_TEMPLE, IMP_FOREST_TEMPLE,
                     IMP_WATER_TEMPLE})
TEMPLE_BASE_SCORE = 100
TEMPLE_LEVEL_SCORE = 50
# The prose says "all Temples level up every two turns", but the page's own
# table gives three-turn bands -- level 1 at 0-2 turns since construction, 2 at
# 3-5, 3 at 6-8, 4 at 9-11, 5 at 12+. The reconciliation is in the same
# paragraph: levels count "how many turns have passed since their construction,
# **including the turn of their construction**". The table is what is implemented.
TEMPLE_TURNS_PER_LEVEL = 3
TEMPLE_MAX_LEVEL = 5

# --- Task ids (see data/improvements.toml `requires_task`) ---
# "A monument is a building awarded for completing a task." Each task is a
# standing condition on the player's whole empire, checked once a turn.
TASK_PACIFIST = 0        # make no attacks for five turns
TASK_WEALTH = 1          # gather 100 stars
TASK_EXPLORER = 2        # find all lighthouses
TASK_KILLER = 3          # kill 10 enemy units
TASK_NETWORK = 4         # connect five cities to the capital
TASK_METROPOLIS = 5      # have a level 5+ city
TASK_GENIUS = 6          # research all technologies
N_TASKS = 7

TASK_NAMES = {
    TASK_PACIFIST: "pacifist", TASK_WEALTH: "wealth", TASK_EXPLORER: "explorer",
    TASK_KILLER: "killer", TASK_NETWORK: "network",
    TASK_METROPOLIS: "metropolis", TASK_GENIUS: "genius",
}
TASK_ID = {v: k for k, v in TASK_NAMES.items()}

PACIFIST_TURNS = 5
KILLER_KILLS = 10
WEALTH_STARS = 100
NETWORK_CITIES = 5
METROPOLIS_LEVEL = 5
MONUMENT_SCORE = 400

# "The number of ruins is based on the size of the map."
RUINS_BY_SIZE = {11: 4, 14: 5, 16: 7, 18: 9, 20: 11, 30: 23}
# "Continents and Pangea have a fixed amount of villages that spawn in the water
# without any land connections."
ISLAND_VILLAGES_BY_SIZE = {11: 0, 14: 1, 16: 2, 18: 3, 20: 4, 30: 9}
STARFISH_PER_WATER_TILES = 25      # "one starfish for every 25 water tiles"
STARFISH_STARS = 8                 # "gives 8 stars"
RUIN_STARS = 10                    # "Resources | Grants 10 Stars"
RUIN_POPULATION = 3                # "Adds 3 free Population to the capital"

# The eight compass directions, clockwise from north. `unit_facing` stores an
# index into this, and a forced unit spawn walks outwards from that index — see
# `push_order` in actions.py.
PUSH_COMPASS = ((0, -1), (1, -1), (1, 0), (1, 1),
                (0, 1), (-1, 1), (-1, 0), (-1, -1))
# "If the city where the unit spawns is on the exact center of the map, the unit
# will be pushed south."
PUSH_SOUTH = 4

# Tiles that carry a city connection. Ports count (the wiki's Port description
# says so explicitly), and city tiles conduct implicitly — see economy.py.
CONNECTORS = frozenset({IMP_ROAD, IMP_BRIDGE, IMP_PORT})

# --- City level-up reward ids (see data/improvements.toml [reward]) ---
RW_WORKSHOP = 0
RW_EXPLORER = 1
RW_CITY_WALL = 2
RW_RESOURCES = 3
RW_POPULATION_GROWTH = 4
RW_BORDER_GROWTH = 5
RW_PARK = 6
RW_SUPER_UNIT = 7

# --- Action types (Phase-2 auto-regressive head 1) ---
A_END_TURN = 0
A_MOVE = 1
A_ATTACK = 2
A_CAPTURE = 3
A_TRAIN = 4
A_RESEARCH = 5
A_BUILD = 6            # harvest a resource or place a building
A_CHOOSE_REWARD = 7    # pick one of two rewards after a city levels up
A_DISBAND = 8          # remove a unit, refund half its cost (needs Free Spirit)
A_UPGRADE = 9          # pay to turn a naval hull into a better one
# Ruins and Starfish share one mechanic: "requires a unit to start the turn on
# the ruin and consumes the unit's actions for that turn (like capturing a
# village)", and Starfish harvesting is "much like if it were examining a ruin".
A_EXAMINE = 10
# "A unit can be promoted to a veteran after it kills three enemy units."
# Promotion is deliberately *not* automatic: it heals the unit completely, so
# holding it back until the unit is hurt is a standard play. See _apply_promote.
A_PROMOTE = 11
# "Heal Others ... heals all adjacent friendly units by up to 4 HP." Mind Bender.
A_HEAL_OTHERS = 12
# Tile-modifying abilities, each unlocked by a technology. One action type with
# four sub-types rather than four types: they share a source (a tile you own)
# and differ only in what they do to it.
A_TILE_ABILITY = 13
# Diplomacy. The sub-type is a PLAYER id, not a unit/tech/improvement -- the only
# two action types that index the subtype head that way.
# "The Strategy technology is required to send, but not respond to, a peace
# treaty", so offering to someone who has already offered you concludes the
# treaty and needs no tech: one action type covers propose and accept.
A_PEACE_OFFER = 14
A_PEACE_BREAK = 15
N_ACTION_TYPES = 16

ACTION_NAMES = {
    A_END_TURN: "END_TURN",
    A_MOVE: "MOVE",
    A_ATTACK: "ATTACK",
    A_CAPTURE: "CAPTURE",
    A_TRAIN: "TRAIN",
    A_RESEARCH: "RESEARCH",
    A_BUILD: "BUILD",
    A_CHOOSE_REWARD: "REWARD",
    A_DISBAND: "DISBAND",
    A_UPGRADE: "UPGRADE",
    A_EXAMINE: "EXAMINE",
    A_PROMOTE: "PROMOTE",
    A_HEAL_OTHERS: "HEAL_OTHERS",
    A_TILE_ABILITY: "TILE_ABILITY",
    A_PEACE_OFFER: "PEACE_OFFER",
    A_PEACE_BREAK: "PEACE_BREAK",
}

# --- tile abilities (sub-types of A_TILE_ABILITY) ---
# "Clear Forest ... removes a forest and gives the player one star" (Forestry)
# "Grow Forest ... turns a field tile into a forest tile at a cost of 5 stars"
#   (Spiritualism)
# "Burn Forest ... replaces an empty Forest with a field with crop at a cost of
#   5 stars" (Construction)
# "Destroy ... removes a building and its benefits at no cost" (Chivalry)
TA_CLEAR_FOREST = 0
TA_GROW_FOREST = 1
TA_BURN_FOREST = 2
TA_DESTROY = 3
N_TILE_ABILITIES = 4

TILE_ABILITY_NAMES = {
    TA_CLEAR_FOREST: "clear_forest",
    TA_GROW_FOREST: "grow_forest",
    TA_BURN_FOREST: "burn_forest",
    TA_DESTROY: "destroy",
}
TILE_ABILITY_TECH = {
    TA_CLEAR_FOREST: "forestry",
    TA_GROW_FOREST: "spiritualism",
    TA_BURN_FOREST: "construction",
    TA_DESTROY: "chivalry",
}
CLEAR_FOREST_STARS = 1        # gained
GROW_FOREST_COST = 5
BURN_FOREST_COST = 5

# "A unit can be promoted to a veteran after it kills three enemy units."
VETERAN_KILLS = 3
VETERAN_BONUS_HP = 5          # "increases its maximum health by five"
HEAL_OTHERS_AMOUNT = 4

# Naval hulls a Raft can be bought up to, and the tech that unlocks each.
# "A Raft can be upgraded into a Scout for five stars, once Sailing is
# researched." Moving onto land drops the upgrade permanently.
NAVAL_UPGRADES = (U_SCOUT, U_RAMMER, U_BOMBER)

# --- dtypes for the flat per-tile / per-player arrays ---
DT_TERRAIN = np.int8
DT_RESOURCE = np.int8
DT_OWNER = np.int8
DT_UNIT_TYPE = np.int8
DT_HP = np.int16
DT_FLAG = np.uint8
DT_STALE = np.int16
DT_STARS = np.int16
DT_TECH = np.uint32
DT_SCORE = np.int32

# City level-up mechanics (simplified economy for M0 — see game.py).
CITY_MAX_LEVEL = 10
POP_PER_LEVEL = 2  # population required to advance one level (level N needs N+1 pop)

# Fog: how far units and cities see (Chebyshev radius).
SIGHT_RADIUS_UNIT = 1
# "Scout: allows a unit to explore a 5x5 area instead of a 3x3 area."
SIGHT_RADIUS_SCOUT = 2
# "Mountain: all units on this terrain have a sight radius of two tiles."
SIGHT_RADIUS_MOUNTAIN = 2
SIGHT_RADIUS_CITY = 1

# Domination turn cap; ties broken by score().
TURN_CAP = 100
# ...at the reference game the cap was tuned on: an 11x11 duel. Real Domination
# has *no* time limit -- "play until there is only one tribe left" -- so the cap
# is a training necessity, not a rule, and the shape of it is ours to choose.
#
# It scales with the board's *linear* size rather than its area, because what a
# bigger map really costs is travel: crossing a 30x30 takes about three times as
# long as crossing an 11x11, not nine. Each extra tribe then adds a share on top,
# since Domination needs every one of them removed.
TURN_CAP_REFERENCE_SIZE = 11
TURN_CAP_PER_EXTRA_PLAYER = 0.22
TURN_CAP_MIN = 30
TURN_CAP_MAX = 600


def turn_cap_for(size: int, n_players: int) -> int:
    """The turn cap a game of this shape gets. See TURN_CAP above."""
    scale = float(size) / TURN_CAP_REFERENCE_SIZE
    crowd = 1.0 + TURN_CAP_PER_EXTRA_PLAYER * max(0, int(n_players) - 2)
    return int(min(TURN_CAP_MAX,
                   max(TURN_CAP_MIN, round(TURN_CAP * scale * crowd))))
