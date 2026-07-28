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
R_WHALE = 6

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

# --- Ownership sentinels ---
NEUTRAL = -1  # unowned territory / neutral (village) city

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
N_ACTION_TYPES = 10

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
}

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
SIGHT_RADIUS_CITY = 1

# Domination turn cap; ties broken by score().
TURN_CAP = 100
