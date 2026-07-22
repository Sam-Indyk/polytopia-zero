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
U_GIANT = 7

# --- Ownership sentinels ---
NEUTRAL = -1  # unowned territory / neutral (village) city

# --- Action types (Phase-2 auto-regressive head 1) ---
A_END_TURN = 0
A_MOVE = 1
A_ATTACK = 2
A_CAPTURE = 3
A_TRAIN = 4
A_RESEARCH = 5
N_ACTION_TYPES = 6

ACTION_NAMES = {
    A_END_TURN: "END_TURN",
    A_MOVE: "MOVE",
    A_ATTACK: "ATTACK",
    A_CAPTURE: "CAPTURE",
    A_TRAIN: "TRAIN",
    A_RESEARCH: "RESEARCH",
}

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
