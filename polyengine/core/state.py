"""GameState — flat-array, cheaply-copyable game state + serialization.

This is the single source of truth consumed by the trainer, baselines, search
booster, viewer, and (Phase 4) the Steam reconciler. Its serialization and
legality contract are frozen earliest and reviewed hardest.

Layout: a square S x S grid, tile index = y * S + x. Polytopia allows at most
one unit per tile, so unit data is stored per-tile (no slot lists). Cities are
likewise pinned to their tile. Everything is a NumPy array or a scalar, so
copy() is a handful of memcpys and the whole state hashes deterministically.
"""
from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset

# 8-adjacency (Polytopia uses a square grid with diagonal movement).
_NEIGHBOR_OFFSETS = ((-1, -1), (0, -1), (1, -1),
                     (-1, 0), (1, 0),
                     (-1, 1), (0, 1), (1, 1))

# Arrays that define ground truth (hashed for determinism / differential replay).
_TILE_ARRAYS = (
    "terrain", "resource", "owner",
    "city_owner", "city_level", "city_pop", "city_walls", "city_capital",
    "unit_type", "unit_owner", "unit_hp", "unit_moved", "unit_attacked",
    "unit_veteran", "unit_kills",
)
_PLAYER_ARRAYS = ("stars", "tech", "tribe", "alive")
_FOG_ARRAYS = (
    "fog_explored", "fog_visible", "fog_staleness",
    "mem_unit_type", "mem_unit_owner", "mem_city_owner", "mem_city_level",
)


class GameState:
    __slots__ = (
        "rules", "size", "n_tiles", "n_players", "turn", "current_player",
        "map_seed", "rng",
        *_TILE_ARRAYS, *_PLAYER_ARRAYS, *_FOG_ARRAYS,
    )

    def __init__(self, rules: Ruleset, size: int, n_players: int,
                 map_seed: int, _alloc: bool = True):
        self.rules = rules
        self.size = int(size)
        self.n_tiles = int(size) * int(size)
        self.n_players = int(n_players)
        self.turn = 0
        self.current_player = 0
        self.map_seed = int(map_seed)
        self.rng = np.random.Generator(np.random.PCG64(map_seed))
        if _alloc:
            self._allocate()

    def _allocate(self) -> None:
        n, p = self.n_tiles, self.n_players
        self.terrain = np.full(n, C.T_FIELD, dtype=C.DT_TERRAIN)
        self.resource = np.zeros(n, dtype=C.DT_RESOURCE)
        self.owner = np.full(n, C.NEUTRAL, dtype=C.DT_OWNER)
        self.city_owner = np.full(n, C.NEUTRAL, dtype=C.DT_OWNER)
        self.city_level = np.zeros(n, dtype=np.int8)
        self.city_pop = np.zeros(n, dtype=np.int8)
        self.city_walls = np.zeros(n, dtype=C.DT_FLAG)
        self.city_capital = np.zeros(n, dtype=C.DT_FLAG)
        self.unit_type = np.full(n, C.U_NONE, dtype=C.DT_UNIT_TYPE)
        self.unit_owner = np.full(n, C.NEUTRAL, dtype=C.DT_OWNER)
        self.unit_hp = np.zeros(n, dtype=C.DT_HP)
        self.unit_moved = np.zeros(n, dtype=C.DT_FLAG)
        self.unit_attacked = np.zeros(n, dtype=C.DT_FLAG)
        self.unit_veteran = np.zeros(n, dtype=C.DT_FLAG)
        self.unit_kills = np.zeros(n, dtype=np.int8)

        self.stars = np.zeros(p, dtype=C.DT_STARS)
        self.tech = np.zeros(p, dtype=C.DT_TECH)
        self.tribe = np.zeros(p, dtype=np.int8)
        self.alive = np.ones(p, dtype=C.DT_FLAG)

        self.fog_explored = np.zeros((p, n), dtype=C.DT_FLAG)
        self.fog_visible = np.zeros((p, n), dtype=C.DT_FLAG)
        self.fog_staleness = np.zeros((p, n), dtype=C.DT_STALE)
        self.mem_unit_type = np.full((p, n), C.U_NONE, dtype=C.DT_UNIT_TYPE)
        self.mem_unit_owner = np.full((p, n), C.NEUTRAL, dtype=C.DT_OWNER)
        self.mem_city_owner = np.full((p, n), C.NEUTRAL, dtype=C.DT_OWNER)
        self.mem_city_level = np.zeros((p, n), dtype=np.int8)

    # --- geometry -----------------------------------------------------------
    def idx(self, x: int, y: int) -> int:
        return y * self.size + x

    def xy(self, i: int) -> tuple[int, int]:
        return i % self.size, i // self.size

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, i: int) -> Iterator[int]:
        x, y = self.xy(i)
        s = self.size
        for dx, dy in _NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < s and 0 <= ny < s:
                yield ny * s + nx

    def chebyshev(self, a: int, b: int) -> int:
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return max(abs(ax - bx), abs(ay - by))

    # --- tech ---------------------------------------------------------------
    def has_tech(self, player: int, tech_id: int) -> bool:
        if tech_id < 0:
            return True  # "no tech required"
        return bool(self.tech[player] & (np.uint32(1) << np.uint32(tech_id)))

    def grant_tech(self, player: int, tech_id: int) -> None:
        self.tech[player] |= (np.uint32(1) << np.uint32(tech_id))

    # --- copy / hash / serialization ---------------------------------------
    def copy(self) -> "GameState":
        s = GameState(self.rules, self.size, self.n_players, self.map_seed,
                      _alloc=False)
        s.turn = self.turn
        s.current_player = self.current_player
        for name in (*_TILE_ARRAYS, *_PLAYER_ARRAYS, *_FOG_ARRAYS):
            setattr(s, name, getattr(self, name).copy())
        # Clone the RNG bitstream position so continuations stay reproducible.
        s.rng = np.random.Generator(np.random.PCG64())
        s.rng.bit_generator.state = self.rng.bit_generator.state
        return s

    def state_hash(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(np.array([self.turn, self.current_player, self.size,
                           self.n_players], dtype=np.int64).tobytes())
        for name in (*_TILE_ARRAYS, *_PLAYER_ARRAYS):
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict:
        d = {
            "ruleset_version": self.rules.version,
            "size": self.size,
            "n_players": self.n_players,
            "turn": self.turn,
            "current_player": self.current_player,
            "map_seed": self.map_seed,
            "rng_state": _jsonify(self.rng.bit_generator.state),
        }
        for name in (*_TILE_ARRAYS, *_PLAYER_ARRAYS, *_FOG_ARRAYS):
            d[name] = getattr(self, name).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict, rules: Ruleset) -> "GameState":
        if d["ruleset_version"] != rules.version:
            raise ValueError("ruleset version mismatch on deserialize")
        s = cls(rules, d["size"], d["n_players"], d["map_seed"], _alloc=False)
        s.turn = d["turn"]
        s.current_player = d["current_player"]
        for name in (*_TILE_ARRAYS, *_PLAYER_ARRAYS, *_FOG_ARRAYS):
            proto = _proto_dtype(name)
            arr = np.asarray(d[name], dtype=proto)
            setattr(s, name, arr)
        s.rng.bit_generator.state = _dejsonify(d["rng_state"])
        return s

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GameState) and self.state_hash() == other.state_hash()

    def __repr__(self) -> str:
        return (f"GameState(size={self.size}, players={self.n_players}, "
                f"turn={self.turn}, current={self.current_player})")


def _proto_dtype(name: str):
    # Reconstruct the dtype used at allocation for a given array name.
    zero_int8 = {"terrain", "resource", "owner", "city_owner", "unit_type",
                 "unit_owner", "mem_unit_type", "mem_unit_owner", "mem_city_owner"}
    if name in zero_int8:
        return np.int8
    if name in {"city_level", "city_pop", "unit_kills", "tribe", "mem_city_level"}:
        return np.int8
    if name in {"unit_hp"}:
        return np.int16
    if name in {"city_walls", "city_capital", "unit_moved", "unit_attacked",
                "unit_veteran", "alive", "fog_explored", "fog_visible"}:
        return np.uint8
    if name in {"fog_staleness"}:
        return np.int16
    if name == "stars":
        return np.int16
    if name == "tech":
        return np.uint32
    return np.int64


def _jsonify(state: dict) -> dict:
    # PCG64 bit-generator state contains numpy ints; make it JSON-safe.
    inner = state["state"]
    return {
        "bit_generator": state["bit_generator"],
        "state": {"state": int(inner["state"]), "inc": int(inner["inc"])},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }


def _dejsonify(d: dict) -> dict:
    return {
        "bit_generator": d["bit_generator"],
        "state": {"state": int(d["state"]["state"]), "inc": int(d["state"]["inc"])},
        "has_uint32": int(d["has_uint32"]),
        "uinteger": int(d["uinteger"]),
    }
