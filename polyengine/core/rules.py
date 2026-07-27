"""Ruleset-as-data loader.

Parses the versioned TOML files in polyengine/data/ into flat NumPy lookup
arrays (indexed by the stable ids in constants.py) plus a few plain dicts. The
engine and kernels read only these arrays, so a balance patch is a data edit —
and a `tribes-2020-compat` ruleset (future work) drops in behind the same API.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from polyengine.core import constants as C

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Ruleset:
    version: str

    # terrain (indexed by terrain id)
    terrain_names: tuple[str, ...]
    terrain_symbol: tuple[str, ...]
    terrain_is_land: np.ndarray          # bool[T]
    terrain_move_cost: np.ndarray        # int8[T]
    terrain_requires_tech: np.ndarray    # int8[T]  (tech id or -1)
    terrain_defense_bonus: np.ndarray    # float32[T]

    resource_names: tuple[str, ...]
    resource_symbol: tuple[str, ...]

    # units (indexed by unit id)
    unit_names: tuple[str, ...]
    unit_symbol: tuple[str, ...]
    unit_cost: np.ndarray                # int16[U]
    unit_max_hp: np.ndarray              # int16[U]
    unit_attack: np.ndarray              # float32[U]
    unit_defense: np.ndarray             # float32[U]
    unit_movement: np.ndarray            # int8[U]
    unit_range: np.ndarray               # int8[U]
    unit_trainable: np.ndarray           # bool[U]
    unit_requires_tech: np.ndarray       # int8[U]  (tech id or -1)
    unit_naval: np.ndarray               # bool[U]
    unit_skills: tuple[frozenset, ...]   # frozenset[str] per unit

    # techs
    tech_names: tuple[str, ...]
    tech_id: dict[str, int]
    tech_tier: np.ndarray                # int8[K]
    tech_requires: np.ndarray            # int8[K]  (tech id or -1)
    tech_cost_override: np.ndarray       # int16[K] (-1 = use tier formula)

    # tribes
    tribes: dict[str, dict]

    # combat constants
    accuracy: float
    defense_bonus_map: dict[str, float]

    @property
    def n_terrain(self) -> int:
        return len(self.terrain_names)

    @property
    def n_units(self) -> int:
        return len(self.unit_names)

    @property
    def n_techs(self) -> int:
        return len(self.tech_names)


def _load_toml(name: str) -> dict:
    with open(DATA_DIR / name, "rb") as fh:
        return tomllib.load(fh)


def _fill_by_id(entries: dict[str, dict], n: int, key: str, dtype, default=0):
    arr = np.full(n, default, dtype=dtype)
    for e in entries.values():
        arr[e["id"]] = e[key]
    return arr


@lru_cache(maxsize=4)
def load_ruleset(name: str = "current") -> Ruleset:
    """Load and cache a ruleset. `name` selects a variant (only 'current' for now)."""
    if name != "current":
        raise ValueError(f"unknown ruleset variant {name!r} (only 'current' exists)")

    terr = _load_toml("terrain.toml")
    units = _load_toml("units.toml")
    techs = _load_toml("techs.toml")
    tribes = _load_toml("tribes.toml")
    combat = _load_toml("combat.toml")

    version = terr["ruleset_version"]
    for src, label in [(units, "units"), (techs, "techs"), (tribes, "tribes"),
                       (combat, "combat")]:
        if src["ruleset_version"] != version:
            raise ValueError(f"ruleset version mismatch in {label}.toml")

    # --- techs first (units/terrain reference tech names) ---
    tech_entries = techs["tech"]
    n_techs = len(tech_entries)
    tech_id = {name: e["id"] for name, e in tech_entries.items()}
    _assert_dense_ids(tech_id.values(), n_techs, "techs")
    tech_names = tuple(sorted(tech_id, key=lambda k: tech_id[k]))
    tech_tier = _fill_by_id(tech_entries, n_techs, "tier", np.int8)
    tech_requires = np.full(n_techs, -1, dtype=np.int8)
    tech_cost_override = np.full(n_techs, -1, dtype=np.int16)
    for name, e in tech_entries.items():
        req = e["requires"]
        if req:
            tech_requires[e["id"]] = tech_id[req]
        if e.get("cost_override", -1) >= 0:
            tech_cost_override[e["id"]] = int(e["cost_override"])

    def tech_or_none(s: str) -> int:
        return tech_id[s] if s else -1

    # --- terrain ---
    terr_entries = terr["terrain"]
    n_terr = len(terr_entries)
    _assert_dense_ids((e["id"] for e in terr_entries.values()), n_terr, "terrain")
    terrain_names = tuple(sorted(terr_entries, key=lambda k: terr_entries[k]["id"]))
    terrain_symbol = _symbols(terr_entries, n_terr)
    terrain_is_land = _fill_by_id(terr_entries, n_terr, "land", bool, False)
    terrain_move_cost = _fill_by_id(terr_entries, n_terr, "move_cost", np.int8, 1)
    terrain_requires_tech = np.full(n_terr, -1, dtype=np.int8)
    dbmap = combat["defense_bonus"]
    terrain_defense_bonus = np.ones(n_terr, dtype=np.float32)
    for e in terr_entries.values():
        terrain_requires_tech[e["id"]] = tech_or_none(e["requires_tech"])
        terrain_defense_bonus[e["id"]] = dbmap[e["defense_bonus"]]

    res_entries = terr["resource"]
    n_res = len(res_entries)
    _assert_dense_ids((e["id"] for e in res_entries.values()), n_res, "resource")
    resource_names = tuple(sorted(res_entries, key=lambda k: res_entries[k]["id"]))
    resource_symbol = _symbols(res_entries, n_res)

    # --- units ---
    unit_entries = units["unit"]
    n_units = len(unit_entries)
    _assert_dense_ids((e["id"] for e in unit_entries.values()), n_units, "units")
    unit_names = tuple(sorted(unit_entries, key=lambda k: unit_entries[k]["id"]))
    unit_symbol = _symbols(unit_entries, n_units)
    unit_cost = _fill_by_id(unit_entries, n_units, "cost", np.int16)
    unit_max_hp = _fill_by_id(unit_entries, n_units, "max_hp", np.int16)
    unit_attack = _fill_by_id(unit_entries, n_units, "attack", np.float32)
    unit_defense = _fill_by_id(unit_entries, n_units, "defense", np.float32)
    unit_movement = _fill_by_id(unit_entries, n_units, "movement", np.int8)
    unit_range = _fill_by_id(unit_entries, n_units, "range", np.int8)
    unit_trainable = _fill_by_id(unit_entries, n_units, "trainable", bool, False)
    unit_naval = _fill_by_id(unit_entries, n_units, "naval", bool, False)
    unit_requires_tech = np.full(n_units, -1, dtype=np.int8)
    skills_by_id: list[frozenset] = [frozenset()] * n_units
    for e in unit_entries.values():
        unit_requires_tech[e["id"]] = tech_or_none(e["requires_tech"])
        skills_by_id[e["id"]] = frozenset(e.get("skills", []))
    unit_skills = tuple(skills_by_id)

    return Ruleset(
        version=version,
        terrain_names=terrain_names,
        terrain_symbol=terrain_symbol,
        terrain_is_land=terrain_is_land,
        terrain_move_cost=terrain_move_cost,
        terrain_requires_tech=terrain_requires_tech,
        terrain_defense_bonus=terrain_defense_bonus,
        resource_names=resource_names,
        resource_symbol=resource_symbol,
        unit_names=unit_names,
        unit_symbol=unit_symbol,
        unit_cost=unit_cost,
        unit_max_hp=unit_max_hp,
        unit_attack=unit_attack,
        unit_defense=unit_defense,
        unit_movement=unit_movement,
        unit_range=unit_range,
        unit_trainable=unit_trainable,
        unit_requires_tech=unit_requires_tech,
        unit_naval=unit_naval,
        unit_skills=unit_skills,
        tech_names=tech_names,
        tech_id=tech_id,
        tech_tier=tech_tier,
        tech_requires=tech_requires,
        tech_cost_override=tech_cost_override,
        tribes=tribes["tribe"],
        accuracy=float(combat["constants"]["accuracy"]),
        defense_bonus_map=dict(dbmap),
    )


def _symbols(entries: dict[str, dict], n: int) -> tuple[str, ...]:
    syms = [" "] * n
    for e in entries.values():
        syms[e["id"]] = e.get("symbol", "?")
    return tuple(syms)


def _assert_dense_ids(ids, n: int, label: str) -> None:
    s = sorted(ids)
    if s != list(range(n)):
        raise ValueError(f"{label}.toml ids must be dense 0..{n - 1}, got {s}")
