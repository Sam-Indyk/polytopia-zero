"""Who a city actually protects.

Every assertion here corresponds to a rule the engine got wrong until it was
reconciled against the wiki, and none of them were covered by a test before —
which is exactly why they survived. From Defence_Bonus / Combat / Cities:

* "cities without a city wall (not including uncaptured villages) provide a
  defence bonus to friendly units (invading units besieging a city do not
  receive any defence bonus)"
* "Cities will only provide a defence bonus (wall or no wall) to units with the
  fortify skill."
* "defenceBonus is 1 when there is no defence bonus, 1.5 for the standard
  forest/mountain/ocean/city bonus, 4 for the city wall bonus."
"""
from __future__ import annotations

import pytest

from polyengine.core import constants as C
from polyengine.core.combat import defense_bonus_for
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

R = load_ruleset()

FORTIFIER = "warrior"       # has the fortify skill
NON_FORTIFIER = "giant"     # explicitly does not, per the wiki


@pytest.fixture
def city():
    """A size-9 game with a known owned city tile, cleared of any unit."""
    game = Game.new(size=9, seed=3)
    s = game.state
    tile = next(t for t in range(s.n_tiles)
                if s.city_level[t] > 0 and s.city_owner[t] != C.NEUTRAL)
    s.unit_type[tile] = C.U_NONE
    s.unit_owner[tile] = C.NEUTRAL
    s.city_walls[tile] = 0
    return s, tile, int(s.city_owner[tile])


def place(s, tile, name, owner):
    s.unit_type[tile] = R.unit_id[name]
    s.unit_owner[tile] = owner
    s.unit_hp[tile] = 10


# --- the besieging bug --------------------------------------------------------
def test_besieging_enemy_gets_no_bonus_from_walls(city):
    """The one that inverted sieges: an invader used to inherit the walls."""
    s, tile, owner = city
    s.city_walls[tile] = 1
    place(s, tile, FORTIFIER, owner=1 - owner)
    assert defense_bonus_for(s, tile, R) == pytest.approx(1.0)


def test_besieging_enemy_gets_no_bonus_from_an_unwalled_city(city):
    s, tile, owner = city
    place(s, tile, FORTIFIER, owner=1 - owner)
    assert defense_bonus_for(s, tile, R) == pytest.approx(1.0)


def test_owner_is_still_fortified_by_walls(city):
    s, tile, owner = city
    s.city_walls[tile] = 1
    place(s, tile, FORTIFIER, owner=owner)
    assert defense_bonus_for(s, tile, R) == pytest.approx(4.0)


# --- unwalled cities used to give nothing ------------------------------------
def test_unwalled_city_fortifies_its_own_units(city):
    s, tile, owner = city
    place(s, tile, FORTIFIER, owner=owner)
    assert defense_bonus_for(s, tile, R) == pytest.approx(1.5)


def test_walls_are_stronger_than_a_bare_city(city):
    s, tile, owner = city
    place(s, tile, FORTIFIER, owner=owner)
    bare = defense_bonus_for(s, tile, R)
    s.city_walls[tile] = 1
    assert defense_bonus_for(s, tile, R) > bare


# --- the fortify skill was never read at all ---------------------------------
def test_a_unit_without_fortify_gets_no_city_bonus(city):
    """A Giant in a walled city is not protected by it."""
    s, tile, owner = city
    s.city_walls[tile] = 1
    place(s, tile, NON_FORTIFIER, owner=owner)
    assert defense_bonus_for(s, tile, R) == pytest.approx(1.0)


# --- villages -----------------------------------------------------------------
def test_uncaptured_village_fortifies_nobody(city):
    s, tile, _owner = city
    s.city_owner[tile] = C.NEUTRAL
    place(s, tile, FORTIFIER, owner=0)
    assert defense_bonus_for(s, tile, R) == pytest.approx(
        float(R.terrain_defense_bonus[s.terrain[tile]]))


# --- terrain still works ------------------------------------------------------
def test_terrain_bonus_applies_off_city_tiles():
    game = Game.new(size=9, seed=5)
    s = game.state
    tile = next(t for t in range(s.n_tiles)
                if s.terrain[t] == C.T_FOREST and s.city_level[t] == 0)
    place(s, tile, FORTIFIER, owner=0)
    assert defense_bonus_for(s, tile, R) == pytest.approx(
        float(R.terrain_defense_bonus[C.T_FOREST]))


def test_empty_tile_falls_back_to_terrain():
    game = Game.new(size=9, seed=5)
    s = game.state
    tile = next(t for t in range(s.n_tiles)
                if s.unit_type[t] == C.U_NONE and s.city_level[t] == 0)
    assert defense_bonus_for(s, tile, R) == pytest.approx(
        float(R.terrain_defense_bonus[s.terrain[tile]]))
