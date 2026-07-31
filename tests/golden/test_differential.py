"""Invariant 4: a recorded replay must re-simulate to the identical state.

`polyengine/testing/differential.py` is the fidelity gate — the thing that will
eventually diff a Steam-captured game against ours frame by frame. It only works
if `resimulate` can rebuild *the same game* from the replay header.

It could not. `resimulate` called `Game.new(size, seed)` and nothing else, so it
defaulted the tribes to Imperius/Bardur and the map type to drylands. Any replay
of a continents game, or of any other tribe pairing, was rebuilt on a completely
different board and failed on action 0 — the gate had been broken since map
types landed, and no test noticed because none of them recorded a non-default
game.

So: one test per map type, and a couple of non-default tribe pairings.
"""
from __future__ import annotations

import pytest

from polyengine.agents.scripted import ScriptedBot
from polyengine.core.game import Game
from polyengine.core.mapgen import MAP_TYPES
from polyengine.replay import record_game
from polyengine.testing.differential import check_determinism, resimulate


def play(seed=3, size=11, map_type="drylands", tribes=None, steps=900):
    game = Game.new(size=size, seed=seed, map_type=map_type, tribes=tribes)
    return record_game(game, [ScriptedBot(seed=1), ScriptedBot(seed=2)],
                       max_steps=steps)


@pytest.mark.parametrize("map_type", sorted(MAP_TYPES))
def test_a_replay_resimulates_identically(map_type):
    rec = play(map_type=map_type)
    assert len(rec.frames) > 1
    assert check_determinism(rec.to_dict()) is None


@pytest.mark.parametrize("tribes", [
    ["xin_xi", "vengir"],
    ["kickoo", "hoodrick"],
    ["luxidoor", "oumaji"],
])
def test_non_default_tribes_resimulate(tribes):
    """The header carries tribe *names*; rebuilding without them silently
    substitutes Imperius and Bardur, whose spawn rates make a different map."""
    rec = play(map_type="lakes", tribes=tribes)
    assert check_determinism(rec.to_dict()) is None


def test_the_header_carries_what_rebuilding_needs():
    """A structural check, so a missing field fails loudly rather than as a
    mysterious illegal action a hundred frames in."""
    rec = play(map_type="continents", tribes=["xin_xi", "vengir"])
    header = rec.to_dict()
    for key in ("size", "map_seed", "map_type", "tribes"):
        assert key in header, key
    assert header["map_type"] == "continents"
    assert header["tribes"] == ["xin_xi", "vengir"]


def test_resimulation_rejects_an_action_that_is_no_longer_legal():
    """The gate's whole purpose: if the legality contract moves, it says so."""
    from polyengine.core import constants as C

    rec = play(map_type="drylands", steps=40)
    blob = rec.to_dict()
    # Splice in a capture of a tile nobody can capture.
    blob["frames"][1]["action"] = {
        "type": int(C.A_CAPTURE), "name": "CAPTURE", "src": 0, "dst": 0,
        "subtype": -1, "desc": "CAPTURE(tile=0)", "player": 0,
    }
    with pytest.raises(AssertionError, match="no longer legal"):
        resimulate(blob)
