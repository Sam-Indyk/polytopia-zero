"""Replay recording, the differential/determinism gate, and the HTML viewer."""
from __future__ import annotations

import json

import pytest

from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core.game import Game
from polyengine.replay import record_game
from polyengine.testing import (
    actions_of,
    check_determinism,
    compare_replays,
    resimulate,
)
from polyengine.viewer import render_html


def a_replay(seed=1, size=9, max_steps=300):
    game = Game.new(size=size, seed=seed)
    agents = [ScriptedBot(seed=seed), RandomAgent(seed=seed + 50)]
    return record_game(game, agents, max_steps=max_steps).to_dict()


# --- recording ----------------------------------------------------------------
def test_replay_shape_and_json_roundtrip():
    rep = a_replay()
    assert rep["format"] == "polyzero-replay/1"
    assert len(rep["terrain"]) == rep["size"] ** 2
    assert len(rep["frames"]) >= 2
    assert rep["frames"][0].get("action") is None      # initial snapshot
    assert all("action" in f for f in rep["frames"][1:])
    assert json.loads(json.dumps(rep)) == rep          # pure-JSON types only


def test_replay_records_every_action():
    game = Game.new(size=9, seed=4)
    agents = [RandomAgent(seed=p) for p in range(game.state.n_players)]
    rec = record_game(game, agents, max_steps=120)
    assert len(actions_of(rec.to_dict())) == len(rec.frames) - 1


def test_replay_fog_masks_are_per_player():
    rep = a_replay()
    explored = rep["frames"][-1]["explored"]
    assert len(explored) == rep["n_players"]
    assert all(len(row) == rep["size"] ** 2 for row in explored)
    # A player should not have explored the whole board in a short game.
    assert any(sum(row) < rep["size"] ** 2 for row in explored)


# --- the fidelity gate ---------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_engine_is_deterministic_under_resimulation(seed):
    """Replaying a recorded action sequence must reproduce it exactly."""
    assert check_determinism(a_replay(seed=seed)) is None


def test_resimulation_rejects_illegal_actions():
    rep = a_replay()
    for frame in rep["frames"]:
        if frame.get("action") and frame["action"]["type"] != 0:
            frame["action"]["src"] = 0
            frame["action"]["dst"] = 0
            break
    with pytest.raises(AssertionError, match="no longer legal"):
        resimulate(rep, strict=True)


def test_compare_replays_localises_a_divergence():
    rep = a_replay()
    other = json.loads(json.dumps(rep))
    target = len(other["frames"]) // 2
    other["frames"][target]["stars"][0] += 7
    div = compare_replays(rep, other)
    assert div is not None
    assert div.frame == target and div.field == "stars"


def test_compare_replays_detects_map_mismatch():
    div = compare_replays(a_replay(seed=1), a_replay(seed=2))
    assert div is not None and div.frame == -1


def test_identical_replays_do_not_diverge():
    rep = a_replay()
    assert compare_replays(rep, json.loads(json.dumps(rep))) is None


# --- viewer -------------------------------------------------------------------
def test_viewer_html_is_self_contained():
    html = render_html(a_replay())
    assert "__REPLAY__" not in html            # payload substituted
    assert "polyzero-replay/1" in html
    assert "</script>" not in html.split("<script>")[1].rsplit("</script>", 1)[0]
    assert "src=\"http" not in html and "href=\"http" not in html
