"""Replay recording, the differential/determinism gate, and the HTML viewer."""
from __future__ import annotations

import json
import re

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


def test_replay_frames_expose_full_mutable_state():
    """The viewer must be able to show everything the engine tracks."""
    rep = a_replay()
    frame = rep["frames"][-1]
    for field in ("owner", "resource", "city_owner", "city_level", "city_pop",
                  "city_walls", "city_capital", "unit_type", "unit_owner",
                  "unit_hp", "unit_moved", "unit_attacked", "unit_veteran",
                  "unit_kills"):
        assert len(frame[field]) == rep["size"] ** 2, f"{field} missing/short"
    for field in ("stars", "tech", "alive"):
        assert len(frame[field]) == rep["n_players"], f"{field} missing/short"


def test_replay_header_carries_tech_tree_for_the_viewer():
    rep = a_replay()
    n = len(rep["tech_names"])
    assert len(rep["tech_tier"]) == n and len(rep["tech_requires"]) == n
    assert len(rep["unit_cost"]) == len(rep["unit_names"])
    assert len(rep["unit_max_hp"]) == len(rep["unit_names"])
    assert rep["resource_names"][0] == "none"
    assert rep["not_simulated"], "unimplemented mechanics must be declared"


def test_researched_techs_are_visible_and_grow():
    """Tech is recorded as a per-player bitmask so the viewer can show the tree."""
    rep = a_replay()
    first = rep["frames"][0]["tech"]
    last = rep["frames"][-1]["tech"]
    popcount = lambda m: bin(m).count("1")
    assert sum(map(popcount, last)) >= sum(map(popcount, first))
    assert any(popcount(m) > 0 for m in last), "nobody researched anything"


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
    # Every template placeholder substituted. Checking one specific token would
    # not have caught the rename from __REPLAY__ to {{BOOT}}, so check the shape.
    for hole in ("{{BOOT}}", "{{MODE}}", "{{TITLE}}", "/*{{CSS}}*/", "/*{{JS}}*/"):
        assert hole not in html, f"unsubstituted {hole}"
    assert "polyzero-replay/1" in html
    assert "</script>" not in html.split("<script>")[1].rsplit("</script>", 1)[0]
    assert "src=\"http" not in html and "href=\"http" not in html


def test_viewer_renders_the_panels_that_prevent_hidden_state():
    html = render_html(a_replay())
    for marker in ("tile inspector", "action log", "not simulated",
                   "renderTile", "renderDiplomacy", "renderTasks",
                   "class TechTree", "class Board"):
        assert marker in html, f"viewer is missing {marker!r}"


def test_viewer_css_colours_are_well_formed():
    import re
    html = render_html(a_replay())
    css = html.split("<style>")[1].split("</style>")[0]
    for colour in re.findall(r"#[0-9a-zA-Z]+", css):
        assert re.fullmatch(r"#[0-9a-fA-F]{3,8}", colour), f"bad colour {colour}"


# --- the "not simulated" panel must stay honest --------------------------------
def test_not_simulated_does_not_name_something_we_do_simulate():
    """The panel exists so an empty log never reads as "nothing happened".

    It went stale once and claimed roads, markets, naval embarking and cloaks
    were missing long after they landed. A panel whose whole job is to tell you
    what you are NOT seeing is the worst possible thing to let drift, so this
    cross-checks it against the names the ruleset actually defines.
    """
    from polyengine.core.rules import load_ruleset
    from polyengine.replay import NOT_SIMULATED

    rules = load_ruleset()
    text = " ".join(NOT_SIMULATED).lower()
    implemented = set(rules.imp_names) | set(rules.unit_names)
    # Whole words, so "port" does not trip on "important"; and singularised,
    # because the stale entries said "roads" and "markets" where the ruleset
    # names are "road" and "market" -- without this the guard misses exactly
    # the wording that went stale.
    words = set(re.findall(r"[a-z_]+", text))
    words |= {w[:-1] for w in words if w.endswith("s")}
    claimed_missing = words & implemented
    assert not claimed_missing, (
        f"NOT_SIMULATED claims these are missing, but the ruleset defines them: "
        f"{sorted(claimed_missing)}")


def test_not_simulated_still_mentions_special_tribes():
    """The one thing genuinely out of scope should stay listed."""
    from polyengine.replay import NOT_SIMULATED
    assert any("tribe" in entry.lower() for entry in NOT_SIMULATED)
