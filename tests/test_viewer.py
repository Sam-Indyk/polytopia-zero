"""The viewer, and the fog-projected frame that makes playing it fair.

Two things are being protected here.

**The page has to keep working from a filesystem.** It is a single self-contained
document on purpose — a replay you cannot open by double-clicking is a replay
nobody looks at — and it is assembled from separate `.js`/`.css` asset files, so
a missing or renamed asset must fail loudly in the suite rather than as a blank
page three weeks later.

**Play mode has to be honest.** `fog_frame` is invariant 1 pointed at the human:
if the page contains the map you have not explored, or the opponent's star
count, then "play against the network" is not a fair game and any conclusion you
draw from it is worthless. These tests read the payload the browser would get
and assert the hidden things are absent from it, not merely undrawn.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from polyengine.agents.scripted import ScriptedBot
from polyengine.core.game import Game
from polyengine.replay import board_frame, fog_frame, record_game
from polyengine.viewer import render_html, render_play_html


def a_game(steps=140, size=11, seed=5, map_type="drylands"):
    g = Game.new(size=size, seed=seed, map_type=map_type)
    bots = [ScriptedBot(seed=1), ScriptedBot(seed=2)]
    for _ in range(steps):
        if g.is_terminal():
            break
        g.step(bots[g.current_player].act(g))
    return g


def a_replay(**kw):
    g = Game.new(size=11, seed=3)
    return record_game(g, [ScriptedBot(seed=1), ScriptedBot(seed=2)],
                       max_steps=120).to_dict()


# --- the page assembles ------------------------------------------------------

def test_every_asset_is_inlined_and_nothing_is_fetched():
    """No CDN, no second file, no network. The whole point of the format."""
    html = render_html(a_replay())
    assert "<script src=" not in html
    assert "<link" not in html
    assert not re.search(r'(src|href)\s*=\s*["\']https?:', html)
    # The four scripts and the stylesheet actually made it in.
    for token in ("class Board", "function drawUnit", "class TechTree",
                  "function renderTile", "--gold:"):
        assert token in html, f"asset missing: {token}"


def test_play_page_renders_and_carries_no_frames():
    """The play page boots from the header alone; the live state is fetched.

    If frames ever leak into it, a browser refresh would silently resurrect a
    stale board."""
    g = a_game(steps=40)
    from polyengine.replay import map_header, ruleset_header
    boot = {**map_header(g.state, g.rules), **ruleset_header(g.rules),
            "you": 0, "opponent": "scripted", "agents": ["you", "scripted"]}
    html = render_play_html(boot)
    assert '"frames"' not in html
    assert "/api/state" in html and "/api/act" in html


def test_assets_reach_the_browser_byte_identical():
    """The reason this package exists.

    The old viewer kept its JavaScript and CSS inside a Python string, where
    `\\25b8` in a CSS rule parsed as an octal escape long before the browser saw
    it — the glyph came out wrong and nothing anywhere said so. The fix was to
    move the assets into real files and inline them verbatim, and *verbatim* is
    the load-bearing word: if anything ever starts transforming them on the way
    through, this is where it should fail."""
    from polyengine.viewer import _ASSETS
    html = render_html(a_replay())
    for asset in sorted(_ASSETS.glob("*.js")):
        if asset.name == "play.js":
            continue                     # play-mode controller, not in a replay
        text = asset.read_text(encoding="utf-8")
        assert text.strip(), f"{asset.name} is empty"
        assert text in html, f"{asset.name} was altered on its way into the page"
    assert (_ASSETS / "style.css").read_text(encoding="utf-8") in html


def test_play_ui_has_a_label_for_every_action_type():
    """The play page renders `enumerate_actions` directly, which is what keeps
    it from drifting from the engine — but an action type it has never heard of
    falls through to the raw `describe()` string, and `TILE_ABILITY(37->-1)` is
    not a button anybody can use. A new action type should land here too."""
    from polyengine.core import constants as C
    from polyengine.viewer import _ASSETS
    js = (_ASSETS / "play.js").read_text(encoding="utf-8")
    for t in range(C.N_ACTION_TYPES):
        name = C.ACTION_NAMES[t]
        assert f"'{name}'" in js, f"play UI has no label for {name}"


def test_css_colours_are_well_formed():
    """`\\25b8` in CSS once parsed as a Python octal escape and the glyph came
    out wrong with nothing to show for it. Assets are files now, but the class
    of bug — a colour literal that is not one — is cheap to keep pinned."""
    html = render_html(a_replay())
    css = html.split("<style>")[1].split("</style>")[0]
    for colour in re.findall(r"#[0-9a-zA-Z]+", css):
        assert re.fullmatch(r"#[0-9a-fA-F]{3,8}", colour), f"bad colour {colour}"


# --- fog projection: invariant 1, pointed at the human ------------------------

def test_fog_frame_erases_unexplored_terrain():
    """Not dims — erases. Dimming leaves the map in the page for anyone who
    opens the network tab, which makes the fog decorative."""
    g = a_game(steps=60)
    f = fog_frame(g, 0)
    explored = g.state.fog_explored[0] == 1
    terrain = np.asarray(f["terrain"])
    assert (terrain[~explored] == -1).all(), "unexplored terrain leaked"
    assert (terrain[explored] == g.state.terrain[explored]).all()
    assert (~explored).sum() > 0, "test needs an unexplored tile to be meaningful"


def test_fog_frame_hides_opponent_stars_score_and_research():
    """"opponent stars are not observable" is invariant 1's own wording."""
    g = a_game(steps=120)
    f = fog_frame(g, 0)
    assert f["stars"][1] == -1 and f["stars"][0] == int(g.state.stars[0])
    assert f["score"][1] == -1
    assert f["tech"][1] == 0
    assert f["kills"][1] == -1 and f["losses"][1] == -1


def test_fog_frame_shows_enemy_units_only_as_remembered():
    """A remembered enemy has a type and an owner and nothing else — memory does
    not record how hurt something was when you last saw it."""
    g = a_game(steps=160)
    f = fog_frame(g, 0)
    truth = board_frame(g)
    seen_enemy = False
    for k in range(g.state.n_tiles):
        if f["unit_type"][k] < 0:
            continue
        if f["unit_owner"][k] == 0:
            assert f["unit_hp"][k] == truth["unit_hp"][k]      # your own, in full
        else:
            seen_enemy = True
            assert f["unit_hp"][k] == -1, "enemy health leaked through fog"
            assert f["unit_veteran"][k] == -1
    if not seen_enemy:
        pytest.skip("no enemy unit in memory at this point in the game")


def test_fog_frame_never_invents_a_unit_the_player_has_not_seen():
    g = a_game(steps=160)
    f = fog_frame(g, 0)
    mem = g.state.mem_unit_type[0]
    explored = g.state.fog_explored[0] == 1
    for k in range(g.state.n_tiles):
        if f["unit_type"][k] >= 0:
            assert explored[k] and mem[k] >= 0, f"tile {k} shows an unremembered unit"


def test_fog_frame_keeps_your_own_side_complete():
    """The projection must not blind you to your own empire — that would make
    the interface unusable, and it is the easy mistake when scrubbing arrays."""
    g = a_game(steps=160)
    f, truth = fog_frame(g, 0), board_frame(g)
    for k in range(g.state.n_tiles):
        if truth["city_owner"][k] == 0 and truth["city_level"][k] > 0:
            assert f["city_level"][k] == truth["city_level"][k]
            assert f["city_pop"][k] == truth["city_pop"][k]
        if truth["unit_owner"][k] == 0 and truth["unit_type"][k] >= 0:
            assert f["unit_type"][k] == truth["unit_type"][k]
            assert f["unit_moved"][k] == truth["unit_moved"][k]


def test_fog_frame_is_json_serialisable():
    """It is posted over HTTP; a stray numpy scalar raises inside the handler
    and surfaces as an opaque 500."""
    g = a_game(steps=80)
    json.dumps(fog_frame(g, 0))
    json.dumps(board_frame(g))


# --- the recorded frame carries what the panels claim to show -----------------

@pytest.mark.parametrize("field", [
    "task_done", "monument_built", "kills", "losses",     # tasks panel
    "improvement_turn",                                   # temple ageing
    "score",                                              # the winning number
    "peace", "peace_offer", "attacked_now", "attacked_prev",   # relations panel
])
def test_frame_carries_the_fields_the_panels_read(field):
    """Each of these backs a panel. A panel reading an absent field renders
    empty, which is indistinguishable from "nothing is happening"."""
    g = a_game(steps=60)
    assert field in board_frame(g)
