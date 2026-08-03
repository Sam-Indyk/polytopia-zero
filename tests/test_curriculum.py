"""The training distribution: what widens, and what must not break when it does.

Before this, every training game ever played was an 11x11 duel between Imperius
and Bardur. The agent had never seen a large board, had met two of the twelve
tribes, and had never been in a game where a peace treaty could pay for itself —
with one opponent a treaty is just a slower draw.

Four things are pinned here:

* the curriculum widens, and never deals a game the engine would refuse;
* **size is constant within an iteration**, because the encoder pads every board
  in a batch up to the largest one in it, so one 18x18 among 64 games makes all
  64 cost 18x18;
* the turn cap grows with the board and the crowd, and survives a round trip
  through the replay header — it is now part of what determines a game, and the
  header losing `map_type` once already broke the differential gate for months;
* the observation carries a row *per seat*, indexed by player id, because
  `A_PEACE_OFFER` takes a player id as its sub-type and an aggregate cannot tell
  you which tribe to offer to.
"""
from __future__ import annotations

import numpy as np
import pytest

import polyengine.obs as O
from polyengine.core import constants as C
from polyengine.core.game import Game
from polyengine.core.mapgen import max_players_for_size
from polyengine.core.rules import load_ruleset
from polytrain import curriculum as K

R = load_ruleset()


# --- the curriculum deals playable games --------------------------------------

@pytest.mark.parametrize("preset", sorted(K.PRESETS))
def test_every_preset_deals_games_the_engine_accepts(preset):
    """`new_game` refuses a map it cannot seat rather than dealing a player with
    no capital. A curriculum that asks for sixteen tribes on a 9x9 would take
    the whole run down at whatever iteration it first widened."""
    cur = K.with_horizon(K.PRESETS[preset], 400)
    for it in (0, 100, 200, 300, 399):
        for i in range(6):
            spec = cur.spec(it, i, seed=3)
            assert 2 <= spec.n_players <= max_players_for_size(spec.size)
            assert spec.map_type in K.ALL_MAP_TYPES
            assert len(spec.tribes) == spec.n_players
            Game.new(size=spec.size, seed=it * 100 + i,
                     map_type=spec.map_type, tribes=list(spec.tribes))


@pytest.mark.parametrize("preset", sorted(K.PRESETS))
def test_size_is_constant_within_an_iteration(preset):
    """The encoder pads a batch to `max(sizes)`, so mixing sizes inside one
    iteration silently pays the largest board's cost for every game in it."""
    cur = K.with_horizon(K.PRESETS[preset], 400)
    for it in (0, 137, 399):
        sizes = {cur.spec(it, i, seed=1).size for i in range(24)}
        assert len(sizes) == 1, f"iteration {it} mixed sizes {sizes}"


def test_the_curriculum_actually_widens():
    """A curriculum that never widens is just a fixed config with ceremony."""
    cur = K.with_horizon(K.PRESETS["ffa"], 400)
    early = {cur.spec(0, i, seed=2).n_players for i in range(24)}
    late = {cur.spec(399, i, seed=2).n_players for i in range(24)}
    assert max(late) > max(early), "seat count never grew"
    assert cur.size_for(399) > cur.size_for(0), "board size never grew"
    early_tribes = {t for i in range(24) for t in cur.spec(0, i, seed=2).tribes}
    late_tribes = {t for i in range(24) for t in cur.spec(399, i, seed=2).tribes}
    assert len(late_tribes) > len(early_tribes), "tribe pool never grew"


def test_stage_boundaries_rescale_to_the_run_length():
    """A preset's `until` values assume a long run. A 200-iteration experiment
    using them unrescaled would never leave stage one and would be a duel-only
    run wearing the name 'ffa'."""
    short = K.with_horizon(K.PRESETS["ffa"], 50)
    assert short.stage_for(49) is not short.stage_for(0)


def test_specs_are_deterministic_in_the_seed():
    a = [K.PRESETS["standard"].spec(7, i, seed=11) for i in range(8)]
    b = [K.PRESETS["standard"].spec(7, i, seed=11) for i in range(8)]
    assert a == b


def test_inline_spec_parses():
    cur = K.parse("11:2,14:2/3,16:2/3/4")
    assert len(cur.stages) == 3
    assert cur.stages[2].players == (2, 3, 4)
    assert K.parse("9:2:drylands/lakes").stages[0].map_types == ("drylands", "lakes")


# --- turn cap -----------------------------------------------------------------

def test_turn_cap_grows_with_board_and_crowd():
    """Real Domination has no time limit; ours is a training device. But a
    30x30 free-for-all cannot be settled in the hundred turns an 11x11 duel
    needs, and capping it there decides every large game on score by default."""
    base = C.turn_cap_for(11, 2)
    assert base == C.TURN_CAP
    assert C.turn_cap_for(20, 2) > base
    assert C.turn_cap_for(11, 8) > base
    assert C.turn_cap_for(6, 2) < base
    assert C.TURN_CAP_MIN <= C.turn_cap_for(30, 16) <= C.TURN_CAP_MAX


def test_game_uses_its_own_cap_not_the_module_constant():
    big = Game.new(size=20, seed=1, tribes=["imperius", "bardur", "kickoo"])
    assert big.state.turn_cap == C.turn_cap_for(20, 3)
    assert big.state.turn_cap > C.TURN_CAP
    pinned = Game.new(size=20, seed=1, turn_cap=40)
    assert pinned.state.turn_cap == 40


def test_turn_cap_survives_the_replay_header():
    """Invariant 4: the header must carry everything needed to rebuild the game.

    It lost `map_type` once and `resimulate` rebuilt continents games as
    drylands, failing on action 0 for months with no test noticing. The turn cap
    is now in the same category — a long game rebuilt under the default cap ends
    early and diverges with no visible cause."""
    from polyengine.agents.scripted import ScriptedBot
    from polyengine.replay import record_game
    from polyengine.testing.differential import check_determinism

    game = Game.new(size=14, seed=5, tribes=["kickoo", "vengir", "xin_xi"])
    assert game.state.turn_cap != C.TURN_CAP, "test needs a non-default cap"
    rec = record_game(game, [ScriptedBot(seed=i) for i in range(3)], max_steps=400)
    blob = rec.to_dict()
    assert blob["turn_cap"] == game.state.turn_cap
    assert check_determinism(blob) is None


def test_state_round_trip_keeps_the_cap():
    from polyengine.core.state import GameState
    g = Game.new(size=16, seed=2, tribes=["imperius", "bardur", "kickoo", "vengir"])
    back = GameState.from_dict(g.state.to_dict(), R)
    assert back.turn_cap == g.state.turn_cap


# --- per-opponent observation --------------------------------------------------

def test_observation_has_one_row_per_seat_indexed_by_player_id():
    """The index has to be the absolute player id: `A_PEACE_OFFER`'s sub-type IS
    a player id, so unless row k describes player k the policy is choosing a
    treaty partner out of a vector it cannot connect to the choice."""
    g = Game.new(size=14, seed=4,
                 tribes=["imperius", "bardur", "kickoo", "vengir"])
    base, count = O.global_layout(R)["opponents"]
    assert count == C.MAX_PLAYERS * O.N_OPP_FEATURES
    gv = O.encode_global(g.state, 1, R)
    f = O.N_OPP_FEATURES
    idx = {name: i for i, name in enumerate(O.OPP_FEATURES)}
    for q in range(C.MAX_PLAYERS):
        row = gv[base + q * f: base + (q + 1) * f]
        exists = q < g.state.n_players
        assert row[idx["seat_exists"]] == (1.0 if exists else 0.0)
        assert row[idx["is_me"]] == (1.0 if q == 1 else 0.0)


def test_peace_and_war_show_up_against_the_right_seat():
    """The whole point: not "60% of my opponents are hostile" but *which one*."""
    from polyengine.core.diplomacy import offer_peace, record_attack

    g = Game.new(size=14, seed=6,
                 tribes=["imperius", "bardur", "kickoo", "vengir"])
    s = g.state
    s.grant_tech(0, R.tech_id["strategy"])
    s.grant_tech(2, R.tech_id["strategy"])
    offer_peace(s, 0, 2, R)          # 0 proposes to 2
    offer_peace(s, 2, 0, R)          # 2 accepts -> treaty
    record_attack(s, 0, 3, R)        # 0 is at war with 3

    base = O.global_layout(R)["opponents"][0]
    f = O.N_OPP_FEATURES
    idx = {name: i for i, name in enumerate(O.OPP_FEATURES)}
    gv = O.encode_global(s, 0, R)

    def row(q):
        return gv[base + q * f: base + (q + 1) * f]

    assert row(2)[idx["at_peace"]] == 1.0
    assert row(1)[idx["at_peace"]] == 0.0
    assert row(3)[idx["at_war"]] == 1.0
    assert row(1)[idx["at_war"]] == 0.0


def test_a_pending_offer_is_visible_and_directional():
    from polyengine.core.diplomacy import offer_peace

    g = Game.new(size=14, seed=6, tribes=["imperius", "bardur", "kickoo"])
    s = g.state
    s.grant_tech(2, R.tech_id["strategy"])
    offer_peace(s, 2, 0, R)                      # 2 offers 0; nobody accepts yet

    base = O.global_layout(R)["opponents"][0]
    f = O.N_OPP_FEATURES
    idx = {name: i for i, name in enumerate(O.OPP_FEATURES)}
    mine = O.encode_global(s, 0, R)[base + 2 * f: base + 3 * f]
    theirs = O.encode_global(s, 2, R)[base + 0 * f: base + 1 * f]
    assert mine[idx["offered_me"]] == 1.0 and mine[idx["i_offered"]] == 0.0
    assert theirs[idx["i_offered"]] == 1.0 and theirs[idx["offered_me"]] == 0.0


def test_opponent_rows_still_hide_what_fog_hides():
    """Invariant 1 applies to the new block too: seat counts come from the
    player's own memory, so an unexplored empire stays invisible."""
    g = Game.new(size=16, seed=8, tribes=["imperius", "bardur", "kickoo"])
    base = O.global_layout(R)["opponents"][0]
    f = O.N_OPP_FEATURES
    idx = {name: i for i, name in enumerate(O.OPP_FEATURES)}
    gv = O.encode_global(g.state, 0, R)
    for q in (1, 2):
        row = gv[base + q * f: base + (q + 1) * f]
        assert row[idx["seen_cities"]] == 0.0, "saw a city it has not explored"
        assert row[idx["seen_units"]] == 0.0


def test_the_global_vector_still_says_nothing_about_the_map_type():
    """The wetness shortcut must stay closed — the new block must not reopen it
    by leaking something that correlates with the map type."""
    dry = Game.new(size=11, seed=11, map_type="drylands")
    wet = Game.new(size=11, seed=11, map_type="waterworld")
    assert np.array_equal(O.encode_global(dry.state, 0, R),
                          O.encode_global(wet.state, 0, R))


# --- eval buckets --------------------------------------------------------------

def test_match_reports_a_breakdown_not_just_an_average():
    """A single number over a curriculum hides 90% on the small duels the agent
    has seen a thousand times and 30% on the crowded boards it barely knows."""
    from polyengine.agents.random_agent import RandomAgent
    from polyengine.agents.scripted import ScriptedBot
    from polytrain.evaluate import play_match

    m = play_match(ScriptedBot(seed=1), RandomAgent(seed=2), n_pairs=4, size=9,
                   max_steps=200, sizes=(9, 11), players=(2, 3),
                   map_types=("drylands", "lakes"))
    assert m.games > 0
    for axis, expected in (("size", {9, 11}), ("players", {2, 3}),
                           ("map", {"drylands", "lakes"})):
        assert {v for v, _, _ in m.breakdown(axis)} == expected
    # Every game lands in exactly one bucket per axis.
    for axis in ("size", "players", "map"):
        assert sum(n for _, _, n in m.breakdown(axis)) == m.games
    assert m.format_breakdown()


def test_match_rotates_the_agent_through_every_seat():
    """With three seats a "mirrored pair" is three games, not two — otherwise
    seat 2 is never played and its corner of the map is never credited."""
    from polyengine.agents.random_agent import RandomAgent
    from polytrain.evaluate import play_pair

    scores = play_pair(RandomAgent(seed=1), RandomAgent(seed=2), seed=3,
                       size=14, max_steps=120,
                       tribes=("imperius", "bardur", "kickoo"), n_players=3)
    assert len(scores) == 3
