"""PFSP league: opponent weighting, Elo, eviction, and the eval protocol.

The league exists to allocate games where they carry signal. These tests pin the
properties that make that true — the weighting really does concentrate on
contested opponents, eviction preserves strength spread rather than recency, and
Elo moves in the right direction.
"""
from __future__ import annotations

import numpy as np
import pytest

from polytrain.evaluate import MatchResult, sprt
from polytrain.league import (
    ANCHOR_ELO,
    League,
    MainExploiter,
    Member,
    pfsp_weights,
)


# --- PFSP weighting -----------------------------------------------------------
def test_even_weighting_peaks_on_contested_opponents():
    w = pfsp_weights(np.array([0.05, 0.5, 0.95]), mode="even")
    assert w.argmax() == 1
    assert w[0] == pytest.approx(w[2], abs=1e-9), "symmetric around p=0.5"
    assert w.sum() == pytest.approx(1.0)


def test_hard_weighting_prefers_opponents_that_beat_you():
    w = pfsp_weights(np.array([0.1, 0.5, 0.9]), mode="hard")
    assert w.argmax() == 0
    assert w[0] > w[1] > w[2]


def test_weights_fall_back_to_uniform_when_everything_is_won():
    w = pfsp_weights(np.ones(4), mode="even")
    assert np.allclose(w, 0.25), "degenerate case must not divide by zero"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown PFSP mode"):
        pfsp_weights(np.array([0.5]), mode="sideways")


# --- member bookkeeping --------------------------------------------------------
def test_unplayed_member_has_a_neutral_prior():
    m = Member(name="x", kind="snapshot")
    assert m.win_rate == 0.5 and m.games_played == 0.0


def test_recorded_results_decay_so_stale_evidence_fades():
    m = Member(name="x", kind="snapshot")
    for _ in range(30):
        m.record(1.0, decay=0.5)
    assert m.win_rate == pytest.approx(1.0, abs=1e-6)
    for _ in range(30):
        m.record(0.0, decay=0.5)
    assert m.win_rate < 0.05, "old wins should have decayed away"


# --- sampling ------------------------------------------------------------------
def test_sample_returns_none_for_mirror_self_play():
    league = League()
    rng = np.random.default_rng(0)
    picks = [league.sample(rng) for _ in range(50)]
    assert all(p is None for p in picks), "empty league can only mirror"


def test_sample_uses_snapshots_once_they_exist():
    league = League(p_self=0.0, p_even=1.0, p_hard=0.0, p_scripted=0.0)
    league.members.append(Member(name="a", kind="snapshot", state={}))
    rng = np.random.default_rng(1)
    picks = [league.sample(rng) for _ in range(20)]
    assert all(p is not None and p.name == "a" for p in picks)


def test_scripted_members_are_only_sampled_when_enabled():
    league = League(p_self=0.5, p_even=0.5, p_hard=0.0, p_scripted=0.0)
    league.add_scripted("scripted", lambda s: None)
    league.members.append(Member(name="snap", kind="snapshot", state={}))
    rng = np.random.default_rng(2)
    picks = [league.sample(rng) for _ in range(200)]
    assert not any(p is not None and p.kind == "scripted" for p in picks)


# --- Elo -----------------------------------------------------------------------
def test_winning_raises_main_elo_and_lowers_the_opponent():
    league = League()
    m = Member(name="snap", kind="snapshot", state={}, elo=0.0)
    league.members.append(m)
    league.record_results(m, [1.0] * 10)
    assert league.main_elo > 0 and m.elo < 0


def test_losing_lowers_main_elo():
    league = League()
    m = Member(name="snap", kind="snapshot", state={}, elo=0.0)
    league.members.append(m)
    league.record_results(m, [0.0] * 10)
    assert league.main_elo < 0


def test_scripted_baselines_are_fixed_elo_anchors():
    league = League()
    m = league.add_scripted("scripted", lambda s: None)
    league.record_results(m, [1.0] * 10)
    assert m.elo == ANCHOR_ELO, "anchors must not drift"
    assert league.main_elo > 0


def test_recording_against_self_play_is_a_no_op():
    league = League()
    league.record_results(None, [1.0, 0.0])
    assert league.main_elo == 0.0


# --- eviction ------------------------------------------------------------------
def test_eviction_preserves_the_strength_extremes():
    league = League(capacity=3)
    for i, elo in enumerate([0, 100, 110, 120, 500]):
        m = Member(name=f"s{i}", kind="snapshot", state={}, elo=float(elo))
        league.members.append(m)
        league._evict()
    elos = sorted(m.elo for m in league.snapshots)
    assert len(elos) == 3
    assert elos[0] == 0 and elos[-1] == 500, "extremes must survive"


def test_eviction_ignores_scripted_members():
    league = League(capacity=1)
    league.add_scripted("scripted", lambda s: None)
    for i in range(4):
        league.members.append(
            Member(name=f"s{i}", kind="snapshot", state={}, elo=float(i)))
        league._evict()
    assert len(league.scripted) == 1
    assert len(league.snapshots) == 1


def test_table_and_render_include_every_member():
    league = League()
    league.members.append(Member(name="snap", kind="snapshot", state={}))
    rows = league.table()
    assert rows[0]["name"] == "main" and len(rows) == 2
    assert "snap" in league.render()


# --- exploiter -----------------------------------------------------------------
def test_exploiter_resets_only_after_enough_dominant_games():
    ex = MainExploiter(reset_threshold=0.7)
    ex.record([1.0] * 5)
    assert not ex.should_reset(min_games=20), "too few games to conclude"
    ex.record([1.0] * 60)
    assert ex.should_reset(min_games=20)
    ex.note_reset()
    assert ex.resets == 1 and ex.win_rate == 0.5


def test_exploiter_does_not_reset_while_losing():
    ex = MainExploiter()
    ex.record([0.0] * 60)
    assert not ex.should_reset()


# --- evaluation protocol -------------------------------------------------------
def test_match_result_reports_the_record_and_elo():
    m = MatchResult(games=200, score=120.0, pairs=100)
    assert m.win_rate == pytest.approx(0.6)
    assert m.record == "120/200"
    assert m.elo_diff > 0


def test_the_record_shows_draws_and_drops_trailing_zeros():
    """Half points are real — a drawn game scores 0.5 — so the record has to
    show them, without printing "31.0/32" for a clean result."""
    assert MatchResult(games=32, score=31.0, pairs=16).record == "31/32"
    assert MatchResult(games=32, score=29.5, pairs=16).record == "29.5/32"
    assert MatchResult(games=0, score=0.0, pairs=0).record == "0/0"


def test_match_result_has_no_error_bar():
    """The binomial standard error described the wrong quantity: the eval plays
    the same fixed maps every time, so the games are not independent draws.
    Measured over five identical repeats, the Random match returned one value
    all five times against a quoted +-3.1%, while Scripted swung 19 points
    against a quoted +-8.8%. It is gone rather than merely unprinted."""
    assert not hasattr(MatchResult(games=2, score=1.0, pairs=1), "stderr")


def test_even_match_is_zero_elo():
    m = MatchResult(games=100, score=50.0, pairs=50)
    assert m.elo_diff == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("wins,games,expected", [
    (300, 500, "accept"),      # clearly better than p0
    (200, 500, "reject"),      # clearly not
    (52, 100, "continue"),     # inconclusive
])
def test_sprt_decisions(wins, games, expected):
    assert sprt(wins, games, p0=0.5, p1=0.55).decision == expected


def test_sprt_with_no_games_is_inconclusive():
    r = sprt(0, 0)
    assert r.decision == "continue" and not r.decided


# --- persistence across a crash -------------------------------------------------
# This box bugchecks under sustained GPU load, so a training run is restarted
# repeatedly from `latest.pt`. If the league did not round-trip, every restart
# would silently drop the opponent pool back to pure self-play and reset Elo.
def test_league_state_dict_round_trips_snapshots_and_elo():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    league = League(capacity=5)
    league.add_scripted("scripted", lambda s: object())
    for it in (10, 20, 30):
        league.add_snapshot(nn.Linear(4, 4), it)
    league.main_elo = 123.5
    for m in league.snapshots:
        m.record(1.0); m.record(0.0)

    blob = league.state_dict()

    restored = League(capacity=5)
    restored.add_scripted("scripted", lambda s: object())
    restored.load_state_dict(blob)

    assert restored.main_elo == pytest.approx(123.5)
    assert [m.name for m in restored.snapshots] == [m.name for m in league.snapshots]
    assert [m.iteration for m in restored.snapshots] == [10, 20, 30]
    for a, b in zip(restored.snapshots, league.snapshots):
        assert a.win_rate == pytest.approx(b.win_rate)
        assert a.elo == pytest.approx(b.elo)


def test_league_load_state_dict_does_not_duplicate_scripted_members():
    league = League(capacity=4)
    league.add_scripted("scripted", lambda s: object())
    league.load_state_dict(league.state_dict())
    assert len(league.scripted) == 1


def test_league_restores_scripted_statistics():
    league = League(capacity=4)
    m = league.add_scripted("scripted", lambda s: object())
    m.record(1.0); m.record(1.0); m.record(0.0)

    restored = League(capacity=4)
    restored.add_scripted("scripted", lambda s: object())
    restored.load_state_dict(league.state_dict())
    assert restored.scripted[0].win_rate == pytest.approx(m.win_rate)


# --- free-for-all scoring: Elo is a two-player rating system -------------------
#
# Everything below became reachable only when games could hold more than two
# tribes. At N = 2 the old rule and the new one are identical by construction,
# which is why none of this showed up before.

def _rank_result(n_players, rank):
    """A learner finishing `rank`-th of `n_players`, scored the new way."""
    return (n_players - 1 - rank) / (n_players - 1)


@pytest.mark.parametrize("n_players", [2, 3, 6, 16])
def test_equal_strength_players_keep_equal_elo(n_players):
    """The bug this fixes, stated as a property.

    Elo asks "what fraction of a head-to-head did you win?". Scoring a
    free-for-all as 1-if-you-won hands an *equal-strength* player 1/N rather
    than the 1/2 Elo expects, so its rating falls and the frozen pool's rises
    with no difference in strength at all. Measured on this very update, the
    learner drifted to -112 Elo at 3 players and -236 at 16.
    """
    import random
    import statistics

    drifts = []
    for seed in range(8):
        lg = League(capacity=8)
        snap = Member(name="s", kind="snapshot", state={}, iteration=0, elo=0.0)
        rng = random.Random(seed)
        for _ in range(2000):
            order = list(range(n_players))
            rng.shuffle(order)                       # equal strength
            lg.record_results(snap, [_rank_result(n_players, order.index(0))])
        drifts.append(lg.main_elo)
    mean = statistics.mean(drifts)
    assert abs(mean) < 60, (
        f"{n_players}-player Elo drifted {mean:+.0f} at equal strength")


def test_pairwise_score_is_exactly_the_old_rule_for_a_duel():
    """Duels and every rating already recorded must be untouched."""
    assert _rank_result(2, 0) == 1.0
    assert _rank_result(2, 1) == 0.0


# --- rank-based scoring against real games ------------------------------------

def test_pairwise_score_matches_win_loss_draw_in_a_duel():
    from polyengine.core.game import Game
    from polytrain.rollout import pairwise_score

    game = Game.new(size=9, seed=4)
    s = game.state
    # Wipe player 1 off the board: player 0 has won outright.
    s.city_owner[s.city_owner == 1] = -1
    s.city_level[s.city_owner == -1] = 0
    assert pairwise_score(game, 0) == 1.0
    assert pairwise_score(game, 1) == 0.0


def test_finishing_second_of_six_beats_finishing_last():
    """The reason for rank-based reward: winner-take-all gave the runner-up and
    the tribe eliminated on turn three an identical -1."""
    from polyengine.core.game import Game
    from polytrain.rollout import pairwise_score

    game = Game.new(size=16, seed=7, tribes=["imperius", "bardur", "kickoo",
                                             "vengir", "xin_xi", "zebasi"])
    scores = sorted((pairwise_score(game, p) for p in range(6)))
    assert scores[0] < scores[-1], "every seat scored the same"
    assert 0.0 <= scores[0] and scores[-1] <= 1.0
    # Pairwise results over a full round robin must average to exactly 0.5.
    assert abs(sum(scores) / 6 - 0.5) < 1e-9


def test_terminal_reward_spans_minus_one_to_one_and_is_unchanged_for_duels():
    from polyengine.core.game import Game
    from polytrain.rollout import pairwise_score

    duel = Game.new(size=9, seed=4)
    s = duel.state
    s.city_owner[s.city_owner == 1] = -1
    s.city_level[s.city_owner == -1] = 0
    assert 2.0 * pairwise_score(duel, 0) - 1.0 == 1.0
    assert 2.0 * pairwise_score(duel, 1) - 1.0 == -1.0


def test_standing_ranks_a_survivor_above_an_eliminated_tribe():
    """Survival outranks score: a tribe still holding a city has not lost to one
    that has been wiped out, whatever its research total says."""
    from polyengine.core.game import Game
    from polytrain.rollout import standing

    game = Game.new(size=11, seed=2)
    s = game.state
    s.city_owner[s.city_owner == 1] = -1
    s.city_level[s.city_owner == -1] = 0
    # Player 1 keeps its techs and explored tiles, so its raw score is not zero
    # — the old truncation path compared score alone and could rank it above a
    # living tribe.
    assert game.score(1) > 0
    assert standing(game, 0) > standing(game, 1)


# --- distinct opponents per seat ----------------------------------------------

def test_sample_opponents_returns_one_draw_per_seat():
    lg = League(capacity=8)
    for i in range(4):
        lg.add_snapshot(_tiny_net(), i)
    rng = np.random.default_rng(0)
    assert len(lg.sample_opponents(rng, 5)) == 5
    assert lg.sample_opponents(rng, 0) == []


def _tiny_net():
    import torch.nn as nn
    return nn.Linear(2, 2)


def test_seat_opponent_fn_gives_different_seats_different_members():
    """The point of the change: a crowded board is a field of different
    checkpoints, not N-1 copies of one."""
    from polytrain.train_league import make_seat_opponent_fn

    class FakeMember:
        def __init__(self, name):
            self.name, self.kind = name, "scripted"
            self.factory = lambda seed, n=name: _Recorder(n)

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def act(self, game):
            seen.append((self.name, game.current_player))
            from polyengine.agents.random_agent import RandomAgent
            return RandomAgent(seed=0).act(game)

    from polyengine.core.game import Game
    seen: list = []
    a, b = FakeMember("A"), FakeMember("B")
    game = Game.new(size=11, seed=1, tribes=["imperius", "bardur", "kickoo"])
    assign = {0: {0: a, 1: b, 2: a}}
    fn = make_seat_opponent_fn(assign, a, None, None, None)
    assert getattr(fn, "wants_indices", False), "must opt in to game indices"
    for seat in (0, 1, 2):
        game.state.current_player = seat
        fn([game], [0])
    assert [n for n, _ in seen] == ["A", "B", "A"], seen


# --- victory is worth more than points ----------------------------------------
#
# Watching a trained net play showed Warriors only, an early peace treaty, and
# all seven monuments farmed for 2,800 points. Every one of those was optimal
# under a reward where leading on score at the turn cap paid the same as
# destroying the opponent. These pin the separation.

def _duel_won_by(seed=4, turn=20):
    from polyengine.core.game import Game
    g = Game.new(size=11, seed=seed)
    s = g.state
    s.turn = turn
    s.city_owner[s.city_owner == 1] = -1
    s.city_level[s.city_owner == -1] = 0
    return g


def test_conquest_pays_far_more_than_the_best_possible_stall():
    from polyengine.core import constants as C
    from polyengine.core.game import Game
    from polytrain.rollout import terminal_rewards

    won = terminal_rewards(_duel_won_by(turn=20))
    assert won[0] == pytest.approx(1.0), "an on-par conquest is a full win"
    assert won[1] == -1.0, "being eliminated is a full loss"

    # The most lopsided possible cap-terminated game still pays a tenth of it.
    stalled = Game.new(size=11, seed=4)
    stalled.state.turn = stalled.state.turn_cap
    best_stall = max(abs(r) for r in terminal_rewards(stalled))
    assert best_stall <= C.STALL_REWARD + 1e-9
    assert min(won) < 0 < max(won)
    assert C.SPEED_FLOOR / C.STALL_REWARD >= 4.0, (
        "even the slowest conquest must clearly beat the best stall")


def test_a_slow_win_is_worth_less_than_a_fast_one():
    """Without this, a win on turn 300 pays exactly what a win on turn 20 does
    and nothing ever pushes the agent to finish."""
    from polytrain.rollout import terminal_rewards
    fast = terminal_rewards(_duel_won_by(turn=20))[0]
    mid = terminal_rewards(_duel_won_by(turn=60))[0]
    slow = terminal_rewards(_duel_won_by(turn=400))[0]
    assert fast > mid > slow, (fast, mid, slow)
    from polyengine.core import constants as C
    assert slow >= C.SPEED_FLOOR, "a slow win is still a win"


def test_a_cap_terminated_game_has_no_winner():
    """`Game.winner()` names a score leader at the cap because that is how a
    capped game is adjudicated. Training must not confuse that with Domination."""
    from polyengine.core.game import Game
    from polytrain.rollout import conquered

    g = Game.new(size=11, seed=4)
    g.state.turn = g.state.turn_cap
    assert g.is_terminal()
    assert g.winner() >= 0, "the engine still adjudicates a capped game"
    assert conquered(g) == -1, "but nobody conquered anyone"
    assert conquered(_duel_won_by()) == 0


def test_monuments_cannot_outweigh_winning():
    """Seven monuments are 2,800 score points, which the old reward valued at a
    full win. They are now worth at most a tenth of one."""
    from polyengine.core import constants as C
    from polyengine.core.game import Game
    from polytrain.rollout import terminal_rewards

    g = Game.new(size=11, seed=4)
    g.state.turn = g.state.turn_cap
    for r in terminal_rewards(g):
        assert abs(r) <= C.STALL_REWARD + 1e-9
