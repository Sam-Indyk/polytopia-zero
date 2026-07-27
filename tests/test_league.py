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
def test_match_result_reports_stderr_and_elo():
    m = MatchResult(games=200, score=120.0, pairs=100)
    assert m.win_rate == pytest.approx(0.6)
    assert 0.0 < m.stderr < 0.05
    assert m.elo_diff > 0


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
