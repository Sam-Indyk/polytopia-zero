"""Baseline agent tests: cloning hygiene, determinism, and strength ordering."""
from __future__ import annotations

import numpy as np
import pytest

from polyengine.agents.heuristic import evaluate
from polyengine.agents.mcts import MCTSAgent
from polyengine.agents.osla import GreedyOSLA
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core import constants as C
from polyengine.core.game import Game


def play(agents, size=9, seed=0, max_steps=600):
    game = Game.new(size=size, seed=seed)
    steps = 0
    while not game.is_terminal() and steps < max_steps:
        game.step(agents[game.current_player].act(game))
        steps += 1
    return game, steps


# --- cloning ------------------------------------------------------------------
def test_clone_is_isolated():
    game = Game.new(size=9, seed=3)
    before = game.state.copy()
    clone = game.clone()
    for _ in range(20):
        if clone.is_terminal():
            break
        clone.step(clone.legal_actions()[0])
    assert np.array_equal(game.state.unit_type, before.unit_type)
    assert np.array_equal(game.state.stars, before.stars)
    assert game.state.turn == before.turn


def test_clone_does_not_grant_income():
    """Cloning must not re-run begin-turn bookkeeping (would duplicate stars)."""
    game = Game.new(size=9, seed=4)
    assert np.array_equal(game.clone().state.stars, game.state.stars)


# --- agents run and terminate --------------------------------------------------
@pytest.mark.parametrize("factory", [
    lambda s: RandomAgent(seed=s),
    lambda s: ScriptedBot(seed=s),
    lambda s: GreedyOSLA(seed=s),
    lambda s: MCTSAgent(n_sims=16, rollout_steps=40, seed=s),
])
def test_agent_plays_legal_game(factory):
    game = Game.new(size=9, seed=7)
    agents = [factory(p) for p in range(game.state.n_players)]
    for _ in range(120):
        if game.is_terminal():
            break
        player = game.current_player
        action = agents[player].act(game)
        assert action in game.legal_actions() or action.atype == C.A_END_TURN
        game.step(action)


@pytest.mark.parametrize("factory", [
    lambda s: ScriptedBot(seed=s),
    lambda s: GreedyOSLA(seed=s),
    lambda s: MCTSAgent(n_sims=16, rollout_steps=40, seed=s),
])
def test_agent_is_deterministic_given_seed(factory):
    g1, s1 = play([factory(p) for p in range(2)], seed=11, max_steps=150)
    g2, s2 = play([factory(p) for p in range(2)], seed=11, max_steps=150)
    assert s1 == s2
    assert np.array_equal(g1.state.unit_type, g2.state.unit_type)
    assert np.array_equal(g1.state.stars, g2.state.stars)


# --- heuristic ----------------------------------------------------------------
def test_evaluate_is_zero_sum_ish_at_symmetric_start():
    game = Game.new(size=9, seed=2)
    assert evaluate(game, 0) == pytest.approx(-evaluate(game, 1), abs=1e-6)


def test_evaluate_prefers_more_territory():
    game = Game.new(size=9, seed=2)
    base = evaluate(game, 0)
    bigger = game.clone()
    free = next(t for t in range(bigger.state.n_tiles)
                if bigger.state.owner[t] != 0)
    bigger.state.owner[free] = 0                  # "20 points per tile"
    assert evaluate(bigger, 0) > base


def test_hoarding_stars_is_not_progress():
    """Stars are NOT part of Polytopia's score.

    The old heuristic gave 0.2 points per star, and a test asserted that
    stockpiling improved your position. It does not: unspent stars are only
    worth what they later buy, and rewarding the pile directly teaches the agent
    to sit on it.
    """
    game = Game.new(size=9, seed=2)
    base = evaluate(game, 0)
    richer = game.clone()
    richer.state.stars[0] += 50
    assert evaluate(richer, 0) == pytest.approx(base)


# --- relative strength ---------------------------------------------------------
def test_scripted_beats_random_over_a_small_match():
    """ScriptedBot should clearly out-perform RandomLegal on material."""
    margin = 0
    for seed in range(6):
        game = Game.new(size=9, seed=seed)
        agents = [ScriptedBot(seed=seed), RandomAgent(seed=seed + 100)]
        for _ in range(400):
            if game.is_terminal():
                break
            game.step(agents[game.current_player].act(game))
        margin += 1 if evaluate(game, 0) > 0 else 0
    assert margin >= 4, f"scripted only led in {margin}/6 games"
