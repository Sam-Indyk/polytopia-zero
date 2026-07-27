"""SyncVecGame and the parallel rollout harness."""
from __future__ import annotations

import numpy as np

from polyengine.agents.random_agent import RandomAgent
from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.vec import SyncVecGame, parallel_rollout


def test_vec_games_are_independent_and_seeded():
    vec = SyncVecGame(n=4, size=9, base_seed=10)
    assert vec.n == 4
    hashes = {g.state.state_hash() for g in vec.games}
    assert len(hashes) == 4, "distinct seeds should give distinct maps"
    # Same base seed reproduces the same batch.
    again = SyncVecGame(n=4, size=9, base_seed=10)
    assert [g.state.state_hash() for g in vec.games] == \
           [g.state.state_hash() for g in again.games]


def test_vec_step_advances_each_game_once():
    vec = SyncVecGame(n=3, size=9, base_seed=1)
    before = [g.state.state_hash() for g in vec.games]
    legal = vec.legal_actions()
    assert all(len(acts) > 0 for acts in legal)
    dones, infos = vec.step([acts[0] for acts in legal])
    assert len(dones) == 3 and len(infos) == 3
    assert [g.state.state_hash() for g in vec.games] != before


def test_vec_runs_to_completion_with_random_agents():
    vec = SyncVecGame(n=4, size=9, base_seed=5)
    agents = [RandomAgent(seed=i) for i in range(4)]
    for _ in range(3000):
        if vec.all_done():
            break
        actions = []
        for g, agent in zip(vec.games, agents):
            actions.append(Action(C.A_END_TURN) if g.is_terminal() else agent.act(g))
        vec.step(actions)
    assert vec.all_done()
    assert all(g.winner() >= -1 for g in vec.games)


def test_vec_matches_single_game_stepping():
    """A game inside the vector must evolve identically to a standalone one."""
    from polyengine.core.game import Game
    vec = SyncVecGame(n=1, size=9, base_seed=42)
    solo = Game.new(size=9, seed=42)
    agent_a, agent_b = RandomAgent(seed=3), RandomAgent(seed=3)
    for _ in range(60):
        if solo.is_terminal():
            break
        vec.step([agent_a.act(vec.games[0])])
        solo.step(agent_b.act(solo))
    assert vec.games[0].state.state_hash() == solo.state.state_hash()


def test_parallel_rollout_single_worker():
    stats = parallel_rollout(n_games=4, size=9, workers=1, seed=0)
    assert stats.total_games == 4
    assert stats.total_steps > 0
    assert stats.steps_per_sec > 0
    assert sum(stats.wins.values()) == 4


def test_parallel_rollout_is_reproducible():
    a = parallel_rollout(n_games=4, size=9, workers=1, seed=7)
    b = parallel_rollout(n_games=4, size=9, workers=1, seed=7)
    assert a.total_steps == b.total_steps and a.wins == b.wins
