"""Serialization, copy independence, and determinism of GameState / Game."""
import pytest

from polyengine.agents.random_agent import RandomAgent
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset
from polyengine.core.state import GameState

RULES = load_ruleset()


def _play(seed, size=9, steps=300):
    game = Game.new(size=size, seed=seed)
    agents = [RandomAgent(seed=100 + p) for p in range(game.state.n_players)]
    hashes = []
    for _ in range(steps):
        if game.is_terminal():
            break
        game.step(agents[game.current_player].act(game))
        hashes.append(game.state.state_hash())
    return game, hashes


def test_copy_is_independent_and_equal():
    game, _ = _play(seed=3, steps=120)
    s = game.state
    c = s.copy()
    assert c.state_hash() == s.state_hash()
    assert c == s
    # Mutating the copy must not touch the original.
    c.unit_hp[:] = 0
    c.stars[:] = 999
    assert c.state_hash() != s.state_hash()


def test_dict_roundtrip():
    game, _ = _play(seed=5, steps=150)
    s = game.state
    restored = GameState.from_dict(s.to_dict(), RULES)
    assert restored.state_hash() == s.state_hash()
    assert restored.turn == s.turn
    assert restored.current_player == s.current_player
    # RNG stream position preserved -> next draws match.
    assert restored.rng.integers(1_000_000) == s.rng.integers(1_000_000)


def test_determinism_same_seed_same_hashes():
    _, h1 = _play(seed=11, steps=250)
    _, h2 = _play(seed=11, steps=250)
    assert h1 == h2


def test_different_seeds_diverge():
    _, h1 = _play(seed=1, steps=250)
    _, h2 = _play(seed=2, steps=250)
    assert h1 != h2
