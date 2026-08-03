"""Property/invariant tests over full random playthroughs.

Plays random games across many seeds and asserts core invariants hold at every
step — the cheap, high-coverage net the plan calls for (stars never negative,
fog never leaks unseen tiles, every legal action applies cleanly, HP in range).
"""
import pytest

from polyengine.agents.random_agent import RandomAgent
from polyengine.core import constants as C
from polyengine.core.actions import apply_action, enumerate_actions
from polyengine.core.combat import effective_max_hp
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset

RULES = load_ruleset()
SEEDS = list(range(12))


def _check_invariants(game):
    s = game.state
    # Stars are never negative.
    assert (s.stars >= 0).all(), f"negative stars: {s.stars}"
    # Live units have HP in (0, max]; empty tiles carry no HP. The ceiling is
    # `effective_max_hp`, not the hull's own stat: a boat is a vehicle, so a
    # Raft carrying a Swordsman has the Swordsman's 15, which is exactly why
    # the wiki lists naval health as "Varies".
    for t in range(s.n_tiles):
        ut = s.unit_type[t]
        if ut != C.U_NONE:
            assert 0 < s.unit_hp[t] <= effective_max_hp(s, t, RULES)
    # Visible implies explored, for every player.
    for p in range(s.n_players):
        vis = s.fog_visible[p]
        expl = s.fog_explored[p]
        assert not ((vis == 1) & (expl == 0)).any(), "visible but unexplored"
        # Fog must not leak: unexplored tiles keep default (empty) memory.
        unexpl = expl == 0
        assert (s.mem_unit_type[p][unexpl] == C.U_NONE).all()
        assert (s.mem_city_owner[p][unexpl] == C.NEUTRAL).all()
    # Score is well-defined and non-negative.
    for p in range(s.n_players):
        assert game.score(p) >= 0


@pytest.mark.parametrize("seed", SEEDS)
def test_random_playthrough_invariants(seed):
    game = Game.new(size=9, seed=seed)
    agents = [RandomAgent(seed=50 + p) for p in range(game.state.n_players)]
    for _ in range(2000):
        if game.is_terminal():
            break
        _check_invariants(game)
        game.step(agents[game.current_player].act(game))
    _check_invariants(game)


@pytest.mark.parametrize("seed", SEEDS[:6])
def test_every_legal_action_applies_without_error(seed):
    """Each enumerated legal action must apply to a copy without raising."""
    game = Game.new(size=9, seed=seed)
    agents = [RandomAgent(seed=70 + p) for p in range(game.state.n_players)]
    checked = 0
    for _ in range(400):
        if game.is_terminal():
            break
        for a in enumerate_actions(game.state, game.current_player, game.rules):
            snapshot = game.state.copy()
            apply_action(snapshot, a, game.rules)  # must not raise
            checked += 1
        game.step(agents[game.current_player].act(game))
    assert checked > 0


def test_games_terminate_within_cap():
    """Random games must not run past the Domination turn cap."""
    for seed in SEEDS:
        game = Game.new(size=9, seed=seed)
        agents = [RandomAgent(seed=90 + p) for p in range(game.state.n_players)]
        for _ in range(100_000):
            if game.is_terminal():
                break
            game.step(agents[game.current_player].act(game))
        assert game.is_terminal()
        assert game.state.turn <= C.TURN_CAP
