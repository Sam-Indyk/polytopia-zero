"""Rollout collection, GAE across turn boundaries, and the PPO update.

The interesting logic is `_gae`: in a turn-based game a player's own decisions
are not contiguous in time, so advantages must be chained per (game, player),
not per wall-clock step. These tests pin that behaviour on hand-built rollouts
where the right answer can be computed by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polyengine import masks as M
from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.rules import load_ruleset
from polynet import ARPolicy, NetConfig, PolyNet
from polytrain.ppo import PPOConfig, PPOLearner
from polytrain.rollout import Rollout, _gae, collect

R = load_ruleset()


def make_policy(device="cpu"):
    cfg = NetConfig.small(n_planes=O.n_planes(R), global_dim=O.global_dim(R),
                          n_action_types=C.N_ACTION_TYPES,
                          subtype_dim=M.subtype_dim(R), n_techs=R.n_techs)
    return ARPolicy(PolyNet(cfg).to(device), R, device=device)


def fake_rollout(values, rewards, traj, terminal) -> Rollout:
    roll = Rollout()
    roll.value = [torch.tensor([v]) for v in values]
    roll.reward = list(rewards)
    roll.traj = list(traj)
    roll.terminal = list(terminal)
    roll.logp = [torch.tensor([0.0])] * len(values)
    return roll


# --- GAE ----------------------------------------------------------------------
def test_gae_on_a_single_terminal_step():
    roll = fake_rollout([0.5], [1.0], [(0, 0)], [True])
    adv, ret = _gae(roll, None, None, gamma=0.99, lam=0.95)
    assert adv[0] == pytest.approx(1.0 - 0.5)      # r - V, no bootstrap
    assert ret[0] == pytest.approx(1.0)


def test_gae_chains_along_one_players_own_decisions():
    """P0's steps are interleaved with P1's; P0's next value is P0's next step."""
    values = [0.0, 99.0, 1.0]        # idx0 P0, idx1 P1 (irrelevant), idx2 P0
    rewards = [0.0, 0.0, 1.0]
    traj = [(0, 0), (0, 1), (0, 0)]
    terminal = [False, False, True]
    gamma, lam = 0.9, 0.5
    adv, _ = _gae(fake_rollout(values, rewards, traj, terminal),
                  None, None, gamma, lam)
    # last P0 step: delta = 1 - 1 = 0
    assert adv[2] == pytest.approx(0.0)
    # first P0 step bootstraps off P0's *next* value (1.0), not P1's 99.0
    delta0 = 0.0 + gamma * 1.0 - 0.0
    assert adv[0] == pytest.approx(delta0 + gamma * lam * 0.0)


def test_gae_ignores_the_other_players_reward():
    values = [0.0, 0.0]
    rewards = [1.0, -1.0]           # P0 won, P1 lost
    adv, _ = _gae(fake_rollout(values, rewards, [(0, 0), (0, 1)], [True, True]),
                  None, None, 0.99, 0.95)
    assert adv[0] > 0 and adv[1] < 0


def test_gae_bootstraps_at_a_segment_boundary():
    """A non-terminal final step uses its own value, giving delta ~ r + (g-1)V."""
    roll = fake_rollout([2.0], [0.0], [(0, 0)], [False])
    adv, _ = _gae(roll, None, None, gamma=1.0, lam=0.95)
    assert adv[0] == pytest.approx(0.0)


def test_gae_separates_distinct_games():
    values = [0.0, 0.0]
    rewards = [1.0, 0.0]
    adv, _ = _gae(fake_rollout(values, rewards, [(0, 0), (1, 0)], [True, False]),
                  None, None, 0.99, 0.95)
    assert adv[0] == pytest.approx(1.0)
    assert adv[1] == pytest.approx(0.0)


# --- collection ----------------------------------------------------------------
def test_collect_produces_a_consistent_batch():
    policy = make_policy()
    roll, stats, games = collect(policy, n_games=4, rollout_steps=24, size=7,
                                 seed=1, max_turns=20)
    n = len(roll)
    assert n > 0
    assert len(stats["adv"]) == n and len(stats["ret"]) == n
    batch = roll.stack(torch.device("cpu"))
    for key in ("spatial", "global", "valid", "type_mask", "atype",
                "old_logp", "old_value", "applies_source"):
        assert batch[key].shape[0] == n, f"{key} has the wrong batch size"
    assert np.isfinite(stats["adv"]).all()


def test_collect_records_only_legal_actions():
    """Every stored action type must have been legal in its stored mask."""
    policy = make_policy()
    roll, _, _ = collect(policy, n_games=3, rollout_steps=20, size=7, seed=2,
                         max_turns=20)
    batch = roll.stack(torch.device("cpu"))
    chosen = batch["atype"]
    legal = batch["type_mask"].gather(1, chosen[:, None]).squeeze(1)
    assert bool(legal.all()), "stored an action type that was masked illegal"


def test_collect_can_resume_from_existing_games():
    policy = make_policy()
    _, _, games = collect(policy, n_games=2, rollout_steps=10, size=7, seed=3,
                          max_turns=20)
    roll2, _, games2 = collect(policy, n_games=2, rollout_steps=10, size=7,
                               seed=4, max_turns=20, games=games)
    assert len(roll2) > 0 and len(games2) == 2


def test_shaping_adds_intermediate_reward():
    policy = make_policy()
    roll, _, _ = collect(policy, n_games=3, rollout_steps=30, size=7, seed=5,
                         shaping=1.0, max_turns=20)
    assert any(r != 0.0 for r in roll.reward), "shaping produced no reward signal"


# --- PPO update ----------------------------------------------------------------
def test_ppo_update_runs_and_changes_weights():
    policy = make_policy()
    roll, stats, _ = collect(policy, n_games=4, rollout_steps=32, size=7, seed=6,
                             max_turns=20)
    batch = roll.stack(torch.device("cpu"))
    learner = PPOLearner(policy, PPOConfig(minibatch=16, epochs=1))
    before = [p.detach().clone() for p in policy.net.parameters()]
    logs = learner.update(batch, stats["adv"], stats["ret"])
    after = list(policy.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert np.isfinite(logs[key]), f"{key} is not finite"


def test_ppo_update_is_numerically_stable_over_several_steps():
    policy = make_policy()
    learner = PPOLearner(policy, PPOConfig(minibatch=16, epochs=1))
    games = None
    for i in range(3):
        roll, stats, games = collect(policy, n_games=4, rollout_steps=24, size=7,
                                     seed=10 + i, max_turns=20, games=games)
        batch = roll.stack(torch.device("cpu"))
        logs = learner.update(batch, stats["adv"], stats["ret"])
        assert np.isfinite(logs["policy_loss"])
    for p in policy.net.parameters():
        assert torch.isfinite(p).all(), "training produced non-finite weights"


# --- crash-safe checkpointing ---------------------------------------------------
# Motivated by a real failure mode on this box: it bugchecks under sustained GPU
# load (WHEA fatal hardware error on the NVIDIA PCIe link), and a crash landing
# inside a multi-megabyte torch.save would otherwise leave a truncated
# `latest.pt` -- costing the run as much as the crash did.
class TestCrashSafeCheckpoints:
    def test_save_atomic_leaves_no_tmp_file(self, tmp_path):
        from polytrain.checkpoint import save_atomic
        path = tmp_path / "latest.pt"
        save_atomic({"x": 1}, path)
        assert path.exists()
        assert not (tmp_path / "latest.pt.tmp").exists()

    def test_save_atomic_keeps_previous_generation(self, tmp_path):
        from polytrain.checkpoint import load_resumable, save_atomic
        path = tmp_path / "latest.pt"
        save_atomic({"iter": 1}, path)
        save_atomic({"iter": 2}, path)
        assert load_resumable(path)["iter"] == 2
        assert load_resumable(path.with_name("latest.pt.bak"))["iter"] == 1

    def test_load_resumable_falls_back_when_primary_is_torn(self, tmp_path):
        from polytrain.checkpoint import load_resumable, save_atomic
        path = tmp_path / "latest.pt"
        save_atomic({"iter": 1}, path)
        save_atomic({"iter": 2}, path)
        # Simulate a crash mid-write: truncate the primary checkpoint.
        path.write_bytes(path.read_bytes()[: 128])
        assert load_resumable(path)["iter"] == 1, "should fall back to .bak"

    def test_load_resumable_returns_none_when_nothing_readable(self, tmp_path):
        from polytrain.checkpoint import load_resumable
        assert load_resumable(tmp_path / "absent.pt") is None


# --- GAE across rollout boundaries --------------------------------------------
#
# A rollout is a fixed number of micro-steps, not a whole game, so most
# trajectories are cut mid-episode. What happens at that cut decides whether a
# terminal reward can influence anything before the last segment.

def test_gae_bootstraps_the_boundary_from_the_next_state():
    """The bug that broke the first real run.

    The boundary used to bootstrap from `values[i]` — the value of the state the
    transition *left*, not the one it arrived at — making
    `delta = r + (gamma - 1) * V(s)`, essentially zero, and self-referential. No
    information could cross a segment boundary, so on large boards (episodes of
    ~1,235 micro-steps against a 160-step rollout) seven eighths of every
    trajectory was cut off from the only unbiased signal in the run.
    """
    import numpy as np

    from polytrain.rollout import Rollout, _gae

    class FakePolicy:
        """Bootstraps every open chain at a large positive value."""
        def state_value(self, pairs):
            return np.full(len(pairs), 10.0, dtype=np.float32)

    import torch

    def build():
        roll = Rollout()
        for _ in range(3):
            roll.logp.append(torch.zeros(1))
            roll.traj.append((0, 0))
            roll.value.append(torch.zeros(1))     # as `collect` stores it
            roll.reward.append(0.0)
            roll.terminal.append(False)
        return roll

    games = [type("G", (), {"state": object()})()]
    roll = build()
    adv_boot, ret_boot = _gae(roll, FakePolicy(), games, gamma=0.99, lam=0.95)
    # Without a policy there is nothing to bootstrap from and the old
    # self-referential behaviour is the fallback.
    adv_none, _ = _gae(build(), None, None, gamma=0.99, lam=0.95)

    assert adv_boot[-1] > 9.0, "boundary did not bootstrap from the next state"
    assert abs(adv_none[-1]) < 1e-6, "fallback should be the degenerate case"
    # And the bootstrapped value must propagate *backwards* through the chain,
    # which is the whole point.
    assert adv_boot[0] > 8.0, "value did not travel back across the segment"


def test_gae_still_zeroes_the_bootstrap_at_a_real_terminal():
    """A finished game has no future to bootstrap from; only the cut does."""
    import numpy as np

    from polytrain.rollout import Rollout, _gae

    class FakePolicy:
        def state_value(self, pairs):
            return np.full(len(pairs), 10.0, dtype=np.float32)

    import torch

    roll = Rollout()
    for r, term in ((0.0, False), (1.0, True)):
        roll.logp.append(torch.zeros(1))
        roll.traj.append((0, 0))
        roll.value.append(torch.zeros(1))
        roll.reward.append(r)
        roll.terminal.append(term)
    games = [type("G", (), {"state": object()})()]
    adv, ret = _gae(roll, FakePolicy(), games, gamma=0.99, lam=0.95)
    # The terminal transition sees its own reward and nothing beyond it.
    assert abs(adv[-1] - 1.0) < 1e-9, "terminal bootstrapped from a future"


def test_state_value_matches_the_value_act_reports():
    """`state_value` skips the action masks; it must not skip the value."""
    import torch

    from polyengine.core.game import Game
    from polyengine.core.rules import load_ruleset
    from polytrain.train import build_policy

    rules = load_ruleset()
    policy, _ = build_policy(rules, torch.device("cpu"), small=True)
    g = Game.new(size=9, seed=3)
    pairs = [(g.state, 0)]
    _, info = policy.act(pairs)
    from_act = float(info["value"].reshape(-1)[0])
    from_value = float(policy.state_value(pairs)[0])
    assert abs(from_act - from_value) < 1e-4, (from_act, from_value)


def test_shaping_decays_to_a_floor_not_to_zero():
    """Terminal rewards get rare as boards grow; the dense signal has to stay.

    Measured on the first real run: 23 episodes per iteration on 9x9, 4.4 on
    14x14, and 1.7 through the collapse. With shaping at exactly zero from
    iteration 100 the value function had almost nothing to learn from."""
    from polytrain.train_league import shaping_for

    s, anneal, floor = 0.01, 100, 0.25
    assert shaping_for(0, s, anneal, floor) == pytest.approx(s)
    assert shaping_for(50, s, anneal, floor) == pytest.approx(s * 0.625)
    late = shaping_for(100, s, anneal, floor)
    assert late == pytest.approx(s * floor)
    assert shaping_for(5000, s, anneal, floor) == pytest.approx(late), \
        "must hold the floor, not keep decaying"
    assert late > 0.0, "annealing to zero is the bug this exists to prevent"
    # floor=0 reproduces the old schedule exactly, so the flag is a superset.
    assert shaping_for(100, s, anneal, 0.0) == pytest.approx(0.0)


def test_the_promotion_gate_is_fixed_for_the_life_of_a_run():
    """A gate that follows the curriculum cannot name the best checkpoint.

    On the first run it froze best.pt at iteration ~275 while true strength
    peaked near 1,340 and then collapsed."""
    from polytrain import train_league as T

    assert isinstance(T.GATE_SIZE, int)
    assert len(T.GATE_MAP_TYPES) == 6, "the gate should span every map type"
    src = (Path(T.__file__).read_text(encoding="utf-8"))
    assert "sizes=(GATE_SIZE,)" in src and "players=(2,)" in src, \
        "the gate eval must pin size and seat count, not read the stage"


def test_learner_seat_follows_a_replaced_game_with_fewer_players():
    """Games persist across `collect` calls so long episodes can finish, and a
    curriculum mixes seat counts — so a persisted 4-player game can be replaced
    mid-call by a 2-player one from the *new* iteration's spec.

    `learner_seats` is computed once, from the games the call started with.
    Left stale it indexed a seat the replacement does not have: `Game.score(2)`
    on a 2-player board raised IndexError and took the run down 29 times.
    Quieter and worse, before crashing it made `is_learner` match nothing, so
    every transition collected in that game was silently discarded.
    """
    import torch

    from polyengine.core.rules import load_ruleset
    from polytrain.curriculum import GameSpec
    from polytrain.rollout import collect
    from polytrain.train import build_policy

    rules = load_ruleset()
    policy, _ = build_policy(rules, torch.device("cpu"), small=True)
    duel = GameSpec(size=11, map_type="drylands", tribes=("imperius", "bardur"),
                    turn_cap_scale=0.05)
    crowd = GameSpec(size=11, map_type="drylands", turn_cap_scale=0.05,
                     tribes=("imperius", "bardur", "kickoo", "vengir"))

    def opponent(games):
        return [g.legal_actions()[0] for g in games]

    # Iteration 1: game index 2 has four seats, so its learner seat is 2.
    _, _, games = collect(policy, n_games=3, rollout_steps=8, size=11, seed=5,
                          opponent=opponent, specs=[duel, duel, crowd],
                          max_turns=None)
    assert games[2].state.n_players == 4

    # Iteration 2: the same games carry over, but the stage now deals duels, so
    # when game 2 finishes it is replaced by a 2-player board — and seat 2 is
    # gone. A tiny cap guarantees the replacement happens inside this call.
    roll, stats, games = collect(policy, n_games=3, rollout_steps=600, size=11,
                                 seed=6, opponent=opponent, games=games,
                                 specs=[duel, duel, duel], max_turns=None)
    assert stats["episodes"] > 0, "nothing was replaced; the test proves nothing"
    assert games[2].state.n_players == 2
    assert len(roll) > 0
