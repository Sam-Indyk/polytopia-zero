"""Rollout collection, GAE across turn boundaries, and the PPO update.

The interesting logic is `_gae`: in a turn-based game a player's own decisions
are not contiguous in time, so advantages must be chained per (game, player),
not per wall-clock step. These tests pin that behaviour on hand-built rollouts
where the right answer can be computed by hand.
"""
from __future__ import annotations

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
