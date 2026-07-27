"""Snapshot opponent pool and frozen-opponent rollouts.

The pool exists because pure mirror self-play lets the policy cycle: the 6x6
smoke run reached 91.7% against ScriptedBot at iteration 100 and fell back to
47% by 150. These tests pin the two properties that make the fix work — frozen
snapshots really are frozen, and rollouts against them store only the learner's
own transitions.
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
from polytrain.pool import FrozenOpponent, SnapshotPool
from polytrain.rollout import collect

R = load_ruleset()


def make_policy(device="cpu"):
    cfg = NetConfig.small(n_planes=O.n_planes(R), global_dim=O.global_dim(R),
                          n_action_types=C.N_ACTION_TYPES,
                          subtype_dim=M.subtype_dim(R), n_techs=R.n_techs)
    return ARPolicy(PolyNet(cfg).to(device), R, device=device)


# --- pool bookkeeping ----------------------------------------------------------
def test_pool_stores_and_samples_snapshots():
    policy = make_policy()
    pool = SnapshotPool(capacity=4)
    assert len(pool) == 0
    for it in (10, 20, 30):
        pool.add(policy.net, it)
    assert len(pool) == 3
    snap, it = pool.sample(np.random.default_rng(0))
    assert it in (10, 20, 30) and isinstance(snap, dict)


def test_pool_respects_capacity_and_keeps_the_anchor():
    policy = make_policy()
    pool = SnapshotPool(capacity=3)
    for it in range(1, 8):
        pool.add(policy.net, it)
    assert len(pool) == 3
    assert pool.iters[0] == 1, "earliest snapshot should be kept as an anchor"
    assert pool.iters[-1] == 7, "most recent snapshot should be kept"


def test_snapshot_is_a_detached_copy():
    """Training after snapshotting must not mutate the stored weights."""
    policy = make_policy()
    pool = SnapshotPool()
    pool.add(policy.net, 1)
    stored = {k: v.clone() for k, v in pool.snapshots[0].items()}
    with torch.no_grad():
        for p in policy.net.parameters():
            p.add_(1.0)
    for k, v in pool.snapshots[0].items():
        assert torch.equal(v, stored[k]), f"snapshot tensor {k} was mutated"


def test_frozen_opponent_requires_no_grad():
    policy = make_policy()
    pool = SnapshotPool()
    pool.add(policy.net, 1)
    frozen = FrozenOpponent(policy.net, pool.snapshots[0], torch.device("cpu"))
    assert not any(p.requires_grad for p in frozen.net.parameters())
    assert not frozen.net.training


def test_frozen_opponent_is_independent_of_the_learner():
    policy = make_policy()
    pool = SnapshotPool()
    pool.add(policy.net, 1)
    frozen = FrozenOpponent(policy.net, pool.snapshots[0], torch.device("cpu"))
    before = [p.detach().clone() for p in frozen.net.parameters()]
    with torch.no_grad():
        for p in policy.net.parameters():
            p.mul_(3.0)
    assert all(torch.equal(a, b.detach())
               for a, b in zip(before, frozen.net.parameters()))


# --- rollouts against a frozen opponent ----------------------------------------
def test_collect_against_opponent_stores_only_learner_transitions():
    policy = make_policy()
    pool = SnapshotPool()
    pool.add(policy.net, 1)
    frozen = FrozenOpponent(policy.net, pool.snapshots[0], torch.device("cpu"))
    frozen_policy = ARPolicy(frozen.net, R, device="cpu")

    def opponent(pairs):
        with torch.no_grad():
            acts, _ = frozen_policy.act(pairs)
        return acts

    seats = [0, 1, 0, 1]
    roll, stats, _ = collect(policy, n_games=4, rollout_steps=40, size=7, seed=3,
                             max_turns=20, opponent=opponent, learner_seats=seats)
    assert len(roll) > 0
    for game_idx, player in roll.traj:
        assert player == seats[game_idx], \
            "stored a transition for the frozen opponent's seat"


def test_self_play_stores_both_seats():
    policy = make_policy()
    roll, _, _ = collect(policy, n_games=2, rollout_steps=40, size=7, seed=4,
                         max_turns=20)
    seats = {player for _, player in roll.traj}
    assert seats == {0, 1}, "mirror self-play should train on both seats"


def test_opponent_rollout_reports_learner_results():
    policy = make_policy()
    pool = SnapshotPool()
    pool.add(policy.net, 1)
    frozen = FrozenOpponent(policy.net, pool.snapshots[0], torch.device("cpu"))
    frozen_policy = ARPolicy(frozen.net, R, device="cpu")

    def opponent(pairs):
        with torch.no_grad():
            acts, _ = frozen_policy.act(pairs)
        return acts

    _, stats, _ = collect(policy, n_games=4, rollout_steps=120, size=6, seed=5,
                          max_turns=15, opponent=opponent)
    if stats["episodes"]:
        assert "learner_result" in stats
        assert all(0.0 <= r <= 1.0 for r in stats["learner_result"])
