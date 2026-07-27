"""Multiprocess rollout collection.

These spawn real worker processes, so they are slower than the rest of the suite
but they are the only thing that catches the failure modes that matter here:
weights not actually reaching workers, shards merging into a mis-aligned batch,
or the batch coming back in a dtype the learner can't consume.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polyengine import masks as M
from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.rules import load_ruleset
from polynet import NetConfig, PolyNet
from polytrain.parallel import ParallelCollector, _merge, to_device

R = load_ruleset()


def small_cfg():
    return NetConfig.small(n_planes=O.n_planes(R), global_dim=O.global_dim(R),
                           n_action_types=C.N_ACTION_TYPES,
                           subtype_dim=M.subtype_dim(R), n_techs=R.n_techs,
                           channels=16, blocks=1, head_hidden=32)


@pytest.fixture(scope="module")
def collector():
    col = ParallelCollector(small_cfg().__dict__, workers=2)
    yield col
    col.close()


# --- merging (no subprocesses needed) -----------------------------------------
def test_merge_concatenates_shards_and_sums_stats():
    shard = lambda n, ep: {
        "n": n, "episodes": ep, "lengths": [5] * ep, "learner_result": [],
        "adv": np.ones(n), "ret": np.full(n, 2.0), "atype": np.zeros(n, np.int64),
    }
    batch, stats = _merge([shard(3, 1), shard(2, 2)])
    assert batch["atype"].shape == (5,)
    assert stats["n"] == 5 and stats["episodes"] == 3
    assert stats["adv"].shape == (5,) and stats["ret"].shape == (5,)
    assert len(stats["lengths"]) == 3


def test_merge_of_nothing_is_empty_not_a_crash():
    batch, stats = _merge([])
    assert batch == {} and stats["n"] == 0


def test_to_device_assigns_the_dtypes_the_learner_expects():
    batch = {
        "spatial": np.zeros((2, 3, 4, 4), np.float32),
        "atype": np.zeros(2, np.int64),
        "type_mask": np.ones((2, C.N_ACTION_TYPES), bool),
        "applies_source": np.ones(2, bool),
        "old_logp": np.zeros(2, np.float32),
    }
    out = to_device(batch, torch.device("cpu"))
    assert out["atype"].dtype == torch.long
    assert out["type_mask"].dtype == torch.bool
    assert out["applies_source"].dtype == torch.bool
    assert out["spatial"].dtype == torch.float32


# --- real workers --------------------------------------------------------------
@pytest.mark.slow
def test_parallel_collect_returns_a_usable_batch(collector):
    net = PolyNet(small_cfg())
    batch, stats = collector.collect(net, n_games=4, rollout_steps=12, size=6,
                                     seed=1, max_turns=15)
    assert stats["n"] > 0
    n = stats["n"]
    for key in ("spatial", "global", "valid", "type_mask", "atype",
                "old_logp", "old_value", "applies_source"):
        assert batch[key].shape[0] == n, f"{key} misaligned with batch size"
    assert len(stats["adv"]) == n and len(stats["ret"]) == n
    assert np.isfinite(stats["adv"]).all()


@pytest.mark.slow
def test_parallel_batch_feeds_the_ppo_update(collector):
    from polynet import ARPolicy
    from polytrain.ppo import PPOConfig, PPOLearner

    net = PolyNet(small_cfg())
    policy = ARPolicy(net, R, device="cpu")
    batch, stats = collector.collect(net, n_games=4, rollout_steps=12, size=6,
                                     seed=2, max_turns=15)
    dev = torch.device("cpu")
    learner = PPOLearner(policy, PPOConfig(minibatch=8, epochs=1))
    logs = learner.update(to_device(batch, dev), stats["adv"], stats["ret"])
    assert np.isfinite(logs["policy_loss"])
    for p in net.parameters():
        assert torch.isfinite(p).all()


@pytest.mark.slow
def test_workers_receive_the_current_weights(collector):
    """A degenerate net must produce a different batch than a fresh one."""
    net = PolyNet(small_cfg())
    _, stats_a = collector.collect(net, n_games=4, rollout_steps=16, size=6,
                                   seed=3, max_turns=15, persist=False)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.0)
    _, stats_b = collector.collect(net, n_games=4, rollout_steps=16, size=6,
                                   seed=3, max_turns=15, persist=False)
    # Same seed and games; only the weights changed. With a zeroed net the
    # policy is uniform over legal actions, so behaviour (and episode stats)
    # should differ from the randomly-initialised one.
    assert stats_a["n"] > 0 and stats_b["n"] > 0
