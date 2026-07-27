"""Network and auto-regressive policy.

The properties that matter: the policy can only ever emit legal actions, the
log-probs it reports are a genuine distribution over the legal set, PopArt
preserves its de-normalised output when statistics move, and the trunk is truly
fully-convolutional (one set of weights, many map sizes).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from polyengine import masks as M
from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.actions import enumerate_actions
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset
from polynet import ARPolicy, NetConfig, PolyNet
from polynet.net import PopArt, masked_log_softmax, sample_masked

R = load_ruleset()


def make_policy(device="cpu", **kw):
    cfg = NetConfig.small(n_planes=O.n_planes(R), global_dim=O.global_dim(R),
                          n_action_types=C.N_ACTION_TYPES,
                          subtype_dim=M.subtype_dim(R), n_techs=R.n_techs, **kw)
    net = PolyNet(cfg).to(device)
    return ARPolicy(net, R, device=device), net, cfg


# --- masked sampling primitives -----------------------------------------------
def test_masked_log_softmax_zeroes_illegal_mass():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    assert probs[0, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-7)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_sample_masked_only_picks_legal_indices():
    logits = torch.randn(64, 8)
    mask = torch.zeros(64, 8, dtype=torch.bool)
    mask[:, 2] = True
    mask[:, 5] = True
    idx, logp, ent = sample_masked(logits, mask)
    assert set(idx.tolist()) <= {2, 5}
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()


def test_fully_masked_row_does_not_produce_nan():
    """Heads that don't apply get an all-False mask; they must stay finite."""
    logits = torch.randn(4, 6)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    idx, logp, ent = sample_masked(logits, mask)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()


# --- PopArt --------------------------------------------------------------------
def test_popart_preserves_denormalised_output():
    pop = PopArt(8, beta=0.5)
    x = torch.randn(16, 8)
    before = pop.denormalise(pop(x))
    pop.update_stats(torch.randn(256) * 7.0 + 3.0)
    after = pop.denormalise(pop(x))
    assert torch.allclose(before, after, atol=1e-4), "PopArt changed its predictions"


def test_popart_normalise_tracks_targets():
    pop = PopArt(4, beta=1.0)
    targets = torch.full((128,), 5.0)
    pop.update_stats(targets)
    assert float(pop.mu) == pytest.approx(5.0, abs=1e-3)


# --- network -------------------------------------------------------------------
def test_trunk_is_fully_convolutional_across_sizes():
    _, net, _ = make_policy()
    for size in (7, 11, 16):
        sp = torch.zeros(2, O.n_planes(R), size, size)
        gv = torch.zeros(2, O.global_dim(R))
        feat, ctx = net.trunk(sp, gv)
        assert feat.shape == (2, net.cfg.channels, size, size)
        assert ctx.shape == (2, net.cfg.channels + O.global_dim(R))


def test_masked_mean_ignores_padding():
    from polynet.net import masked_mean
    x = torch.ones(1, 3, 4, 4)
    x[:, :, 2:, :] = 100.0
    valid = torch.zeros(1, 1, 4, 4)
    valid[:, :, :2, :] = 1.0
    assert torch.allclose(masked_mean(x, valid), torch.ones(1, 3))


def test_value_and_aux_shapes():
    _, net, cfg = make_policy()
    ctx = torch.randn(5, cfg.channels + O.global_dim(R))
    value, score, tech = net.value_and_aux(ctx)
    assert value.shape == (5,) and score.shape == (5,)
    assert tech.shape == (5, R.n_techs)


# --- policy --------------------------------------------------------------------
@pytest.mark.parametrize("greedy", [False, True])
def test_policy_only_emits_legal_actions(greedy):
    policy, _, _ = make_policy()
    games = [Game.new(size=9, seed=s) for s in range(6)]
    for _ in range(12):
        live = [g for g in games if not g.is_terminal()]
        if not live:
            break
        pairs = [(g.state, g.current_player) for g in live]
        actions, _ = policy.act(pairs, greedy=greedy)
        for game, action in zip(live, actions):
            legal = enumerate_actions(game.state, game.current_player, R)
            assert action in legal, f"illegal action {action}"
            game.step(action)


def test_policy_handles_mixed_map_sizes_in_one_batch():
    policy, _, _ = make_policy()
    games = [Game.new(size=7, seed=1), Game.new(size=11, seed=2)]
    pairs = [(g.state, g.current_player) for g in games]
    actions, info = policy.act(pairs)
    assert len(actions) == 2
    assert info["logp"].shape == (2,)
    for game, action in zip(games, actions):
        assert action in enumerate_actions(game.state, game.current_player, R)


def test_act_reports_finite_logp_and_value():
    policy, _, _ = make_policy()
    pairs = [(Game.new(size=9, seed=s).state, 0) for s in range(4)]
    _, info = policy.act(pairs)
    assert torch.isfinite(info["logp"]).all()
    assert torch.isfinite(info["value"]).all()
    assert (info["logp"] <= 1e-5).all(), "log-probs must be <= 0"


def test_evaluate_reproduces_act_logp():
    """The PPO update must score stored actions exactly as sampling did."""
    policy, _, _ = make_policy()
    pairs = [(Game.new(size=9, seed=s).state, 0) for s in range(4)]
    _, info = policy.act(pairs)

    stored = {k: info[k] for k in ("spatial", "global", "valid", "type_mask",
                                   "src_mask", "tgt_mask", "sub_mask",
                                   "atype", "src_idx", "tgt_idx", "sub_idx")}
    for head, key in (("source", M.NEEDS_SOURCE), ("target", M.NEEDS_TARGET),
                      ("subtype", M.NEEDS_SUBTYPE)):
        stored[f"applies_{head}"] = torch.tensor(
            [int(a) in key for a in info["atype"]], dtype=torch.bool)

    with torch.no_grad():
        out = policy.evaluate(stored)
    assert torch.allclose(out["logp"], info["logp"], atol=1e-5)


def test_entropy_is_non_negative():
    policy, _, _ = make_policy()
    pairs = [(Game.new(size=9, seed=s).state, 0) for s in range(4)]
    _, info = policy.act(pairs)
    stored = {k: info[k] for k in ("spatial", "global", "valid", "type_mask",
                                   "src_mask", "tgt_mask", "sub_mask",
                                   "atype", "src_idx", "tgt_idx", "sub_idx")}
    for head, key in (("source", M.NEEDS_SOURCE), ("target", M.NEEDS_TARGET),
                      ("subtype", M.NEEDS_SUBTYPE)):
        stored[f"applies_{head}"] = torch.tensor(
            [int(a) in key for a in info["atype"]], dtype=torch.bool)
    with torch.no_grad():
        out = policy.evaluate(stored)
    assert (out["entropy"] >= -1e-6).all()
