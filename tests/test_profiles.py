"""Hardware profiles.

The load-bearing property is that a profile never silently overrides a flag the
operator typed — that class of bug already bit once, when `--smoke` clobbered an
explicit `--eval-games`.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from polytrain import profiles


def args_stub(**kw):
    base = dict(channels=None, blocks=None, games=1, rollout=1, minibatch=1,
                epochs=1, size=1, max_turns=1, league_size=1, workers=0,
                worker_device="cpu", precision="fp32")
    base.update(kw)
    return SimpleNamespace(**base)


def test_all_profiles_are_self_consistent():
    for name, p in profiles.PROFILES.items():
        assert p.name == name
        assert p.channels > 0 and p.blocks > 0
        assert p.games > 0 and p.rollout > 0
        assert p.minibatch > 0 and p.epochs > 0
        assert p.precision in ("fp32", "bf16", "fp8")
        assert p.worker_device in ("cpu", "cuda")
        assert p.notes, f"{name} should explain itself"


def test_get_rejects_unknown_profiles():
    with pytest.raises(SystemExit, match="unknown profile"):
        profiles.get("supercomputer")


def test_apply_fills_unset_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--profile", "workstation"])
    args = args_stub()
    profiles.WORKSTATION.apply(args)
    assert args.channels == profiles.WORKSTATION.channels
    assert args.games == profiles.WORKSTATION.games
    assert args.workers == profiles.WORKSTATION.env_workers
    assert args.precision == profiles.WORKSTATION.precision


def test_apply_never_overrides_an_explicit_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["train.py", "--profile", "workstation", "--games", "7",
                         "--minibatch=13"])
    args = args_stub(games=7, minibatch=13)
    profiles.WORKSTATION.apply(args)
    assert args.games == 7, "explicit --games was overwritten"
    assert args.minibatch == 13, "explicit --minibatch=13 was overwritten"
    assert args.channels == profiles.WORKSTATION.channels   # this one was unset


def test_workstation_scales_up_from_laptop():
    lap, ws = profiles.LAPTOP, profiles.WORKSTATION
    assert ws.channels > lap.channels and ws.blocks > lap.blocks
    assert ws.games > lap.games
    assert ws.env_workers > lap.env_workers
    assert ws.league_size > lap.league_size


def test_smoke_profile_is_actually_small():
    s = profiles.SMOKE
    assert s.size <= 6 and s.games <= 16


def test_autodetect_returns_a_real_profile():
    assert profiles.autodetect() in profiles.PROFILES


def test_workstation_model_fits_one_gpu_many_times_over():
    """The reason we use data-parallel rather than FSDP/ZeRO sharding."""
    torch = pytest.importorskip("torch")
    from polyengine import masks as M
    from polyengine import obs as O
    from polyengine.core.rules import load_ruleset
    from polynet import NetConfig, PolyNet

    R = load_ruleset()
    w = profiles.WORKSTATION
    cfg = NetConfig(n_planes=O.n_planes(R), global_dim=O.global_dim(R),
                    n_action_types=6, subtype_dim=M.subtype_dim(R),
                    n_techs=R.n_techs, channels=w.channels, blocks=w.blocks)
    params = sum(p.numel() for p in PolyNet(cfg).parameters())
    gb_fp32 = params * 4 / 1e9
    assert gb_fp32 < 5.0, (
        f"workstation net is {gb_fp32:.1f} GB; if it ever approaches 96 GB the "
        "data-parallel assumption in profiles.py needs revisiting")
