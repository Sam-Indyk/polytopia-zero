"""Load a trained checkpoint as a playable agent, and save one crash-safely.

Lets the network drop into the same places the scripted baselines go — the
arena, the replay recorder, the viewer — so "is it actually playing sensibly?"
is a question you answer by watching a game, not by reading a win rate.

`save_atomic` exists because this dev box bugchecks under sustained GPU load
(WHEA fatal hardware error on the NVIDIA PCIe link; see CLAUDE.md). A machine
that can die *during* a multi-megabyte `torch.save` will otherwise leave a
truncated `latest.pt`, which costs the run exactly as much as the crash did.
Write-then-rename is atomic on NTFS, so a reader sees either the old file or
the new one, never a half-written one.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from polyengine.core.rules import load_ruleset
from polynet import ARPolicy, NetConfig, PolyNet


def save_atomic(blob: dict, path: str | Path, keep_backup: bool = True) -> None:
    """Save `blob` so an abrupt power loss can never leave a torn file.

    With `keep_backup`, the previous checkpoint is rotated to `.bak` first, so a
    crash in the rename window still leaves one complete generation behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Flush through the same handle torch.save wrote: os.replace orders the
    # *rename*, not the data, and fsync needs a writable descriptor on Windows.
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if keep_backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def load_resumable(path: str | Path, map_location="cpu") -> dict | None:
    """Load a checkpoint, falling back to its `.bak` if the primary is corrupt.

    Returns None if neither can be read — the caller should then start fresh
    rather than die, since this runs unattended after a reboot.
    """
    path = Path(path)
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.exists():
            continue
        try:
            return torch.load(candidate, map_location=map_location,
                              weights_only=False)
        except Exception as exc:                       # truncated / torn file
            print(f"  checkpoint {candidate.name} unreadable ({exc}); "
                  f"trying older generation")
    return None


def load_policy(path: str | Path, device: str | None = None, rules=None):
    """Rebuild the policy stored at `path` (config travels with the weights)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    rules = rules or load_ruleset()
    cfg = NetConfig(**blob["cfg"])
    net = PolyNet(cfg).to(device)
    net.load_state_dict(blob["model"])
    net.eval()
    return ARPolicy(net, rules, device=device), blob.get("iter", -1)


class CheckpointAgent:
    """A saved network, playing through the standard `act(game)` interface."""

    def __init__(self, path: str | Path, device: str | None = None,
                 greedy: bool = False, rules=None):
        self.policy, self.iter = load_policy(path, device, rules)
        self.greedy = greedy

    def act(self, game):
        actions, _ = self.policy.act([(game.state, game.current_player)],
                                     greedy=self.greedy)
        return actions[0]
