"""Load a trained checkpoint as a playable agent.

Lets the network drop into the same places the scripted baselines go — the
arena, the replay recorder, the viewer — so "is it actually playing sensibly?"
is a question you answer by watching a game, not by reading a win rate.
"""
from __future__ import annotations

from pathlib import Path

import torch

from polyengine.core.rules import load_ruleset
from polynet import ARPolicy, NetConfig, PolyNet


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
