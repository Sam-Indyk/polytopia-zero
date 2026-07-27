"""Snapshot opponent pool — the minimal fix for self-play strategy cycling.

Pure mirror self-play optimises against *the current* opponent, so the policy is
free to drift in circles: it can unlearn how to beat an earlier style because
nothing in the objective remembers that style. The 6x6 smoke run showed exactly
this — 91.7% against ScriptedBot at iteration 100, then back to 47% by 150.

The standard remedy (and the base of the AlphaStar-lite league scheduled for M3)
is to keep frozen snapshots of past selves and play a fraction of games against
them. This module is that pool: a bounded, uniformly-sampled set of past weights
that the learner must stay good against, plus book-keeping for "best so far".

Deliberately *not* included: the scripted baselines. They are the M2 evaluation
gate, and training against your own benchmark measures nothing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SnapshotPool:
    """Frozen past policies the learner is periodically matched against."""
    capacity: int = 8
    snapshots: list[dict] = field(default_factory=list)   # state_dicts (CPU)
    iters: list[int] = field(default_factory=list)

    def add(self, net, iteration: int) -> None:
        snap = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        self.snapshots.append(snap)
        self.iters.append(iteration)
        if len(self.snapshots) > self.capacity:
            # Drop the oldest interior snapshot, always keeping the earliest
            # (a stable anchor) and the most recent ones.
            drop = 1 if len(self.snapshots) > 2 else 0
            self.snapshots.pop(drop)
            self.iters.pop(drop)

    def __len__(self) -> int:
        return len(self.snapshots)

    def sample(self, rng: np.random.Generator) -> tuple[dict, int]:
        i = int(rng.integers(len(self.snapshots)))
        return self.snapshots[i], self.iters[i]


class FrozenOpponent:
    """A past snapshot loaded into a throwaway net, used only for acting."""

    def __init__(self, template_net, state_dict, device):
        self.net = copy.deepcopy(template_net).to(device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def act_batch(self, policy_cls, rules, device, pairs, pad_to=None):
        from polynet.policy import ARPolicy
        pol = ARPolicy(self.net, rules, device=device, pad_to=pad_to)
        actions, _ = pol.act(pairs)
        return actions
