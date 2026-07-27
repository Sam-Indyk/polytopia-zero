"""Self-play rollout collection and GAE.

A subtlety this file exists to get right: in a turn-based game with interleaved
players, a player's *own* trajectory is not contiguous in wall-clock steps. Each
transition is therefore tagged with `(game, player)`, and advantages are
computed along those per-player chains — so the value target for one of P0's
decisions is P0's next decision, however many of P1's actions happened in
between. That is what "GAE runs across turn boundaries" means in the plan.

Rewards are terminal ±1 (Domination / turn-cap score tiebreak) plus an optional
small potential-based shaping term on the material heuristic, annealed to zero
early in training so the final objective is pure win/loss.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from polyengine.agents.heuristic import evaluate
from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.core.game import Game

STORE_KEYS = ("spatial", "global", "valid", "type_mask", "src_mask",
              "tgt_mask", "sub_mask", "atype", "src_idx", "tgt_idx", "sub_idx")


@dataclass
class Rollout:
    """Flat storage of transitions plus per-trajectory bookkeeping."""
    data: dict = field(default_factory=lambda: defaultdict(list))
    logp: list = field(default_factory=list)
    value: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    traj: list = field(default_factory=list)       # (game, player)
    applies: dict = field(default_factory=lambda: defaultdict(list))
    terminal: list = field(default_factory=list)   # trajectory ended here

    def __len__(self):
        return len(self.logp)

    def stack(self, device) -> dict:
        out = {}
        for k in STORE_KEYS:
            arr = self.data[k]
            out[k] = torch.cat([a.detach().cpu() for a in arr]).to(device) \
                if isinstance(arr[0], torch.Tensor) else torch.as_tensor(
                    np.concatenate(arr), device=device)
        for k in ("source", "target", "subtype"):
            out[f"applies_{k}"] = torch.cat(
                [a.detach().cpu() for a in self.applies[k]]).to(device)
        out["old_logp"] = torch.cat([v.detach().cpu() for v in self.logp]).to(device)
        out["old_value"] = torch.cat([v.detach().cpu() for v in self.value]).to(device)
        return out


def collect(policy, n_games: int, rollout_steps: int, size: int = 9,
            seed: int = 0, gamma: float = 0.999, lam: float = 0.95,
            shaping: float = 0.0, max_turns: int = 100,
            games: list[Game] | None = None) -> tuple[Rollout, dict, list[Game]]:
    """Play `rollout_steps` micro-steps across `n_games` self-play games."""
    rules = policy.rules
    if games is None:
        games = [Game.new(size=size, seed=seed + i) for i in range(n_games)]
    roll = Rollout()
    # per (game, player): index into roll of that trajectory's previous transition
    last_idx: dict[tuple[int, int], int] = {}
    prev_material: dict[tuple[int, int], float] = {}
    stats = {"episodes": 0, "wins": defaultdict(int), "lengths": []}
    steps_played = [0] * len(games)

    for _ in range(rollout_steps):
        live = [i for i, g in enumerate(games) if not g.is_terminal()]
        if not live:
            break
        pairs = [(games[i].state, games[i].current_player) for i in live]
        actions, info = policy.act(pairs)

        for slot, gi in enumerate(live):
            game = games[gi]
            player = game.current_player
            key = (gi, player)
            idx = len(roll)

            for k in STORE_KEYS:
                roll.data[k].append(info[k][slot:slot + 1])
            for k in ("source", "target", "subtype"):
                roll.applies[k].append(_applies(info["atype"][slot], k))
            roll.logp.append(info["logp"][slot:slot + 1])
            roll.value.append(info["value"][slot:slot + 1])
            roll.traj.append(key)
            roll.reward.append(0.0)
            roll.terminal.append(False)

            if shaping:
                mat = evaluate(game, player)
                if key in prev_material:
                    roll.reward[last_idx[key]] += shaping * (mat - prev_material[key])
                prev_material[key] = mat

            last_idx[key] = idx

        for slot, gi in enumerate(live):
            game = games[gi]
            game.step(actions[slot])
            steps_played[gi] += 1
            if game.is_terminal() or game.state.turn >= max_turns:
                _finish(game, gi, roll, last_idx, stats, steps_played[gi])
                games[gi] = Game.new(size=size, seed=seed + 10_000 + gi * 7919
                                     + stats["episodes"])
                steps_played[gi] = 0
                for p in range(game.state.n_players):
                    last_idx.pop((gi, p), None)
                    prev_material.pop((gi, p), None)

    advantages, returns = _gae(roll, policy, games, gamma, lam)
    return roll, {**stats, "adv": advantages, "ret": returns}, games


def _applies(atype, head) -> torch.Tensor:
    from polyengine import masks as M
    sets = {"source": M.NEEDS_SOURCE, "target": M.NEEDS_TARGET,
            "subtype": M.NEEDS_SUBTYPE}
    return torch.tensor([int(atype) in sets[head]], dtype=torch.bool)


def _finish(game, gi, roll, last_idx, stats, length):
    """Credit terminal reward to each player's final transition."""
    winner = game.winner() if game.is_terminal() else _score_winner(game)
    stats["episodes"] += 1
    stats["wins"][winner] += 1
    stats["lengths"].append(length)
    for p in range(game.state.n_players):
        key = (gi, p)
        if key not in last_idx:
            continue
        idx = last_idx[key]
        roll.reward[idx] += 1.0 if winner == p else (-1.0 if winner >= 0 else 0.0)
        roll.terminal[idx] = True


def _score_winner(game) -> int:
    """Turn-cap tiebreak on material."""
    scores = [evaluate(game, p) for p in range(game.state.n_players)]
    best = max(scores)
    leaders = [p for p, s in enumerate(scores) if s == best]
    return leaders[0] if len(leaders) == 1 else -1


def _gae(roll: Rollout, policy, games, gamma: float, lam: float):
    """Generalised advantage estimation along each (game, player) chain."""
    n = len(roll)
    values = np.array([float(v) for v in roll.value], dtype=np.float64)
    rewards = np.array(roll.reward, dtype=np.float64)
    adv = np.zeros(n, dtype=np.float64)

    chains: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(roll.traj):
        chains[key].append(i)

    for key, idxs in chains.items():
        last_gae = 0.0
        for pos in range(len(idxs) - 1, -1, -1):
            i = idxs[pos]
            if roll.terminal[i]:
                next_value, next_nonterminal = 0.0, 0.0
            elif pos + 1 < len(idxs):
                next_value, next_nonterminal = values[idxs[pos + 1]], 1.0
            else:
                # Segment boundary: bootstrap from the stored value (the game
                # continues into the next rollout).
                next_value, next_nonterminal = values[i], 1.0
            delta = rewards[i] + gamma * next_value * next_nonterminal - values[i]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[i] = last_gae
    returns = adv + values
    return adv, returns
