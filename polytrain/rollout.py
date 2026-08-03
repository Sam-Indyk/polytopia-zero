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

    def to_numpy(self) -> dict:
        """Flatten to plain NumPy arrays — the form that crosses a process boundary.

        Worker processes return this; the parent concatenates shards and uploads
        once. Sending tensors instead would pickle CUDA handles or force
        per-shard device transfers.
        """
        out = {}
        for k in STORE_KEYS:
            arr = self.data[k]
            out[k] = np.concatenate([
                a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
                for a in arr])
        for k in ("source", "target", "subtype"):
            out[f"applies_{k}"] = np.concatenate(
                [a.detach().cpu().numpy() for a in self.applies[k]])
        out["old_logp"] = np.concatenate([v.detach().cpu().numpy() for v in self.logp])
        out["old_value"] = np.concatenate([v.detach().cpu().numpy() for v in self.value])
        return out

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


DEFAULT_MAP_TYPES = ("drylands", "lakes", "continents", "archipelago")


def _make_game(size: int, seed: int, map_types, spec=None) -> Game:
    """One game, with its world chosen from the rotation by seed.

    Cycling map types across the batch is what teaches the policy that the plan
    depends on the world: fishing, Ports and boats on an archipelago, Riders and
    roads on drylands. Nothing announces the type: the observation carries water
    tile by tile and the policy has to infer the shape of the world from that,
    the same way a player does on turn one with most of the map still dark.

    A `spec` from `polytrain.curriculum` additionally fixes the board size, the
    seat count and which tribes are dealt — the axes that used to be frozen at
    an 11x11 Imperius-versus-Bardur duel for the whole of training.
    """
    if spec is not None:
        return Game.new(size=spec.size, seed=seed, map_type=spec.map_type,
                        tribes=list(spec.tribes))
    kind = map_types[seed % len(map_types)]
    return Game.new(size=size, seed=seed, map_type=kind)


def collect(policy, n_games: int, rollout_steps: int, size: int = 9,
            seed: int = 0, gamma: float = 0.999, lam: float = 0.95,
            shaping: float = 0.0, max_turns: int = 100,
            games: list[Game] | None = None,
            opponent=None, learner_seats: list[int] | None = None,
            map_types=DEFAULT_MAP_TYPES, specs=None
            ) -> tuple[Rollout, dict, list[Game]]:
    """Play `rollout_steps` micro-steps across `n_games` games.

    With `opponent=None` this is mirror self-play and *both* seats' transitions
    are trained on. With an `opponent` (a frozen snapshot or a scripted
    baseline), only the seat in `learner_seats[i]` is stored — the opponent's
    moves shape the environment but must not be treated as the learner's own
    decisions. `opponent` is called with a list of Game objects and returns one
    Action per game.
    """
    rules = policy.rules
    if games is None:
        games = [_make_game(size, seed + i, map_types,
                            specs[i] if specs else None)
                 for i in range(n_games)]
    if opponent is not None and learner_seats is None:
        # Which seat the learner holds. Round-robin over the seats that game
        # actually has: `i % 2` silently pinned every game to seat 0 or 1, so on
        # a six-player board four tribes were never learned from and the learner
        # never once sat in them.
        learner_seats = [i % g.state.n_players for i, g in enumerate(games)]
    roll = Rollout()
    # per (game, player): index into roll of that trajectory's previous transition
    last_idx: dict[tuple[int, int], int] = {}
    prev_material: dict[tuple[int, int], float] = {}
    stats = {"episodes": 0, "wins": defaultdict(int), "lengths": []}
    steps_played = [0] * len(games)

    def is_learner(gi: int, player: int) -> bool:
        return opponent is None or player == learner_seats[gi]

    for _ in range(rollout_steps):
        live = [i for i, g in enumerate(games) if not g.is_terminal()]
        if not live:
            break

        mine = [i for i in live if is_learner(i, games[i].current_player)]
        theirs = [i for i in live if i not in set(mine)]

        actions_by_game: dict[int, Action] = {}
        info = None
        if mine:
            pairs = [(games[i].state, games[i].current_player) for i in mine]
            acts, info = policy.act(pairs)
            actions_by_game.update(dict(zip(mine, acts)))
        if theirs:
            # Opponents receive Game objects, not (state, player) pairs, so a
            # scripted baseline and a frozen network satisfy the same interface.
            opp_acts = opponent([games[i] for i in theirs])
            actions_by_game.update(dict(zip(theirs, opp_acts)))

        for slot, gi in enumerate(mine):
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

        for gi in live:
            game = games[gi]
            game.step(actions_by_game[gi])
            steps_played[gi] += 1
            # `max_turns=None` defers to the game's own cap, which now scales
            # with board size and seat count — a fixed 100 would cut every large
            # or crowded game off long before it could be decided.
            limit = game.state.turn_cap if max_turns is None else max_turns
            if game.is_terminal() or game.state.turn >= limit:
                seat = learner_seats[gi] if opponent is not None else None
                _finish(game, gi, roll, last_idx, stats, steps_played[gi], seat)
                # Replacements keep the same world: a curriculum stage is a
                # property of the iteration, not of one episode ending early.
                games[gi] = _make_game(
                    size, seed + 10_000 + gi * 7919 + stats["episodes"],
                    map_types, specs[gi] if specs else None)
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


def _finish(game, gi, roll, last_idx, stats, length, learner_seat=None):
    """Credit terminal reward to each stored player's final transition."""
    winner = game.winner() if game.is_terminal() else _score_winner(game)
    stats["episodes"] += 1
    stats["wins"][winner] += 1
    stats["lengths"].append(length)
    if learner_seat is not None:
        stats.setdefault("learner_result", []).append(
            1.0 if winner == learner_seat else (0.5 if winner < 0 else 0.0))
    for p in range(game.state.n_players):
        key = (gi, p)
        if key not in last_idx:
            continue
        idx = last_idx[key]
        roll.reward[idx] += 1.0 if winner == p else (-1.0 if winner >= 0 else 0.0)
        roll.terminal[idx] = True


def _score_winner(game) -> int:
    """Turn-cap tiebreak, using the ENGINE's score and nothing else.

    This used to call the material heuristic instead, which disagreed with
    `Game.score` on 15% of turn-capped games -- so training's terminal reward
    was pointing at a different objective from the one the engine adjudicates.
    Most 11x11 games end on the cap rather than by Domination, so that was a
    large slice of the signal.
    """
    scores = [game.score(p) for p in range(game.state.n_players)]
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
