"""Flat/tree UCT baseline on the forward model.

A determinized, full-information UCT (it searches ground truth, ignoring fog) —
the "MCTS-1k/10k" rung of the baseline ladder. It leans on GameState.copy() being
cheap. Values are stored from the root player's perspective; at each node the
player to move picks the child that is best *for them* (max Q for the root
player, min Q for opponents). Rollouts are random and truncated, then scored by
the shared heuristic (or the exact result at a real terminal).

This is a baseline/opponent and an eval-time reference, NOT the M2 training loop
(which is PPO); the in-loop IS-MCTS booster in the plan is a separate, later thing.
"""
from __future__ import annotations

import math

import numpy as np

from polyengine.agents.heuristic import evaluate
from polyengine.core import constants as C
from polyengine.core.actions import Action


class _Node:
    __slots__ = ("game", "mover", "untried", "children", "N", "W")

    def __init__(self, game, rng):
        self.game = game
        self.mover = game.current_player
        self.untried = game.legal_actions() if not game.is_terminal() else []
        rng.shuffle(self.untried)
        self.children: dict[Action, _Node] = {}
        self.N = 0
        self.W = 0.0  # summed value, root-player perspective


class MCTSAgent:
    def __init__(self, n_sims: int = 200, c_puct: float = 1.4,
                 rollout_steps: int = 240, eval_scale: float = 40.0,
                 end_turn_bias: float = 0.2, seed: int = 0):
        self.n_sims = n_sims
        self.c = c_puct
        self.rollout_steps = rollout_steps
        self.eval_scale = eval_scale
        self.end_turn_bias = end_turn_bias
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def act(self, game) -> Action:
        if game.is_terminal():
            return Action(C.A_END_TURN)
        root_player = game.current_player
        root = _Node(game.clone(), self.rng)
        if len(root.untried) == 1 and not root.children:
            return root.untried[0]

        for _ in range(self.n_sims):
            path = [root]
            node = root
            # --- selection ---
            while not node.untried and node.children:
                node = self._select(node, root_player)
                path.append(node)
                if node.game.is_terminal():
                    break
            # --- expansion ---
            if node.untried and not node.game.is_terminal():
                a = node.untried.pop()
                child_game = node.game.clone()
                child_game.step(a)
                child = _Node(child_game, self.rng)
                node.children[a] = child
                path.append(child)
                node = child
            # --- simulation + backprop ---
            value = self._rollout(node.game, root_player)
            for nd in path:
                nd.N += 1
                nd.W += value

        # Most-visited root action.
        best = max(root.children.items(), key=lambda kv: kv[1].N)
        return best[0]

    def _select(self, node: _Node, root_player: int) -> _Node:
        sign = 1.0 if node.mover == root_player else -1.0
        logN = math.log(node.N + 1)
        best, best_score = None, -1e18
        for child in node.children.values():
            q = (child.W / child.N) if child.N else 0.0
            score = sign * q + self.c * math.sqrt(logN / (child.N + 1))
            if score > best_score:
                best_score, best = score, child
        return best

    def _rollout(self, game, root_player: int) -> float:
        sim = game.clone()
        for _ in range(self.rollout_steps):
            if sim.is_terminal():
                break
            actions = sim.legal_actions()
            if not actions:
                sim.step(Action(C.A_END_TURN))
                continue
            non_end = [a for a in actions if a.atype != C.A_END_TURN]
            if not non_end or self.rng.random() < self.end_turn_bias:
                sim.step(Action(C.A_END_TURN))
            else:
                sim.step(non_end[int(self.rng.integers(len(non_end)))])
        if sim.is_terminal():
            w = sim.winner()
            return 1.0 if w == root_player else (-1.0 if w >= 0 else 0.0)
        return math.tanh(evaluate(sim, root_player) / self.eval_scale)
