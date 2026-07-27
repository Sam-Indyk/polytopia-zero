"""Vectorized environments.

Two surfaces, matching how the plan uses them:

* ``SyncVecGame`` — an in-process vector of independent games with a gym-style
  reset/step/legal API. Deterministic, easy to test, the substrate the PPO
  rollout worker will drive in M2.

* ``parallel_rollout`` — mass self-play across worker processes to hit the
  aggregate-throughput target (>=100k micro-steps/s on a many-core box). Each
  worker owns a shard of games and plays them to completion with a picklable
  agent, so there is no per-step IPC; only aggregate stats come back. Numba's
  on-disk cache (cache=True on the kernels) means workers don't recompile.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable

from polyengine.agents.random_agent import RandomAgent
from polyengine.core.actions import Action
from polyengine.core.game import Game


class SyncVecGame:
    """A synchronous vector of independent games (single process)."""

    def __init__(self, n: int, size: int = 11, tribes: list[str] | None = None,
                 base_seed: int = 0, ruleset: str = "current"):
        self.n = n
        self.size = size
        self.tribes = tribes
        self.base_seed = base_seed
        self.ruleset = ruleset
        self.games: list[Game] = []
        self.reset()

    def reset(self) -> None:
        self.games = [
            Game.new(size=self.size, tribes=self.tribes,
                     seed=self.base_seed + i, ruleset=self.ruleset)
            for i in range(self.n)
        ]

    @property
    def current_players(self) -> list[int]:
        return [g.current_player for g in self.games]

    def legal_actions(self) -> list[list[Action]]:
        return [g.legal_actions() if not g.is_terminal() else [] for g in self.games]

    def step(self, actions: list[Action]) -> tuple[list[bool], list[dict]]:
        """Apply one action per game; auto-reset terminated games. Returns (dones, infos)."""
        dones, infos = [], []
        for i, (game, action) in enumerate(zip(self.games, actions)):
            if game.is_terminal():
                dones.append(True)
                infos.append({"terminal": True, "winner": game.winner()})
                continue
            info = game.step(action)
            done = game.is_terminal()
            if done:
                info = {**info, "terminal": True, "winner": game.winner()}
            dones.append(done)
            infos.append(info)
        return dones, infos

    def all_done(self) -> bool:
        return all(g.is_terminal() for g in self.games)


# --- mass parallel rollout ----------------------------------------------------
@dataclass
class RolloutStats:
    total_games: int
    total_steps: int
    seconds: float
    wins: dict[int, int]

    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / self.seconds if self.seconds else 0.0


_AGENTS: dict[str, Callable[[int], object]] = {
    "random": lambda seed: RandomAgent(seed=seed),
}


def _play_shard(args) -> tuple[int, int, dict]:
    n_games, size, tribes, seed0, agent_name, ruleset = args
    make_agent = _AGENTS[agent_name]
    steps = 0
    wins: dict[int, int] = {}
    for g in range(n_games):
        game = Game.new(size=size, tribes=tribes, seed=seed0 + g, ruleset=ruleset)
        agents = [make_agent(seed0 * 131 + g * 7 + p)
                  for p in range(game.state.n_players)]
        while not game.is_terminal():
            game.step(agents[game.current_player].act(game))
            steps += 1
        w = game.winner()
        wins[w] = wins.get(w, 0) + 1
    return n_games, steps, wins


def parallel_rollout(n_games: int, size: int = 11, workers: int = 4,
                     tribes: list[str] | None = None, seed: int = 0,
                     agent_name: str = "random", ruleset: str = "current") -> RolloutStats:
    """Play `n_games` random games across `workers` processes; report aggregate stats."""
    per = max(1, n_games // workers)
    shards = []
    assigned = 0
    for w in range(workers):
        k = per if w < workers - 1 else n_games - assigned
        if k <= 0:
            break
        shards.append((k, size, tribes, seed + assigned, agent_name, ruleset))
        assigned += k

    t0 = time.perf_counter()
    if workers == 1:
        results = [_play_shard(s) for s in shards]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_play_shard, shards)
    dt = time.perf_counter() - t0

    total_games = sum(r[0] for r in results)
    total_steps = sum(r[1] for r in results)
    wins: dict[int, int] = {}
    for _, _, w in results:
        for k, v in w.items():
            wins[k] = wins.get(k, 0) + v
    return RolloutStats(total_games, total_steps, dt, wins)
