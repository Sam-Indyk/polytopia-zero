"""Rough single-core micro-step throughput probe.

Not the real VecGame benchmark (that arrives in M1); this just tells you whether
the pure-Python forward model is in the right ballpark on this machine and where
the hot loops are. Reports micro-actions applied per second under random play.
"""
from __future__ import annotations

import argparse
import time

from polyengine.agents.random_agent import RandomAgent
from polyengine.core.game import Game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    steps = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        game = Game.new(size=args.size, seed=args.seed + g)
        agents = [RandomAgent(seed=g * 10 + p) for p in range(game.state.n_players)]
        while not game.is_terminal():
            game.step(agents[game.current_player].act(game))
            steps += 1
    dt = time.perf_counter() - t0

    print(f"{args.games} games, size {args.size}: {steps} micro-steps in {dt:.2f}s")
    print(f"  {steps / dt:,.0f} micro-steps/s (single core, pure-Python path)")
    print("  (M1 target: >=5k steps/s/core once hot loops are Numba kernels + VecGame)")


if __name__ == "__main__":
    main()
