"""Micro-step throughput benchmark: single-core and multi-process.

    python scripts/bench_throughput.py --size 11 --games 20
    python scripts/bench_throughput.py --parallel --workers 8 --games 200

Single-core measures the forward model with the Numba grid kernels in the hot
loops. `--parallel` measures aggregate self-play throughput via
polyengine.vec.parallel_rollout, which is the number that matters for feeding
the M2 PPO trainer.
"""
from __future__ import annotations

import argparse
import os
import time

from polyengine.agents.random_agent import RandomAgent
from polyengine.core.game import Game
from polyengine.vec import parallel_rollout

SINGLE_TARGET = 5_000
AGG_TARGET = 100_000


def bench_single(size: int, games: int, seed: int) -> tuple[int, float]:
    steps = 0
    t0 = time.perf_counter()
    for g in range(games):
        game = Game.new(size=size, seed=seed + g)
        agents = [RandomAgent(seed=g * 10 + p) for p in range(game.state.n_players)]
        while not game.is_terminal():
            game.step(agents[game.current_player].act(game))
            steps += 1
    return steps, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    # Warm the Numba JIT so the timing measures steady state, not compilation.
    warm = Game.new(size=args.size, seed=99_991)
    warm_agent = RandomAgent(seed=0)
    for _ in range(30):
        if warm.is_terminal():
            break
        warm.step(warm_agent.act(warm))

    if args.parallel:
        stats = parallel_rollout(n_games=args.games, size=args.size,
                                 workers=args.workers, seed=args.seed)
        print(f"{stats.total_games} games, size {args.size}, "
              f"{args.workers} workers: {stats.total_steps:,} micro-steps "
              f"in {stats.seconds:.2f}s")
        print(f"  {stats.steps_per_sec:,.0f} micro-steps/s aggregate")
        print(f"  {stats.steps_per_sec / args.workers:,.0f} /s/worker")
        met = stats.steps_per_sec >= AGG_TARGET
        print(f"  target: >={AGG_TARGET:,}/s aggregate ({'MET' if met else 'NOT MET'})")
        if not met:
            need = AGG_TARGET / max(1.0, stats.steps_per_sec / args.workers)
            print(f"  note: aggregate throughput is core-bound; at this per-worker "
                  f"rate the target needs ~{need:.0f} cores. On hybrid CPUs the "
                  f"per-worker rate drops once E-cores are recruited.")
    else:
        steps, dt = bench_single(args.size, args.games, args.seed)
        rate = steps / dt
        print(f"{args.games} games, size {args.size}: {steps:,} micro-steps in {dt:.2f}s")
        print(f"  {rate:,.0f} micro-steps/s (single core, Numba grid kernels)")
        print(f"  target: >={SINGLE_TARGET:,}/s/core "
              f"({'MET' if rate >= SINGLE_TARGET else 'below target'})")


if __name__ == "__main__":
    main()
