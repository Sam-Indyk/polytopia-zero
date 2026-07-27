"""Benchmark PPO rollout throughput: single-process GPU vs multiprocess CPU.

    python scripts/bench_rollout.py --size 9 --steps 32

Rollout collection — not the gradient update — was the training bottleneck: the
single-process loop ran at ~2.3k micro-steps/s with the GPU idle 93% of the time,
slower than the bare engine on one core. This measures what the fixes bought.
"""
from __future__ import annotations

import argparse
import time

import torch

from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset
from polytrain.parallel import ParallelCollector
from polytrain.rollout import collect
from polytrain.train import build_policy


def bench_single(policy, n_games, steps, size, device):
    games = [Game.new(size=size, seed=i) for i in range(n_games)]
    collect(policy, n_games, 2, size=size, seed=0, games=games)   # warm
    if device.type == "cuda":
        torch.cuda.synchronize()
    games = [Game.new(size=size, seed=i) for i in range(n_games)]
    t0 = time.perf_counter()
    roll, _, _ = collect(policy, n_games, steps, size=size, seed=0, games=games)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return len(roll), time.perf_counter() - t0


def bench_parallel(cfg, net, n_games, steps, size, workers):
    col = ParallelCollector(cfg.__dict__, workers=workers)
    try:
        col.collect(net, n_games, 2, size=size, seed=0)            # warm + spawn
        t0 = time.perf_counter()
        _, stats = col.collect(net, n_games, steps, size=size, seed=1)
        return stats["n"], time.perf_counter() - t0
    finally:
        col.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--small", action="store_true", default=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--games", type=int, nargs="+", default=[64, 256])
    args = ap.parse_args()

    rules = load_ruleset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = build_policy(rules, device, args.small)

    print(f"size={args.size} rollout_steps={args.steps} net={cfg.channels}x{cfg.blocks}\n")
    print("  single-process (GPU inference)")
    for n in args.games:
        steps, dt = bench_single(policy, n, args.steps, args.size, device)
        print(f"    games={n:4d}: {steps:6d} steps in {dt:5.2f}s -> {steps/dt:8.0f} micro-steps/s")

    print("\n  multiprocess (CPU inference in workers)")
    for w in args.workers:
        for n in args.games:
            if n < w:
                continue
            steps, dt = bench_parallel(cfg, policy.net, n, args.steps, args.size, w)
            print(f"    workers={w:3d} games={n:4d} ({n//w:3d}/worker): "
                  f"{steps:6d} steps in {dt:5.2f}s -> {steps/dt:8.0f} micro-steps/s")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
