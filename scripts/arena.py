"""Round-robin arena for the baseline ladder.

    python scripts/arena.py --games 20 --size 9
    python scripts/arena.py --agents random scripted osla --games 40

Plays each pair with colours swapped (so map/first-move advantage cancels) and
reports a win-rate matrix. This is the ladder the PPO agent is measured against
in M2; RandomLegal is the Elo anchor.
"""
from __future__ import annotations

import argparse
import itertools
import time

from polyengine.agents.heuristic import evaluate
from polyengine.agents.mcts import MCTSAgent
from polyengine.agents.osla import GreedyOSLA
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core.game import Game

AGENTS = {
    "random": lambda seed: RandomAgent(seed=seed),
    "scripted": lambda seed: ScriptedBot(seed=seed),
    "osla": lambda seed: GreedyOSLA(seed=seed),
    "mcts": lambda seed: MCTSAgent(n_sims=50, seed=seed),   # ~0.25 s/move
}


def play_one(a_name: str, b_name: str, seed: int, size: int, max_steps: int) -> float:
    """Return 1.0 if `a` (player 0) wins, 0.0 if `b` wins, 0.5 for a draw."""
    game = Game.new(size=size, seed=seed)
    agents = [AGENTS[a_name](seed * 3 + 1), AGENTS[b_name](seed * 3 + 2)]
    steps = 0
    while not game.is_terminal() and steps < max_steps:
        game.step(agents[game.current_player].act(game))
        steps += 1
    if game.is_terminal():
        w = game.winner()
        return 1.0 if w == 0 else (0.0 if w == 1 else 0.5)
    # Truncated: score on material, the same signal the ladder optimises.
    margin = evaluate(game, 0)
    return 1.0 if margin > 1e-6 else (0.0 if margin < -1e-6 else 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["random", "scripted", "osla"],
                    choices=list(AGENTS))
    ap.add_argument("--games", type=int, default=20, help="games per ordered pair")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = args.agents
    score: dict[tuple[str, str], float] = {}
    t0 = time.perf_counter()
    for a, b in itertools.permutations(names, 2):
        pts = sum(play_one(a, b, args.seed + g, args.size, args.max_steps)
                  for g in range(args.games))
        score[(a, b)] = pts / args.games
        print(f"  {a:>9} (P0) vs {b:<9} : {pts / args.games:5.1%}")

    print(f"\nwin-rate matrix (row as P0, {args.games} games/pair, "
          f"{time.perf_counter() - t0:.1f}s)")
    print(f"{'':>10}" + "".join(f"{n:>10}" for n in names))
    for a in names:
        row = "".join("     --   " if a == b else f"{score[(a, b)]:>9.1%} "
                      for b in names)
        print(f"{a:>10}{row}")

    print("\noverall (both colours):")
    for n in names:
        played = [(x, y) for (x, y) in score if x == n or y == n]
        pts = sum(score[(x, y)] if x == n else 1 - score[(x, y)] for x, y in played)
        print(f"  {n:>9}: {pts / len(played):5.1%}")


if __name__ == "__main__":
    main()
