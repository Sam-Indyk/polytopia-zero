"""M0 watch target: a random-vs-random game printed turn by turn.

    python scripts/demo_random.py --size 11 --seed 7 --show-every 10
"""
from __future__ import annotations

import argparse

from polyengine.agents.random_agent import RandomAgent
from polyengine.core import constants as C
from polyengine.core.game import Game
from polyengine.core.render import render_ascii, scoreboard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tribes", nargs="+", default=["imperius", "bardur"])
    ap.add_argument("--show-every", type=int, default=10,
                    help="print the board every N rounds (0 = only final)")
    ap.add_argument("--max-actions", type=int, default=200_000)
    args = ap.parse_args()

    game = Game.new(size=args.size, tribes=args.tribes, seed=args.seed)
    agents = [RandomAgent(seed=1000 + p) for p in range(game.state.n_players)]

    last_shown = -1
    for _ in range(args.max_actions):
        if game.is_terminal():
            break
        turn = game.state.turn
        if args.show_every and turn != last_shown and turn % args.show_every == 0:
            last_shown = turn
            print(f"\n=== round {turn} | {scoreboard(game)} ===")
            print(render_ascii(game.state, game.rules))
        action = agents[game.current_player].act(game)
        game.step(action)

    print(f"\n=== FINAL (round {game.state.turn}) | {scoreboard(game)} ===")
    print(render_ascii(game.state, game.rules))
    w = game.winner()
    print(f"\nWinner: {'player ' + str(w) if w >= 0 else 'undecided/draw'}"
          f" after {game.state.turn} rounds.")


if __name__ == "__main__":
    main()
