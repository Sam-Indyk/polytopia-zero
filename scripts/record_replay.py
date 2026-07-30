"""Play a game between two baselines and write a JSON replay + HTML viewer.

    python scripts/record_replay.py --p0 scripted --p1 random --size 11 --open

The HTML is self-contained: open it straight from disk, no server needed.
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from polyengine.agents.mcts import MCTSAgent
from polyengine.agents.osla import GreedyOSLA
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core.game import Game
from polyengine.replay import record_game
from polyengine.viewer import save_html

AGENTS = {
    "random": lambda seed: RandomAgent(seed=seed),
    "scripted": lambda seed: ScriptedBot(seed=seed),
    "osla": lambda seed: GreedyOSLA(seed=seed),
    "mcts": lambda seed: MCTSAgent(n_sims=100, seed=seed),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default="scripted")
    ap.add_argument("--p1", default="random")
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-type", default="drylands",
                    help="drylands|lakes|continents|pangea|archipelago|"
                         "waterworld -- how much water the map gets")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--out", default="replays/game")
    ap.add_argument("--open", action="store_true", help="open the viewer in a browser")
    ap.add_argument("--net", help="path to a checkpoint (.pt) to play as agent 'net'")
    ap.add_argument("--net-greedy", action="store_true")
    args = ap.parse_args()

    if args.net:
        from polytrain.checkpoint import CheckpointAgent
        agent = CheckpointAgent(args.net, greedy=args.net_greedy)
        AGENTS["net"] = lambda seed, _a=agent: _a
        print(f"loaded checkpoint {args.net} (iter {agent.iter})")

    for name in (args.p0, args.p1):
        if name not in AGENTS:
            ap.error(f"unknown agent {name!r}; choose from {sorted(AGENTS)}"
                     + ("" if args.net else " (pass --net PATH to enable 'net')"))

    game = Game.new(size=args.size, seed=args.seed,
                    map_type=args.map_type)
    names = [args.p0, args.p1]
    agents = [AGENTS[names[p % 2]](args.seed * 17 + p)
              for p in range(game.state.n_players)]

    rec = record_game(game, agents, max_steps=args.max_steps,
                      agent_names=[names[p % 2] for p in range(game.state.n_players)])
    data = rec.to_dict()

    out = Path(args.out)
    json_path = rec.save(out.with_suffix(".json"))
    html_path = save_html(data, out.with_suffix(".html"))

    print(f"{len(rec.frames)} frames · terminal={data['terminal']} · winner={data['winner']}")
    print(f"replay: {json_path}")
    print(f"viewer: {html_path}")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
