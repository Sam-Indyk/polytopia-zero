"""Play a game yourself, in the browser, against the network or a baseline.

    .venv\\Scripts\\python.exe scripts/play.py --opponent scripted --map-type lakes --open
    .venv\\Scripts\\python.exe scripts/play.py --opponent net --net checkpoints_big/best.pt --open

(Go through the venv. A bare `python` is the system interpreter and has no
`polyengine` on its path.)

The interesting parts live in `polyengine/viewer/session.py`; this is a CLI.

By default you see the board through **your own fog memory**, rebuilt from the
same `mem_*` arrays `obs.py` reads, so you and the network work from the same
information. `--reveal` turns that off for debugging.
"""
from __future__ import annotations

import argparse
import webbrowser

from polyengine.viewer.session import PlaySession, serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="scripted",
                    choices=["scripted", "crazy", "net", "osla", "mcts", "random"])
    ap.add_argument("--net", help="checkpoint (.pt) for --opponent net")
    ap.add_argument("--greedy", action="store_true", help="net plays argmax")
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-type", default="drylands",
                    help="drylands|lakes|continents|pangea|archipelago|waterworld")
    ap.add_argument("--seat", type=int, default=0, help="which seat you play")
    ap.add_argument("--reveal", action="store_true",
                    help="show ground truth instead of your own fog memory")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.opponent == "net" and not args.net:
        ap.error("--opponent net needs --net PATH")

    session = PlaySession(size=args.size, seed=args.seed, map_type=args.map_type,
                          seat=args.seat, opponent=args.opponent, net=args.net,
                          greedy=args.greedy, reveal=args.reveal)
    server = serve(session, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"you are {session.header()['tribes'][session.you]} "
          f"vs {args.opponent} on {args.size}x{args.size} {args.map_type}")
    print(f"playing at {url}   (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
