"""Play a game yourself, in the browser, against the network or a baseline.

    python scripts/play.py --opponent scripted --size 11 --map-type continents
    python scripts/play.py --net checkpoints_big/best.pt --open

Serves the viewer on localhost and drives one `Game` behind it. The browser
never decides what is legal: every button it draws is one entry from
`enumerate_actions`, so this is also a fairly brutal test of invariant 2 — an
action the engine enumerates but cannot apply shows up immediately as a button
that raises.

By default you see the board through **your own fog memory**, rebuilt from the
same `mem_*` arrays `obs.py` reads, so you and the network are working from the
same information. `--reveal` (or the checkbox) turns that off for debugging.
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from polyengine.core import constants as C
from polyengine.core.game import Game
from polyengine.replay import (NOT_SIMULATED, board_frame, fog_frame,
                               map_header, ruleset_header)
from polyengine.viewer import render_play_html

# A bot turn is many micro-actions; the client animates them so you can see
# where its army went. Past this many we stop collecting intermediate frames
# rather than post a several-megabyte turn — the count is reported either way.
MAX_ANIMATED_STEPS = 60


def build_agent(kind: str, seed: int, net: str | None, greedy: bool):
    if kind == "net":
        if not net:
            raise SystemExit("--opponent net needs --net PATH")
        from polytrain.checkpoint import CheckpointAgent
        return CheckpointAgent(net, greedy=greedy)
    if kind == "scripted":
        from polyengine.agents.scripted import ScriptedBot
        return ScriptedBot(seed=seed)
    if kind == "crazy":
        from polyengine.agents.scripted import ScriptedBot
        return ScriptedBot.crazy(seed=seed)
    if kind == "osla":
        from polyengine.agents.osla import GreedyOSLA
        return GreedyOSLA(seed=seed)
    if kind == "mcts":
        from polyengine.agents.mcts import MCTSAgent
        return MCTSAgent(n_sims=100, seed=seed)
    from polyengine.agents.random_agent import RandomAgent
    return RandomAgent(seed=seed)


class Session:
    """One game, plus the log the viewer shows. Guarded by a lock because the
    HTTP server is threaded and a half-applied action is not a state."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.reveal = args.reveal
        self.seed = args.seed
        self.new_game(self.seed)

    def new_game(self, seed: int):
        self.game = Game.new(size=self.args.size, seed=seed,
                             map_type=self.args.map_type)
        self.you = self.args.seat % self.game.state.n_players
        self.agents = {}
        for p in range(self.game.state.n_players):
            if p != self.you:
                self.agents[p] = build_agent(self.args.opponent, seed * 17 + p,
                                             self.args.net, self.args.greedy)
                # Difficulty handicaps are applied once, before play — a bot
                # that never gets `apply_handicap` is just a normal bot.
                h = getattr(self.agents[p], "apply_handicap", None)
                if h:
                    h(self.game, p)
        self.log: list[dict] = []
        self.last_action = None

    # --- payloads --------------------------------------------------------
    def header(self) -> dict:
        s, r = self.game.state, self.game.rules
        return {
            **map_header(s, r), **ruleset_header(r),
            "you": self.you,
            "opponent": self.args.opponent,
            "agents": [self.args.opponent if p != self.you else "you"
                       for p in range(s.n_players)],
            "not_simulated": NOT_SIMULATED,
        }

    def frame(self) -> dict:
        return (board_frame(self.game) if self.reveal
                else fog_frame(self.game, self.you))

    def legal(self) -> list[dict]:
        """The legal action list, indexed — the index is what the client posts
        back, so there is no way for it to name an action the engine did not
        just offer."""
        if self.game.is_terminal() or self.game.current_player != self.you:
            return []
        out = []
        for i, a in enumerate(self.game.legal_actions()):
            out.append({"i": i, "type": int(a.atype),
                        "name": C.ACTION_NAMES[a.atype],
                        "src": int(a.src), "dst": int(a.dst),
                        "subtype": int(a.subtype),
                        "desc": a.describe(self.game.rules)})
        return out

    def state(self) -> dict:
        g = self.game
        return {
            "frame": self.frame(),
            "cur": int(g.current_player),
            "terminal": bool(g.is_terminal()),
            "winner": int(g.winner()) if g.is_terminal() else -1,
            "legal": self.legal(),
            "last_action": self.last_action,
            "log": self.log[-200:],
        }

    def _record(self, action, player: int):
        self.last_action = {
            "name": C.ACTION_NAMES[action.atype],
            "src": int(action.src), "dst": int(action.dst),
            "desc": action.describe(self.game.rules), "player": int(player),
        }
        self.log.append({"action": dict(self.last_action),
                         "turn": int(self.game.state.turn)})

    # --- moves -----------------------------------------------------------
    def apply(self, index: int) -> dict:
        with self.lock:
            actions = self.game.legal_actions()
            if not 0 <= index < len(actions):
                raise ValueError("that action is no longer available — "
                                 "the board moved under the click")
            a = actions[index]
            player = int(self.game.current_player)
            self.game.step(a)
            self._record(a, player)
            return self.state()

    def run_opponents(self) -> dict:
        """Play every non-human seat until it is your move again."""
        with self.lock:
            steps, n = [], 0
            while (not self.game.is_terminal()
                   and self.game.current_player != self.you):
                p = int(self.game.current_player)
                a = self.agents[p].act(self.game)
                self.game.step(a)
                self._record(a, p)
                n += 1
                if len(steps) < MAX_ANIMATED_STEPS:
                    steps.append(self.state())
                if n > 4000:                      # a bot that will not end its turn
                    break
            return {"steps": steps, "state": self.state(), "count": n}


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                # keep the console quiet
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, separators=(",", ":")), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                # Rendered per request, not once at startup: the header carries
                # the terrain, and "restart game" deals a different map.
                self._send(200, render_play_html(session.header()), "text/html")
            elif self.path == "/api/state":
                self._json(session.state())
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or "{}")
            try:
                if self.path == "/api/act":
                    self._json(session.apply(int(body["index"])))
                elif self.path == "/api/ai":
                    self._json(session.run_opponents())
                elif self.path == "/api/new":
                    with session.lock:
                        session.seed += 1
                        session.new_game(session.seed)
                    self._json(session.state())
                elif self.path == "/api/reveal":
                    session.reveal = bool(body.get("on"))
                    self._json({"ok": True})
                else:
                    self._send(404, "not found", "text/plain")
            except Exception as exc:              # surfaced in the page as a toast
                self._json({"error": str(exc)}, 400)
    return Handler


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

    session = Session(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
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
