"""The server-side half of play mode: one game, and the HTTP surface over it.

This lives in the package rather than in `scripts/play.py` so it can be tested.
`scripts/play.py` is a CLI and nothing else.

The whole design rests on one rule: **the client never names a move.** It posts
back an *index* into the legal action list the server just sent it, so there is
no vocabulary in which the browser could express an action the engine did not
offer. That is invariant 2 enforced by construction rather than by agreement —
and it is why the play UI needs no knowledge of movement, zone of control or
build costs, and why new action types appear in it for free.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from polyengine.core import constants as C
from polyengine.core.game import Game
from polyengine.replay import (NOT_SIMULATED, board_frame, fog_frame,
                               map_header, ruleset_header)

# A bot turn is many micro-actions and the client animates them so you can see
# where its army went. Past this many we stop collecting intermediate frames
# rather than post several megabytes; the true count is reported either way.
MAX_ANIMATED_STEPS = 60

# A bot that will not end its turn would otherwise hang the request.
MAX_OPPONENT_ACTIONS = 4000


def build_agent(kind: str, seed: int, net: str | None = None,
                greedy: bool = False):
    if kind == "net":
        if not net:
            raise ValueError("opponent 'net' needs a checkpoint path")
        from polytrain.checkpoint import CheckpointAgent
        return CheckpointAgent(net, greedy=greedy)
    if kind in ("scripted", "crazy"):
        from polyengine.agents.scripted import ScriptedBot
        return ScriptedBot.crazy(seed=seed) if kind == "crazy" else ScriptedBot(seed=seed)
    if kind == "osla":
        from polyengine.agents.osla import GreedyOSLA
        return GreedyOSLA(seed=seed)
    if kind == "mcts":
        from polyengine.agents.mcts import MCTSAgent
        return MCTSAgent(n_sims=100, seed=seed)
    from polyengine.agents.random_agent import RandomAgent
    return RandomAgent(seed=seed)


class PlaySession:
    """One game plus the log the viewer shows.

    Guarded by a lock because the HTTP server is threaded and a half-applied
    action is not a state anyone should be able to read.
    """

    def __init__(self, *, size=11, seed=0, map_type="drylands", seat=0,
                 opponent="scripted", net=None, greedy=False, reveal=False):
        self.size, self.map_type = size, map_type
        self.opponent, self.net, self.greedy = opponent, net, greedy
        self.seat, self.reveal, self.seed = seat, reveal, seed
        self.lock = threading.Lock()
        self.new_game(seed)

    def new_game(self, seed: int) -> None:
        self.seed = seed
        self.game = Game.new(size=self.size, seed=seed, map_type=self.map_type)
        self.you = self.seat % self.game.state.n_players
        self.agents = {}
        for p in range(self.game.state.n_players):
            if p == self.you:
                continue
            self.agents[p] = build_agent(self.opponent, seed * 17 + p,
                                         self.net, self.greedy)
            # Difficulty handicaps are applied once, before play. A bot that
            # never gets `apply_handicap` is just a normal bot wearing the name.
            handicap = getattr(self.agents[p], "apply_handicap", None)
            if handicap:
                handicap(self.game, p)
        self.log: list[dict] = []
        self.last_action = None

    # --- payloads --------------------------------------------------------
    def header(self) -> dict:
        s, r = self.game.state, self.game.rules
        return {
            **map_header(s, r), **ruleset_header(r),
            "you": self.you,
            "opponent": self.opponent,
            "agents": ["you" if p == self.you else self.opponent
                       for p in range(s.n_players)],
            "not_simulated": NOT_SIMULATED,
        }

    def frame(self) -> dict:
        return (board_frame(self.game) if self.reveal
                else fog_frame(self.game, self.you))

    def legal(self) -> list[dict]:
        """The indexed legal action list. The index is the client's whole
        vocabulary — see the module docstring."""
        if self.game.is_terminal() or self.game.current_player != self.you:
            return []
        return [{"i": i, "type": int(a.atype), "name": C.ACTION_NAMES[a.atype],
                 "src": int(a.src), "dst": int(a.dst), "subtype": int(a.subtype),
                 "desc": a.describe(self.game.rules)}
                for i, a in enumerate(self.game.legal_actions())]

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

    def _record(self, action, player: int) -> None:
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
            action = actions[index]
            player = int(self.game.current_player)
            self.game.step(action)
            self._record(action, player)
            return self.state()

    def run_opponents(self) -> dict:
        """Play every non-human seat until it is your move again."""
        with self.lock:
            steps, n = [], 0
            while (not self.game.is_terminal()
                   and self.game.current_player != self.you):
                p = int(self.game.current_player)
                action = self.agents[p].act(self.game)
                self.game.step(action)
                self._record(action, p)
                n += 1
                if len(steps) < MAX_ANIMATED_STEPS:
                    steps.append(self.state())
                if n > MAX_OPPONENT_ACTIONS:
                    break
            return {"steps": steps, "state": self.state(), "count": n}


def make_handler(session: PlaySession):
    from polyengine.viewer import render_play_html

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
            self._send(code, json.dumps(obj, separators=(",", ":")),
                       "application/json")

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
                        session.new_game(session.seed + 1)
                    self._json(session.state())
                elif self.path == "/api/reveal":
                    session.reveal = bool(body.get("on"))
                    self._json({"ok": True})
                else:
                    self._send(404, "not found", "text/plain")
            except Exception as exc:          # surfaced in the page as a toast
                self._json({"error": str(exc)}, 400)

    return Handler


def serve(session: PlaySession, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
