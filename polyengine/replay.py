"""Replay recording — the shared debugging/inspection format.

A replay is a JSON document: the static header (map, ruleset, tribes) plus one
frame per micro-action, each carrying the action taken and a compact snapshot of
the mutable board. Frames are ground-truth; the viewer can additionally dim
tiles a chosen player hasn't explored, so you can eyeball what the agent's fog
actually looked like — the same `mem_*` contract the observation encoder uses.

The format doubles as the fidelity harness's diff target: a Steam-captured game
and an engine-simulated one serialize to the same shape, so `polyengine.testing.
differential` can compare them frame by frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from polyengine.core import constants as C
from polyengine.core.actions import Action


def _board(state) -> dict:
    """Mutable per-frame board data (terrain lives in the static header).

    This is deliberately *everything* mutable the engine tracks, so the viewer
    can show the full state with nothing hidden: city level/pop/walls, unit
    hp/veteran/spent flags, per-player stars and the researched-tech bitmask,
    and the resource layer (which becomes mutable once harvesting lands).
    """
    return {
        "turn": int(state.turn),
        "cur": int(state.current_player),
        "owner": state.owner.tolist(),
        "resource": state.resource.tolist(),
        "improvement": state.improvement.tolist(),
        "city_owner": state.city_owner.tolist(),
        "city_level": state.city_level.tolist(),
        "city_pop": state.city_pop.tolist(),
        "city_walls": state.city_walls.tolist(),
        "city_capital": state.city_capital.tolist(),
        "city_workshop": state.city_workshop.tolist(),
        "city_park": state.city_park.tolist(),
        "city_conn_pop": state.city_conn_pop.tolist(),
        "improvement_owner": state.improvement_owner.tolist(),
        "city_pending_reward": state.city_pending_reward.tolist(),
        "unit_type": state.unit_type.tolist(),
        "unit_owner": state.unit_owner.tolist(),
        "unit_hp": state.unit_hp.tolist(),
        "unit_moved": state.unit_moved.tolist(),
        "unit_attacked": state.unit_attacked.tolist(),
        "unit_veteran": state.unit_veteran.tolist(),
        "unit_kills": state.unit_kills.tolist(),
        "unit_home": state.unit_home.tolist(),
        "unit_carried": state.unit_carried.tolist(),
        "unit_hidden": state.unit_hidden.tolist(),
        "stars": state.stars.tolist(),
        "tech": [int(t) for t in state.tech],       # bitmask per player
        "alive": state.alive.tolist(),
        "explored": [row.tolist() for row in state.fog_explored],
    }


@dataclass
class ReplayRecorder:
    """Records a game to a JSON replay. Call `capture` before the first action."""
    game: object
    agent_names: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    header: dict = field(default_factory=dict)

    def __post_init__(self):
        s, r = self.game.state, self.game.rules
        tribe_names = [name for name, _ in
                       sorted(r.tribes.items(), key=lambda kv: kv[1]["id"])]
        self.header = {
            "format": "polyzero-replay/1",
            "size": int(s.size),
            "n_players": int(s.n_players),
            "map_seed": int(s.map_seed),
            "ruleset_version": r.version,
            "terrain": s.terrain.tolist(),
            "tribes": [tribe_names[t] if 0 <= t < len(tribe_names) else str(t)
                       for t in s.tribe.tolist()],
            "unit_names": list(r.unit_names),
            "unit_cost": [int(c) for c in r.unit_cost],
            "unit_max_hp": [int(h) for h in r.unit_max_hp],
            "terrain_names": list(r.terrain_names),
            "resource_names": list(r.resource_names),
            "improvement_names": list(r.imp_names),
            "reward_names": list(r.reward_names),
            "tech_names": list(r.tech_names),
            "tech_tier": [int(t) for t in r.tech_tier],
            "tech_requires": [int(t) for t in r.tech_requires],
            "agents": list(self.agent_names),
            # Mechanics the engine does not simulate yet, surfaced in the viewer
            # so an empty panel never reads as "nothing happened".
            "not_simulated": [
                "markets, trade routes and city connections",
                "roads",
                "naval embarking via ports (the Port building exists; boarding does not)",
                "cloak infiltration and diplomacy relations",
                "special tribes",
            ],
        }

    def capture(self, action: Action | None = None) -> None:
        frame = _board(self.game.state)
        if action is not None:
            frame["action"] = {
                "type": int(action.atype),
                "name": C.ACTION_NAMES[action.atype],
                "src": int(action.src),
                "dst": int(action.dst),
                "subtype": int(action.subtype),
                "desc": action.describe(self.game.rules),
                "player": frame["cur"],
            }
        self.frames.append(frame)

    def step(self, action: Action) -> dict:
        """Apply an action to the game and record the resulting frame."""
        actor = int(self.game.state.current_player)
        info = self.game.step(action)
        self.capture(action)
        self.frames[-1]["action"]["player"] = actor
        return info

    def to_dict(self) -> dict:
        return {
            **self.header,
            "winner": int(self.game.winner()) if self.game.is_terminal() else -1,
            "terminal": bool(self.game.is_terminal()),
            "frames": self.frames,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")),
                        encoding="utf-8")
        return path


def record_game(game, agents, max_steps: int = 4000,
                agent_names: list[str] | None = None) -> ReplayRecorder:
    """Play `game` to completion with `agents`, recording every micro-action."""
    names = agent_names or [type(a).__name__ for a in agents]
    rec = ReplayRecorder(game, agent_names=names)
    rec.capture()
    steps = 0
    while not game.is_terminal() and steps < max_steps:
        rec.step(agents[game.current_player].act(game))
        steps += 1
    return rec
