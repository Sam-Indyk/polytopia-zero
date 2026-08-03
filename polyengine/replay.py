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

import numpy as np

from polyengine.core import constants as C
from polyengine.core.actions import Action


# Mechanics the engine does not simulate, surfaced in the viewer so an empty
# panel never reads as "nothing happened".
#
# KEEP THIS HONEST IN BOTH DIRECTIONS. It went stale once and claimed roads,
# markets, naval embarking and cloaks were missing months after they landed,
# which is worse than saying nothing: the one panel whose entire job is to tell
# you what you are not seeing was the thing lying.
NOT_SIMULATED = [
    "special tribes (Aquarion, Polaris, Cymanti, ...) and their unique units",
    "tribe-specific technology replacements (Waterways, Polarism, Pescetism)",
]


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
        # When each building went down. The viewer reads it to age temples,
        # which are worth 100 points plus 50 for every three turns since.
        "improvement_turn": state.improvement_turn.tolist(),
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
        "unit_facing": state.unit_facing.tolist(),
        "stars": state.stars.tolist(),
        "tech": [int(t) for t in state.tech],       # bitmask per player
        "alive": state.alive.tolist(),
        # Tasks are latched: earned once, never revoked. `monument_built` is the
        # subset already placed, so the viewer can show "earned, unbuilt".
        "task_done": [int(v) for v in state.task_done],
        "monument_built": [int(v) for v in state.monument_built],
        "kills": state.kills.tolist(),
        "losses": state.losses.tolist(),
        # Tribe relations, as bitmasks over players (see core/diplomacy.py).
        "peace": [int(v) for v in state.peace],
        "peace_offer": [int(v) for v in state.peace_offer],
        "attacked_now": [int(v) for v in state.attacked_now],
        "attacked_prev": [int(v) for v in state.attacked_prev],
        "explored": [row.tolist() for row in state.fog_explored],
    }


def ruleset_header(rules) -> dict:
    """Every static rules table the viewer needs, in one place.

    Shared by the recorder and the interactive play server so the two can never
    describe the same ruleset differently. The viewer draws unit stats, build
    costs, tech prerequisites and the "what does this unlock" lines in the tech
    tree straight out of this — nothing about the rules is hardcoded in JS.
    """
    return {
        "ruleset_version": rules.version,
        "unit_names": list(rules.unit_names),
        "unit_cost": [int(c) for c in rules.unit_cost],
        "unit_max_hp": [int(h) for h in rules.unit_max_hp],
        "unit_attack": [float(a) for a in rules.unit_attack],
        "unit_defense": [float(d) for d in rules.unit_defense],
        "unit_movement": [int(m) for m in rules.unit_movement],
        "unit_range": [int(v) for v in rules.unit_range],
        "unit_naval": [bool(b) for b in rules.unit_naval],
        "unit_trainable": [bool(b) for b in rules.unit_trainable],
        "unit_requires_tech": [int(t) for t in rules.unit_requires_tech],
        # Sorted so the viewer's skill chips are in a stable order.
        "unit_skills": [sorted(s) for s in rules.unit_skills],
        "terrain_names": list(rules.terrain_names),
        "terrain_requires_tech": [int(t) for t in rules.terrain_requires_tech],
        "terrain_move_cost": [int(c) for c in rules.terrain_move_cost],
        "terrain_rough": [bool(b) for b in rules.terrain_rough],
        "terrain_defense_bonus": [float(b) for b in rules.terrain_defense_bonus],
        "terrain_defense_tech": [int(t) for t in rules.terrain_defense_tech],
        "resource_names": list(rules.resource_names),
        "improvement_names": list(rules.imp_names),
        "imp_kind": list(rules.imp_kind),
        "imp_cost": [int(c) for c in rules.imp_cost],
        "imp_population": [int(p) for p in rules.imp_population],
        "imp_requires_tech": [int(t) for t in rules.imp_requires_tech],
        "imp_map_feature": [bool(b) for b in rules.imp_map_feature],
        "reward_names": list(rules.reward_names),
        "tech_names": list(rules.tech_names),
        "tech_tier": [int(t) for t in rules.tech_tier],
        "tech_requires": [int(t) for t in rules.tech_requires],
        "task_names": [C.TASK_NAMES[t] for t in range(C.N_TASKS)],
        "tile_ability_names": [C.TILE_ABILITY_NAMES[t]
                               for t in range(C.N_TILE_ABILITIES)],
        # Which technology each ability is behind, and which reveal a resource
        # that is hidden until then. The tech tree lists both as unlocks, so a
        # tech that "does nothing" in the viewer really does nothing.
        "tile_ability_tech": {C.TILE_ABILITY_NAMES[t]: C.TILE_ABILITY_TECH[t]
                              for t in range(C.N_TILE_ABILITIES)},
        "hidden_resource_tech": {rules.resource_names[r]: list(techs)
                                 for r, techs in C.HIDDEN_RESOURCE_TECHS.items()},
        "action_names": [C.ACTION_NAMES[t] for t in range(C.N_ACTION_TYPES)],
        # Capture and Examine both require a unit that has not yet acted this
        # turn, so moving onto a village and taking it are two separate turns.
        # The play UI explains that when the button is missing — gated on the
        # flag rather than assumed, so turning the rule off silences the hint.
        "capture_requires_fresh_unit": bool(rules.capture_requires_fresh_unit),
    }


def map_header(state, rules) -> dict:
    """The static map: what generation fixed and play never changes."""
    tribe_names = [name for name, _ in
                   sorted(rules.tribes.items(), key=lambda kv: kv[1]["id"])]
    return {
        "size": int(state.size),
        "n_players": int(state.n_players),
        "map_seed": int(state.map_seed),
        # Needed to re-simulate: a replay of a continents game
        # rebuilt as drylands is a different map entirely. The turn cap is here
        # for exactly the same reason — it varies with board size and seat
        # count now, and rebuilding a 400-turn game under a 100-turn cap ends it
        # early and reports a divergence with no visible cause. That is not
        # hypothetical: this header lost `map_type` once and the differential
        # gate failed on action 0 for months without a test noticing.
        "map_type": state.map_type,
        "turn_cap": int(state.turn_cap),
        "terrain": state.terrain.tolist(),
        # Which tribe's biome each tile is. Fixed at generation and
        # never changes hands, unlike `owner`.
        "tile_tribe": state.tile_tribe.tolist(),
        "tribe_names": list(tribe_names),
        "tribes": [tribe_names[t] if 0 <= t < len(tribe_names) else str(t)
                   for t in state.tribe.tolist()],
    }


def board_frame(game) -> dict:
    """One frame: the mutable board plus the derived numbers the viewer shows.

    Score is the one thing here the state does not store — it is `Game.score`,
    the published Polytopia formula, and it is what decides a turn-capped game.
    Recording it per frame means the viewer can graph the lead over time rather
    than only revealing who won at the end.
    """
    frame = _board(game.state)
    frame["score"] = [int(game.score(p)) for p in range(game.state.n_players)]
    return frame


def fog_frame(game, player: int) -> dict:
    """A frame containing only what `player` can actually know.

    This exists so a human can play the network on even terms. A replay frame is
    ground truth and the viewer merely *dims* what you have not explored — fine
    for studying a finished game, useless as a fair playing surface, because the
    truth is still sitting in the page for anyone who looks.

    So this rebuilds the frame the way `obs.py` builds the observation, from the
    same arrays: cities and units come from `mem_*` (last seen, which is why a
    remembered enemy has a type and an owner but no HP), and everything on an
    unexplored tile — terrain included — is erased rather than hidden. Invariant
    1 says the observation may only contain what a human can see; this is the
    same rule pointed the other way.
    """
    s = game.state
    f = board_frame(game)
    explored = s.fog_explored[player] == 1
    mem_ut, mem_uo = s.mem_unit_type[player], s.mem_unit_owner[player]
    mem_co, mem_cl = s.mem_city_owner[player], s.mem_city_level[player]

    terrain = s.terrain.copy()
    terrain[~explored] = -1                 # -1 renders as unexplored cloud
    f["terrain"] = terrain.tolist()
    f["explored"] = [explored.astype(int).tolist()]
    f["fog_of"] = 0                         # index into the single row above

    # Units: only what memory holds. HP is not remembered, so it is sent as -1
    # and the viewer draws no health bar rather than inventing one.
    live = (mem_ut != C.U_NONE) & explored
    for name, src in (("unit_type", mem_ut), ("unit_owner", mem_uo)):
        arr = np.where(live, src, -1 if name == "unit_owner" else C.U_NONE)
        f[name] = arr.tolist()
    own_unit = live & (mem_uo == player)
    for name in ("unit_hp", "unit_moved", "unit_attacked", "unit_veteran",
                 "unit_kills", "unit_home", "unit_carried", "unit_hidden"):
        arr = np.asarray(f[name])
        # Your own units you know completely; a remembered enemy is a silhouette.
        f[name] = np.where(own_unit, arr, np.where(live, -1, arr * 0 - 1)).tolist()

    cities = (mem_cl > 0) & explored
    f["city_owner"] = np.where(cities, mem_co, -1).tolist()
    f["city_level"] = np.where(cities, mem_cl, 0).tolist()
    own_city = cities & (mem_co == player)
    for name in ("city_pop", "city_walls", "city_capital", "city_workshop",
                 "city_park", "city_conn_pop", "city_pending_reward"):
        f[name] = np.where(own_city, np.asarray(f[name]), 0).tolist()

    for name in ("owner", "resource", "improvement", "improvement_owner",
                 "improvement_turn"):
        blank = 0 if name == "resource" else -1
        f[name] = np.where(explored, np.asarray(f[name]), blank).tolist()

    # Opponent stars are not observable — see invariant 1. Neither is their
    # score, their research, nor their running kill count, all of which are
    # computed from things you cannot count from your own screen. -1 means
    # "unknown" and the panels print it as a question mark.
    for name, unknown in (("stars", -1), ("score", -1), ("kills", -1),
                          ("losses", -1), ("task_done", 0),
                          ("monument_built", 0), ("tech", 0)):
        f[name] = [v if p == player else unknown for p, v in enumerate(f[name])]
    return f


@dataclass
class ReplayRecorder:
    """Records a game to a JSON replay. Call `capture` before the first action."""
    game: object
    agent_names: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    header: dict = field(default_factory=dict)

    def __post_init__(self):
        s, r = self.game.state, self.game.rules
        self.header = {
            "format": "polyzero-replay/1",
            **map_header(s, r),
            **ruleset_header(r),
            "agents": list(self.agent_names),
            # Mechanics the engine does not simulate yet, surfaced in the viewer
            # so an empty panel never reads as "nothing happened".
            "not_simulated": NOT_SIMULATED,
        }

    def capture(self, action: Action | None = None) -> None:
        frame = board_frame(self.game)
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
