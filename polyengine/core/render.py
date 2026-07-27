"""ASCII renderer — the M0 "watch it play" surface.

render_ascii(state, rules) draws god's-eye ground truth. Pass `player=` to draw
that player's fogged view (unexplored = '.', explored-but-hidden shows dimmed
memory), which is what the observation encoder will later consume.

Per tile the glyph priority is unit > city > resource > terrain; the leading
character tags ownership (a player digit, '*' for a neutral village, ' ' else).
"""
from __future__ import annotations

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset


def _tile_cell(state, rules: Ruleset, i: int) -> str:
    if state.unit_type[i] != C.U_NONE:
        return f"{int(state.unit_owner[i])}{rules.unit_symbol[state.unit_type[i]]}"
    if state.city_level[i] > 0:
        owner = state.city_owner[i]
        tag = "*" if owner == C.NEUTRAL else str(int(owner))
        return f"{tag}#"
    if state.resource[i] != C.R_NONE:
        return f" {rules.resource_symbol[state.resource[i]]}"
    return f" {rules.terrain_symbol[state.terrain[i]]}"


def render_ascii(state, rules: Ruleset, player: int | None = None) -> str:
    s = state.size
    lines = []
    header = "    " + " ".join(f"{x:>2}" for x in range(s))
    lines.append(header)
    for y in range(s):
        row = [f"{y:>2} "]
        for x in range(s):
            i = y * s + x
            if player is not None and not state.fog_explored[player][i]:
                row.append(" .")
            elif player is not None and not state.fog_visible[player][i]:
                # Explored but not currently visible: show remembered city only.
                if state.mem_city_owner[player][i] != C.NEUTRAL:
                    row.append(f"{int(state.mem_city_owner[player][i])}#")
                else:
                    row.append(" :")
            else:
                row.append(_tile_cell(state, rules, i))
        lines.append(" ".join(row))
    return "\n".join(lines)


def scoreboard(game) -> str:
    parts = []
    for p in range(game.state.n_players):
        alive = "" if game.owns_city(p) else " (out)"
        parts.append(f"P{p}: score={game.score(p)} stars={int(game.state.stars[p])}{alive}")
    return " | ".join(parts)
