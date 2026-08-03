"""Differential replay checking — the ruleset fidelity gate.

Two uses, one mechanism:

1. **Determinism.** Re-simulate a recorded action sequence and assert the engine
   reproduces it frame for frame. Catches accidental nondeterminism (dict order,
   RNG draws, uninitialised memory) and unintended rule changes — a diff here
   after a refactor means behaviour moved.

2. **Fidelity vs. the real game.** A Steam-captured game, exported to the same
   `polyzero-replay/1` shape, is compared against the engine replaying those same
   actions. The first divergent frame localises exactly which rule is wrong —
   this is what turns "the wiki says X" into a test.

`compare_replays` returns a structured Divergence (or None), naming the frame,
the field, and the differing tiles, so a failure points at a rule rather than a
hash mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

from polyengine.core.actions import Action
from polyengine.core.game import Game

# Frame fields that must match exactly, in the order they're checked.
COMPARED_FIELDS = (
    "turn", "cur", "stars", "tech", "alive",
    "peace", "peace_offer", "attacked_now", "attacked_prev",
    "owner", "resource", "improvement",
    "city_owner", "city_level", "city_pop", "city_walls", "city_capital",
    "city_workshop", "city_park", "city_pending_reward", "city_conn_pop",
    "improvement_owner",
    "unit_type", "unit_owner", "unit_hp", "unit_veteran", "unit_home",
    "unit_carried", "unit_hidden", "unit_facing",
)


@dataclass
class Divergence:
    frame: int
    field: str
    detail: str
    expected: object = None
    actual: object = None

    def __str__(self) -> str:
        return f"frame {self.frame}: {self.field} differs — {self.detail}"


def _diff_field(name: str, exp, act) -> str | None:
    if isinstance(exp, list) and isinstance(act, list):
        if len(exp) != len(act):
            return f"length {len(exp)} != {len(act)}"
        bad = [i for i, (a, b) in enumerate(zip(exp, act)) if a != b]
        if not bad:
            return None
        head = ", ".join(f"[{i}] {exp[i]}!={act[i]}" for i in bad[:6])
        more = f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""
        return f"{len(bad)} entries: {head}{more}"
    return None if exp == act else f"{exp!r} != {act!r}"


def compare_replays(expected: dict, actual: dict,
                    fields: tuple[str, ...] = COMPARED_FIELDS) -> Divergence | None:
    """First point where two replays disagree, or None if they match."""
    for key in ("size", "n_players"):
        if expected[key] != actual[key]:
            return Divergence(-1, key, f"{expected[key]} != {actual[key]}")
    if expected["terrain"] != actual["terrain"]:
        d = _diff_field("terrain", expected["terrain"], actual["terrain"])
        return Divergence(-1, "terrain", d or "differs")

    ef, af = expected["frames"], actual["frames"]
    for i in range(min(len(ef), len(af))):
        ea, aa = ef[i].get("action"), af[i].get("action")
        if (ea is None) != (aa is None):
            return Divergence(i, "action", f"{ea} vs {aa}")
        if ea and aa:
            for k in ("type", "src", "dst", "subtype"):
                if ea[k] != aa[k]:
                    return Divergence(i, f"action.{k}", f"{ea[k]} != {aa[k]}",
                                      ea.get("desc"), aa.get("desc"))
        for name in fields:
            d = _diff_field(name, ef[i][name], af[i][name])
            if d:
                return Divergence(i, name, d)
    if len(ef) != len(af):
        return Divergence(min(len(ef), len(af)), "frames",
                          f"length {len(ef)} != {len(af)}")
    return None


def actions_of(replay: dict) -> list[Action]:
    """Extract the action sequence from a replay (skips the initial frame)."""
    return [Action(f["action"]["type"], f["action"]["src"],
                   f["action"]["dst"], f["action"]["subtype"])
            for f in replay["frames"] if f.get("action")]


def resimulate(replay: dict, strict: bool = True) -> dict:
    """Replay a recorded action sequence through a fresh engine instance.

    With `strict`, every recorded action must still be legal when re-applied —
    an illegality means the engine's legality contract moved, which is exactly
    what the gate exists to catch.
    """
    from polyengine.replay import ReplayRecorder

    # Tribes and map type are as load-bearing as the seed: rebuilding a
    # continents game as the default drylands two-tribe map produces a
    # different board, and every recorded action then looks illegal. The turn
    # cap belongs to the same list — it varies with size and seat count, so a
    # long game rebuilt under the default ends early and diverges.
    game = Game.new(size=replay["size"], seed=replay["map_seed"],
                    tribes=list(replay.get("tribes") or []) or None,
                    map_type=replay.get("map_type", "drylands"),
                    turn_cap=replay.get("turn_cap"))
    rec = ReplayRecorder(game, agent_names=replay.get("agents", []))
    rec.capture()
    for n, action in enumerate(actions_of(replay)):
        if strict and action.atype != 0 and action not in game.legal_actions():
            raise AssertionError(
                f"action {n} ({action.describe(game.rules)}) is no longer legal")
        rec.step(action)
    return rec.to_dict()


def check_determinism(replay: dict) -> Divergence | None:
    """Re-simulate `replay` and return the first divergence, if any."""
    return compare_replays(replay, resimulate(replay))
