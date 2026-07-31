"""Tribe relations: war, neutral, peace — and what a treaty changes.

Sources: `.wikicache/Strategy.wiki` (the treaty itself), `.wikicache/Tribe
Relations.wiki` (the three-state relation), `.wikicache/Embassy.wiki`.

    "Against every other player, you are either at war ..., neutral, or at
     peace. To be at war, either side must have attacked the other side at most
     a turn ago. After that, the relation goes back to neutral. To be at peace
     with another tribe, a peace treaty must be sent and the other tribe must
     have accepted it."

    "The Peace Treaty offers another player the opportunity to become friends
     indefinitely. The Strategy technology is required to send, but not respond
     to, a peace treaty. While two tribes are at peace, they cannot perform any
     hostile actions, such as attacking, occupying cities, etc. Their units can
     also freely move past each other and use each other's roads. And income
     from Embassies is doubled. A peace treaty can be unilaterally terminated,
     but the player that breaks the peace treaty cannot attack the other player
     on that turn. Upon breaking the treaty, all of the aggressor's active units
     will be frozen for the rest of the turn, and any of their units within the
     enemy's borders will be disbanded without any star refund."

**War is not stored, it is remembered.** There is no "declare war" — attacking
*is* the declaration, and it lapses on its own after a turn. `attacked_now` and
`attacked_prev` are the two generations that implement "at most a turn ago";
they roll over at the start of each player's turn.

The treaty itself is symmetric and stored twice (a bit in each player's row), so
every query is a single mask test and no code has to remember which way round a
pair is written.
"""
from __future__ import annotations

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset


_ALL_BITS = 0xFFFFFFFF


def _bit(player: int) -> int:
    return 1 << int(player)


def _without(player: int) -> int:
    """Mask that clears `player`'s bit. Python's `~` yields a negative int, which
    NumPy refuses to store in a uint32 — so the complement is taken explicitly."""
    return _ALL_BITS ^ _bit(player)


# --- the three states ------------------------------------------------------------
def at_peace(state, a: int, b: int) -> bool:
    """Is there a treaty between these two? Symmetric, and never with yourself."""
    if a == b or a < 0 or b < 0:
        return False
    return bool(int(state.peace[a]) & _bit(b))


def at_war(state, a: int, b: int) -> bool:
    """"To be at war, either side must have attacked the other side at most a
    turn ago." Either direction counts, and it lapses by itself."""
    if a == b or a < 0 or b < 0:
        return False
    for p, q in ((a, b), (b, a)):
        recent = int(state.attacked_now[p]) | int(state.attacked_prev[p])
        if recent & _bit(q):
            return True
    return False


def relation(state, a: int, b: int) -> str:
    if at_peace(state, a, b):
        return "peace"
    return "war" if at_war(state, a, b) else "neutral"


def offered(state, a: int, b: int) -> bool:
    """Has `a` put a treaty on the table for `b`?"""
    if a == b or a < 0 or b < 0:
        return False
    return bool(int(state.peace_offer[a]) & _bit(b))


# --- bookkeeping -------------------------------------------------------------------
def record_attack(state, attacker: int, defender: int, rules: Ruleset) -> None:
    """Note a hostile act, which is the only way war is ever entered.

    Also settles the diplomatic fallout: "attacking a tribe (a unit or a city)
    will destroy the attacker's embassy in the defender's capital, if it exists,
    as well as the defender's embassy in the attacker's capital, if it exists."

    Called from `resolve_attack` (which covers the Mind Bender's conversion --
    "converting a unit with a Mind Bender is considered an attack, as it resets
    the Pacifist task and also destroys embassies") and from infiltration, which
    *is* the Cloak's attack.

    **Deliberately NOT called on capture.** Taking a city is obviously hostile,
    but the wiki defines war narrowly -- "either side must have attacked the
    other side" -- and lists "attacking" and "occupying cities" as two separate
    things a treaty forbids. Walking into an undefended city is occupying, not
    attacking. If a real game turns out to declare war on capture, this is the
    one line to change; it is left out rather than invented.
    """
    if attacker < 0 or defender < 0 or attacker == defender:
        return
    state.attacked_now[attacker] |= _bit(defender)
    _raze_embassies_between(state, attacker, defender)


def roll_turn(state, player: int) -> None:
    """Age `player`'s attack memory by one turn. Called when their turn begins."""
    state.attacked_prev[player] = state.attacked_now[player]
    state.attacked_now[player] = 0


def _raze_embassies_between(state, a: int, b: int) -> None:
    for tile in range(state.n_tiles):
        if int(state.improvement[tile]) != C.IMP_EMBASSY:
            continue
        builder = int(state.improvement_owner[tile])
        host = int(state.city_owner[tile])
        if {builder, host} == {a, b}:
            state.improvement[tile] = C.IMP_NONE
            state.improvement_owner[tile] = C.NEUTRAL
            state.improvement_turn[tile] = -1


# --- the treaty --------------------------------------------------------------------
def can_offer_peace(state, player: int, other: int, rules: Ruleset) -> bool:
    """May `player` offer `other` a treaty (or accept theirs) right now?

    "The Strategy technology is required to send, but **not respond to**, a
    peace treaty" — so answering an offer already on the table needs no tech,
    which is what lets a tribe that never researched Strategy still make peace.
    """
    if player == other or other < 0 or player < 0:
        return False
    if not state.alive[player] or not state.alive[other]:
        return False
    if at_peace(state, player, other):
        return False
    if offered(state, player, other):
        return False                     # already on the table; do not repeat
    if offered(state, other, player):
        return True                      # responding: no technology needed
    strategy = rules.tech_id.get("strategy", -1)
    return strategy >= 0 and state.has_tech(player, strategy)


def offer_peace(state, player: int, other: int, rules: Ruleset) -> dict:
    """Offer a treaty. If they have already offered, this concludes it."""
    if not can_offer_peace(state, player, other, rules):
        raise ValueError(f"player {player} cannot offer peace to {other}")
    if offered(state, other, player):
        state.peace[player] |= _bit(other)
        state.peace[other] |= _bit(player)
        state.peace_offer[other] &= _without(player)
        return {"type": "peace", "with": other, "accepted": True}
    state.peace_offer[player] |= _bit(other)
    return {"type": "peace_offer", "to": other, "accepted": False}


def can_break_peace(state, player: int, other: int) -> bool:
    return at_peace(state, player, other)


def break_peace(state, player: int, other: int, rules: Ruleset) -> dict:
    """Tear up a treaty, and pay for it immediately.

    "The player that breaks the peace treaty cannot attack the other player on
    that turn. Upon breaking the treaty, all of the aggressor's active units
    will be frozen for the rest of the turn, and any of their units within the
    enemy's borders will be disbanded without any star refund."

    That is a real price: you cannot betray and strike in the same turn, and the
    army you walked through their open borders is simply gone.
    """
    if not can_break_peace(state, player, other):
        raise ValueError(f"player {player} has no treaty with {other}")
    from polyengine.core.actions import _clear_unit

    state.peace[player] &= _without(other)
    state.peace[other] &= _without(player)
    state.peace_offer[player] &= _without(other)
    state.peace_offer[other] &= _without(player)

    disbanded, frozen = [], []
    for tile in range(state.n_tiles):
        if int(state.unit_owner[tile]) != player:
            continue
        if state.unit_type[tile] == C.U_NONE:
            continue
        if int(state.owner[tile]) == other:
            disbanded.append(tile)
            state.losses[player] += 1
            _clear_unit(state, tile)
            continue
        state.unit_moved[tile] = 1
        state.unit_attacked[tile] = 1
        frozen.append(tile)
    return {"type": "break_peace", "with": other,
            "disbanded": disbanded, "frozen": len(frozen)}


# --- what a treaty changes -----------------------------------------------------------
def friendly_to(state, player: int, other: int) -> bool:
    """Treat `other`'s units and land as passable for `player`?

    "Their units can also freely move past each other and use each other's
    roads." Your own tiles count, so callers can use one predicate.
    """
    return player == other or at_peace(state, player, other)


def embassy_income(state, tile: int, player: int, rules: Ruleset) -> int:
    """Stars this Embassy pays `player` per turn, 0 if it is not theirs.

    "+2 income to both sender and receiver. Doubled to +4 each if a Peace Treaty
    is in effect between both parties."
    """
    if int(state.improvement[tile]) != C.IMP_EMBASSY:
        return 0
    builder = int(state.improvement_owner[tile])
    host = int(state.city_owner[tile])
    if player not in (builder, host):
        return 0
    flat = int(rules.imp_income_flat[C.IMP_EMBASSY])
    return flat * 2 if at_peace(state, builder, host) else flat
