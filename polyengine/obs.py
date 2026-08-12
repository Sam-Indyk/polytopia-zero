"""Observation encoding — the frozen contract shared by sim and screen parser.

Everything the network sees comes from the acting player's **fog memory**, never
ground truth: unit and city planes are rendered from `mem_*` (last-seen
snapshots) plus a staleness scalar, which is exactly what the real client shows
a human. That constraint is deliberate — in M5 the vision layer must be able to
produce these same tensors from pixels, so the encoder may not use anything a
player cannot see on screen.

Layout (C x H x W float32, C = N_PLANES):
  terrain one-hot                     n_terrain
  resource one-hot                    n_resource
  improvement one-hot                 n_improvements  (harvests + buildings)
  territory                           4   own / enemy / neutral / off-board
  city                                9   present, own, enemy, level, pop, walls,
                                          capital, workshop, support-used fraction
  remembered units                    n_units + 6
                                          type one-hot (from memory)
                                          own, enemy, hp fraction, veteran, moved, attacked
  cloak_sense                         1   the eye symbol: one of OUR units has a
                                          hidden enemy Cloak on an adjacent tile.
                                          It narrows the Cloak to 8 tiles and no
                                          further -- exactly what a player sees.
  fog                                 3   visible, explored, staleness (normalised)
  geometry                            3   x, y, edge distance

Opponents occupy fixed relative slots ("own" vs "enemy") so the encoding is
perspective-invariant and self-play stays symmetric.

The global vector carries the non-spatial state: stars, tech bitmask as bits,
city/unit counts, turn progress, and the same for the opponent *as far as it is
known*.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset

# Staleness is squashed by this constant; ~20 turns saturates the plane.
STALENESS_SCALE = 20.0


def plane_layout(rules: Ruleset) -> dict[str, tuple[int, int]]:
    """Name -> (start, count) for every block of planes, for debugging/tests."""
    n_t, n_r, n_u = rules.n_terrain, len(rules.resource_names), rules.n_units
    n_i = rules.n_improvements
    blocks = [
        ("terrain", n_t), ("resource", n_r), ("improvement", n_i),
        ("territory", 4), ("city", 9),
        ("unit_type", n_u), ("unit_attrs", 6), ("cloak_sense", 1),
        ("promote_ready", 1),
        ("fog", 3), ("geometry", 3),
    ]
    out, start = {}, 0
    for name, count in blocks:
        out[name] = (start, count)
        start += count
    return out


def n_planes(rules: Ruleset) -> int:
    start, count = plane_layout(rules)["geometry"]
    return start + count


# --- per-opponent block -------------------------------------------------------
# One fixed-width row per seat, indexed by ABSOLUTE player id. That indexing is
# not a detail: `A_PEACE_OFFER` and `A_PEACE_BREAK` take a *player id* as their
# sub-type, so unless row k describes player k the policy is choosing who to
# make peace with out of a vector it cannot connect to the choice. The
# aggregate "share of opponents at war" that used to be the only relation
# signal is fine for sensing the mood of a game and useless for picking a
# target -- which is why diplomacy was unlearnable before this block existed.
#
# Every feature is something a human reads off their own screen: the tribe list
# shows who is still alive, the diplomacy panel shows treaties and pending
# offers, and the counts come from the player's own fog memory.
OPP_FEATURES = (
    "seat_exists",      # this slot is a real seat in this game
    "is_me",            # ... and it is the acting player
    "alive",
    "at_peace",
    "at_war",
    "offered_me",       # they have proposed a treaty to me
    "i_offered",        # I have proposed one to them
    "seen_cities",      # their cities in my memory, normalised
    "seen_units",
)
N_OPP_FEATURES = len(OPP_FEATURES)


def global_layout(rules: Ruleset) -> dict[str, tuple[int, int]]:
    """Name -> (start, count) for each block of the global vector.

    The spatial planes have had `plane_layout` for debugging since the start;
    the global vector was addressed by hand-computed strides scattered across
    the encoder and its tests, which is how `2 * stride + 3` ends up meaning
    something different in two places."""
    stride = 4 + rules.n_techs
    blocks = [
        ("own", stride),
        ("opponent_seen", stride),
        ("turn", 1), ("size", 1), ("peace_share", 1), ("war_share", 1),
        ("opponents", C.MAX_PLAYERS * N_OPP_FEATURES),
    ]
    out, start = {}, 0
    for name, count in blocks:
        out[name] = (start, count)
        start += count
    return out


def global_dim(rules: Ruleset) -> int:
    # own: stars, cities, units, techs-known + tech bits
    # opp: stars(known), cities seen, units seen, techs seen + tech bits
    # plus turn progress, map size, the two relation summaries, and one row per
    # seat (see OPP_FEATURES).
    #
    # Deliberately NOT here: any summary of how wet the world is, or which map
    # type is in play. The net sees water tile by tile in the spatial planes and
    # has to draw its own conclusion about whether this is a Ports-and-boats map
    # or a Riders-and-roads one. Handing it an aggregate would be a shortcut the
    # screen-reading agent of M5 has no way to reproduce.
    #
    # Relations ARE fair game: your own treaties are on your own screen, always,
    # and the fraction of opponents you are at peace or at war with is something
    # a player reads straight off the tribe list.
    start, count = global_layout(rules)["opponents"]
    return start + count


@njit(cache=True)
def _encode_spatial(size, player, n_terrain, n_resource, n_units, n_improvement,
                    mem_terrain, mem_resource, mem_improvement, mem_owner,
                    mem_city_owner, mem_city_level, city_pop, city_walls,
                    city_capital, city_workshop, city_support_frac,
                    mem_unit_type, mem_unit_owner, unit_hp, unit_veteran,
                    unit_moved, unit_attacked, unit_max_hp,
                    visible, explored, staleness,
                    out):
    n = size * size
    # block offsets
    o_terr = 0
    o_res = o_terr + n_terrain
    o_imp = o_res + n_resource
    o_terr_own = o_imp + n_improvement
    o_city = o_terr_own + 4
    o_utype = o_city + 9
    o_uattr = o_utype + n_units
    o_fog = o_uattr + 6
    o_geo = o_fog + 3

    inv = 1.0 / max(1.0, float(size - 1))
    for i in range(n):
        x = i % size
        y = i // size
        seen = explored[i] == 1

        # Terrain, resources and buildings, all from memory. These read ground
        # truth for every tile until now, which handed the player the entire
        # map on turn 0: every mountain and ocean, every ruin, before walking a
        # step. That also quietly falsified the design claim that the policy
        # infers the map type "water tile by tile, through fog" — it was seeing
        # all of the water immediately, and the full terrain grid *is* the map
        # type. An unexplored tile now sets no terrain, resource or improvement
        # plane at all.
        if seen:
            out[o_terr + mem_terrain[i], y, x] = 1.0
            out[o_res + mem_resource[i], y, x] = 1.0
            if mem_improvement[i] >= 0:
                out[o_imp + mem_improvement[i], y, x] = 1.0

        # Territory, from memory — the same contract as cities and units.
        # An unexplored tile sets none of the three, which reads as "unknown"
        # and is exactly correlated with the explored plane, so it leaks
        # nothing; asserting "neutral" there would be a lie.
        if seen:
            if mem_owner[i] == player:
                out[o_terr_own + 0, y, x] = 1.0
            elif mem_owner[i] >= 0:
                out[o_terr_own + 1, y, x] = 1.0
            else:
                out[o_terr_own + 2, y, x] = 1.0
        # slot 3 is the off-board mask, set by the padding helper

        # city, from memory
        if seen and mem_city_level[i] > 0:
            out[o_city + 0, y, x] = 1.0
            if mem_city_owner[i] == player:
                out[o_city + 1, y, x] = 1.0
            elif mem_city_owner[i] >= 0:
                out[o_city + 2, y, x] = 1.0
            out[o_city + 3, y, x] = mem_city_level[i] / 10.0
            if visible[i] == 1:          # pop/walls only readable when in sight
                out[o_city + 4, y, x] = city_pop[i] / 10.0
                out[o_city + 5, y, x] = 1.0 if city_walls[i] else 0.0
            out[o_city + 6, y, x] = 1.0 if city_capital[i] else 0.0
            if visible[i] == 1:
                out[o_city + 7, y, x] = 1.0 if city_workshop[i] else 0.0
                # How full this city's unit-support is — the scarce resource
                # that decides whether another unit can be trained at all.
                out[o_city + 8, y, x] = city_support_frac[i]

        # units, from memory
        ut = mem_unit_type[i]
        if seen and ut >= 0:
            out[o_utype + ut, y, x] = 1.0
            if mem_unit_owner[i] == player:
                out[o_uattr + 0, y, x] = 1.0
            else:
                out[o_uattr + 1, y, x] = 1.0
            if visible[i] == 1:
                mx = unit_max_hp[ut]
                if mx > 0:
                    out[o_uattr + 2, y, x] = unit_hp[i] / mx
                out[o_uattr + 3, y, x] = 1.0 if unit_veteran[i] else 0.0
                out[o_uattr + 4, y, x] = 1.0 if unit_moved[i] else 0.0
                out[o_uattr + 5, y, x] = 1.0 if unit_attacked[i] else 0.0

        # fog
        out[o_fog + 0, y, x] = 1.0 if visible[i] else 0.0
        out[o_fog + 1, y, x] = 1.0 if explored[i] else 0.0
        s = staleness[i] / 20.0
        out[o_fog + 2, y, x] = s if s < 1.0 else 1.0

        # geometry
        out[o_geo + 0, y, x] = x * inv
        out[o_geo + 1, y, x] = y * inv
        ex = x if x < size - 1 - x else size - 1 - x
        ey = y if y < size - 1 - y else size - 1 - y
        e = ex if ex < ey else ey
        out[o_geo + 2, y, x] = e * inv


def encode_spatial(state, player: int, rules: Ruleset,
                   pad_to: int | None = None) -> np.ndarray:
    """Encode `player`'s view as a (C, H, W) float32 tensor.

    `pad_to` bottom-right pads to a fixed size for size-bucketed batching; the
    off-board mask plane marks the padding so the conv trunk can ignore it.
    """
    size = state.size
    out_size = pad_to or size
    if out_size < size:
        raise ValueError(f"pad_to={pad_to} smaller than map size {size}")
    out = np.zeros((n_planes(rules), out_size, out_size), dtype=np.float32)
    # Fraction of each city's unit-support capacity already in use.
    support = np.zeros(state.n_tiles, dtype=np.float32)
    cities = np.flatnonzero(state.city_level > 0)
    if cities.size:
        used = (state.unit_home[:, None] == cities[None, :]).sum(axis=0)
        cap = state.city_level[cities].astype(np.float32) + 1.0
        support[cities] = np.minimum(used / cap, 1.0)
    _encode_spatial(
        size, player, rules.n_terrain, len(rules.resource_names), rules.n_units,
        rules.n_improvements,
        state.mem_terrain[player], state.mem_resource[player],
        state.mem_improvement[player], state.mem_owner[player],
        state.mem_city_owner[player], state.mem_city_level[player],
        state.city_pop, state.city_walls, state.city_capital,
        state.city_workshop, support,
        state.mem_unit_type[player], state.mem_unit_owner[player],
        state.unit_hp, state.unit_veteran, state.unit_moved, state.unit_attacked,
        rules.unit_max_hp.astype(np.float32),
        state.fog_visible[player], state.fog_explored[player],
        state.fog_staleness[player],
        out)
    layout = plane_layout(rules)

    # Resources the player has not unlocked are not on their screen at all:
    # "crop is hidden until Organization or Farming has been researched", and
    # Climbing "reveals metal". Scrubbing the plane is not enough -- the
    # one-hot has to fall back to "no resource", or the absence of every flag
    # would itself mark the tile as interesting.
    res_at = layout["resource"][0]
    for rid, techs in C.HIDDEN_RESOURCE_TECHS.items():
        if any(state.has_tech(player, rules.tech_id[t])
               for t in techs if t in rules.tech_id):
            continue
        concealed = out[res_at + rid] > 0
        if concealed.any():
            out[res_at + rid][concealed] = 0.0
            out[res_at + C.R_NONE][concealed] = 1.0

    from polyengine.core.fog import cloak_adjacent
    sense_at = layout["cloak_sense"][0]
    sensed = cloak_adjacent(state, player)
    if sensed.any():
        out[sense_at, :size, :size] = sensed.reshape(size, size)

    # Which of OUR units are sitting on an unclaimed promotion. The real client
    # shows this as a star on the unit, and it is a live decision every turn --
    # promoting is a free full heal, so the policy needs to see when one is
    # banked. Own units only: an enemy's kill count is not on screen.
    from polyengine.core.actions import can_promote
    ready_at = layout["promote_ready"][0]
    for tile in np.flatnonzero((state.unit_owner == player)
                               & (state.unit_type != C.U_NONE)):
        if can_promote(state, int(tile), player, rules):
            out[ready_at, int(tile) // size, int(tile) % size] = 1.0
    if out_size > size:
        off = plane_layout(rules)["territory"][0] + 3
        out[off, size:, :] = 1.0
        out[off, :, size:] = 1.0
    return out


def encode_global(state, player: int, rules: Ruleset,
                  max_turns: int = 100) -> np.ndarray:
    """Non-spatial features: own status, believed opponent status, turn progress."""
    vec = np.zeros(global_dim(rules), dtype=np.float32)
    n_techs = rules.n_techs

    def fill(base: int, stars, cities, units, techs_mask):
        vec[base + 0] = stars / 50.0
        vec[base + 1] = cities / 10.0
        vec[base + 2] = units / 20.0
        vec[base + 3] = bin(int(techs_mask)).count("1") / max(1, n_techs)
        for t in range(n_techs):
            if int(techs_mask) >> t & 1:
                vec[base + 4 + t] = 1.0

    own_cities = int(((state.city_owner == player) & (state.city_level > 0)).sum())
    own_units = int(((state.unit_owner == player) & (state.unit_type != C.U_NONE)).sum())
    fill(0, float(state.stars[player]), own_cities, own_units, state.tech[player])

    # Opponent block: only what this player could know. Stars are hidden in the
    # real client, so they are not included; counts come from fog memory.
    mem_city_owner = state.mem_city_owner[player]
    mem_unit_owner = state.mem_unit_owner[player]
    explored = state.fog_explored[player] == 1
    seen_cities = int((explored & (mem_city_owner >= 0)
                       & (mem_city_owner != player)
                       & (state.mem_city_level[player] > 0)).sum())
    seen_units = int((explored & (mem_unit_owner >= 0)
                      & (mem_unit_owner != player)
                      & (state.mem_unit_type[player] != C.U_NONE)).sum())
    stride = 4 + n_techs
    fill(stride, 0.0, seen_cities, seen_units, 0)

    # The turn cap is per game now (it scales with map size and seat count), so
    # progress is measured against *this* game's cap rather than a constant the
    # encoder happened to be built with.
    cap = float(getattr(state, "turn_cap", 0) or max_turns)
    vec[2 * stride + 0] = min(1.0, state.turn / cap)
    vec[2 * stride + 1] = state.size / 32.0
    # Own relations, which a player always knows: the share of live opponents
    # this player has a treaty with, and the share who attacked (or were
    # attacked by) them within the last turn.
    from polyengine.core.diplomacy import at_peace, at_war, offered
    others = [q for q in range(state.n_players)
              if q != player and state.alive[q]]
    if others:
        vec[2 * stride + 2] = sum(at_peace(state, player, q)
                                  for q in others) / len(others)
        vec[2 * stride + 3] = sum(at_war(state, player, q)
                                  for q in others) / len(others)

    # --- one row per seat, indexed by player id ---------------------------
    base = 2 * stride + 4
    live_city = explored & (state.mem_city_level[player] > 0)
    live_unit = explored & (state.mem_unit_type[player] != C.U_NONE)
    for q in range(C.MAX_PLAYERS):
        row = base + q * N_OPP_FEATURES
        if q >= state.n_players:
            continue                       # slot unused on a smaller map
        vec[row + 0] = 1.0
        if q == player:
            vec[row + 1] = 1.0
            vec[row + 2] = 1.0 if state.alive[q] else 0.0
            continue                       # the rest describe a *relationship*
        vec[row + 2] = 1.0 if state.alive[q] else 0.0
        vec[row + 3] = 1.0 if at_peace(state, player, q) else 0.0
        vec[row + 4] = 1.0 if at_war(state, player, q) else 0.0
        vec[row + 5] = 1.0 if offered(state, q, player) else 0.0
        vec[row + 6] = 1.0 if offered(state, player, q) else 0.0
        vec[row + 7] = float((live_city & (mem_city_owner == q)).sum()) / 10.0
        vec[row + 8] = float((live_unit & (mem_unit_owner == q)).sum()) / 20.0
    return vec


def encode(state, player: int, rules: Ruleset, pad_to: int | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: (spatial, global) for one player."""
    return (encode_spatial(state, player, rules, pad_to),
            encode_global(state, player, rules))
