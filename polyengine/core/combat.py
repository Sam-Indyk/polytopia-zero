"""Combat resolution: turns the kernel's damage numbers into state changes.

Handles the full melee exchange (hit + retaliation), ranged (no retaliation),
death, and moving the attacker into a vacated tile on a killing melee blow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from polyengine.core import constants as C
from polyengine.core.rules import Ruleset
from polyengine.kernels.combat_kernel import damage_numbers


@dataclass
class CombatResult:
    attacker_tile: int
    defender_tile: int
    damage_to_defender: int
    retaliation_to_attacker: int
    defender_died: bool
    attacker_died: bool
    attacker_advanced: bool  # melee kill on an adjacent tile -> attacker moved in
    converted: bool = False                     # Mind Bender took the defender
    splashed: list = field(default_factory=list)   # (tile, damage) from a Bomber


def _terrain_bonus_for(state, tile: int, rules: Ruleset) -> float:
    """Terrain defence, honouring the techs that unlock each one.

    The wiki's table gives forest -> Archery, mountain -> Climbing, and shallow
    water / ocean -> Aquatism: "Tiles of certain terrain types provide a defence
    bonus once the corresponding technology is researched." Water was hard-coded
    to 1.0 before because naval combat did not exist.
    """
    terrain = int(state.terrain[tile])
    bonus = float(rules.terrain_defense_bonus[terrain])
    gate = int(rules.terrain_defense_tech[terrain])
    if bonus <= 1.0 or gate < 0:
        return bonus
    owner = int(state.unit_owner[tile])
    if owner == C.NEUTRAL:
        return bonus            # no defender to hold the tech; report the terrain
    if not state.has_tech(owner, gate):
        return rules.defense_bonus_map["none"]
    return bonus


def defense_bonus_for(state, tile: int, rules: Ruleset) -> float:
    """Defense multiplier for the unit standing on `tile` (it is the defender).

    Cities are pickier than they look, and the previous version got all three
    conditions wrong:

    * A city fortifies **only its owner's** units. "Invading units besieging a
      city do not receive any defence bonus" — so occupying an enemy city must
      not hand the invader that city's walls, which is what used to happen and
      made a captured city harder to take back than an open field.
    * A city fortifies **only units with the `fortify` skill**. A Giant standing
      in a city gets nothing (it may still get a terrain bonus elsewhere).
    * A city **without** walls still fortifies (1.5x); walls make it 4x. We
      previously gave unwalled cities nothing at all.

    Uncaptured neutral villages never fortify anyone, which falls out of the
    ownership test since `city_owner` is NEUTRAL for them.
    """
    utype = int(state.unit_type[tile])
    terrain_bonus = _terrain_bonus_for(state, tile, rules)
    if utype == C.U_NONE:
        return terrain_bonus

    city_owner = int(state.city_owner[tile])
    if state.city_level[tile] > 0 and city_owner != C.NEUTRAL:
        garrisoned = (city_owner == int(state.unit_owner[tile])
                      and "fortify" in rules.unit_skills[utype])
        if not garrisoned:
            return rules.defense_bonus_map["none"]
        key = "city_walls" if state.city_walls[tile] else "city"
        return rules.defense_bonus_map[key]
    return terrain_bonus


def effective_max_hp(state, tile: int, rules: Ruleset) -> int:
    """Max HP of the unit on `tile`, following a naval hull to its passenger.

    The wiki lists naval health as "Varies" because a hull does not have health
    of its own -- it carries the passenger's. Reading `unit_max_hp[hull]` would
    silently use the placeholder value in units.toml.
    """
    utype = int(state.unit_type[tile])
    carried = int(state.unit_carried[tile])
    if carried != C.U_NONE and rules.unit_hull_hp_varies[utype]:
        base = int(rules.unit_max_hp[carried])
    else:
        base = int(rules.unit_max_hp[utype])
    # "The health stat of veteran units is increased by 5 HP."
    if state.unit_veteran[tile]:
        base += C.VETERAN_BONUS_HP
    return base


HEAL_FRIENDLY = 4
HEAL_ELSEWHERE = 2


def heal_amount(state, tile: int, rules: Ruleset) -> int:
    """HP the unit on `tile` would recover this turn.

    Wiki: "Heals a unit by up to 4 HP in friendly territory and 2 HP elsewhere.
    (Cannot heal past maximum health.)"
    """
    utype = int(state.unit_type[tile])
    if utype == C.U_NONE:
        return 0
    cap = effective_max_hp(state, tile, rules)
    missing = cap - int(state.unit_hp[tile])
    if missing <= 0:
        return 0
    owner = int(state.unit_owner[tile])
    rate = HEAL_FRIENDLY if int(state.owner[tile]) == owner else HEAL_ELSEWHERE
    return min(missing, rate)


def recover_idle_units(state, player: int, rules: Ruleset) -> list[tuple[int, int]]:
    """Heal every unit of `player` that did nothing on its last turn.

    Wiki: "If a unit does not take any action in a given turn, it will
    automatically heal if it can do so." That makes an explicit Recover action
    redundant -- declining to move *is* recovering -- so this is modelled as an
    end-of-turn consequence rather than as a ninth action type.

    Must run BEFORE the per-turn move/attack flags are cleared, since those
    flags are the record of what the unit did last turn.
    """
    healed = []
    for tile in range(state.n_tiles):
        if int(state.unit_owner[tile]) != player:
            continue
        if state.unit_type[tile] == C.U_NONE:
            continue
        if state.unit_moved[tile] or state.unit_attacked[tile]:
            continue
        amount = heal_amount(state, tile, rules)
        if amount > 0:
            state.unit_hp[tile] += amount
            healed.append((tile, amount))
    return healed


def resolve_attack(state, src: int, dst: int, rules: Ruleset) -> CombatResult:
    """Apply an attack from the unit on `src` against the unit on `dst`.

    Mutates `state` in place. Assumes legality was already checked (there is an
    attacker on src, an enemy on dst, and dst is within the attacker's range).
    """
    a_type = state.unit_type[src]
    d_type = state.unit_type[dst]
    a_atk = float(rules.unit_attack[a_type])
    d_def = float(rules.unit_defense[d_type])
    a_maxhp = effective_max_hp(state, src, rules)
    d_maxhp = effective_max_hp(state, dst, rules)
    a_range = int(rules.unit_range[a_type])
    dbonus = defense_bonus_for(state, dst, rules)

    atk_res, def_res = damage_numbers(
        a_atk, int(state.unit_hp[src]), a_maxhp,
        d_def, int(state.unit_hp[dst]), d_maxhp,
        dbonus, rules.accuracy,
    )

    defender_died = False
    attacker_died = False
    attacker_advanced = False
    converted = False
    splashed: list[tuple[int, int]] = []

    a_skills = rules.unit_skills[a_type]
    d_skills = rules.unit_skills[d_type]

    # "Convert: allows a unit to convert an enemy unit into a friendly unit by
    # attacking it." The Mind Bender deals no damage at all -- its attack value
    # is zero -- so this replaces the damage step rather than following it.
    if "convert" in a_skills:
        _convert_unit(state, src, dst, rules)
        state.unit_moved[src] = 1
        state.unit_attacked[src] = 1
        return CombatResult(src, dst, 0, 0, False, False, False, converted=True)

    state.unit_hp[dst] -= atk_res
    # "Splash: allows a unit to damage or poison enemy units adjacent to the
    # targeted unit." Resolved before the target's death so the blast is not
    # skipped when the primary shot kills.
    if "splash" in a_skills:
        splashed = _apply_splash(state, src, dst, atk_res, rules)

    if state.unit_hp[dst] <= 0:
        defender_died = True
        state.unit_kills[src] += 1
        state.kills[state.unit_owner[src]] += 1      # Killer task: 10 kills
        state.losses[state.unit_owner[dst]] += 1
        _clear_unit(state, dst)
        # Melee units advance into the vacated tile; ranged do not.
        if a_range == 1:
            _move_unit(state, src, dst)
            attacker_advanced = True
    else:
        # Retaliation is melee-only, and two skills switch it off:
        #   "Stiff: prevents a unit from being able to do retaliation attacks
        #    when attacked by an enemy unit."
        #   "Surprise: prevents a unit from triggering retaliation attacks when
        #    attacking an enemy unit."
        if (a_range == 1 and "stiff" not in d_skills
                and "surprise" not in a_skills):
            state.unit_hp[src] -= def_res
            if state.unit_hp[src] <= 0:
                attacker_died = True
                # A retaliation kill counts, and it can promote the defender.
                state.unit_kills[dst] += 1
                state.kills[state.unit_owner[dst]] += 1
                state.losses[state.unit_owner[src]] += 1
                _clear_unit(state, src)
        else:
            def_res = 0

    # An attack consumes the attacker's turn -- unless `persist` earned it
    # another one, or `escape` leaves the move available.
    #   "Persist: allows a unit to attack again immediately after killing an
    #    enemy unit. There is no limit on the number of kills in a single turn."
    #   "Escape: allows a unit to move after attacking in the same turn. The
    #    unit cannot attack again in the same turn if they have dash."
    tile = dst if attacker_advanced else src
    if not attacker_died:
        keeps_attack = defender_died and "persist" in a_skills
        state.unit_attacked[tile] = 0 if keeps_attack else 1
        state.unit_moved[tile] = 0 if "escape" in a_skills else 1
        if "escape" in a_skills and "dash" in a_skills:
            # An escaping unit that also has dash may move but not attack again,
            # so persist and escape cannot be combined into an endless turn.
            state.unit_attacked[tile] = 1
        # "Ranged units get pushed in the direction of their last move or last
        # attack." A melee attacker either advanced (which set its facing) or
        # never left its tile, so only shooting needs recording here.
        if a_range > 1:
            state.unit_facing[src] = _facing_toward(state, src, dst)

    return CombatResult(src, dst, atk_res, def_res if not defender_died else 0,
                        defender_died, attacker_died, attacker_advanced,
                        converted=converted, splashed=splashed)


def _apply_splash(state, src: int, dst: int, primary: int,
                  rules: Ruleset) -> list[tuple[int, int]]:
    """Bomber blast: the same damage again to enemies neighbouring the target."""
    owner = int(state.unit_owner[src])
    hit = []
    for nb in list(state.neighbors(dst)):
        nb = int(nb)
        if nb == src or state.unit_type[nb] == C.U_NONE:
            continue
        if int(state.unit_owner[nb]) == owner:
            continue
        state.unit_hp[nb] -= primary
        hit.append((nb, int(primary)))
        if state.unit_hp[nb] <= 0:
            state.unit_kills[src] += 1
            state.kills[owner] += 1
            state.losses[int(state.unit_owner[nb])] += 1
            _clear_unit(state, nb)
    return hit


def apply_stomp(state, tile: int, rules: Ruleset) -> list[tuple[int, int]]:
    """"Stomp: causes a unit to deal damage to all adjacent enemy units when it
    moves." The Juggernaut's signature, and how it clears a coastline."""
    utype = int(state.unit_type[tile])
    if utype == C.U_NONE or "stomp" not in rules.unit_skills[utype]:
        return []
    owner = int(state.unit_owner[tile])
    damage = int(rules.unit_attack[utype])
    hit = []
    for nb in list(state.neighbors(tile)):
        nb = int(nb)
        if state.unit_type[nb] == C.U_NONE or int(state.unit_owner[nb]) == owner:
            continue
        state.unit_hp[nb] -= damage
        hit.append((nb, damage))
        if state.unit_hp[nb] <= 0:
            state.unit_kills[tile] += 1
            state.kills[owner] += 1
            state.losses[int(state.unit_owner[nb])] += 1
            _clear_unit(state, nb)
    return hit


def _convert_unit(state, src: int, dst: int, rules: Ruleset) -> None:
    """Turn the enemy on `dst` into a unit of the attacker's tribe.

    "Converted units take up population in the mind bender's capital (except
    units with Independent skill). If no capital is owned, then a random city
    starting from the southernmost to northernmost city is chosen."
    """
    owner = int(state.unit_owner[src])
    state.unit_owner[dst] = owner
    state.unit_moved[dst] = 1
    state.unit_attacked[dst] = 1
    state.unit_kills[dst] = 0
    state.unit_veteran[dst] = 0
    state.unit_hidden[dst] = 0
    if "independent" in rules.unit_skills[int(state.unit_type[dst])]:
        state.unit_home[dst] = -1
        return
    home = -1
    for tile in range(state.n_tiles):
        if int(state.city_owner[tile]) == owner and state.city_capital[tile]:
            home = tile
            break
    if home < 0:
        # "starting from the southernmost to northernmost city": tile index runs
        # north to south, so the southernmost owned city is the highest index.
        owned = [t for t in range(state.n_tiles)
                 if int(state.city_owner[t]) == owner and state.city_level[t] > 0]
        home = owned[-1] if owned else -1
    state.unit_home[dst] = home


def _clear_unit(state, tile: int) -> None:
    state.unit_type[tile] = C.U_NONE
    state.unit_owner[tile] = C.NEUTRAL
    state.unit_hp[tile] = 0
    state.unit_moved[tile] = 0
    state.unit_attacked[tile] = 0
    # Free the supporting city's slot — a dead unit stops taking up space.
    state.unit_home[tile] = -1
    state.unit_veteran[tile] = 0
    state.unit_kills[tile] = 0
    state.unit_carried[tile] = C.U_NONE
    # Stale flags would otherwise be inherited by the next unit to stand here:
    # invisible to the enemy, and pushed in a direction it never travelled.
    state.unit_hidden[tile] = 0
    state.unit_facing[tile] = -1


def _move_unit(state, src: int, dst: int) -> None:
    state.unit_type[dst] = state.unit_type[src]
    state.unit_owner[dst] = state.unit_owner[src]
    state.unit_hp[dst] = state.unit_hp[src]
    state.unit_veteran[dst] = state.unit_veteran[src]
    state.unit_kills[dst] = state.unit_kills[src]
    state.unit_moved[dst] = state.unit_moved[src]
    state.unit_attacked[dst] = state.unit_attacked[src]
    state.unit_carried[dst] = state.unit_carried[src]
    state.unit_hidden[dst] = state.unit_hidden[src]
    # A melee advance IS a move, so it sets the direction a later forced spawn
    # would shove this unit.
    state.unit_facing[dst] = _facing_toward(state, src, dst)
    _clear_unit(state, src)


def _facing_toward(state, src: int, dst: int) -> int:
    sx, sy = state.xy(src)
    dx, dy = state.xy(dst)
    step = ((dx > sx) - (dx < sx), (dy > sy) - (dy < sy))
    return C.PUSH_COMPASS.index(step) if step in C.PUSH_COMPASS else -1
