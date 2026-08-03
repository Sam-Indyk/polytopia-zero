"""Static state evaluation: what "doing well" means, in one scalar.

Two things read this — the baseline agents (Scripted / OSLA / MCTS) and PPO's
potential-based reward shaping — so it is effectively the definition of progress
the agent is nudged toward between terminal rewards. It should therefore agree
with how the game is actually won:

    winning = destroy every opponent, or lead on score when the turn cap hits

So the base term is Polytopia's **own score**, differenced against the strongest
opponent. `Game.score` implements the wiki's published formula, which already
rewards what a good player does: taking villages, levelling cities, holding
territory, exploring, teching, and building monuments.

On top of that sits a **Domination term**, because this bot is meant to win by
elimination rather than to farm points for 30 turns. Its weights are not
invented: Polytopia rates a Domination game on four equally-weighted parts (the
wiki's Game Modes page), and three of them can be read off a mid-game position:

    battle skills     (enemy killed + 1) / (friendly lost + enemy killed + 1)
    tribes destroyed  tribes eliminated / total opponents
    speed skills      100% at 10 turns per enemy

`battle skills` is used verbatim. Its being a *ratio* is the important part:
trading two units to kill one is not progress, and a raw kill count would say it
was. Tribes-destroyed becomes pressure on the opponent's surviving cities, and
speed becomes a mild penalty on letting the clock run.

The previous version of this file was material-based and predated most of the
ruleset — it knew nothing about monuments, connections, markets or parks, and it
disagreed with `Game.winner` about who had won 15% of turn-capped games.
"""
from __future__ import annotations

# A city is 100 points and a monument 400, which is the scale this tilt is
# measured against: a maximal Domination advantage is worth about two cities.
# Enough to break an even position toward the aggressive line, not enough to
# talk the agent into trading an army for one kill.
DOMINATION_WEIGHT = 200.0

# Equal split across the three mid-game-observable rating parts, matching "four
# equally weighted parts" (the fourth, difficulty, is fixed for a given game).
W_BATTLE = 1.0 / 3.0
W_TRIBES = 1.0 / 3.0
W_SPEED = 1.0 / 3.0


def battle_skill(state, player: int) -> float:
    """Polytopia's own battle-skills ratio, verbatim.

        (enemy killed + 1) / (friendly lost + enemy killed + 1)

    The +1 makes an untested army score 1.0 rather than 0/0, so a peaceful
    opening is not punished; it decays only once trades start going badly.
    """
    killed = float(state.kills[player])
    lost = float(state.losses[player])
    return (killed + 1.0) / (lost + killed + 1.0)


def tribes_destroyed(game, player: int) -> float:
    """Fraction of opponents already eliminated (no cities left)."""
    n = game.state.n_players
    opponents = [p for p in range(n) if p != player]
    if not opponents:
        return 0.0
    dead = sum(1 for p in opponents if not game.owns_city(p))
    return dead / len(opponents)


def speed_skill(game) -> float:
    """Decays as the clock runs, since Domination rates a fast kill higher.

    "The time limit for 100% is 10 turns per enemy", so the reference horizon
    scales with how many opponents there were to remove.
    """
    n_opponents = max(1, game.state.n_players - 1)
    budget = 10.0 * n_opponents
    return max(0.0, 1.0 - float(game.state.turn) / budget)


def elimination_pressure(game, player: int) -> float:
    """How close the nearest opponent is to losing their last city.

    Tribes-destroyed only moves in whole steps, which leaves a one-step search no
    gradient to climb until a tribe actually dies. This supplies the slope: an
    opponent cut from four cities to one is most of the way to elimination.
    """
    s = game.state
    opponents = [p for p in range(s.n_players)
                 if p != player and game.owns_city(p)]
    if not opponents:
        return 1.0
    weakest = min(int(((s.city_owner == p) & (s.city_level > 0)).sum())
                  for p in opponents)
    return 1.0 / (1.0 + weakest)


def domination_term(game, player: int) -> float:
    """The combat tilt, expressed in score-equivalent points."""
    progress = max(tribes_destroyed(game, player),
                   elimination_pressure(game, player))
    parts = (W_BATTLE * battle_skill(game.state, player)
             + W_TRIBES * progress
             + W_SPEED * speed_skill(game))
    return DOMINATION_WEIGHT * parts


def evaluate(game, player: int) -> float:
    """Score lead over the strongest opponent, tilted toward Domination.

    Positive means `player` is ahead. Everything is in score points, so the two
    halves are directly comparable and DOMINATION_WEIGHT means what it says.
    """
    n = game.state.n_players
    own = float(game.score(player))
    opp = max((float(game.score(p)) for p in range(n) if p != player), default=0.0)

    own_tilt = domination_term(game, player)
    opp_tilt = max((domination_term(game, p) for p in range(n) if p != player),
                   default=0.0)
    return (own - opp) + (own_tilt - opp_tilt)
