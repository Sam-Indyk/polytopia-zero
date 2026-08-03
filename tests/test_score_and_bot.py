"""Scoring, the Domination-tilted heuristic, and the difficulty ladder.

The score formula is quoted from the wiki's Score page, which conveniently
supplies its own checkable anchor: "level 1 city + 9 tiles territory = 280
points". Everything else here follows from that page or from Game Modes, which
defines the Domination rating the heuristic borrows its weights from.
"""
from __future__ import annotations

import pytest

from polyengine.agents.heuristic import (
    DOMINATION_WEIGHT,
    battle_skill,
    domination_term,
    elimination_pressure,
    evaluate,
    speed_skill,
    tribes_destroyed,
)
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import DIFFICULTIES, ScriptedBot
from polyengine.core import constants as C
from polyengine.core.game import Game

N = 11


def bare(seed=3):
    """A board stripped to nothing, so each score component can be isolated."""
    game = Game.new(size=N, seed=seed)
    s = game.state
    for t in range(s.n_tiles):
        s.unit_type[t] = C.U_NONE
        s.unit_owner[t] = C.NEUTRAL
        s.unit_carried[t] = C.U_NONE
        s.city_level[t] = 0
        s.city_owner[t] = C.NEUTRAL
        s.city_capital[t] = 0
        s.city_park[t] = 0
        s.improvement[t] = C.IMP_NONE
        s.improvement_owner[t] = C.NEUTRAL
        s.owner[t] = C.NEUTRAL
    for p in range(s.n_players):
        s.tech[p] = 0
        s.fog_explored[p][:] = 0
        s.kills[p] = 0
        s.losses[p] = 0
    return game, s


# --- the score formula --------------------------------------------------------
class TestScore:
    def test_the_wikis_own_worked_example(self):
        """"level 1 city + 9 tiles territory = 280 points"."""
        game, s = bare()
        city = 5 * N + 5
        s.city_owner[city] = 0
        s.city_level[city] = 1
        s.owner[city] = 0
        for nb in s.neighbors(city):
            s.owner[nb] = 0
        assert int((s.owner == 0).sum()) == 9
        assert game.score(0) == 280

    def test_territory_is_twenty_a_tile(self):
        game, s = bare()
        base = game.score(0)
        s.owner[5 * N + 5] = 0
        assert game.score(0) - base == 20

    def test_exploration_is_five_a_tile(self):
        game, s = bare()
        base = game.score(0)
        s.fog_explored[0][5 * N + 5] = 1
        assert game.score(0) - base == 5

    def test_a_city_is_a_hundred_plus_fifty_per_level_above_one(self):
        game, s = bare()
        city = 5 * N + 5
        s.city_owner[city] = 0
        s.city_level[city] = 1
        assert game.score(0) == 100
        s.city_level[city] = 3
        assert game.score(0) == 100 + 2 * 50

    def test_units_score_five_per_star_of_cost(self):
        game, s = bare()
        tile = 5 * N + 5
        s.unit_type[tile] = C.U_ARCHER
        s.unit_owner[tile] = 0
        s.unit_hp[tile] = 10
        assert game.score(0) == 5 * int(game.rules.unit_cost[C.U_ARCHER])

    def test_a_super_unit_is_fifty(self):
        """"Super units are worth 50 points each."""
        game, s = bare()
        tile = 5 * N + 5
        s.unit_type[tile] = C.U_GIANT
        s.unit_owner[tile] = 0
        s.unit_hp[tile] = 40
        assert game.score(0) == 50

    def test_a_hull_scores_as_its_passenger(self):
        """"Ship and battleship upgrades do not give extra score."""
        game, s = bare()
        tile = 5 * N + 5
        s.unit_type[tile] = C.U_SCOUT
        s.unit_carried[tile] = C.U_WARRIOR
        s.unit_owner[tile] = 0
        s.unit_hp[tile] = 10
        assert game.score(0) == 5 * int(game.rules.unit_cost[C.U_WARRIOR])

    def test_technology_is_a_hundred_per_tier(self):
        game, s = bare()
        tier2 = next(t for t in range(game.rules.n_techs)
                     if int(game.rules.tech_tier[t]) == 2)
        s.grant_tech(0, tier2)
        assert game.score(0) == 200

    def test_a_monument_is_four_hundred(self):
        game, s = bare()
        tile = 5 * N + 5
        s.improvement[tile] = C.IMP_EMPERORS_TOMB
        s.improvement_owner[tile] = 0
        assert game.score(0) == 400

    def test_a_park_is_two_hundred_and_fifty(self):
        game, s = bare()
        city = 5 * N + 5
        s.city_owner[city] = 0
        s.city_level[city] = 1
        s.city_park[city] = 1
        assert game.score(0) == 100 + 250

    def test_stars_are_not_score(self):
        """Absent from the wiki's four categories, and the old formula added them."""
        game, s = bare()
        base = game.score(0)
        s.stars[0] += 500
        assert game.score(0) == base


# --- one definition of winning ------------------------------------------------
def test_training_and_the_engine_agree_on_the_turn_cap_winner():
    """These used to be two different functions, disagreeing on 15% of games."""
    from polytrain.rollout import _score_winner
    for seed in range(12):
        game = Game.new(size=9, seed=4000 + seed)
        agents = [RandomAgent(seed=seed * 5 + p)
                  for p in range(game.state.n_players)]
        for _ in range(500):
            if game.is_terminal():
                break
            game.step(agents[game.current_player].act(game))
        engine = max(range(game.state.n_players), key=game.score)
        trained = _score_winner(game)
        if trained >= 0:
            assert trained == engine


# --- the Domination tilt ------------------------------------------------------
class TestDominationTerm:
    def test_battle_skill_is_the_wikis_ratio(self):
        """(enemy killed + 1) / (friendly lost + enemy killed + 1)."""
        _, s = bare()
        s.kills[0], s.losses[0] = 3, 1
        assert battle_skill(s, 0) == pytest.approx(4 / 5)

    def test_an_untested_army_scores_one(self):
        _, s = bare()
        assert battle_skill(s, 0) == pytest.approx(1.0)

    def test_trading_badly_lowers_battle_skill(self):
        _, s = bare()
        s.kills[0], s.losses[0] = 1, 0
        good = battle_skill(s, 0)
        s.losses[0] = 4
        assert battle_skill(s, 0) < good

    def test_killing_without_losing_beats_trading_evenly(self):
        _, s = bare()
        s.kills[0], s.losses[0] = 5, 0
        clean = battle_skill(s, 0)
        s.losses[0] = 5
        assert clean > battle_skill(s, 0)

    def test_speed_decays_with_the_turn_counter(self):
        game, _ = bare()
        game.state.turn = 0
        early = speed_skill(game)
        game.state.turn = 8
        assert speed_skill(game) < early

    def test_speed_never_goes_negative(self):
        game, _ = bare()
        game.state.turn = 10_000
        assert speed_skill(game) == 0.0

    def test_pressure_rises_as_an_opponent_loses_cities(self):
        game, s = bare()
        for k in range(3):
            tile = 2 * N + 2 + k
            s.city_owner[tile] = 1
            s.city_level[tile] = 1
        weak = elimination_pressure(game, 0)
        s.city_level[2 * N + 3] = 0
        s.city_level[2 * N + 4] = 0
        assert elimination_pressure(game, 0) > weak

    def test_eliminating_a_tribe_is_full_credit(self):
        game, _ = bare()          # nobody owns a city, so player 1 is gone
        assert tribes_destroyed(game, 0) == pytest.approx(1.0)

    def test_the_tilt_is_bounded_by_its_weight(self):
        game, s = bare()
        s.kills[0], s.losses[0] = 50, 0
        game.state.turn = 0
        assert domination_term(game, 0) <= DOMINATION_WEIGHT + 1e-6


# --- the combined evaluation --------------------------------------------------
class TestEvaluate:
    def test_it_is_zero_sum_between_two_players(self):
        game = Game.new(size=9, seed=2)
        assert evaluate(game, 0) == pytest.approx(-evaluate(game, 1), abs=1e-6)

    def test_score_still_dominates_a_large_lead(self):
        """A monument (400) must outweigh the whole combat tilt (<=200)."""
        game, s = bare()
        tile = 5 * N + 5
        s.improvement[tile] = C.IMP_EMPERORS_TOMB
        s.improvement_owner[tile] = 0
        assert evaluate(game, 0) > 0

    def test_a_favourable_kill_ratio_breaks_an_even_position(self):
        game, s = bare()
        for p in (0, 1):
            city = (3 + p) * N + 5
            s.city_owner[city] = p
            s.city_level[city] = 1
        level = evaluate(game, 0)
        s.kills[0], s.losses[1] = 4, 4
        assert evaluate(game, 0) > level


# --- the difficulty ladder ----------------------------------------------------
class TestDifficulty:
    def test_aggression_and_income_both_rise_with_difficulty(self):
        order = ["easy", "normal", "hard", "crazy"]
        aggr = [DIFFICULTIES[d][0] for d in order]
        income = [DIFFICULTIES[d][1] for d in order]
        assert aggr == sorted(aggr)
        assert income == sorted(income)

    def test_crazy_matches_the_wikis_five_stars_per_turn(self):
        """"crazy bots receive five" SPT; a level-1 capital already gives 2."""
        from polyengine.core.economy import player_income
        game = Game.new(size=9, seed=1)
        bot = ScriptedBot.crazy(seed=0)
        bot.apply_handicap(game, 0)
        assert player_income(game.state, 0, game.rules) == 5

    def test_an_unknown_difficulty_is_rejected(self):
        with pytest.raises(ValueError):
            ScriptedBot(difficulty="impossible")

    def test_a_more_aggressive_bot_keeps_a_bigger_army(self):
        assert (ScriptedBot(difficulty="crazy").min_army
                > ScriptedBot(difficulty="easy").min_army)

    def test_the_bot_only_ever_emits_legal_actions(self):
        from polyengine.core.actions import enumerate_actions
        for difficulty in DIFFICULTIES:
            game = Game.new(size=9, seed=11)
            bot = ScriptedBot(seed=1, difficulty=difficulty)
            opponent = RandomAgent(seed=2)
            for _ in range(200):
                if game.is_terminal():
                    break
                actor = bot if game.current_player == 0 else opponent
                action = actor.act(game)
                legal = enumerate_actions(game.state, game.current_player,
                                          game.rules)
                assert action in legal, f"{difficulty} emitted {action}"
                game.step(action)
