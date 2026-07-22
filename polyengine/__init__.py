"""polyengine — a from-scratch simulator of the current Battle of Polytopia ruleset.

Public surface (the load-bearing invariant of the whole project):
    GameState  — copyable flat-array state + forward model
    Game       — thin orchestration wrapper (legality API + apply)
    load_ruleset / Ruleset — versioned data-driven rules
"""

from polyengine.core.rules import Ruleset, load_ruleset
from polyengine.core.state import GameState
from polyengine.core.game import Game

__all__ = ["GameState", "Game", "Ruleset", "load_ruleset"]
