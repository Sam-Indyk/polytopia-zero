"""Self-play rollout collection and GAE.

A subtlety this file exists to get right: in a turn-based game with interleaved
players, a player's *own* trajectory is not contiguous in wall-clock steps. Each
transition is therefore tagged with `(game, player)`, and advantages are
computed along those per-player chains — so the value target for one of P0's
decisions is P0's next decision, however many of P1's actions happened in
between. That is what "GAE runs across turn boundaries" means in the plan.

Rewards are terminal ±1 (Domination / turn-cap score tiebreak) plus an optional
small potential-based shaping term on the material heuristic, annealed to zero
early in training so the final objective is pure win/loss.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from polyengine.agents.heuristic import evaluate
from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.core.game import Game

STORE_KEYS = ("spatial", "global", "valid", "type_mask", "src_mask",
              "tgt_mask", "sub_mask", "atype", "src_idx", "tgt_idx", "sub_idx")


@dataclass
class Rollout:
    """Flat storage of transitions plus per-trajectory bookkeeping."""
    data: dict = field(default_factory=lambda: defaultdict(list))
    logp: list = field(default_factory=list)
    value: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    traj: list = field(default_factory=list)       # (game, player)
    applies: dict = field(default_factory=lambda: defaultdict(list))
    terminal: list = field(default_factory=list)   # trajectory ended here

    def __len__(self):
        return len(self.logp)

    def to_numpy(self) -> dict:
        """Flatten to plain NumPy arrays — the form that crosses a process boundary.

        Worker processes return this; the parent concatenates shards and uploads
        once. Sending tensors instead would pickle CUDA handles or force
        per-shard device transfers.
        """
        out = {}
        for k in STORE_KEYS:
            arr = self.data[k]
            out[k] = np.concatenate([
                a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
                for a in arr])
        for k in ("source", "target", "subtype"):
            out[f"applies_{k}"] = np.concatenate(
                [a.detach().cpu().numpy() for a in self.applies[k]])
        out["old_logp"] = np.concatenate([v.detach().cpu().numpy() for v in self.logp])
        out["old_value"] = np.concatenate([v.detach().cpu().numpy() for v in self.value])
        return out

    def stack(self, device) -> dict:
        out = {}
        for k in STORE_KEYS:
            arr = self.data[k]
            out[k] = torch.cat([a.detach().cpu() for a in arr]).to(device) \
                if isinstance(arr[0], torch.Tensor) else torch.as_tensor(
                    np.concatenate(arr), device=device)
        for k in ("source", "target", "subtype"):
            out[f"applies_{k}"] = torch.cat(
                [a.detach().cpu() for a in self.applies[k]]).to(device)
        out["old_logp"] = torch.cat([v.detach().cpu() for v in self.logp]).to(device)
        out["old_value"] = torch.cat([v.detach().cpu() for v in self.value]).to(device)
        return out


DEFAULT_MAP_TYPES = ("drylands", "lakes", "continents", "archipelago")


class TypeNovelty:
    """A count-based exploration bonus over *action types*.

    The classic `beta / sqrt(N)` intrinsic reward, counted per action type and
    persisted across iterations so it reflects a trailing window of the run
    rather than one batch.

    Action types are the right granularity because that is the axis that
    collapsed: measured on run 3, the type head put 0.000005 of its mass on
    RESEARCH, which was legal at a third of all decisions. At that rate the
    agent needs ~200,000 decisions to research once by accident and an episode
    is a thousand. Bonusing rare *types* pays for the first step onto an
    unexplored branch -- and researching is exactly a first step whose payoff
    arrives dozens of actions later, which is the case undirected exploration
    is worst at.

    It is deliberately small, so it tilts a close call rather than becoming an
    objective of its own.

    Counts **decay**, so the bonus tracks how rare an action is *now* rather
    than over the run's whole history. Cumulative counts had the wrong sign for
    the failure this exists to prevent: a verb the policy abandons keeps its
    large historical count forever, so the moment it most needs a bonus is
    exactly when it can no longer earn one. With a decay the counts converge to
    `share / (1 - decay)` and an abandoned verb becomes tempting again.

    **`beta` must be annealed to zero by the caller, and this is not optional.**
    Cumulative counts carry a guarantee that is easy to miss until it is gone:
    `beta / sqrt(N)` vanishes as N grows, so the intrinsic reward disappears and
    the optimal policy is asymptotically the true one. A decay destroys that --
    the counts reach a steady state, so the bonus is *permanent* and therefore
    changes what the agent is optimising, forever.

    Run 5 measured the cost. Expected bonus per step is

        sum_i s_i * beta/sqrt(s_i * T)  =  beta/sqrt(T) * sum_i sqrt(s_i)

    for shares `s_i` and steady-state total `T = 1/(1-decay)` -- and that is
    **maximised when the shares are uniform**. At the defaults it is 0.0017 a
    step, or +0.58 over a 350-step episode against a terminal reward of +-1.
    So the agent was paid more than half a win to play a uniformly random mix
    of verbs, and it did: 675 iterations with no improvement against any
    baseline, losing 37-11 to run 4's iteration-650 snapshot. Under cumulative
    counts the same quantity was +0.03, which is why run 4 was unaffected.

    Annealed, it keeps what it is for -- run 5's first hundred iterations moved
    median P(RESEARCH) from 0.0007 to 0.0159 and techs per game from 6.8 to
    14.8 -- and then gets out of the way.
    """

    def __init__(self, beta: float = 0.05, prior: float = 50.0,
                 decay: float = 0.9999):
        self.beta = beta
        self.decay = decay
        self.counts = np.full(C.N_ACTION_TYPES, prior, dtype=np.float64)

    def per_episode_scale(self, steps: int = 350) -> float:
        """Roughly what this bonus is worth over an episode, against +-1.

        Exists to be *looked at*: the failure above was invisible in every log
        line and obvious the moment the number was computed.
        """
        share = self.counts / self.counts.sum()
        return float(self.beta / np.sqrt(self.counts.sum())
                     * np.sqrt(share).sum() * steps)

    def bonus(self, atype: int) -> float:
        return self.beta / float(np.sqrt(self.counts[atype]))

    def record(self, atype: int) -> None:
        # Every count ages, not just the one being recorded -- otherwise the
        # decay would be a no-op on precisely the abandoned verbs it is for.
        self.counts *= self.decay
        self.counts[atype] += 1.0

    def report(self) -> str:
        share = self.counts / self.counts.sum()
        rare = np.argsort(share)[:3]
        return " ".join(f"{C.ACTION_NAMES[i][:5].lower()}:{share[i]:.4f}"
                        for i in rare)


def _make_game(size: int, seed: int, map_types, spec=None) -> Game:
    """One game, with its world chosen from the rotation by seed.

    Cycling map types across the batch is what teaches the policy that the plan
    depends on the world: fishing, Ports and boats on an archipelago, Riders and
    roads on drylands. Nothing announces the type: the observation carries water
    tile by tile and the policy has to infer the shape of the world from that,
    the same way a player does on turn one with most of the map still dark.

    A `spec` from `polytrain.curriculum` additionally fixes the board size, the
    seat count and which tribes are dealt — the axes that used to be frozen at
    an 11x11 Imperius-versus-Bardur duel for the whole of training.
    """
    if spec is not None:
        cap = None
        if getattr(spec, "turn_cap_scale", 1.0) != 1.0:
            cap = max(C.TURN_CAP_MIN, int(round(
                C.turn_cap_for(spec.size, spec.n_players) * spec.turn_cap_scale)))
        return Game.new(size=spec.size, seed=seed, map_type=spec.map_type,
                        tribes=list(spec.tribes), turn_cap=cap)
    kind = map_types[seed % len(map_types)]
    return Game.new(size=size, seed=seed, map_type=kind)


def collect(policy, n_games: int, rollout_steps: int, size: int = 9,
            seed: int = 0, gamma: float = 0.999, lam: float = 0.95,
            shaping: float = 0.0, max_turns: int = 100,
            games: list[Game] | None = None,
            opponent=None, learner_seats: list[int] | None = None,
            map_types=DEFAULT_MAP_TYPES, specs=None, seat_members=None,
            novelty=None
            ) -> tuple[Rollout, dict, list[Game]]:
    """Play `rollout_steps` micro-steps across `n_games` games.

    With `opponent=None` this is mirror self-play and *both* seats' transitions
    are trained on. With an `opponent` (a frozen snapshot or a scripted
    baseline), only the seat in `learner_seats[i]` is stored — the opponent's
    moves shape the environment but must not be treated as the learner's own
    decisions. `opponent` is called with a list of Game objects and returns one
    Action per game.
    """
    rules = policy.rules
    if games is None:
        games = [_make_game(size, seed + i, map_types,
                            specs[i] if specs else None)
                 for i in range(n_games)]
    if opponent is not None and learner_seats is None:
        # Which seat the learner holds. Round-robin over the seats that game
        # actually has: `i % 2` silently pinned every game to seat 0 or 1, so on
        # a six-player board four tribes were never learned from and the learner
        # never once sat in them.
        learner_seats = [i % g.state.n_players for i, g in enumerate(games)]
    roll = Rollout()
    # per (game, player): index into roll of that trajectory's previous transition
    last_idx: dict[tuple[int, int], int] = {}
    prev_material: dict[tuple[int, int], float] = {}
    stats = {"episodes": 0, "wins": defaultdict(int), "lengths": [],
             "type_counts": defaultdict(int)}
    steps_played = [0] * len(games)

    def is_learner(gi: int, player: int) -> bool:
        return opponent is None or player == learner_seats[gi]

    for _ in range(rollout_steps):
        live = [i for i, g in enumerate(games) if not g.is_terminal()]
        if not live:
            break

        mine = [i for i in live if is_learner(i, games[i].current_player)]
        theirs = [i for i in live if i not in set(mine)]

        actions_by_game: dict[int, Action] = {}
        info = None
        if mine:
            pairs = [(games[i].state, games[i].current_player) for i in mine]
            acts, info = policy.act(pairs)
            actions_by_game.update(dict(zip(mine, acts)))
        if theirs:
            # Opponents receive Game objects, not (state, player) pairs, so a
            # scripted baseline and a frozen network satisfy the same interface.
            # An opponent that fills different seats with *different* league
            # members needs to know which game each action is for; it opts in
            # by setting `wants_indices` rather than by having its arity
            # sniffed, since these are all closures with bound defaults.
            subset = [games[i] for i in theirs]
            opp_acts = (opponent(subset, theirs)
                        if getattr(opponent, "wants_indices", False)
                        else opponent(subset))
            actions_by_game.update(dict(zip(theirs, opp_acts)))

        for slot, gi in enumerate(mine):
            game = games[gi]
            player = game.current_player
            key = (gi, player)
            idx = len(roll)

            for k in STORE_KEYS:
                roll.data[k].append(info[k][slot:slot + 1])
            for k in ("source", "target", "subtype"):
                roll.applies[k].append(_applies(info["atype"][slot], k))
            roll.logp.append(info["logp"][slot:slot + 1])
            roll.value.append(info["value"][slot:slot + 1])
            roll.traj.append(key)
            # Count-based novelty on the action *type*. Paid on the transition
            # that took it, so the credit lands where the choice was made.
            at = int(actions_by_game[gi].atype)
            roll.reward.append(novelty.bonus(at) if novelty else 0.0)
            if novelty:
                novelty.record(at)
                stats["type_counts"][at] += 1
            roll.terminal.append(False)

            if shaping:
                # Strictly potential-based: gamma * Phi(s') - Phi(s), and scaled
                # into win-equivalents. Both matter.
                #
                # Without the gamma this is not the Ng-Harada-Russell form and
                # the policy-invariance guarantee does not hold -- the agent
                # gains by *sitting* on a high potential rather than converting
                # it, which is precisely the stall-and-hoard behaviour we were
                # trying to remove. Without the scale, Phi is in raw score
                # points (thousands), and shaping measured **96% of all reward
                # magnitude** -- the heuristic was the objective and the game
                # outcome was a rounding error.
                mat = evaluate(game, player) / C.SHAPING_SCALE
                if key in prev_material:
                    roll.reward[last_idx[key]] += shaping * (
                        gamma * mat - prev_material[key])
                prev_material[key] = mat

            last_idx[key] = idx

        for gi in live:
            game = games[gi]
            game.step(actions_by_game[gi])
            steps_played[gi] += 1
            # `max_turns=None` defers to the game's own cap, which now scales
            # with board size and seat count — a fixed 100 would cut every large
            # or crowded game off long before it could be decided.
            limit = game.state.turn_cap if max_turns is None else max_turns
            if game.is_terminal() or game.state.turn >= limit:
                seat = learner_seats[gi] if opponent is not None else None
                _finish(game, gi, roll, last_idx, stats, steps_played[gi], seat,
                        seat_members[gi] if seat_members else None)
                # Replacements keep the same world: a curriculum stage is a
                # property of the iteration, not of one episode ending early.
                games[gi] = _make_game(
                    size, seed + 10_000 + gi * 7919 + stats["episodes"],
                    map_types, specs[gi] if specs else None)
                # ...but NOT the same seat count. `learner_seats` was computed
                # once from the games this call started with, and a replacement
                # can have fewer seats -- a finished 4-player game becomes a
                # 2-player one and the stale seat index 2 no longer exists.
                # That crashed in `Game.score` with an out-of-bounds player, and
                # before crashing it did something quieter and worse: the
                # learner's seat matched nothing, so every transition in that
                # game was silently discarded.
                if learner_seats is not None:
                    learner_seats[gi] = gi % games[gi].state.n_players
                steps_played[gi] = 0
                for p in range(game.state.n_players):
                    last_idx.pop((gi, p), None)
                    prev_material.pop((gi, p), None)

    advantages, returns = _gae(roll, policy, games, gamma, lam)
    return roll, {**stats, "adv": advantages, "ret": returns}, games


def _applies(atype, head) -> torch.Tensor:
    from polyengine import masks as M
    sets = {"source": M.NEEDS_SOURCE, "target": M.NEEDS_TARGET,
            "subtype": M.NEEDS_SUBTYPE}
    return torch.tensor([int(atype) in sets[head]], dtype=torch.bool)


def standing(game, player: int) -> tuple[int, int]:
    """How well `player` finished, higher is better: survival first, then score.

    Survival outranks score because that is what Domination adjudicates — a
    tribe still holding a city has not lost to one that has been wiped out,
    whatever its research total says. Among survivors it is `Game.score`, the
    engine's own published formula and the thing that decides a turn-capped
    game, so this agrees with `Game.winner()` by construction rather than by
    coincidence. (The old truncation path compared raw score alone, which could
    rank an eliminated tribe above a living one.)
    """
    return (1 if game.owns_city(player) else 0, game.score(player))


def head_to_head(game, a: int, b: int) -> float:
    """`a`'s result against `b` alone: 1 above, 0.5 level, 0 below."""
    sa, sb = standing(game, a), standing(game, b)
    return 1.0 if sa > sb else (0.5 if sa == sb else 0.0)


def pairwise_score(game, seat: int) -> float:
    """`seat`'s average result against each opponent individually.

    This is the fix for a bug that only became reachable when games could hold
    more than two tribes. Elo asks "what fraction of a head-to-head did you
    win?", and scoring a free-for-all as 1-if-you-won gives an *equal-strength*
    player 1/N rather than the 1/2 Elo expects. Measured on the real update, an
    identical-strength learner drifted to -112 Elo at 3 players and -218 at 16,
    with the frozen pool inflating to match — pure seat-count artifact, and it
    feeds PFSP's opponent sampling.

    Averaging the pairwise results instead is unbiased at any N, and at N = 2 it
    is *identically* the old 1 / 0.5 / 0, so duels and every rating already
    recorded are unaffected.
    """
    n = game.state.n_players
    if n < 2:
        return 1.0
    return sum(head_to_head(game, seat, q)
               for q in range(n) if q != seat) / (n - 1)


def conquered(game) -> int:
    """The seat that destroyed everyone, or -1 if the game merely ran out.

    `Game.winner()` deliberately conflates the two — it returns the score
    leader at the cap, because that is how a capped game is adjudicated. For
    *training* they must be told apart: one is winning Domination, the other is
    having the most points when nobody won.
    """
    alive = [p for p in range(game.state.n_players) if game.owns_city(p)]
    return alive[0] if len(alive) == 1 else -1


def speed_factor(game) -> float:
    """How much of a victory survives the time it took.

    "Speed skills: 100% at 10 turns per enemy." A win at or inside par is worth
    full value and decays toward `SPEED_FLOOR` after that, so a grinding
    hundred-turn conquest still beats stalling by a wide margin but a decisive
    one is plainly better. Without this the agent has no reason to ever finish:
    a win on turn 300 pays exactly what a win on turn 20 does.
    """
    par = C.SPEED_PAR_TURNS_PER_ENEMY * max(1, game.state.n_players - 1)
    turns = max(1, int(game.state.turn))
    # Decays toward the floor rather than onto it: par/turns alone collapses to
    # the floor almost immediately and stops distinguishing a 40-turn win from
    # a 300-turn one.
    return C.SPEED_FLOOR + (1.0 - C.SPEED_FLOOR) * min(1.0, par / turns)


def terminal_rewards(game) -> list[float]:
    """One reward per seat, with **victory worth far more than points**.

    Three outcomes, deliberately far apart:

      conquest      the winner takes +1 scaled by speed; everyone else -1
      eliminated    -1, whether you came second or died on turn three
      ran out       nobody won: +-STALL_REWARD by relative standing

    The old rule was `2 * pairwise - 1` with `standing = (alive, score)`, which
    made a score lead at the cap worth exactly as much as destroying the
    opponent. Watching a trained net play showed precisely what that buys:
    Warriors only, an early peace treaty, and monuments farmed to 2,800 points.
    Every one of those is optimal under a reward that pays for points.
    """
    n = game.state.n_players
    champion = conquered(game)
    if champion >= 0:
        speed = speed_factor(game)
        return [speed if p == champion else -1.0 for p in range(n)]
    # Nobody won: everyone is penalised. A stall must cost more than the
    # expected value of fighting, or a symmetric self-play agent is paid to
    # avoid combat -- which is precisely what it learned to do, making peace on
    # contact in nearly every game. Rank still tilts it slightly, so a hopeless
    # position is worse than a strong one and there is a gradient when winning
    # is out of reach.
    return [-C.STALL_PENALTY
            + C.STALL_RANK_TILT * (2.0 * pairwise_score(game, p) - 1.0)
            for p in range(n)]


def _finish(game, gi, roll, last_idx, stats, length, learner_seat=None,
            seat_members=None):
    """Credit terminal reward to each stored player's final transition."""
    winner = game.winner() if game.is_terminal() else _score_winner(game)
    stats["episodes"] += 1
    stats["wins"][winner] += 1
    stats["lengths"].append(length)
    if learner_seat is not None:
        stats.setdefault("learner_result", []).append(
            pairwise_score(game, learner_seat))
        # One result per opponent *identity*, so a league member is rated on how
        # the learner did against it specifically rather than on a number
        # smeared across however many seats it happened to fill.
        if seat_members:
            per = stats.setdefault("member_results", {})
            for seat, name in seat_members.items():
                if seat == learner_seat or seat >= game.state.n_players:
                    continue
                per.setdefault(name, []).append(
                    head_to_head(game, learner_seat, seat))
    rewards = terminal_rewards(game)
    stats.setdefault("conquests", 0)
    stats.setdefault("stalls", 0)
    if conquered(game) >= 0:
        stats["conquests"] += 1
    else:
        stats["stalls"] += 1
    # A treaty makes a game unwinnable by construction -- neither side may take
    # the other's cities -- so peace-locked games are exactly the ones that run
    # the cap and pay nothing. Counted so the log can show it happening.
    if any(int(v) for v in game.state.peace):
        stats["peace_games"] = stats.get("peace_games", 0) + 1
    for p in range(game.state.n_players):
        key = (gi, p)
        if key not in last_idx:
            continue
        roll.reward[last_idx[key]] += rewards[p]
        roll.terminal[last_idx[key]] = True


def _score_winner(game) -> int:
    """Turn-cap tiebreak, using the ENGINE's score and nothing else.

    This used to call the material heuristic instead, which disagreed with
    `Game.score` on 15% of turn-capped games -- so training's terminal reward
    was pointing at a different objective from the one the engine adjudicates.
    Most 11x11 games end on the cap rather than by Domination, so that was a
    large slice of the signal.
    """
    scores = [game.score(p) for p in range(game.state.n_players)]
    best = max(scores)
    leaders = [p for p, s in enumerate(scores) if s == best]
    return leaders[0] if len(leaders) == 1 else -1


def _gae(roll: Rollout, policy, games, gamma: float, lam: float):
    """Generalised advantage estimation along each (game, player) chain.

    The subtle part is the **segment boundary**: a rollout is a fixed number of
    micro-steps, not a whole game, so most chains are cut mid-episode and have
    to bootstrap from the value of the state the player will act from next.

    That bootstrap used to read `values[i]` — the value of the *current* state,
    not the next one — which makes `delta = r + (gamma - 1) * V(s)`, very nearly
    zero. Self-referential, so **no information could travel backwards across a
    boundary at all**: the terminal reward could only ever reach the segment it
    landed in. That was survivable while episodes were short. It stopped being
    survivable when the curriculum grew the boards: measured on the first real
    run, episodes went from 256 micro-steps in stage 1 to 1,235 in stage 3
    against a 160-step rollout, so an episode spanned about eight segments and
    seven eighths of every trajectory was cut off from the only unbiased signal
    in the run. Strength peaked around iteration 1,340 and fell apart after.

    `policy` and `games` were already parameters, for exactly this, and were
    never used.
    """
    n = len(roll)
    values = np.array([float(v) for v in roll.value], dtype=np.float64)
    rewards = np.array(roll.reward, dtype=np.float64)
    adv = np.zeros(n, dtype=np.float64)

    chains: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(roll.traj):
        chains[key].append(i)

    # One batched value pass for every chain still running at the cut. A chain
    # whose last transition is terminal needs no bootstrap, and a game that was
    # replaced after finishing already terminated its chain.
    open_chains = [key for key, idxs in chains.items() if not roll.terminal[idxs[-1]]]
    boot: dict[tuple[int, int], float] = {}
    if open_chains and games is not None and policy is not None:
        pairs = [(games[gi].state, p) for gi, p in open_chains
                 if gi < len(games)]
        keys = [k for k in open_chains if k[0] < len(games)]
        if pairs:
            vals = policy.state_value(pairs)
            boot = {k: float(v) for k, v in zip(keys, vals)}

    for key, idxs in chains.items():
        last_gae = 0.0
        for pos in range(len(idxs) - 1, -1, -1):
            i = idxs[pos]
            if roll.terminal[i]:
                next_value, next_nonterminal = 0.0, 0.0
            elif pos + 1 < len(idxs):
                next_value, next_nonterminal = values[idxs[pos + 1]], 1.0
            else:
                # Segment boundary: the game continues into the next rollout,
                # so bootstrap from where this player will act next.
                next_value, next_nonterminal = boot.get(key, values[i]), 1.0
            delta = rewards[i] + gamma * next_value * next_nonterminal - values[i]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[i] = last_gae
    returns = adv + values
    return adv, returns
