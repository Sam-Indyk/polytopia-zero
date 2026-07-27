"""M3 league training: PFSP opponent selection with Elo tracking.

    python -u -m polytrain.train_league --smoke --iters 300

The difference from `polytrain.train` is opponent *allocation*. That trainer
either mirrors itself or picks a past snapshot uniformly; this one samples by how
informative each opponent currently is (see `polytrain.league`), tracks an
anchored Elo, and keeps the pool spread across strength levels.

Held-out evaluation: when scripted baselines are in the league (PLAN.md puts them
at 5%), ScriptedBot is no longer a clean gate, so `Greedy-OSLA` is reported too
and is *never* trained against.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from polyengine.agents.osla import GreedyOSLA
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core.rules import load_ruleset
from polynet import ARPolicy
from polytrain.evaluate import play_match
from polytrain.league import League, MainExploiter
from polytrain.pool import FrozenOpponent
from polytrain.ppo import PPOConfig, PPOLearner
from polytrain.rollout import collect
from polytrain.train import NetAgent, build_policy


def make_opponent_fn(member, policy_template, rules, device):
    """Turn a league member into `opponent(games) -> [Action]`."""
    if member.kind == "scripted":
        agent = member.factory(0)
        return lambda games, _a=agent: [_a.act(g) for g in games]
    frozen = FrozenOpponent(policy_template.net, member.state, device)
    frozen_policy = ARPolicy(frozen.net, rules, device=device)

    def fn(games, _p=frozen_policy):
        with torch.no_grad():
            acts, _ = _p.act([(g.state, g.current_player) for g in games])
        return acts
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--shaping", type=float, default=0.01)
    ap.add_argument("--shaping-anneal", type=int, default=100)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--league-size", type=int, default=20)
    ap.add_argument("--league-scripted", action="store_true",
                    help="include scripted baselines in the league (PLAN: 5%%)")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-pairs", type=int, default=12,
                    help="mirrored seed pairs per eval (2 games each)")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--out", default="checkpoints_league")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        given = {a.lstrip("-").replace("-", "_")
                 for a in __import__("sys").argv[1:] if a.startswith("--")}
        preset = dict(size=6, games=16, rollout=64, small=True, snapshot_every=10,
                      eval_every=20, eval_pairs=12, max_turns=40)
        for k, v in preset.items():
            if k not in given:
                setattr(args, k, v)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rules = load_ruleset()
    device = torch.device(args.device)
    policy, cfg = build_policy(rules, device, args.small)
    print(f"device={device} params={sum(p.numel() for p in policy.net.parameters())/1e6:.2f}M "
          f"size={args.size} league_size={args.league_size}")

    learner = PPOLearner(policy, PPOConfig(lr=args.lr, minibatch=args.minibatch,
                                           epochs=args.epochs,
                                           entropy_coef=args.entropy))
    league = League(capacity=args.league_size)
    if args.league_scripted:
        league.add_scripted("scripted", lambda s: ScriptedBot(seed=s))
        print("league includes ScriptedBot -> OSLA is the held-out gate")
    exploiter = MainExploiter()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    games_self = None
    games_vs = None
    best = -1.0
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
        shaping = args.shaping * max(0.0, 1 - it / max(1, args.shaping_anneal))
        member = league.sample(rng)

        if member is None:
            roll, stats, games_self = collect(
                policy, args.games, args.rollout, size=args.size,
                seed=args.seed + it * 1000, shaping=shaping,
                max_turns=args.max_turns, games=games_self)
            label = "self"
        else:
            opp_fn = make_opponent_fn(member, policy, rules, device)
            roll, stats, games_vs = collect(
                policy, args.games, args.rollout, size=args.size,
                seed=args.seed + it * 1000, shaping=shaping,
                max_turns=args.max_turns, games=games_vs, opponent=opp_fn)
            label = member.name
            league.record_results(member, stats.get("learner_result", []))
            if member.kind == "snapshot":
                exploiter.record([1.0 - r for r in stats.get("learner_result", [])])

        if len(roll) == 0:
            continue
        logs = learner.update(roll.stack(device), stats["adv"], stats["ret"])

        res = stats.get("learner_result", [])
        wr = f"{np.mean(res):5.1%}" if res else "  --  "
        print(f"it {it:4d} | {label:>12} | wr {wr} | steps {len(roll):5d} | "
              f"eps {stats['episodes']:3d} | pl {logs['policy_loss']:+.4f} | "
              f"ent {logs['entropy']:.3f} | elo {league.main_elo:+6.0f} | "
              f"{time.perf_counter()-t0:.0f}s")

        if args.snapshot_every and it % args.snapshot_every == 0:
            league.add_snapshot(policy.net, it)

        if it % args.eval_every == 0 or it == args.iters:
            agent = NetAgent(policy)
            vs_rand = play_match(agent, RandomAgent(seed=7), args.eval_pairs,
                                 args.size, seed=4242, max_steps=400)
            vs_osla = play_match(agent, GreedyOSLA(seed=7), args.eval_pairs,
                                 args.size, seed=5353, max_steps=400)
            vs_scr = play_match(agent, ScriptedBot(seed=7), args.eval_pairs,
                                args.size, seed=6464, max_steps=400)
            print(f"  eval@{it}: Random {vs_rand.win_rate:5.1%}+-{vs_rand.stderr:.1%} | "
                  f"OSLA {vs_osla.win_rate:5.1%}+-{vs_osla.stderr:.1%} (held out) | "
                  f"Scripted {vs_scr.win_rate:5.1%}+-{vs_scr.stderr:.1%} | "
                  f"elo {league.main_elo:+.0f}")
            blob = {"model": policy.net.state_dict(), "cfg": cfg.__dict__,
                    "iter": it, "elo": league.main_elo,
                    "wr_scripted": vs_scr.win_rate, "wr_osla": vs_osla.win_rate}
            torch.save(blob, out_dir / "latest.pt")
            if vs_scr.win_rate > best:
                best = vs_scr.win_rate
                torch.save(blob, out_dir / "best.pt")

        if exploiter.should_reset():
            print(f"  exploiter beat main at {exploiter.win_rate:.0%} — resetting")
            exploiter.note_reset()

    print("\n" + league.render())
    print(f"\ndone in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
