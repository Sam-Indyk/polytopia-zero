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
from polytrain.checkpoint import load_resumable, save_atomic
from polytrain import curriculum
from polytrain.evaluate import play_match
from polytrain.league import League, MainExploiter
from polytrain.pool import FrozenOpponent
from polytrain import profiles
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
    ap.add_argument("--curriculum", default=None,
                    help="widen the training distribution over iterations: a "
                         "preset (legacy|duel|standard|ffa) or an inline spec "
                         "like '11:2,14:2/3,16:2/3/4'. Drives board size, seat "
                         "count and tribes, overriding --size and --map-types. "
                         "Omit for the old fixed-size, fixed-tribe behaviour.")
    ap.add_argument("--map-types", default="drylands,lakes,continents,archipelago",
                    help="comma-separated rotation. Games cycle through these, so "
                         "the policy has to plan for water as well as land: "
                         "fishing/Ports/boats on an archipelago, Riders/roads on "
                         "drylands. The type is never told to the net -- it has to "
                         "read the water off the board like a player does.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--channels", type=int)
    ap.add_argument("--blocks", type=int)
    ap.add_argument("--out", default="checkpoints_league")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=None,
                    help="hardware profile: smoke | laptop | workstation | auto")
    ap.add_argument("--precision", default="fp32", choices=("fp32", "bf16"),
                    help="autocast precision for the trunk")
    ap.add_argument("--resume", metavar="CKPT",
                    help="continue from a checkpoint (e.g. checkpoints_x/latest.pt)")
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume from <out>/latest.pt if it exists. Safe to pass "
                         "on a fresh run; this is what the crash supervisor uses.")
    ap.add_argument("--save-every", type=int, default=5,
                    help="save latest.pt every N iterations, not just at eval. "
                         "Bounds what a hard crash costs.")
    ap.add_argument("--league-save-every", type=int, default=20,
                    help="persist the opponent pool every N iterations (it is "
                         "large: capacity x model size)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.profile:
        name = profiles.autodetect() if args.profile == "auto" else args.profile
        prof = profiles.get(name)
        prof.apply(args)
        print(profiles.describe(prof))

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
    map_types = tuple(m.strip() for m in args.map_types.split(",") if m.strip())
    # The curriculum, if any, owns size / seats / tribes from here on. Its stage
    # boundaries are rescaled to this run's length, or a 200-iteration
    # experiment would never leave stage one and would be a duel-only run
    # wearing the name "ffa".
    cur = None
    if args.curriculum:
        cur = curriculum.with_horizon(curriculum.parse(args.curriculum), args.iters)
        print(f"curriculum {args.curriculum}: "
              + " | ".join(cur.describe(i) for i in
                           sorted({0, args.iters // 2, args.iters})))
    device = torch.device(args.device)
    policy, cfg = build_policy(rules, device, args.small,
                               getattr(args, 'channels', None), getattr(args, 'blocks', None))
    print(f"device={device} params={sum(p.numel() for p in policy.net.parameters())/1e6:.2f}M "
          f"size={args.size} league_size={args.league_size}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    learner = PPOLearner(policy, PPOConfig(lr=args.lr, minibatch=args.minibatch,
                                           epochs=args.epochs,
                                           entropy_coef=args.entropy,
                                           precision=args.precision))
    league = League(capacity=args.league_size)
    if args.league_scripted:
        league.add_scripted("scripted", lambda s: ScriptedBot(seed=s))
        print("league includes ScriptedBot -> OSLA is the held-out gate")
    exploiter = MainExploiter()
    rng = np.random.default_rng(args.seed)

    resume_from = args.resume
    if args.auto_resume and not resume_from and (out_dir / "latest.pt").exists():
        resume_from = str(out_dir / "latest.pt")

    start_iter = 1
    best = -1.0
    if resume_from:
        blob = load_resumable(resume_from, map_location=device)
        if blob is None:
            print(f"could not read {resume_from} or its backup; starting fresh")
        else:
            policy.net.load_state_dict(blob["model"])
            start_iter = int(blob.get("iter", 0)) + 1
            # Adam moments matter: dropping them puts a visible transient in the
            # loss at every restart, and this box restarts often.
            if blob.get("optim"):
                try:
                    learner.opt.load_state_dict(blob["optim"])
                except ValueError as exc:
                    print(f"  optimiser state incompatible ({exc}); reinitialised")
            if blob.get("rng") is not None:
                rng = np.random.default_rng()
                rng.bit_generator.state = blob["rng"]
            if blob.get("torch_rng") is not None:
                torch.set_rng_state(
                    torch.as_tensor(blob["torch_rng"], dtype=torch.uint8))
            best = blob.get("best", -1.0)
            print(f"resumed {resume_from} at iteration {start_iter} "
                  f"(elo {blob.get('elo', 0):+.0f}, scripted {blob.get('wr_scripted')})")

        lg = load_resumable(out_dir / "league.pt", map_location="cpu")
        if lg is not None:
            league.load_state_dict(lg)
            print(f"  league restored: {len(league.snapshots)} snapshots, "
                  f"elo {league.main_elo:+.0f}")

    # Never let a resumed run overwrite a better best.pt with a worse one.
    prev_best = load_resumable(out_dir / "best.pt", map_location="cpu")
    if prev_best is not None:
        prior = prev_best.get("gate", prev_best.get("wr_scripted"))
        best = max(best, prior if prior is not None else -1.0)

    games_self = None
    games_vs = None

    def snapshot_blob(it, elo, wr_scr=None, wr_osla=None):
        return {"model": policy.net.state_dict(), "cfg": cfg.__dict__,
                "iter": it, "elo": elo, "best": best,
                "optim": learner.opt.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state().numpy(),
                "wr_scripted": wr_scr, "wr_osla": wr_osla}

    t0 = time.perf_counter()

    stage_size = None
    for it in range(start_iter, args.iters + 1):
        shaping = args.shaping * max(0.0, 1 - it / max(1, args.shaping_anneal))
        member = league.sample(rng)

        size, specs = args.size, None
        if cur is not None:
            size = cur.size_for(it)
            specs = [cur.spec(it, i, args.seed) for i in range(args.games)]
            # Games persist across iterations to avoid re-generating maps, but a
            # carried-over 11x11 board would quietly ignore a stage that just
            # moved to 16x16. Size changes force a rebuild; the other axes vary
            # per game and are picked up as episodes end.
            if size != stage_size:
                games_self = games_vs = None
                stage_size = size

        if member is None:
            roll, stats, games_self = collect(
                policy, args.games, args.rollout, size=size,
                seed=args.seed + it * 1000, shaping=shaping,
                max_turns=args.max_turns, games=games_self,
                map_types=map_types, specs=specs)
            label = "self"
        else:
            opp_fn = make_opponent_fn(member, policy, rules, device)
            roll, stats, games_vs = collect(
                policy, args.games, args.rollout, size=size,
                seed=args.seed + it * 1000, shaping=shaping,
                max_turns=args.max_turns, games=games_vs, opponent=opp_fn,
                map_types=map_types, specs=specs)
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

        # Cheap, frequent save so a bugcheck costs minutes rather than the night.
        # Evaluation is the expensive part, so this deliberately skips it and
        # carries the last known win rates forward.
        if args.save_every and it % args.save_every == 0 and it % args.eval_every:
            save_atomic(snapshot_blob(it, league.main_elo), out_dir / "latest.pt")

        if args.league_save_every and it % args.league_save_every == 0:
            save_atomic(league.state_dict(), out_dir / "league.pt",
                        keep_backup=False)

        if it % args.eval_every == 0 or it == args.iters:
            agent = NetAgent(policy)
            # Eval rotates the same worlds training saw, so a policy that only
            # learned small dry duels cannot hide behind a small dry duel eval.
            stage = cur.stage_for(it) if cur is not None else None
            ev = dict(map_types=(stage.map_types if stage else map_types),
                      sizes=(stage.sizes if stage else None),
                      players=(stage.players if stage else (2,)),
                      tribes=(stage.tribes if stage else None))
            vs_rand = play_match(agent, RandomAgent(seed=7), args.eval_pairs,
                                 args.size, seed=4242, max_steps=400, **ev)
            vs_osla = play_match(agent, GreedyOSLA(seed=7), args.eval_pairs,
                                 args.size, seed=5353, max_steps=400, **ev)
            vs_scr = play_match(agent, ScriptedBot(seed=7), args.eval_pairs,
                                args.size, seed=6464, max_steps=400, **ev)
            print(f"  eval@{it}: Random {vs_rand.win_rate:5.1%}+-{vs_rand.stderr:.1%} | "
                  f"OSLA {vs_osla.win_rate:5.1%}+-{vs_osla.stderr:.1%} (held out) | "
                  f"Scripted {vs_scr.win_rate:5.1%}+-{vs_scr.stderr:.1%} | "
                  f"elo {league.main_elo:+.0f}")
            # Where it is weak, not just how weak. One average over a curriculum
            # hides 90% on the small duels and 30% on the crowded boards.
            detail = vs_osla.format_breakdown()
            if detail:
                print(f"    OSLA by bucket: {detail}")
            # Select on a gate we do NOT train against. With --league-scripted,
            # ScriptedBot is a training opponent, so promoting on it selects for
            # having just sparred with it rather than for general strength.
            gate = vs_osla.win_rate if args.league_scripted else vs_scr.win_rate
            blob = snapshot_blob(it, league.main_elo, vs_scr.win_rate,
                                 vs_osla.win_rate)
            blob["gate"] = gate
            blob["gate_name"] = "osla" if args.league_scripted else "scripted"
            save_atomic(blob, out_dir / "latest.pt")
            if gate > best:
                best = gate
                blob["best"] = best
                save_atomic(blob, out_dir / "best.pt")
            save_atomic(league.state_dict(), out_dir / "league.pt",
                        keep_backup=False)

        if exploiter.should_reset():
            print(f"  exploiter beat main at {exploiter.win_rate:.0%} — resetting")
            exploiter.note_reset()

    print("\n" + league.render())
    print(f"\ndone in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
