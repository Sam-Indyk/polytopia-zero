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
from collections import defaultdict
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


# The promotion yardstick. Fixed for the life of a run — a gate that moves with
# the curriculum cannot say which checkpoint is best, only which is best right
# now. 11x11 duels across every map type: big enough that naval play matters
# (continents and waterworld are where the first run was weakest, at 22%), small
# enough that the eval is cheap, and never saturating the way 9x9 drylands did.
GATE_SIZE = 11
GATE_MAP_TYPES = ("drylands", "lakes", "continents",
                  "archipelago", "pangea", "waterworld")


def shaping_for(it: int, scale: float, anneal: int, floor: float) -> float:
    """Reward-shaping weight at iteration `it`, decaying to a *floor*, not zero.

    The shaping term is potential-based, so it cannot change which policy is
    optimal — only how quickly the value function finds it. Annealing it to
    exactly zero was fine while episodes were short and terminal rewards
    plentiful (23 per iteration on 9x9). On the boards the curriculum reaches
    they arrive about four times per iteration across 48 games, and a value
    function updated 160 steps at a time with almost no terminal anchor drifts.
    """
    decay = max(0.0, 1.0 - it / max(1, anneal))
    return scale * (floor + (1.0 - floor) * decay)


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


def make_seat_opponent_fn(assign, primary, policy_template, rules, device,
                          built=None):
    """An opponent that fills different seats with *different* league members.

    `assign` maps game index -> {seat: member}. Members are grouped per call so
    each distinct opponent still runs one batched forward pass, which is what
    keeps a sixteen-tribe game from costing sixteen times a duel.

    This replaces filling every non-learner seat with copies of one checkpoint.
    Five identical opponents teach the learner to beat one thing five times at
    once; five different ones are the position it will actually be in, and each
    gets its own Elo result rather than sharing a number.

    **`built` must be owned by the caller and reused across iterations.** Each
    entry is a `FrozenOpponent`, and that constructor does
    `deepcopy(net).to(device)` — a fresh net on the GPU. When this cache was
    local, a league iteration allocated one per *distinct member drawn*, so a
    full 20-snapshot pool meant ~20 new GPU nets every iteration instead of the
    single one the old single-opponent code built. Measured on the first real
    run: VRAM climbed from 4.4 GB at iteration 120 to 7.9 GB of 8.1 GB by
    iteration 2160, heading for an out-of-memory crash.
    """
    if built is None:
        built = {}

    def agent_for(member):
        if member.name not in built:
            if member.kind == "scripted":
                built[member.name] = ("scripted", member.factory(0))
            else:
                frozen = FrozenOpponent(policy_template.net, member.state, device)
                built[member.name] = ("net", ARPolicy(frozen.net, rules,
                                                      device=device))
        return built[member.name]

    def fn(games, gis):
        out: list = [None] * len(games)
        groups: dict[str, list[int]] = defaultdict(list)
        members: dict[str, object] = {}
        for k, (g, gi) in enumerate(zip(games, gis)):
            m = assign.get(gi, {}).get(g.current_player, primary)
            groups[m.name].append(k)
            members[m.name] = m
        for name, ks in groups.items():
            kind, obj = agent_for(members[name])
            if kind == "scripted":
                for k in ks:
                    out[k] = obj.act(games[k])
            else:
                with torch.no_grad():
                    acts, _ = obj.act([(games[k].state, games[k].current_player)
                                       for k in ks])
                for k, a in zip(ks, acts):
                    out[k] = a
        return out

    fn.wants_indices = True
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
    ap.add_argument("--shaping-floor", type=float, default=0.25,
                    help="fraction of --shaping that survives the anneal. "
                         "Annealing to exactly zero leaves the terminal reward "
                         "as the only signal, and on large boards that arrives "
                         "about four times per iteration across 48 games "
                         "(measured: 23/iteration on 9x9, 4.4 on 14x14). The "
                         "shaping is potential-based, so a standing floor does "
                         "not change which policy is optimal.")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--league-size", type=int, default=20)
    ap.add_argument("--league-scripted", action="store_true",
                    help="include scripted baselines in the league (PLAN: 5%%)")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-pairs", type=int, default=12,
                    help="mirrored seed pairs per eval (2 games each)")
    ap.add_argument("--max-turns", type=int, default=0,
                    help="episode truncation. 0 (default) defers to each game's "
                         "own turn cap, which scales with board size and seat "
                         "count; a fixed value pins every game to it, which on "
                         "a curriculum cuts the large boards off before they "
                         "can be decided.")
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

    # Never let a resumed run overwrite a better best.pt with a worse one --
    # unless the gate itself changed, in which case the old number is not a
    # measurement of the same thing and keeping it would silently freeze
    # promotion for the rest of the run.
    gate_name = f"osla@{GATE_SIZE}x{GATE_SIZE}-fixed"
    prev_best = load_resumable(out_dir / "best.pt", map_location="cpu")
    if prev_best is not None:
        if prev_best.get("gate_name") != gate_name:
            print(f"  best.pt was gated on {prev_best.get('gate_name')!r}, now "
                  f"{gate_name!r}; the old score is not comparable, resetting")
            best = -1.0
        else:
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
    stage_key = None
    opp_cache: dict[str, tuple[str, object]] = {}
    for it in range(start_iter, args.iters + 1):
        shaping = shaping_for(it, args.shaping, args.shaping_anneal,
                              args.shaping_floor)
        member = league.sample(rng)

        size, specs = args.size, None
        if cur is not None:
            size = cur.size_for(it)
            stage_key = cur.describe(it)
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
                max_turns=(args.max_turns or None), games=games_self,
                map_types=map_types, specs=specs)
            label = "self"
        else:
            # Who sits where. Every seat gets its own draw from the league,
            # falling back to the primary member when the draw says "self" — a
            # league iteration should not quietly become self-play in half its
            # seats.
            #
            # Drawn for *every* seat rather than for "the other N-1", because
            # games persist across iterations to let long episodes finish, so
            # the game at index gi may have a different seat count from this
            # iteration's spec. `collect` picks the learner's seat from the real
            # game and `_finish` ignores seats the game does not have, so an
            # entry per seat is always enough and never wrong.
            max_seats = max([s.n_players for s in specs] if specs else [2])
            assign, seat_members = {}, []
            for gi in range(args.games):
                drawn = league.sample_opponents(rng, max_seats)
                picks = {q: (m or member) for q, m in enumerate(drawn)}
                assign[gi] = picks
                seat_members.append({q: m.name for q, m in picks.items()})
            # Cached across iterations, and pruned to the members still in the
            # pool so an evicted snapshot's net does not sit on the GPU forever.
            live_names = {m.name for m in league.members}
            for gone in [k for k in opp_cache if k not in live_names]:
                opp_cache.pop(gone, None)
            opp_fn = make_seat_opponent_fn(assign, member, policy, rules, device,
                                           built=opp_cache)
            roll, stats, games_vs = collect(
                policy, args.games, args.rollout, size=size,
                seed=args.seed + it * 1000, shaping=shaping,
                max_turns=(args.max_turns or None), games=games_vs, opponent=opp_fn,
                map_types=map_types, specs=specs, seat_members=seat_members)
            label = member.name
            # Elo per opponent identity. `learner_result` is now the learner's
            # *average pairwise* result, which is unbiased at any seat count;
            # `member_results` splits it by who was actually beaten.
            by_member = stats.get("member_results", {})
            by_name = {m.name: m for gi in assign for m in assign[gi].values()}
            for name, results in by_member.items():
                league.record_results(by_name.get(name), results)
            if not by_member:
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
            print(f"  eval@{it}: Random {vs_rand.record:>9} | "
                  f"OSLA {vs_osla.record:>9} (held out) | "
                  f"Scripted {vs_scr.record:>9} | "
                  f"elo {league.main_elo:+.0f}")
            # Where it is weak, not just how weak. One average over a curriculum
            # hides 90% on the small duels and 30% on the crowded boards.
            detail = vs_osla.format_breakdown()
            if detail:
                print(f"    OSLA by bucket: {detail}")
            # --- the promotion gate --------------------------------------
            # Measured on a FIXED reference: a held-out opponent on a fixed
            # distribution that never moves with the curriculum.
            #
            # The eval above deliberately follows the curriculum, which makes it
            # the wrong thing to promote on. On the first real run the net hit
            # 100% vs OSLA at iteration 275 on 9x9 duels; the curriculum then
            # widened and the same policy scored 55-70% on the harder
            # distribution, so `best.pt` froze at iteration ~275 and stayed a
            # stage-1 checkpoint for 1,800 iterations. Resetting the bar per
            # stage does not fix it either: strength actually peaked around
            # iteration 1,340 and then collapsed, and a per-stage best keeps the
            # best of the *current* stage, which by then was much weaker.
            # Only a yardstick that never changes can say "this is the best
            # checkpoint of the run".
            ref = play_match(agent, GreedyOSLA(seed=11), args.eval_pairs,
                             GATE_SIZE, seed=9191, max_steps=400,
                             sizes=(GATE_SIZE,), players=(2,),
                             map_types=GATE_MAP_TYPES)
            gate = ref.win_rate
            blob = snapshot_blob(it, league.main_elo, vs_scr.win_rate,
                                 vs_osla.win_rate)
            blob["gate"] = gate
            blob["gate_name"] = f"osla@{GATE_SIZE}x{GATE_SIZE}-fixed"
            blob["stage"] = stage_key
            save_atomic(blob, out_dir / "latest.pt")
            print(f"    gate (fixed {GATE_SIZE}x{GATE_SIZE} duel vs OSLA): "
                  f"{ref.record}"
                  f"{'  <- new best' if gate > best else f'  (best {best:.1%})'}")
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
