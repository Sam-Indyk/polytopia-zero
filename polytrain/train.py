"""PPO self-play training loop.

    # smoke test: tiny map, should beat RandomLegal within a few minutes
    python -m polytrain.train --smoke

    # a real (small) run
    python -m polytrain.train --size 9 --games 32 --rollout 128 --iters 200

Evaluates against the M1 baseline ladder on a fixed cadence and checkpoints the
network, so the Elo curve in the plan has something to plot.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from polyengine import masks as M
from polyengine import obs as O
from polyengine.agents.random_agent import RandomAgent
from polyengine.agents.scripted import ScriptedBot
from polyengine.core import constants as C
from polyengine.core.actions import Action
from polyengine.core.game import Game
from polyengine.core.rules import load_ruleset
from polynet import ARPolicy, NetConfig, PolyNet
from polytrain.ppo import PPOConfig, PPOLearner
from polytrain.rollout import collect


def build_policy(rules, device, small: bool, channels=None, blocks=None):
    kw = dict(n_planes=O.n_planes(rules), global_dim=O.global_dim(rules),
              n_action_types=C.N_ACTION_TYPES, subtype_dim=M.subtype_dim(rules),
              n_techs=rules.n_techs)
    cfg = NetConfig.small(**kw) if small else NetConfig(**kw)
    if channels:
        cfg.channels = channels
    if blocks:
        cfg.blocks = blocks
    net = PolyNet(cfg).to(device)
    return ARPolicy(net, rules, device=device), cfg


class NetAgent:
    """Wraps the policy so it can play in the arena / replay recorder."""

    def __init__(self, policy, greedy: bool = False):
        self.policy = policy
        self.greedy = greedy

    def act(self, game) -> Action:
        actions, _ = self.policy.act([(game.state, game.current_player)],
                                     greedy=self.greedy)
        return actions[0]


def evaluate_vs(policy, opponent_factory, n_games: int, size: int,
                max_steps: int = 800, seed: int = 12345) -> float:
    """Win rate of the network (as both colours) against a baseline."""
    from polyengine.agents.heuristic import evaluate as material
    agent = NetAgent(policy, greedy=False)
    wins = 0.0
    for g in range(n_games):
        net_seat = g % 2
        game = Game.new(size=size, seed=seed + g)
        opp = opponent_factory(seed + g)
        steps = 0
        while not game.is_terminal() and steps < max_steps:
            actor = agent if game.current_player == net_seat else opp
            game.step(actor.act(game))
            steps += 1
        if game.is_terminal():
            w = game.winner()
            wins += 1.0 if w == net_seat else (0.5 if w < 0 else 0.0)
        else:
            m = material(game, net_seat)
            wins += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
    return wins / n_games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--shaping", type=float, default=0.01)
    ap.add_argument("--shaping-anneal", type=int, default=60,
                    help="iterations over which shaping decays to zero")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--small", action="store_true", help="small net config")
    ap.add_argument("--channels", type=int)
    ap.add_argument("--blocks", type=int)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny preset: 6x6, small net, quick RandomLegal gate")
    args = ap.parse_args()

    if args.smoke:
        args.size, args.games, args.rollout = 6, 16, 64
        args.iters = args.iters if args.iters != 200 else 40
        args.small = True
        args.eval_every, args.eval_games = 5, 12
        args.max_turns = 40

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rules = load_ruleset()
    device = torch.device(args.device)
    policy, cfg = build_policy(rules, device, args.small, args.channels, args.blocks)
    n_params = sum(p.numel() for p in policy.net.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M channels={cfg.channels} "
          f"blocks={cfg.blocks} planes={cfg.n_planes} size={args.size}")

    learner = PPOLearner(policy, PPOConfig(
        lr=args.lr, minibatch=args.minibatch, epochs=args.epochs,
        entropy_coef=args.entropy))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    games = None
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        shaping = args.shaping * max(0.0, 1 - it / max(1, args.shaping_anneal))
        roll, stats, games = collect(
            policy, args.games, args.rollout, size=args.size,
            seed=args.seed + it * 1000, shaping=shaping,
            max_turns=args.max_turns, games=games)
        if len(roll) == 0:
            continue
        batch = roll.stack(device)
        logs = learner.update(batch, stats["adv"], stats["ret"])

        if it % 1 == 0:
            ep = stats["episodes"]
            avg_len = np.mean(stats["lengths"]) if stats["lengths"] else 0
            print(f"it {it:4d} | steps {len(roll):5d} | eps {ep:3d} | "
                  f"len {avg_len:5.1f} | pl {logs['policy_loss']:+.4f} | "
                  f"vl {logs['value_loss']:.4f} | ent {logs['entropy']:.3f} | "
                  f"kl {logs['approx_kl']:+.4f} | shape {shaping:.4f} | "
                  f"{time.perf_counter()-t0:.0f}s")

        if it % args.eval_every == 0 or it == args.iters:
            wr_rand = evaluate_vs(policy, lambda s: RandomAgent(seed=s),
                                  args.eval_games, args.size)
            msg = f"  eval@{it}: vs RandomLegal {wr_rand:.1%}"
            if wr_rand > 0.8:
                wr_script = evaluate_vs(policy, lambda s: ScriptedBot(seed=s),
                                        args.eval_games, args.size)
                msg += f" | vs ScriptedBot {wr_script:.1%}"
            print(msg)
            torch.save({"model": policy.net.state_dict(), "cfg": cfg.__dict__,
                        "iter": it}, out_dir / "latest.pt")

    print(f"done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
