"""Multiprocess rollout collection.

Profiling the single-process trainer showed ~93% of a rollout step was CPU-side
Python (engine stepping, mask generation, observation encoding) with the GPU idle
— the loop ran slower than the bare engine on one core. The fix is to put the
CPU work on many cores.

Design: each worker owns its own copy of the policy and a shard of the games, and
runs a whole rollout segment without talking to the parent. Only weights go out
(once per iteration) and transitions come back. That suits PPO, which needs the
batch only at the end of a segment, and it avoids per-step IPC entirely.

**Which mode wins depends on network size — measured, 9x9, 24-step segments:**

    net          single-process GPU    16 CPU workers    4 CUDA workers
    48x4  (0.25M)      3,800 st/s          9,600 st/s        --
    160x12 (6.2M)      2,660 st/s            660 st/s      1,660 st/s

For a small network the CPU work dominates and spreading it over cores wins
outright. For the full-size network CPU inference collapses (~134 decisions/s per
worker) and several processes sharing one GPU just contend, so the plain
single-process loop is fastest. Hence `--workers 0` (single-process) is the
default and is the right choice for the full config; use CPU workers for small
nets and toy maps.

The mode that would beat both — workers doing env/encode/mask on many cores while
the parent batches GPU inference — needs a shared-memory transport rather than
per-step pickling, and is the next throughput step if the full-config run turns
out to be rollout-bound.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp

# Per-process state, initialised once per worker (see `_init_worker`).
_W: dict = {}


def _init_worker(cfg_dict: dict, ruleset_name: str, threads: int,
                 device: str = "cpu") -> None:
    torch.set_num_threads(max(1, threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, threads)))
    from polyengine.core.rules import load_ruleset
    from polynet import ARPolicy, NetConfig, PolyNet

    rules = load_ruleset(ruleset_name)
    dev = torch.device(device)
    net = PolyNet(NetConfig(**cfg_dict)).to(dev)
    net.eval()
    _W["rules"] = rules
    _W["net"] = net
    _W["device"] = dev
    _W["policy"] = ARPolicy(net, rules, device=dev)
    _W["games"] = None
    _W["pool_net"] = None


def _run_shard(job: dict) -> dict:
    """Collect one rollout shard. Returns plain NumPy so it pickles cheaply."""
    from polytrain.rollout import collect

    policy = _W["policy"]
    dev = _W["device"]
    policy.net.load_state_dict(
        {k: v.to(dev) for k, v in job["weights"].items()})

    opponent = None
    if job.get("opponent_weights") is not None:
        import copy
        if _W["pool_net"] is None:
            _W["pool_net"] = copy.deepcopy(_W["net"]).to(dev)
        _W["pool_net"].load_state_dict(
            {k: v.to(dev) for k, v in job["opponent_weights"].items()})
        _W["pool_net"].eval()
        from polynet import ARPolicy
        opp_policy = ARPolicy(_W["pool_net"], _W["rules"], device=dev)

        def opponent(pairs, _p=opp_policy):
            with torch.no_grad():
                acts, _ = _p.act(pairs)
            return acts

    roll, stats, games = collect(
        policy, job["n_games"], job["rollout_steps"], size=job["size"],
        seed=job["seed"], shaping=job["shaping"], max_turns=job["max_turns"],
        games=_W["games"] if job["persist"] else None, opponent=opponent)
    _W["games"] = games if job["persist"] else None

    if len(roll) == 0:
        return {"n": 0}
    data = roll.to_numpy()
    data["adv"] = stats["adv"]
    data["ret"] = stats["ret"]
    data["n"] = len(roll)
    data["episodes"] = stats["episodes"]
    data["lengths"] = list(stats["lengths"])
    data["learner_result"] = list(stats.get("learner_result", []))
    return data


@dataclass
class ParallelCollector:
    """A persistent pool of rollout workers."""
    cfg_dict: dict
    workers: int = 8
    ruleset_name: str = "current"
    threads_per_worker: int = 1
    worker_device: str = "cpu"
    _pool: object = None

    def __post_init__(self):
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.workers, initializer=_init_worker,
            initargs=(self.cfg_dict, self.ruleset_name, self.threads_per_worker,
                      self.worker_device))

    def collect(self, net, n_games: int, rollout_steps: int, size: int, seed: int,
                shaping: float = 0.0, max_turns: int = 100, persist: bool = True,
                opponent_state: dict | None = None) -> tuple[dict, dict]:
        weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        opp = ({k: v.detach().cpu() for k, v in opponent_state.items()}
               if opponent_state is not None else None)
        per = max(1, n_games // self.workers)
        jobs = [{
            "weights": weights, "opponent_weights": opp,
            "n_games": per, "rollout_steps": rollout_steps, "size": size,
            "seed": seed + w * 7919, "shaping": shaping, "max_turns": max_turns,
            "persist": persist,
        } for w in range(self.workers)]

        shards = [s for s in self._pool.map(_run_shard, jobs) if s.get("n", 0) > 0]
        return _merge(shards)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def _merge(shards: list[dict]) -> tuple[dict, dict]:
    """Concatenate worker shards into one batch plus aggregate stats."""
    if not shards:
        return {}, {"episodes": 0, "lengths": [], "adv": np.zeros(0),
                    "ret": np.zeros(0), "n": 0}
    keys = [k for k in shards[0]
            if k not in ("n", "episodes", "lengths", "adv", "ret", "learner_result")]
    batch = {k: np.concatenate([s[k] for s in shards]) for k in keys}
    stats = {
        "adv": np.concatenate([s["adv"] for s in shards]),
        "ret": np.concatenate([s["ret"] for s in shards]),
        "episodes": sum(s["episodes"] for s in shards),
        "lengths": [x for s in shards for x in s["lengths"]],
        "learner_result": [x for s in shards for x in s["learner_result"]],
        "n": sum(s["n"] for s in shards),
    }
    return batch, stats


def to_device(batch: dict, device) -> dict:
    """Upload a merged NumPy batch to the training device."""
    bool_keys = {"type_mask", "src_mask", "tgt_mask", "sub_mask",
                 "applies_source", "applies_target", "applies_subtype"}
    long_keys = {"atype", "src_idx", "tgt_idx", "sub_idx"}
    out = {}
    for k, v in batch.items():
        if k in bool_keys:
            dtype = torch.bool
        elif k in long_keys:
            dtype = torch.long
        else:
            dtype = torch.float32
        out[k] = torch.as_tensor(np.ascontiguousarray(v), dtype=dtype, device=device)
    return out
