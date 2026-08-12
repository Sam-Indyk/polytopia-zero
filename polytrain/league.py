"""AlphaStar-lite league: PFSP opponent selection, Elo, and pool management.

The M2 runs showed why this is needed. Mirror self-play cycled (91.7% against
ScriptedBot, then 47%); a uniformly-sampled snapshot pool roughly doubled
sustained strength but still drifted (88% best vs 69% final). Uniform sampling
spends most of its games on opponents the learner already crushes, which carry
almost no gradient signal.

Prioritised fictitious self-play fixes the allocation: opponents are sampled in
proportion to how *informative* they are, measured by the learner's win rate p
against each one.

    p(1-p)   "even"   — maximal at p=0.5, the opponents still in contention
    (1-p)^2  "hard"   — concentrates on the ones currently beating the learner

Per PLAN.md the mixture is 50% mirror self, 35% PFSP-even, 10% hardest, 5%
scripted baselines. Win rates are tracked with exponential decay so a snapshot
the learner *used* to beat becomes eligible again as the policy drifts — which
is precisely the cycling failure this is meant to catch.

Eviction preserves Elo spread rather than dropping the oldest: a pool that is all
recent snapshots is a pool of near-identical opponents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Sampling mixture (PLAN.md Phase 3).
P_SELF = 0.50
P_EVEN = 0.35
P_HARD = 0.10
P_SCRIPTED = 0.05

ELO_K = 16.0
ANCHOR_ELO = 0.0          # RandomLegal is the anchor at 0


@dataclass
class Member:
    """One league opponent: a frozen snapshot or a scripted baseline."""
    name: str
    kind: str                       # "snapshot" | "scripted"
    iteration: int = -1
    state: dict | None = None       # weights, for snapshots
    factory: object = None          # callable(seed) -> agent, for scripted
    elo: float = 0.0
    _wins: float = 0.0              # exponentially-decayed
    _games: float = 0.0

    @property
    def win_rate(self) -> float:
        """Learner's win rate against this member (0.5 prior when unplayed)."""
        if self._games < 1e-6:
            return 0.5
        return self._wins / self._games

    @property
    def games_played(self) -> float:
        return self._games

    def record(self, result: float, decay: float = 0.99) -> None:
        """Record a learner result in [0,1], decaying older evidence."""
        self._wins = self._wins * decay + result
        self._games = self._games * decay + 1.0


def pfsp_weights(win_rates: np.ndarray, mode: str = "even") -> np.ndarray:
    """Sampling weights over opponents from the learner's win rate against each."""
    p = np.clip(win_rates, 0.0, 1.0)
    if mode == "even":
        w = p * (1.0 - p)
    elif mode == "hard":
        w = (1.0 - p) ** 2
    else:
        raise ValueError(f"unknown PFSP mode {mode!r}")
    total = w.sum()
    if total <= 1e-12:                 # e.g. every opponent at p=1.0
        return np.full(len(p), 1.0 / max(1, len(p)))
    return w / total


@dataclass
class League:
    """The opponent pool plus its sampling policy and Elo book-keeping."""
    capacity: int = 20
    p_self: float = P_SELF
    p_even: float = P_EVEN
    p_hard: float = P_HARD
    p_scripted: float = P_SCRIPTED
    members: list[Member] = field(default_factory=list)
    main_elo: float = 0.0
    history: list[dict] = field(default_factory=list)

    # --- membership -----------------------------------------------------------
    def add_snapshot(self, net, iteration: int) -> Member:
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        m = Member(name=f"snap@{iteration}", kind="snapshot", iteration=iteration,
                   state=state, elo=self.main_elo)
        self.members.append(m)
        self._evict()
        return m

    def add_scripted(self, name: str, factory) -> Member:
        m = Member(name=name, kind="scripted", factory=factory, elo=ANCHOR_ELO)
        self.members.append(m)
        return m

    @property
    def snapshots(self) -> list[Member]:
        return [m for m in self.members if m.kind == "snapshot"]

    @property
    def scripted(self) -> list[Member]:
        return [m for m in self.members if m.kind == "scripted"]

    def _evict(self) -> None:
        """Drop the snapshot whose removal costs the least Elo coverage.

        Keeps the extremes (weakest and strongest) and removes whichever interior
        snapshot is closest to its neighbours, so the pool stays spread across
        strength levels instead of collapsing onto recent, near-identical copies.
        """
        snaps = self.snapshots
        if len(snaps) <= self.capacity:
            return
        snaps.sort(key=lambda m: m.elo)
        if len(snaps) < 3:
            # No interior members to choose between (e.g. capacity 1): keep the
            # strongest, which is the more useful sparring partner.
            self.members.remove(snaps[0])
            return
        gaps = [(snaps[i + 1].elo - snaps[i - 1].elo, i) for i in range(1, len(snaps) - 1)]
        _, idx = min(gaps)
        self.members.remove(snaps[idx])

    # --- sampling -------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Member | None:
        """Pick an opponent, or None meaning 'mirror self-play this iteration'."""
        snaps, scripted = self.snapshots, self.scripted
        buckets, probs = [], []
        buckets.append(("self", None)); probs.append(self.p_self)
        if snaps:
            buckets.append(("even", snaps)); probs.append(self.p_even)
            buckets.append(("hard", snaps)); probs.append(self.p_hard)
        if scripted:
            buckets.append(("scripted", scripted)); probs.append(self.p_scripted)

        probs = np.asarray(probs, dtype=np.float64)
        probs /= probs.sum()
        kind, pool = buckets[int(rng.choice(len(buckets), p=probs))]
        if kind == "self":
            return None
        if kind == "scripted":
            return pool[int(rng.integers(len(pool)))]
        w = pfsp_weights(np.array([m.win_rate for m in pool]), mode=kind)
        return pool[int(rng.choice(len(pool), p=w))]

    def sample_opponents(self, rng: np.random.Generator, k: int) -> list:
        """`k` opponents for the other seats of one game.

        Drawn independently, so a crowded board is a field of *different*
        checkpoints rather than N-1 copies of one. Facing five identical
        snapshots teaches the learner to beat one opponent five times at once;
        facing five different ones is the position it will actually be in, and
        it gives PFSP a separate result per member instead of one number smeared
        across all of them.

        A None in the list means "the live net" — self-play in that seat, which
        keeps a crowded game from being the learner alone against the whole
        pool when the pool is still small.
        """
        return [self.sample(rng) for _ in range(max(0, k))]

    # --- results and Elo ------------------------------------------------------
    def record_results(self, member: Member | None, results: list[float]) -> None:
        """Record learner outcomes (1 win / 0.5 draw / 0 loss) against `member`."""
        if member is None or not results:
            return
        for r in results:
            member.record(r)
            expected = 1.0 / (1.0 + 10 ** ((member.elo - self.main_elo) / 400.0))
            delta = ELO_K * (r - expected)
            self.main_elo += delta
            if member.kind == "snapshot":
                member.elo -= delta          # frozen opponents still move
            # Scripted baselines are anchors: their Elo is held fixed.

    # --- persistence ----------------------------------------------------------
    # Without this, `--resume` rebuilt an empty pool: the learner would go back
    # to pure mirror self-play until snapshots accumulated again, and Elo across
    # a resume boundary was meaningless. On a box that reboots several times a
    # night that is not a footnote, it is most of the training signal.
    def state_dict(self) -> dict:
        """Serialisable league state. Snapshot weights travel with it."""
        return {
            "capacity": self.capacity,
            "main_elo": self.main_elo,
            "snapshots": [
                {"name": m.name, "iteration": m.iteration, "state": m.state,
                 "elo": m.elo, "wins": m._wins, "games": m._games}
                for m in self.snapshots
            ],
            # Scripted members are rebuilt from their factories by the caller;
            # only their accumulated statistics are worth carrying over.
            "scripted_stats": {
                m.name: {"elo": m.elo, "wins": m._wins, "games": m._games}
                for m in self.scripted
            },
        }

    def load_state_dict(self, blob: dict) -> None:
        """Restore snapshots and Elo. Call *after* re-adding scripted members."""
        self.main_elo = float(blob.get("main_elo", 0.0))
        self.members = [m for m in self.members if m.kind == "scripted"]
        for row in blob.get("snapshots", []):
            self.members.append(Member(
                name=row["name"], kind="snapshot", iteration=row["iteration"],
                state=row["state"], elo=row["elo"],
                _wins=row["wins"], _games=row["games"]))
        for m in self.scripted:
            stats = blob.get("scripted_stats", {}).get(m.name)
            if stats:
                m.elo, m._wins, m._games = (
                    stats["elo"], stats["wins"], stats["games"])

    def table(self) -> list[dict]:
        rows = [{"name": "main", "kind": "main", "elo": self.main_elo,
                 "win_rate": float("nan"), "games": float("nan")}]
        for m in sorted(self.members, key=lambda x: -x.elo):
            rows.append({"name": m.name, "kind": m.kind, "elo": m.elo,
                         "win_rate": m.win_rate, "games": m.games_played})
        return rows

    def render(self) -> str:
        lines = [f"{'opponent':>14} {'kind':>9} {'elo':>7} {'p(win)':>7} {'games':>6}"]
        for r in self.table():
            wr = "  --  " if r["win_rate"] != r["win_rate"] else f"{r['win_rate']:6.1%}"
            gp = "  --  " if r["games"] != r["games"] else f"{r['games']:6.1f}"
            lines.append(f"{r['name']:>14} {r['kind']:>9} {r['elo']:7.0f} {wr} {gp}")
        return "\n".join(lines)


# --- exploiter ----------------------------------------------------------------
@dataclass
class MainExploiter:
    """A second learner that trains only against the current main policy.

    Its job is to find degenerate strategies the main policy is vulnerable to —
    the main learner then has to answer them because the exploiter's snapshots go
    into the league. Reinitialised whenever it beats main convincingly, so it
    keeps hunting for *new* holes rather than milking one.
    """
    reset_threshold: float = 0.70
    decay: float = 0.98
    _wins: float = 0.0
    _games: float = 0.0
    resets: int = 0

    @property
    def win_rate(self) -> float:
        return 0.5 if self._games < 1e-6 else self._wins / self._games

    def record(self, results: list[float]) -> None:
        for r in results:
            self._wins = self._wins * self.decay + r
            self._games = self._games * self.decay + 1.0

    def should_reset(self, min_games: float = 20.0) -> bool:
        return self._games >= min_games and self.win_rate >= self.reset_threshold

    def note_reset(self) -> None:
        self._wins = 0.0
        self._games = 0.0
        self.resets += 1
