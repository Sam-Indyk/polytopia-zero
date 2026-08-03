"""A small always-on window showing training progress. No dependencies.

    .venv\\Scripts\\pythonw.exe scripts/watch_training.py
    .venv\\Scripts\\python.exe  scripts/watch_training.py --log runs_big.log --target 4000

Reads the trainer's log file and nothing else, so it is safe to open, close, and
reopen at any time, and it cannot disturb the run it is watching. Use `pythonw`
to launch it without a console window attached.

It reports the two things worth knowing at a glance -- how far along the run is,
and whether the win rates are still going up -- plus the restart count, which on
this machine is how you notice the box has been bugchecking overnight (see the
hardware note in CLAUDE.md).
"""
from __future__ import annotations

import argparse
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

# `it   53 | scripted | wr 27.8% | steps 6763 | eps 36 | ... | elo -175 | 920s`
ITER_RE = re.compile(
    r"^it\s+(\d+)\s*\|\s*([^|]+?)\s*\|.*?\belo\s*([+-]?\d+)\s*\|\s*(\d+)s")
# `  eval@50: Random 96.9%+-3.1% | OSLA 90.6%+-5.2% (held out) | Scripted 43.8%...`
EVAL_RE = re.compile(
    r"eval@(\d+):\s*Random\s+([\d.]+)%.*?OSLA\s+([\d.]+)%.*?"
    r"Scripted\s+([\d.]+)%.*?elo\s*([+-]?\d+)")
ATTEMPT_RE = re.compile(r"attempt\s+(\d+)\s+started")

BG, FG, DIM = "#12141a", "#e8eaf0", "#8b93a7"
ACCENT, GOOD, WARN = "#6ea8fe", "#5dd39e", "#ffd166"


@dataclass
class Snapshot:
    iteration: int = 0
    opponent: str = "-"
    elo: int = 0
    attempts: int = 0
    secs_per_iter: float = 0.0
    evals: list = None            # (iter, random, osla, scripted, elo)
    stale: bool = False

    def __post_init__(self):
        if self.evals is None:
            self.evals = []


def parse(path: Path) -> Snapshot:
    snap = Snapshot()
    if not path.exists():
        return snap
    try:
        # utf-8-sig: the supervisor's first Add-Content writes a BOM.
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return snap

    marks: list[tuple[int, int]] = []          # (iteration, elapsed seconds)
    for line in text.splitlines():
        m = ITER_RE.match(line)
        if m:
            snap.iteration = int(m.group(1))
            snap.opponent = m.group(2).strip()
            snap.elo = int(m.group(3))
            marks.append((int(m.group(1)), int(m.group(4))))
            continue
        m = EVAL_RE.search(line)
        if m:
            snap.evals.append((int(m.group(1)), float(m.group(2)),
                               float(m.group(3)), float(m.group(4)),
                               int(m.group(5))))
            continue
        m = ATTEMPT_RE.search(line)
        if m:
            snap.attempts += 1

    # Pace, from consecutive iterations inside one attempt. The elapsed counter
    # restarts at every resume, so pairs that go backwards are dropped rather
    # than averaged in as negative time.
    deltas = []
    for (i0, t0), (i1, t1) in zip(marks, marks[1:]):
        if i1 > i0 and t1 > t0:
            deltas.append((t1 - t0) / (i1 - i0))
    if deltas:
        recent = deltas[-25:]
        snap.secs_per_iter = sum(recent) / len(recent)
    return snap


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "--"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


class Watcher(tk.Tk):
    def __init__(self, log: Path, target: int, every_ms: int):
        super().__init__()
        self.log, self.target, self.every_ms = log, target, every_ms
        self.title("PolyZero - training")
        self.configure(bg=BG)
        self.geometry("470x680")
        self.minsize(420, 560)
        self.attributes("-topmost", True)

        pad = dict(bg=BG, fg=FG)
        tk.Label(self, text=log.name, font=("Consolas", 9), bg=BG, fg=DIM,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 0))

        self.big = tk.Label(self, text="--", font=("Segoe UI", 46, "bold"),
                            bg=BG, fg=FG)
        self.big.pack(pady=(2, 0))
        self.sub = tk.Label(self, text="", font=("Segoe UI", 10), bg=BG, fg=DIM)
        self.sub.pack()

        self.bar = tk.Canvas(self, height=8, bg="#20242e", highlightthickness=0)
        self.bar.pack(fill="x", padx=14, pady=(12, 4))
        self.pace = tk.Label(self, text="", font=("Segoe UI", 9), bg=BG, fg=DIM)
        self.pace.pack()

        tk.Frame(self, height=1, bg="#262b36").pack(fill="x", padx=14, pady=10)

        self.latest = tk.Label(self, text="no eval yet", justify="left",
                               font=("Consolas", 11), bg=BG, fg=FG, anchor="w")
        self.latest.pack(fill="x", padx=14)

        self.chart = tk.Canvas(self, height=130, bg="#171a21",
                               highlightthickness=0)
        self.chart.pack(fill="x", padx=14, pady=10)

        # Pack the status strip against the bottom edge *before* the table, so
        # the table's expand can never push it off the window.
        self.status = tk.Label(self, text="", font=("Segoe UI", 9), bg=BG,
                               fg=DIM, anchor="w")
        self.status.pack(side="bottom", fill="x", padx=14, pady=(0, 10))

        self.table = tk.Label(self, text="", justify="left",
                              font=("Consolas", 9), bg=BG, fg=DIM, anchor="nw")
        self.table.pack(fill="both", expand=True, padx=14)

        self.refresh()

    def refresh(self) -> None:
        s = parse(self.log)
        self.big.config(text=str(s.iteration) if s.iteration else "--")
        self.sub.config(text=f"of {self.target}   |   vs {s.opponent}   "
                             f"|   elo {s.elo:+d}")

        frac = min(1.0, s.iteration / self.target) if self.target else 0.0
        self.bar.delete("all")
        w = max(1, self.bar.winfo_width())
        self.bar.create_rectangle(0, 0, w * frac, 8, fill=ACCENT, width=0)

        if s.secs_per_iter:
            left = (self.target - s.iteration) * s.secs_per_iter
            self.pace.config(text=f"{s.secs_per_iter:.0f}s / iteration   "
                                  f"|   {frac:.1%} done   |   ~{fmt_eta(left)} left")
        else:
            self.pace.config(text="waiting for the first iteration...")

        if s.evals:
            it, rnd, osla, scr, _elo = s.evals[-1]
            prev = s.evals[-2] if len(s.evals) > 1 else None
            def delta(cur, idx):
                if not prev:
                    return "     "
                d = cur - prev[idx]
                return f" {d:+5.1f}"
            self.latest.config(
                text=(f"eval @ {it}\n"
                      f"  Random    {rnd:5.1f}%{delta(rnd, 1)}\n"
                      f"  OSLA      {osla:5.1f}%{delta(osla, 2)}   (held out)\n"
                      f"  Scripted  {scr:5.1f}%{delta(scr, 3)}"))
            rows = ["  iter   rand   osla   scr"]
            for e in s.evals[-7:]:
                rows.append(f"  {e[0]:>4}  {e[1]:5.1f}  {e[2]:5.1f}  {e[3]:5.1f}")
            self.table.config(text="\n".join(rows))
        self.draw_chart(s)

        bits = [f"{s.attempts} start(s)"]
        if s.attempts > 1:
            bits.append(f"{s.attempts - 1} restart(s)")
        self.status.config(text="   |   ".join(bits),
                           fg=WARN if s.attempts > 1 else DIM)

        self.after(self.every_ms, self.refresh)

    def draw_chart(self, s: Snapshot) -> None:
        c = self.chart
        c.delete("all")
        w, h = max(1, c.winfo_width()), 130
        if len(s.evals) < 2:
            c.create_text(w // 2, h // 2, text="chart appears after two evals",
                          fill=DIM, font=("Segoe UI", 9))
            return
        pad_l, pad_r, pad_t, pad_b = 30, 10, 12, 18
        xs = [e[0] for e in s.evals]
        x0, x1 = min(xs), max(xs)
        span = max(1, x1 - x0)

        for pct in (0, 50, 100):                    # gridlines
            y = pad_t + (h - pad_t - pad_b) * (1 - pct / 100)
            c.create_line(pad_l, y, w - pad_r, y, fill="#242833")
            c.create_text(pad_l - 6, y, text=f"{pct}", fill=DIM, anchor="e",
                          font=("Consolas", 7))

        for idx, colour, label in ((2, GOOD, "OSLA"), (3, ACCENT, "Scripted")):
            pts = []
            for e in s.evals:
                x = pad_l + (w - pad_l - pad_r) * (e[0] - x0) / span
                y = pad_t + (h - pad_t - pad_b) * (1 - e[idx] / 100)
                pts += [x, y]
            if len(pts) >= 4:
                # Deliberately NOT smoothed: a spline through few points rounds
                # off real movement, which made a genuine drop in the ScriptedBot
                # win rate read as a flat line.
                c.create_line(*pts, fill=colour, width=2)
            c.create_oval(pts[-2] - 3, pts[-1] - 3, pts[-2] + 3, pts[-1] + 3,
                          fill=colour, width=0)
        c.create_text(pad_l + 4, pad_t + 2, text="OSLA", fill=GOOD, anchor="nw",
                      font=("Segoe UI", 8))
        c.create_text(pad_l + 44, pad_t + 2, text="Scripted", fill=ACCENT,
                      anchor="nw", font=("Segoe UI", 8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs_big.log")
    ap.add_argument("--target", type=int, default=4000,
                    help="total iterations, for the progress bar and ETA")
    ap.add_argument("--every", type=int, default=5,
                    help="refresh interval in seconds")
    args = ap.parse_args()
    Watcher(Path(args.log), args.target, max(1, args.every) * 1000).mainloop()


if __name__ == "__main__":
    main()
