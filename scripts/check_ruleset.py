"""Diff the ruleset TOMLs against the cached Polytopia wiki pages.

    python scripts/fetch_wiki.py --all      # populate .wikicache/
    python scripts/check_ruleset.py

Exits non-zero if anything mismatches, so it can gate a ruleset change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyengine.core.rules import load_ruleset
from polyengine.testing.wiki_check import check_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".wikicache")
    ap.add_argument("--strict", action="store_true",
                    help="treat notes (e.g. skill-name drift) as failures too")
    args = ap.parse_args()

    rules = load_ruleset()
    report = check_all(rules, Path(args.cache))
    print(f"ruleset {rules.version} vs wiki cache {args.cache}\n")
    print(report.render())
    bad = not report.clean or (args.strict and report.notes)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
