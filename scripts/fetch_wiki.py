"""Fetch reference pages from the Polytopia wiki via the MediaWiki API.

The wiki blocks generic scraping, but `api.php` responds normally. This pulls
page wikitext into a local cache so the ruleset TOMLs can be reconciled against
the community-documented numbers instead of hand-entered guesses.

    python scripts/fetch_wiki.py --list
    python scripts/fetch_wiki.py Warrior Archery Technology
    python scripts/fetch_wiki.py --all --out .wikicache

Cached pages are plain wikitext; `polyengine.testing.wiki_check` parses the
relevant infoboxes out of them.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://polytopia.fandom.com/api.php"
UA = "polyzero-ruleset-check/0.1 (research; contact via github.com/Sam-Indyk/polytopia-zero)"

# The 25 non-special-tribe technologies, by tier (Whaling and Engineering were
# removed from the game and are deliberately absent).
TECH_PAGES = [
    # T1
    "Climbing", "Fishing", "Hunting", "Organization", "Riding",
    # T2
    "Archery", "Ramming", "Farming", "Forestry", "Free_Spirit",
    "Meditation", "Mining", "Roads_(Technology)", "Sailing", "Strategy",
    # T3
    "Aquatism", "Chivalry", "Construction", "Diplomacy", "Mathematics",
    "Navigation", "Smithery", "Spiritualism", "Trade", "Philosophy",
]

UNIT_PAGES = [
    "Warrior", "Archer", "Rider", "Defender", "Swordsman", "Catapult",
    "Knight", "Mind_Bender", "Giant", "Cloak", "Dagger",
    "Raft", "Scout", "Rammer", "Bomber", "Juggernaut",
]

# Pages that carry the numbers the engine encodes.
CORE_PAGES = [
    "Technology", "List_of_Units", "Units", "Unit_Skills", "Combat",
    "Cities", "Villages", "Terrain", "Resources", "Buildings", "Economy",
    "Stars", "Population", "Movement", "Defence_Bonus", "Ruins",
    *TECH_PAGES, *UNIT_PAGES,
]


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def fetch_page(title: str) -> str | None:
    """Return a page's wikitext, or None if it doesn't exist."""
    data = _get({"action": "query", "prop": "revisions", "rvprop": "content",
                 "rvslots": "main", "titles": title, "redirects": 1})
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1" or "revisions" not in page:
            return None
        return page["revisions"][0]["slots"]["main"]["*"]
    return None


def cache_pages(titles: list[str], out: Path, delay: float = 0.4) -> dict[str, bool]:
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    for title in titles:
        path = out / f"{title.replace('/', '_')}.wiki"
        if path.exists() and path.stat().st_size > 0:
            results[title] = True
            continue
        text = fetch_page(title)
        if text is None:
            results[title] = False
            print(f"  MISSING {title}")
        else:
            path.write_text(text, encoding="utf-8")
            results[title] = True
            print(f"  ok {title} ({len(text)} chars)")
        time.sleep(delay)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*")
    ap.add_argument("--all", action="store_true", help="fetch the core page set")
    ap.add_argument("--list", action="store_true", help="print the core page set")
    ap.add_argument("--out", default=".wikicache")
    args = ap.parse_args()

    if args.list:
        print("\n".join(CORE_PAGES))
        return
    titles = args.titles or (CORE_PAGES if args.all else [])
    if not titles:
        ap.error("give page titles, or --all")
    got = cache_pages(titles, Path(args.out))
    print(f"\n{sum(got.values())}/{len(got)} pages cached in {args.out}")


if __name__ == "__main__":
    main()
