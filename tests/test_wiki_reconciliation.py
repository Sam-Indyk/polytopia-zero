"""The ruleset must stay reconciled with the wiki.

These run only when a wiki cache is present (`python scripts/fetch_wiki.py --all`),
so the suite still passes offline / in CI without network. When the cache *is*
there, any drift between the TOMLs and the community-documented numbers fails
the build — that is what keeps the data files from silently rotting as the game
gets patched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from polyengine.core.rules import load_ruleset
from polyengine.testing.wiki_check import (
    check_all,
    find_template,
    load_cache,
    parse_params,
    parse_tech_page,
    parse_unit_page,
)

CACHE = Path(".wikicache")
pages = load_cache(CACHE)
needs_cache = pytest.mark.skipif(not pages, reason="no .wikicache (run scripts/fetch_wiki.py --all)")
R = load_ruleset()


# --- parser unit tests (no cache needed) --------------------------------------
def test_find_template_is_brace_balanced():
    text = "intro {{Infobox Unit|name=X|cost={{nested|1}}|attack=2}} tail"
    tpl = find_template(text, "Infobox Unit")
    assert tpl.endswith("}}") and "attack=2" in tpl and "tail" not in tpl


def test_parse_params_splits_on_top_level_pipes_only():
    tpl = "{{Infobox Unit|name=Scout|skills=Water, Dash|link=[[A|B]]|attack=2}}"
    p = parse_params(tpl)
    assert p["name"] == "Scout"
    assert p["skills"] == "Water, Dash"
    assert p["attack"] == "2"
    assert p["link"] == "[[A|B]]"          # pipe inside a wikilink not split


def test_parse_unit_page_reads_an_inline_infobox():
    text = ("{{Infobox_Unit|name=Rammer\n| type = Naval\n"
            "|attack=3|defence=3|movement=3|range=1|health=Varies"
            "|skills=Water, Dash, Carry|cost=5 stars to upgrade from a Raft}}")
    u = parse_unit_page("Rammer", text)
    assert (u.attack, u.defence, u.movement, u.range) == (3.0, 3.0, 3.0, 1.0)
    assert u.cost == 5.0 and u.kind == "Naval" and u.health == "Varies"
    assert "carry" in u.skills


def test_parse_tech_page_maps_tier_words():
    text = "{{Infobox Technology| tier = Three| preceded_by = [[Ramming]]}}"
    t = parse_tech_page("Aquatism", text)
    assert t.tier == 3 and t.preceded_by == "ramming"


# --- reconciliation against the real cache ------------------------------------
@needs_cache
def test_ruleset_matches_the_wiki():
    report = check_all(R, CACHE)
    assert report.clean, "\n" + report.render()


@needs_cache
def test_reconciliation_actually_checked_a_lot():
    """Guard against the checker silently no-oping (e.g. a parser regression)."""
    report = check_all(R, CACHE)
    assert len(report.ok) > 100, f"only {len(report.ok)} checks ran"


@needs_cache
def test_every_engine_tech_exists_on_the_wiki():
    wiki_techs = set()
    for page, text in pages.items():
        t = parse_tech_page(page, text)
        if t and t.tier:
            from polyengine.testing.wiki_check import _slug
            wiki_techs.add(_slug(t.name))
    missing = set(R.tech_names) - wiki_techs
    assert not missing, f"techs with no wiki page: {sorted(missing)}"


@needs_cache
def test_every_engine_unit_exists_on_the_wiki():
    wiki_units = set()
    for page, text in pages.items():
        u = parse_unit_page(page, text)
        if u and u.attack is not None:
            from polyengine.testing.wiki_check import _slug
            wiki_units.add(_slug(u.name))
    missing = set(R.unit_names) - wiki_units
    assert not missing, f"units with no wiki page: {sorted(missing)}"
