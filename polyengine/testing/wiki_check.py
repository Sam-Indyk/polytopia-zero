"""Reconcile the ruleset TOMLs against the Polytopia wiki.

The wiki blocks generic scraping but its MediaWiki `api.php` responds normally,
so `scripts/fetch_wiki.py` caches page wikitext locally and this module parses
the infoboxes out of it and diffs them against the loaded `Ruleset`.

That turns the data files from hand-entered numbers into numbers with a
checkable source. It is a *reference* check, not the fidelity gate: the wiki is
community-maintained and self-inconsistent in places (unit pages and the
Technology page disagree about a few unlock techs), so conflicts are reported
rather than silently resolved. The authority for behaviour remains the replay
differential in `polyengine.testing.differential`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE = Path(".wikicache")
TIER_WORDS = {"one": 1, "two": 2, "three": 3}


# --- wikitext parsing ---------------------------------------------------------
def find_template(text: str, name: str) -> str | None:
    """Extract the first {{name ...}} template body, brace-balanced."""
    m = re.search(r"\{\{\s*" + name.replace(" ", r"[ _]") + r"\b", text, re.I)
    if not m:
        return None
    i, depth = m.start(), 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return text[m.start():i]
        else:
            i += 1
    return None


def parse_params(template: str) -> dict[str, str]:
    """Split a template body into its named parameters."""
    body = template[2:-2]
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "{" or ch == "[":
            depth += 1
        elif ch == "}" or ch == "]":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)

    out: dict[str, str] = {}
    for part in parts[1:]:                      # parts[0] is the template name
        if "=" not in part:
            continue
        key, _, val = part.partition("=")
        out[key.strip().lower()] = val.strip()
    return out


def clean(value: str) -> str:
    """Strip wiki markup down to readable text."""
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.S)
    value = re.sub(r"<ref[^>]*/>", "", value)
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("'''", "").replace("''", "")
    return value.strip()


def first_number(value: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(m.group()) if m else None


# --- parsed records -----------------------------------------------------------
@dataclass
class WikiUnit:
    page: str
    name: str = ""
    kind: str = ""                 # "Land" / "Naval"
    unlock: str = ""
    cost: float | None = None
    health: str = ""
    attack: float | None = None
    defence: float | None = None
    movement: float | None = None
    range: float | None = None
    skills: tuple[str, ...] = ()


@dataclass
class WikiTech:
    page: str
    name: str = ""
    tier: int | None = None
    preceded_by: str = ""
    unlocks: str = ""


def _slug(text: str) -> str:
    """'Free Spirit' / 'Roads (Technology)' -> 'free_spirit' / 'roads'."""
    text = re.sub(r"\(.*?\)", "", clean(text))
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def parse_unit_page(page: str, text: str) -> WikiUnit | None:
    tpl = find_template(text, "Infobox Unit")
    if not tpl:
        return None
    p = parse_params(tpl)
    skills = tuple(s.strip().lower() for s in clean(p.get("skills", "")).split(",")
                   if s.strip())
    return WikiUnit(
        page=page,
        name=clean(p.get("name", page)),
        kind=clean(p.get("type", "")),
        unlock=clean(p.get("unlock", "")),
        cost=first_number(p.get("cost", "")),
        health=clean(p.get("health", "")),
        attack=first_number(p.get("attack", "")),
        defence=first_number(p.get("defence", p.get("defense", ""))),
        movement=first_number(p.get("movement", "")),
        range=first_number(p.get("range", "")),
        skills=skills,
    )


def parse_tech_page(page: str, text: str) -> WikiTech | None:
    tpl = find_template(text, "Infobox Technology")
    if not tpl:
        return None
    p = parse_params(tpl)
    tier_raw = clean(p.get("tier", "")).lower()
    tier = TIER_WORDS.get(tier_raw) or (int(first_number(tier_raw))
                                        if first_number(tier_raw) else None)
    return WikiTech(
        page=page,
        name=clean(p.get("name", page.replace("_", " "))),
        tier=tier,
        preceded_by=_slug(p.get("preceded_by", "")),
        unlocks=clean(p.get("unlocks", "")),
    )


def load_cache(cache: Path = DEFAULT_CACHE) -> dict[str, str]:
    if not cache.exists():
        return {}
    return {p.stem: p.read_text(encoding="utf-8") for p in cache.glob("*.wiki")}


# --- reconciliation -----------------------------------------------------------
@dataclass
class Report:
    ok: list[str] = field(default_factory=list)
    diffs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.diffs and not self.missing

    def render(self) -> str:
        out = []
        if self.diffs:
            out.append(f"MISMATCHES ({len(self.diffs)})")
            out += [f"  {d}" for d in self.diffs]
        if self.missing:
            out.append(f"\nMISSING FROM RULESET ({len(self.missing)})")
            out += [f"  {m}" for m in self.missing]
        if self.notes:
            out.append(f"\nNOTES ({len(self.notes)})")
            out += [f"  {n}" for n in self.notes]
        out.append(f"\n{len(self.ok)} checks passed, {len(self.diffs)} mismatched, "
                   f"{len(self.missing)} missing")
        return "\n".join(out)


def _cmp(report: Report, label: str, wiki, ours, tol: float = 1e-6) -> None:
    if wiki is None:
        return
    if abs(float(wiki) - float(ours)) <= tol:
        report.ok.append(label)
    else:
        report.diffs.append(f"{label}: wiki={wiki:g} ours={ours:g}")


def check_units(rules, pages: dict[str, str]) -> Report:
    rep = Report()
    for page, text in sorted(pages.items()):
        unit = parse_unit_page(page, text)
        if not unit or unit.attack is None:
            continue
        key = _slug(unit.name)
        if key not in rules.unit_id:
            rep.missing.append(f"unit {key!r} (wiki page {page}) not in units.toml")
            continue
        u = rules.unit_id[key]
        _cmp(rep, f"{key}.attack", unit.attack, rules.unit_attack[u])
        _cmp(rep, f"{key}.defense", unit.defence, rules.unit_defense[u])
        _cmp(rep, f"{key}.movement", unit.movement, rules.unit_movement[u])
        if unit.range is not None:
            _cmp(rep, f"{key}.range", unit.range, rules.unit_range[u])
        if unit.cost is not None and rules.unit_cost[u] > 0:
            _cmp(rep, f"{key}.cost", unit.cost, rules.unit_cost[u])
        is_naval = unit.kind.lower() == "naval"
        if is_naval != bool(rules.unit_naval[u]):
            rep.diffs.append(f"{key}.naval: wiki={is_naval} ours={bool(rules.unit_naval[u])}")
        else:
            rep.ok.append(f"{key}.naval")
        if unit.health.lower().startswith("varies"):
            rep.notes.append(
                f"{key}: wiki health='Varies' (naval hull inherits the carried "
                f"unit's HP); ours={int(rules.unit_max_hp[u])}")
        else:
            _cmp(rep, f"{key}.max_hp", first_number(unit.health),
                 rules.unit_max_hp[u])
        extra = set(unit.skills) - set(rules.unit_skills[u])
        gone = set(rules.unit_skills[u]) - set(unit.skills)
        if extra or gone:
            rep.notes.append(f"{key}.skills: wiki-only={sorted(extra)} "
                             f"ours-only={sorted(gone)}")
    return rep


def check_techs(rules, pages: dict[str, str]) -> Report:
    rep = Report()
    for page, text in sorted(pages.items()):
        tech = parse_tech_page(page, text)
        if not tech or tech.tier is None:
            continue
        key = _slug(tech.name)
        if key not in rules.tech_id:
            rep.missing.append(f"tech {key!r} (T{tech.tier}, wiki page {page}) "
                               f"not in techs.toml")
            continue
        t = rules.tech_id[key]
        ours_tier = int(rules.tech_tier[t])
        if tech.tier != ours_tier:
            rep.diffs.append(f"{key}.tier: wiki=T{tech.tier} ours=T{ours_tier}")
        else:
            rep.ok.append(f"{key}.tier")
        if tech.preceded_by:
            ours_req = int(rules.tech_requires[t])
            ours_name = rules.tech_names[ours_req] if ours_req >= 0 else ""
            if tech.preceded_by != ours_name:
                rep.diffs.append(f"{key}.requires: wiki={tech.preceded_by!r} "
                                 f"ours={ours_name!r}")
            else:
                rep.ok.append(f"{key}.requires")
    return rep


def check_all(rules, cache: Path = DEFAULT_CACHE) -> Report:
    pages = load_cache(cache)
    if not pages:
        rep = Report()
        rep.notes.append(f"no wiki cache at {cache} — run scripts/fetch_wiki.py --all")
        return rep
    techs, units = check_techs(rules, pages), check_units(rules, pages)
    return Report(ok=techs.ok + units.ok, diffs=techs.diffs + units.diffs,
                  missing=techs.missing + units.missing,
                  notes=techs.notes + units.notes)
