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


# Wiki row label -> our improvement name. The Population table names actions
# ("Harvest fruit") where we name improvements ("harvest_fruit").
POPULATION_ROWS = {
    "harvest fruit": "harvest_fruit",
    "hunt animal": "hunt_game",
    "fishing": "fishing",
    "lumber hut": "lumber_hut",
    "farm/aqua farm": "farm",
    "mine": "mine",
    "port": "port",
    "temple": "temple",
}


def _cell_value(cell: str) -> str:
    """Drop a table cell's HTML attributes, keeping only its content.

    Rows like `rowspan="6" |1` would otherwise yield 6 (the span) rather than 1
    (the value) — which is exactly how the Temple row first mis-compared.
    """
    if "|" in cell and re.search(r'\b(row|col)span\s*=', cell):
        return cell.rsplit("|", 1)[1]
    return cell


def parse_population_table(text: str) -> dict[str, tuple[int, int]]:
    """Parse the Population page's table into {improvement: (population, cost)}.

    Rows are `Item | Population | Cost | pop-per-star | stars-per-pop | Unlocked by`.
    Rows using rowspan (the Temple block) or describing combinations
    ("Grow Forest + Farm") are skipped — only the plain single-improvement rows
    are unambiguous enough to check automatically.
    """
    out: dict[str, tuple[int, int]] = {}
    for block in re.split(r"\n\|-", text):
        cells = [c.strip() for c in block.split("\n|") if c.strip()]
        if len(cells) < 3:
            continue
        label = clean(cells[0]).lower().strip()
        key = POPULATION_ROWS.get(label)
        if key is None:
            continue
        pop = first_number(_cell_value(cells[1]))
        cost = first_number(_cell_value(cells[2]))
        if pop is None or cost is None:
            continue
        out[key] = (int(pop), int(cost))
    return out


def check_improvements(rules, pages: dict[str, str]) -> Report:
    """Diff improvement costs and population yields against the wiki table."""
    rep = Report()
    text = pages.get("Population")
    if not text:
        rep.notes.append("no Population page cached — improvements unchecked")
        return rep
    table = parse_population_table(text)
    if not table:
        rep.notes.append("Population table parsed to nothing (format changed?)")
        return rep
    for name, (pop, cost) in sorted(table.items()):
        if name not in rules.imp_id:
            rep.missing.append(f"improvement {name!r} on the wiki but not in "
                               f"improvements.toml")
            continue
        i = rules.imp_id[name]
        _cmp(rep, f"{name}.cost", cost, rules.imp_cost[i])
        _cmp(rep, f"{name}.population", pop, rules.imp_population[i])
    return rep


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
    """Compare a wiki value with ours. Numbers within `tol`, anything else by
    equality — the temple and skill checks compare names and terrain sets."""
    if wiki is None:
        return
    try:
        same = abs(float(wiki) - float(ours)) <= tol
        detail = f"wiki={float(wiki):g} ours={float(ours):g}"
    except (TypeError, ValueError):
        same = wiki == ours
        detail = f"wiki={wiki!r} ours={ours!r}"
    if same:
        report.ok.append(label)
    else:
        report.diffs.append(f"{label}: {detail}")


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


def parse_defence_bonuses(text: str) -> dict[str, float] | None:
    """The three defence multipliers, stated in one sentence of Defence_Bonus.

    "defenceBonus is 1 when there is no defence bonus, 1.5 for the standard
    forest/mountain/ocean/city bonus, 4 for the city wall bonus."

    Each number is anchored to the phrase that follows it rather than taken
    positionally, so unrelated numbers elsewhere on the page cannot shift them.
    """
    patterns = {
        "none": r"([\d.]+)\s+when there is no defence bonus",
        "standard": r"([\d.]+)\s+for the standard",
        "wall": r"([\d.]+)\s+for the city wall",
    }
    found = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            found[key] = float(m.group(1))
    return found or None


def check_combat(rules, pages: dict[str, str]) -> Report:
    """Defence multipliers.

    These went unchecked for a long time, and the city-wall bonus was wrong by
    a factor of 2.7 the whole while (1.5 instead of 4) — 164 other checks passed
    around it. Anything the combat maths reads from data belongs here.
    """
    rep = Report()
    text = pages.get("Defence_Bonus") or pages.get("Combat")
    if not text:
        rep.notes.append("no Defence_Bonus page cached; defence bonuses unchecked")
        return rep
    wiki = parse_defence_bonuses(text)
    if not wiki:
        rep.notes.append("could not parse the defence-bonus sentence")
        return rep

    ours = rules.defense_bonus_map
    if "none" in wiki:
        _cmp(rep, "defence_bonus none", wiki["none"], ours["none"])
    if "standard" in wiki:
        # One documented value covers the standard terrain and bare-city bonus.
        for key in ("city", "mountain", "forest"):
            if key in ours:
                _cmp(rep, f"defence_bonus {key}", wiki["standard"], ours[key])
    if "wall" in wiki and "city_walls" in ours:
        _cmp(rep, "defence_bonus city_walls", wiki["wall"], ours["city_walls"])
    return rep


def parse_other_improvement_costs(text: str) -> dict[str, int]:
    """Costs from the "Other improvements" table on Buildings.

    The rows are pipe-delimited wikitext ending in the star cost, e.g.
    "|Roads |...|Field, Forest |Roads |3". Anchored on the improvement name so
    row order cannot shift the numbers.
    """
    section = text.split("Other improvements", 1)[-1]
    out: dict[str, int] = {}
    for block in re.split(r"\n\|-", section):
        cells = [c.strip() for c in block.split("\n|") if c.strip()]
        if len(cells) < 3:
            continue
        # Image | Name | Description | Terrain | Unlocked by | Cost.
        name = clean(cells[1]).lower()
        cost = first_number(cells[-1])
        if cost is None or not name:
            continue
        out[name] = int(cost)
    return out


def check_new_improvements(rules, pages: dict[str, str]) -> Report:
    """Costs for roads, bridges and the market."""
    rep = Report()
    text = pages.get("Buildings")
    if not text:
        rep.notes.append("no Buildings page cached; road/bridge/market costs unchecked")
        return rep
    wiki = parse_other_improvement_costs(text)
    for wiki_name, our_name in (("roads", "road"), ("bridge", "bridge"),
                                ("market", "market")):
        if wiki_name in wiki and our_name in rules.imp_id:
            _cmp(rep, f"{our_name} cost", wiki[wiki_name],
                 int(rules.imp_cost[rules.imp_id[our_name]]))

    # The Market's rate is stated differently on three pages; surface the
    # disagreement rather than silently trusting the majority.
    resources = pages.get("Resources", "")
    m = re.search(r"\[\[Market\]\]\s*\|\s*(\d+) per each level", resources)
    if m:
        ours = int(rules.imp_income_per_level[rules.imp_id["market"]])
        # Not a mismatch: Resources says 2, Buildings says 1, and 1 is correct
        # (confirmed by the project owner). Reported as a note so the next
        # session does not "fix" it back.
        rep.notes.append(
            f"market rate: Resources says {m.group(1)} per level, Buildings "
            f"says 1; ours={ours}. 1 is correct -- do not change it back.")
    return rep


def check_temples(rules, pages: dict[str, str]) -> Report:
    """The four temples: terrain, unlocking technology and cost.

    The Population page lists them with `rowspan`s the generic table parser
    cannot follow, which is exactly how three of the four temples sat missing
    from the ruleset without a single check complaining. The Temples page has
    the same data in a flat table, so it is read directly.
    """
    rep = Report()
    text = pages.get("Temples")
    if not text:
        rep.notes.append("no Temples page cached; temple costs unchecked")
        return rep

    # Rows read: | Name | Terrain | [[Tech]] | cost
    pattern = re.compile(
        r"\|\s*([A-Za-z ]*Temple)\s*\n"
        r"\|\s*([A-Za-z, ]+)\s*\n"
        r"\|\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]\s*\n"
        r"\|\s*(\d+)")
    seen = 0
    for name, terrain, tech, cost in pattern.findall(text):
        key = name.strip().lower().replace(" ", "_")
        if key not in rules.imp_id:
            # Ice Temple is Polaris-only, and special tribes are excluded.
            if "ice" not in key:
                rep.missing.append(f"{name} on the wiki but not in "
                                   f"improvements.toml")
            continue
        seen += 1
        imp = rules.imp_id[key]
        _cmp(rep, f"{key} cost", int(cost), int(rules.imp_cost[imp]))
        wiki_tech = tech.strip().lower().replace(" ", "_")
        our_tech = int(rules.imp_requires_tech[imp])
        our_name = rules.tech_names[our_tech] if our_tech >= 0 else "-"
        _cmp(rep, f"{key} tech", wiki_tech, our_name)
        # Terrain, as named in the same row.
        wiki_terrains = {t.strip().lower() for t in terrain.split(",")}
        wiki_terrains = {"water" if t == "shallow water" else t
                         for t in wiki_terrains}
        ours = {rules.terrain_names[t] for t in rules.imp_terrains[imp]}
        _cmp(rep, f"{key} terrain", sorted(wiki_terrains), sorted(ours))
    if seen == 0:
        rep.notes.append("Temples table parsed to nothing (format changed?)")
    return rep


def check_unit_skills(rules, pages: dict[str, str]) -> Report:
    """Which units carry which skill, against the Unit Skills page's own table.

    Eleven skills once sat in `units.toml` read by no code, and nothing noticed
    because nothing compared the *skill lists* to the wiki. This checks the
    assignment; `tests/test_unit_skills.py` checks that each one does something.
    """
    rep = Report()
    text = pages.get("Unit Skills")
    if not text:
        rep.notes.append("no Unit Skills page cached; skills unchecked")
        return rep

    ours = {name: set() for name in rules.unit_names}
    for uid, name in enumerate(rules.unit_names):
        ours[name] = set(rules.unit_skills[uid])

    # Each block is: |'''Skill''' ... | description | list of [[Unit]] links.
    blocks = re.split(r"\n\|-\n", text)
    checked = 0
    for block in blocks:
        m = re.match(r"\|'''([A-Za-z ]+)'''", block.strip())
        if not m:
            continue
        skill = m.group(1).strip().lower()
        if skill not in {s for skills in ours.values() for s in skills}:
            continue                       # a skill no base unit has
        listed = set()
        for target, shown in re.findall(
                r"\*\s*\[\[([^\]|]+)(?:\|([^\]]*))?\]\]", block):
            # `[[Swordsman|Mantis]]` is Cymanti's unit linked through the
            # Swordsman page — the *display* name is the unit, so a piped link
            # must never be credited to its target. Special tribes are excluded,
            # so those names simply match nothing.
            name = (shown or target).strip()
            key = name.lower().replace(" ", "_").replace("-", "_")
            if key in ours:
                listed.add(key)
        if not listed:
            continue
        checked += 1
        for unit in listed:
            if skill not in ours[unit]:
                rep.diffs.append(
                    f"{unit}.{skill}: wiki lists it, units.toml does not")
            else:
                rep.ok.append(f"{unit}.{skill}")
    if checked == 0:
        rep.notes.append("Unit Skills table parsed to nothing (format changed?)")
    return rep


def check_embassy(rules, pages: dict[str, str]) -> Report:
    """The Embassy's cost and income, which no other check covers.

    It is absent from the Population table (it produces no population), so it
    sat at cost 4 / income 1 against the wiki's 5 / 2 with 177 checks reporting
    nothing wrong.
    """
    rep = Report()
    text = pages.get("Embassy")
    if not text or "embassy" not in rules.imp_id:
        rep.notes.append("no Embassy page cached; embassy unchecked")
        return rep
    imp = rules.imp_id["embassy"]

    m = re.search(r"\|\s*cost\s*=\s*(\d+)", text)
    if m:
        _cmp(rep, "embassy cost", int(m.group(1)), int(rules.imp_cost[imp]))
    m = re.search(r"produces (\w+) \[\[star\]\]s per turn", text)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    if m and m.group(1) in words:
        _cmp(rep, "embassy income", words[m.group(1)],
             int(rules.imp_income_flat[imp]))
    # "doubled to four stars per turn to each party if they have a Peace Treaty"
    m = re.search(r"doubled to (\w+) stars per turn", text)
    if m and m.group(1) in words:
        _cmp(rep, "embassy peace income", words[m.group(1)],
             int(rules.imp_income_flat[imp]) * 2)
    return rep


def check_all(rules, cache: Path = DEFAULT_CACHE) -> Report:
    pages = load_cache(cache)
    if not pages:
        rep = Report()
        rep.notes.append(f"no wiki cache at {cache} — run scripts/fetch_wiki.py --all")
        return rep
    parts = [check_techs(rules, pages), check_units(rules, pages),
             check_improvements(rules, pages), check_combat(rules, pages),
             check_new_improvements(rules, pages), check_temples(rules, pages),
             check_unit_skills(rules, pages), check_embassy(rules, pages)]
    return Report(ok=[x for p in parts for x in p.ok],
                  diffs=[x for p in parts for x in p.diffs],
                  missing=[x for p in parts for x in p.missing],
                  notes=[x for p in parts for x in p.notes])
