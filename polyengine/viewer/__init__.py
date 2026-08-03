"""The board viewer — replay playback and interactive play, one renderer.

    render_html(replay)      -> a standalone replay page; opens straight off disk
    render_play_html(boot)   -> the same board wired to a local play server

**Why this is a package of real asset files.** The previous viewer kept its
JavaScript and CSS inside a triple-quoted Python string, and Python's escape
rules quietly ate them: `\\25b8` in a CSS `content:` rule parsed as an octal
escape long before the browser saw it, and the glyph came out wrong with nothing
in any log to say so. Every backslash in that file had to be written twice and
one day someone would forget. The assets now live in `assets/` as `.js` and
`.css` files and are inlined verbatim at render time, so what you read is what
the browser gets, your editor highlights it, and the escaping trap is gone.

The page is still a **single self-contained document** — no server, no CDN, no
CORS — because a replay you cannot open by double-clicking it is a replay nobody
looks at.
"""
from __future__ import annotations

import json
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"

# Order matters: `iso` defines the renderer the panels and controllers call.
_SCRIPTS = ("iso.js", "sprites.js", "tech.js", "panels.js")


def _read(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _embed(obj) -> str:
    """JSON for a `<script>` block: no literal `</` may survive inside one."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _page(*, title: str, mode: str, boot: dict, controller: str) -> str:
    """Assemble the single-document page from the template and its assets."""
    scripts = "\n".join(_read(n) for n in _SCRIPTS) + "\n" + _read(controller)
    return (_read("index.html")
            .replace("/*{{CSS}}*/", _read("style.css"))
            .replace("/*{{JS}}*/", scripts)
            .replace('"{{BOOT}}"', _embed(boot))
            .replace("{{MODE}}", mode)
            .replace("{{TITLE}}", title))


def render_html(replay: dict) -> str:
    """A standalone replay viewer with the replay embedded."""
    return _page(title="PolyZero replay", mode="replay",
                 boot=replay, controller="replay.js")


def render_play_html(boot: dict) -> str:
    """The play page. `boot` carries the static header only — the live state is
    fetched from the server, so the same document survives a new game."""
    return _page(title="PolyZero", mode="play",
                 boot=boot, controller="play.js")


def save_html(replay: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(replay), encoding="utf-8")
    return path


__all__ = ["render_html", "render_play_html", "save_html"]
