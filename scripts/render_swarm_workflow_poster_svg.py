#!/usr/bin/env python3
"""Render the public swarm workflow poster as a standalone SVG."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pack_config import load_pack_config, pack_value  # noqa: E402


PACK = load_pack_config(REPO_ROOT)
POSTER = pack_value(PACK, "presentation.workflow_poster", dict)


WIDTH = 1600
HEIGHT = 4040
MARGIN = 30

COLORS = {
    "paper": "#f4ecda",
    "paper_page": "#f8f2e4",
    "paper_page_deep": "#efe3cb",
    "ink": "#142231",
    "muted": "#4d5b68",
    "line": "rgba(20,34,49,0.14)",
    "rule": "rgba(20,34,49,0.24)",
    "teal": "#0f6a70",
    "teal_soft": "#d5ece6",
    "rust": "#b2542d",
    "rust_soft": "#f4d9cb",
    "gold": "#b58a18",
    "gold_soft": "#f4e7bb",
    "slate": "#2d4a66",
    "slate_soft": "#d7e3ef",
    "green": "#2d6546",
    "green_soft": "#d7eadb",
    "white": "#ffffff",
}


def q(value: str) -> str:
    return escape(value, quote=True)


def wrap_text(text: str, width: float, font_size: int, *, factor: float = 0.66) -> list[str]:
    max_chars = max(10, int(width / max(1.0, font_size * factor)))
    return textwrap.wrap(
        text,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )


def fit_text_to_width(
    text: str,
    width: float,
    size: int,
    *,
    factor: float = 0.62,
    min_size: int = 11,
) -> tuple[str, int]:
    current = size
    while current > min_size and len(text) * current * factor > width:
        current -= 1
    if len(text) * current * factor <= width:
        return text, current

    max_chars = max(4, int(width / max(1.0, current * factor)))
    if len(text) <= max_chars:
        return text, current
    return text[: max(1, max_chars - 1)] + "…", current


class SvgWriter:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rx: float = 0,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float = 1,
        opacity: float | None = None,
        filter_ref: str | None = None,
    ) -> None:
        attrs = [
            f'x="{x}"',
            f'y="{y}"',
            f'width="{w}"',
            f'height="{h}"',
            f'rx="{rx}"',
            f'fill="{fill}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{stroke_width}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        if filter_ref:
            attrs.append(f'filter="url(#{filter_ref})"')
        self.add(f"<rect {' '.join(attrs)} />")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 2,
        marker_end: str | None = None,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x1="{x1}"',
            f'y1="{y1}"',
            f'x2="{x2}"',
            f'y2="{y2}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width}"',
            'stroke-linecap="round"',
        ]
        if marker_end:
            attrs.append(f'marker-end="url(#{marker_end})"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add(f"<line {' '.join(attrs)} />")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int,
        fill: str,
        weight: int | str = 400,
        family: str = "Trebuchet MS, Segoe UI, sans-serif",
        letter_spacing: float | None = None,
        anchor: str = "start",
        transform: str | None = None,
    ) -> None:
        attrs = [
            f'x="{x}"',
            f'y="{y}"',
            f'font-size="{size}"',
            f'fill="{fill}"',
            f'font-weight="{weight}"',
            f'font-family="{q(family)}"',
            f'text-anchor="{anchor}"',
            'dominant-baseline="hanging"',
        ]
        if letter_spacing is not None:
            attrs.append(f'letter-spacing="{letter_spacing}"')
        if transform is not None:
            attrs.append(f'transform="{transform}"')
        self.add(f"<text {' '.join(attrs)}>{escape(text)}</text>")

    def wrapped_text(
        self,
        x: float,
        y: float,
        width: float,
        text: str,
        *,
        size: int,
        fill: str,
        line_height: float,
        weight: int | str = 400,
        family: str = "Trebuchet MS, Segoe UI, sans-serif",
        factor: float = 0.66,
        max_lines: int | None = None,
    ) -> float:
        lines = wrap_text(text, width, size, factor=factor)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                lines[-1] = lines[-1].rstrip(".") + "..."
        for idx, line in enumerate(lines):
            self.text(
                x,
                y + idx * line_height,
                line,
                size=size,
                fill=fill,
                weight=weight,
                family=family,
            )
        return y + len(lines) * line_height

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fill: str,
        stroke: str,
        text_fill: str,
        size: int = 18,
        weight: int | str = 600,
    ) -> None:
        self.rect(x, y, w, h, rx=h / 2, fill=fill, stroke=stroke, stroke_width=1)
        fitted_text, fitted_size = fit_text_to_width(text, w - 18, size)
        self.text(x + w / 2, y + (h - fitted_size) / 2 - 1, fitted_text, size=fitted_size, fill=text_fill, weight=weight, anchor="middle")

    def svg(self) -> str:
        return "".join(self.parts)


def card(
    writer: SvgWriter,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    border: str,
    title: str,
    body: str,
    title_size: int = 28,
    body_size: int = 19,
    path: str | None = None,
    accent_bar: str | None = None,
    title_family: str = "Trebuchet MS, Segoe UI, sans-serif",
) -> None:
    writer.rect(x, y, w, h, rx=22, fill=fill, stroke=border, stroke_width=1)
    if accent_bar:
        writer.rect(x + 14, y + 18, 8, h - 36, rx=4, fill=accent_bar)
    tx = x + 26 if not accent_bar else x + 36
    title_bottom = writer.wrapped_text(
        tx,
        y + 20,
        w - (tx - x) - 22,
        title,
        size=title_size,
        fill=COLORS["ink"],
        line_height=title_size + 6,
        weight=700,
        family=title_family,
        factor=0.62,
    )
    writer.wrapped_text(
        tx,
        title_bottom + 10,
        w - (tx - x) - 22,
        body,
        size=body_size,
        fill=COLORS["muted"],
        line_height=body_size + 8,
        factor=0.66,
    )
    if path:
        pill_h = 36
        path_w = w - 44
        writer.pill(
            x + 22,
            y + h - pill_h - 18,
            path_w,
            pill_h,
            path,
            fill="rgba(20,34,49,0.08)",
            stroke="rgba(20,34,49,0.08)",
            text_fill=COLORS["ink"],
            size=15,
            weight=500,
        )


def flow_node(
    writer: SvgWriter,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    label_fill: str,
    title: str,
    body: str,
) -> None:
    writer.rect(x, y, w, h, rx=18, fill="rgba(255,255,255,0.58)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    label_w = 18 + len(label) * 12
    writer.pill(x + 18, y + 16, label_w, 28, label.upper(), fill=f"{label_fill}22", stroke=f"{label_fill}22", text_fill=label_fill, size=13, weight=700)
    writer.text(x + 18, y + 54, title, size=24, fill=COLORS["ink"], weight=700)
    writer.wrapped_text(x + 18, y + 86, w - 36, body, size=15, fill=COLORS["muted"], line_height=21, factor=0.72)


def add_marker_defs(writer: SvgWriter) -> None:
    writer.add(
        """
<defs>
  <linearGradient id="pageGradient" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" stop-color="#f8f2e4" />
    <stop offset="100%" stop-color="#efe3cb" />
  </linearGradient>
  <linearGradient id="posterGradient" x1="0%" y1="0%" x2="0%" y2="100%">
    <stop offset="0%" stop-color="rgba(255,255,255,0.54)" />
    <stop offset="100%" stop-color="rgba(255,255,255,0.30)" />
  </linearGradient>
  <radialGradient id="heroGlow" cx="78%" cy="6%" r="24%">
    <stop offset="0%" stop-color="rgba(15,106,112,0.14)" />
    <stop offset="100%" stop-color="rgba(15,106,112,0)" />
  </radialGradient>
  <radialGradient id="footerGlow" cx="12%" cy="92%" r="26%">
    <stop offset="0%" stop-color="rgba(178,84,45,0.12)" />
    <stop offset="100%" stop-color="rgba(178,84,45,0)" />
  </radialGradient>
  <pattern id="diagPattern" width="18" height="18" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="18" height="18" fill="transparent" />
    <rect width="2" height="18" fill="rgba(20,34,49,0.03)" />
  </pattern>
  <filter id="posterShadow" x="-10%" y="-10%" width="130%" height="130%">
    <feDropShadow dx="0" dy="24" stdDeviation="24" flood-color="rgba(18,28,40,0.13)" />
  </filter>
  <marker id="arrowHead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M0,0 L12,6 L0,12 z" fill="rgba(20,34,49,0.42)" />
  </marker>
</defs>
"""
    )


def render(out_path: Path) -> None:
    s = SvgWriter(WIDTH, HEIGHT)
    s.add('<?xml version="1.0" encoding="UTF-8"?>')
    s.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">'
    )
    s.add("<title id=\"title\">The Repo Is the Shared Memory</title>")
    s.add(
        "<desc id=\"desc\">Public poster showing the workflow of the Autonomous Agentic Research Swarm framework, from locked definitions through Planner, Worker, Judge execution and a concrete research artifact pipeline.</desc>"
    )
    add_marker_defs(s)

    s.rect(0, 0, WIDTH, HEIGHT, fill="url(#pageGradient)")
    s.rect(0, 0, WIDTH, HEIGHT, fill="url(#diagPattern)")
    s.rect(0, 0, WIDTH, HEIGHT, fill="url(#heroGlow)")
    s.rect(0, 0, WIDTH, HEIGHT, fill="url(#footerGlow)")

    poster_x = MARGIN
    poster_y = MARGIN
    poster_w = WIDTH - 2 * MARGIN
    poster_h = HEIGHT - 2 * MARGIN
    s.rect(poster_x, poster_y, poster_w, poster_h, rx=28, fill=COLORS["paper"], stroke="rgba(20,34,49,0.18)", stroke_width=1, filter_ref="posterShadow")
    s.rect(poster_x, poster_y, poster_w, poster_h, rx=28, fill="url(#posterGradient)")

    # Eyebrow
    content_x = poster_x + 40
    content_w = poster_w - 80
    s.text(content_x, 44, "AUTONOMOUS AGENTIC RESEARCH SWARM", size=15, fill=COLORS["teal"], weight=700, letter_spacing=3.2)
    s.pill(poster_x + poster_w - 245, 34, 180, 38, "Public Workflow Poster", fill="rgba(15,106,112,0.08)", stroke="rgba(15,106,112,0.18)", text_fill=COLORS["teal"], size=15, weight=700)
    s.line(content_x, 92, poster_x + poster_w - 40, 92, stroke="rgba(20,34,49,0.24)", stroke_width=1)

    # Hero
    left_x = content_x
    right_x = poster_x + poster_w - 520
    title_family = "Iowan Old Style, Palatino Linotype, Book Antiqua, Georgia, serif"
    s.text(left_x, 120, "THE REPO IS THE", size=82, fill=COLORS["ink"], weight=700, family=title_family)
    s.text(left_x, 198, "SHARED MEMORY", size=82, fill=COLORS["ink"], weight=700, family=title_family)
    hero_body = (
        "A file-based workflow for running multiple research agents in parallel without letting "
        "definitions drift, tasks collide, or outputs become unauditable. This poster maps the "
        "actual framework in this repository, from locked protocol to reproducible figure."
    )
    hero_body_end = s.wrapped_text(left_x, 306, 820, hero_body, size=20, fill=COLORS["muted"], line_height=31, factor=0.7)

    hero_pill_y = hero_body_end + 40
    hero_pill_w = 270
    hero_pill_h = 124
    hero_pills = [
        ("LOCK FIRST", COLORS["teal"], "Protocol, contracts, and workstreams are fixed before workers execute."),
        ("ONE TASK, ONE SCOPE", COLORS["rust"], "Each worker gets a narrow task file, explicit outputs, and an isolated branch."),
        ("AUDIT TRAIL", COLORS["slate"], "Gates, manifests, notes, and git history turn the repo into durable memory."),
    ]
    for idx, (label, label_color, body) in enumerate(hero_pills):
        x = left_x + idx * (hero_pill_w + 16)
        s.rect(x, hero_pill_y, hero_pill_w, hero_pill_h, rx=16, fill="rgba(255,255,255,0.48)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
        s.text(x + 14, hero_pill_y + 12, label, size=14, fill=label_color, weight=700, letter_spacing=2.1)
        s.wrapped_text(x + 14, hero_pill_y + 34, hero_pill_w - 28, body, size=14, fill=COLORS["muted"], line_height=20, factor=0.7)

    # Right aside
    s.rect(right_x, 120, 450, 206, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.text(right_x + 18, 136, "No agent-to-agent chat. Files, tasks,", size=30, fill=COLORS["ink"], weight=700)
    s.text(right_x + 18, 170, "and git history do the coordination.", size=30, fill=COLORS["ink"], weight=700)
    aside_body = (
        "The framework works by making ambiguity expensive: definitions are locked first, "
        "ownership is explicit, and every task is judged against deterministic gates."
    )
    quote_bottom = s.wrapped_text(right_x + 18, 222, 390, aside_body, size=15, fill=COLORS["muted"], line_height=20, factor=0.7)

    fact_cards = [
        ("3", "roles: Planner, Worker, Judge"),
        ("7", "workstreams with strict path ownership"),
        ("1", "shared memory: the repository itself"),
        ("0", "tolerance for silent scope creep"),
    ]
    fact_w = 214
    fact_h = 124
    start_y = quote_bottom + 30
    for idx, (num, body) in enumerate(fact_cards):
        col = idx % 2
        row = idx // 2
        x = right_x + col * (fact_w + 22)
        y = start_y + row * (fact_h + 18)
        s.rect(x, y, fact_w, fact_h, rx=18, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
        s.text(x + 18, y + 16, num, size=38, fill=COLORS["ink"], weight=700)
        s.wrapped_text(x + 18, y + 58, fact_w - 36, body, size=16, fill=COLORS["muted"], line_height=21)

    # Section 01
    fact_bottom = start_y + 2 * fact_h + 18
    sec1_y = max(hero_pill_y + hero_pill_h, fact_bottom) + 88
    s.text(content_x, sec1_y, "01", size=14, fill=COLORS["rust"], weight=700, letter_spacing=2.3)
    s.text(content_x + 38, sec1_y - 8, "Lock the Definitions Before Work Begins", size=50, fill=COLORS["ink"], weight=700, family=title_family)
    sec1_note = (
        "The framework starts by reducing scientific ambiguity. Definitions, schemas, ownership "
        "boundaries, and task contracts are all written down before workers execute."
    )
    s.wrapped_text(content_x, sec1_y + 48, 1180, sec1_note, size=18, fill=COLORS["muted"], line_height=27)

    chain_y = sec1_y + 108
    card_w = 320
    card_h = 240
    gap = 34
    card_specs = [
        (COLORS["teal_soft"], "rgba(15,106,112,0.18)", "Protocol Lock", "Primary metric, units, inclusion criteria, regimes, and tolerances.", "docs/protocol.md"),
        (COLORS["slate_soft"], "rgba(45,74,102,0.18)", "Contracts", "Schemas, decisions, assumptions, and project mode define the canonical interface.", "contracts/*"),
        (COLORS["gold_soft"], "rgba(181,138,24,0.22)", "Workstreams", "Each path has an owner so multiple agents can work without clobbering each other.", ".orchestrator/workstreams.md"),
        (COLORS["rust_soft"], "rgba(178,84,45,0.18)", "Task Files", "One task, one owner, one I/O contract, one set of allowed paths.", ".orchestrator/backlog/*.md"),
    ]
    card_xs = [
        content_x,
        content_x + card_w + gap,
        content_x + 2 * (card_w + gap),
        content_x + 3 * (card_w + gap),
    ]
    for idx, (fill, border, title, body, path) in enumerate(card_specs):
        card(s, card_xs[idx], chain_y, card_w, card_h, fill=fill, border=border, title=title, body=body, path=path)
    for idx in range(3):
        ax1 = card_xs[idx] + card_w + 10
        ax2 = card_xs[idx + 1] - 12
        ay = chain_y + 95
        s.line(ax1, ay, ax2, ay, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")

    # Three column middle
    mid_y = chain_y + card_h + 74
    left_panel_x = content_x
    center_panel_x = 410
    right_panel_x = 1148
    panel_h = 1120
    s.rect(left_panel_x, mid_y, 340, panel_h, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.text(left_panel_x + 22, mid_y + 18, "Three Roles", size=40, fill=COLORS["ink"], weight=700, family=title_family)
    role_cards = [
        ("Planner", COLORS["teal"], "Creates small tasks, defines ownership, and moves lifecycle folders.", [
            "Scopes work into explicit contracts",
            "Owns the control plane under .orchestrator/",
            "Unblocks downstream tasks by sweeping state",
        ]),
        ("Worker", COLORS["rust"], "Executes exactly one task inside an isolated branch or worktree.", [
            "Writes only inside allowed_paths",
            "Produces artifacts, notes, and reproduction commands",
            "Never coordinates by chat with other agents",
        ]),
        ("Judge", COLORS["slate"], "Runs deterministic checks and decides whether work can move forward.", [
            "Runs gates and tests",
            "Checks path ownership and declared outputs",
            "Blocks or approves with explicit reasons",
        ]),
    ]
    role_y = mid_y + 78
    role_w = 292
    role_h = 306
    for idx, (title, accent, body, bullets) in enumerate(role_cards):
        y = role_y + idx * (role_h + 18)
        s.rect(left_panel_x + 24, y, role_w, role_h, rx=18, fill="rgba(255,255,255,0.58)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
        s.rect(left_panel_x + 24 + 14, y + 18, 8, role_h - 36, rx=4, fill=accent)
        tx = left_panel_x + 24 + 34
        s.text(tx, y + 18, title, size=24, fill=COLORS["ink"], weight=700)
        next_y = s.wrapped_text(tx, y + 54, role_w - 48, body, size=14, fill=COLORS["muted"], line_height=20, factor=0.7)
        bullet_y = next_y + 10
        for bullet in bullets:
            s.text(tx, bullet_y, "\u2022", size=18, fill=COLORS["ink"], weight=700)
            bullet_y = s.wrapped_text(tx + 14, bullet_y, role_w - 62, bullet, size=14, fill=COLORS["ink"], line_height=19, factor=0.7)

    s.rect(center_panel_x, mid_y, 680, panel_h, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.text(center_panel_x + 22, mid_y + 18, "One Task, End to End", size=40, fill=COLORS["ink"], weight=700, family=title_family)
    box_w = 260
    box_h = 234
    flow_node(s, center_panel_x + 20, mid_y + 72, box_w, box_h, label="Plan", label_fill=COLORS["teal"], title="Plan", body="A task is written in backlog with dependencies, allowed paths, outputs, and stop conditions.")
    flow_node(s, center_panel_x + 340, mid_y + 72, box_w, box_h, label="Isolate", label_fill=COLORS["rust"], title="Isolate", body="The worker gets a dedicated git branch and worktree so parallel runs do not trample each other.")
    s.line(center_panel_x + 286, mid_y + 170, center_panel_x + 330, mid_y + 170, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")
    flow_node(s, center_panel_x + 20, mid_y + 338, 580, 166, label="Build", label_fill=COLORS["rust"], title="Build", body="Codex executes the task, writes only inside scope, and leaves durable notes in the repo.")
    flow_node(s, center_panel_x + 20, mid_y + 548, box_w, box_h, label="Judge", label_fill=COLORS["slate"], title="Judge", body="make gate, optional make test, path ownership checks, and output existence all run before state changes.")
    flow_node(s, center_panel_x + 340, mid_y + 548, box_w, box_h, label="Publish", label_fill=COLORS["green"], title="Publish", body="The run is committed, pushed, optionally opened as a PR, and recorded as a swarm run manifest.")
    s.line(center_panel_x + 286, mid_y + 665, center_panel_x + 330, mid_y + 665, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")
    flow_node(s, center_panel_x + 20, mid_y + 806, 580, 144, label="Sweep", label_fill=COLORS["green"], title="Sweep", body="Only the Planner moves the task file across lifecycle folders until it reaches review or done.")
    s.rect(center_panel_x + 20, mid_y + 972, 620, 116, rx=18, fill="rgba(255,255,255,0.32)", stroke="rgba(20,34,49,0.26)", stroke_width=1)
    s.text(center_panel_x + 40, mid_y + 990, "Automation Entry Points", size=20, fill=COLORS["ink"], weight=700)
    commands = [
        "python scripts/swarm.py plan",
        "python scripts/swarm.py tick",
        "python scripts/swarm.py run-task --task-id ...",
    ]
    cmd_y = mid_y + 1028
    cmd_x = center_panel_x + 40
    for idx, cmd in enumerate(commands):
        w = min(310, 26 + len(cmd) * 8.5)
        s.pill(cmd_x + (idx % 2) * 300, cmd_y + (idx // 2) * 34, w, 28, cmd, fill="rgba(20,34,49,0.08)", stroke="rgba(20,34,49,0.08)", text_fill=COLORS["ink"], size=13, weight=500)

    s.rect(right_panel_x, mid_y, 382, panel_h, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.wrapped_text(right_panel_x + 22, mid_y + 18, 330, "Why It Does Not Collapse", size=31, fill=COLORS["ink"], line_height=36, weight=700, family=title_family, factor=0.62)
    bullets = [
        "Source-of-truth precedence is explicit: protocol, contracts, workstreams, task.",
        "Raw snapshots are append-only; they are never overwritten in place.",
        "Network access is limited to ETL workstreams configured in framework policy.",
        "Workers cannot silently touch unrelated files without being blocked.",
        "Tasks cannot claim success unless declared outputs actually exist.",
        "Git history becomes a durable audit trail for both science and operations.",
    ]
    bullet_y = mid_y + 118
    for bullet in bullets:
        s.text(right_panel_x + 24, bullet_y, "\u2022", size=20, fill=COLORS["ink"], weight=700)
        bullet_y = s.wrapped_text(right_panel_x + 42, bullet_y, 310, bullet, size=17, fill=COLORS["muted"], line_height=24)
        bullet_y += 12

    # Section 02
    sec2_y = mid_y + panel_h + 78
    s.text(content_x, sec2_y, "02", size=14, fill=COLORS["rust"], weight=700, letter_spacing=2.3)
    s.text(content_x + 38, sec2_y - 8, "Artifact Pipeline: What a Research Slice Actually Produces", size=48, fill=COLORS["ink"], weight=700, family=title_family)
    sec2_note = (
        "The public-facing result is not just code execution. The framework turns one task chain "
        "into evidence, validation, and a visible output."
    )
    s.wrapped_text(content_x, sec2_y + 46, 1180, sec2_note, size=18, fill=COLORS["muted"], line_height=27)

    pipe_y = sec2_y + 114
    pipe_w = 240
    pipe_h = 250
    pipe_gap = 42
    pipeline_cards = [
        ("Raw Snapshot", "Immutable pull from the source system.", "data/raw/<source>/YYYY-MM-DD/...", COLORS["green_soft"], "rgba(45,101,70,0.18)"),
        ("Provenance Manifest", "Hash list, UTC fetch date, and exact reproduction command.", "data/raw_manifest/<source>_YYYY-MM-DD.json", COLORS["green_soft"], "rgba(45,101,70,0.18)"),
        ("Golden Sample", "Tiny tracked dataset used by gates and tests.", "data/samples/<source>/...", COLORS["green_soft"], "rgba(45,101,70,0.18)"),
        ("Validation Report", "Machine-readable JSON plus a short human-readable audit summary.", "reports/validation/*", COLORS["green_soft"], "rgba(45,101,70,0.18)"),
        ("Figure / Table", "Stable, reproducible output for papers, decks, or public presentation.", "reports/figures/*", COLORS["green_soft"], "rgba(45,101,70,0.18)"),
    ]
    pipe_xs = [content_x + idx * (pipe_w + pipe_gap) for idx in range(5)]
    for idx, (title, body, path, fill, border) in enumerate(pipeline_cards):
        card(s, pipe_xs[idx], pipe_y, pipe_w, pipe_h, fill=fill, border=border, title=title, body=body, path=path, title_size=18, body_size=15)
    for idx in range(4):
        s.line(pipe_xs[idx] + pipe_w + 8, pipe_y + pipe_h / 2, pipe_xs[idx + 1] - 8, pipe_y + pipe_h / 2, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")

    # Section 03
    sec3_y = pipe_y + pipe_h + 126
    s.rect(content_x, sec3_y, content_w, 742, rx=24, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.text(content_x + 24, sec3_y + 20, "03", size=14, fill=COLORS["rust"], weight=700, letter_spacing=2.3)
    s.text(content_x + 62, sec3_y + 10, "Concrete Vertical Slice in This Repository", size=44, fill=COLORS["ink"], weight=700, family=title_family)

    slice_x = content_x + 24
    slice_y = sec3_y + 84
    slice_w = 400
    slice_h = 218
    slice_gap_x = 74
    slice_gap_y = 18
    slice_cards = [
        (item["task_id"], COLORS[item["accent"]], item["title"], item["description"])
        for item in POSTER["vertical_slice"]
    ]
    positions = [
        (slice_x, slice_y),
        (slice_x + slice_w + slice_gap_x, slice_y),
        (slice_x, slice_y + slice_h + slice_gap_y),
        (slice_x + slice_w + slice_gap_x, slice_y + slice_h + slice_gap_y),
    ]
    for idx, (task_id, accent, title, body) in enumerate(slice_cards):
        x, y = positions[idx]
        s.rect(x, y, slice_w, slice_h, rx=18, fill="rgba(255,255,255,0.55)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
        badge_w = 74
        s.pill(x + 22, y + 18, badge_w, 28, task_id, fill=accent, stroke=accent, text_fill=COLORS["white"], size=13, weight=700)
        s.wrapped_text(x + 22, y + 64, slice_w - 44, title, size=24, fill=COLORS["ink"], line_height=30, weight=700, factor=0.64)
        s.wrapped_text(x + 22, y + 116, slice_w - 44, body, size=15, fill=COLORS["muted"], line_height=21, factor=0.68)

    s.line(slice_x + slice_w + 12, slice_y + 102, slice_x + slice_w + slice_gap_x - 12, slice_y + 102, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")
    s.line(slice_x + slice_w + 12, slice_y + slice_h + slice_gap_y + 102, slice_x + slice_w + slice_gap_x - 12, slice_y + slice_h + slice_gap_y + 102, stroke="rgba(20,34,49,0.42)", stroke_width=3, marker_end="arrowHead")

    info_x = content_x + 920
    info_y = sec3_y + 84
    info_w = 560
    info_h = 132
    info_cards = [
        ("Primary Research Question", POSTER["research_question"]),
        ("Primary Metric", POSTER["metric_definition"]),
        ("Public Message", "This is not 'AI agents chatting.' It is a reproducible workflow where the repo behaves like a disciplined research operating system."),
    ]
    for idx, (title, body) in enumerate(info_cards):
        y = info_y + idx * (info_h + 18)
        s.rect(info_x, y, info_w, info_h, rx=18, fill="rgba(255,255,255,0.55)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
        s.text(info_x + 18, y + 18, title, size=22, fill=COLORS["ink"], weight=700)
        if idx == 1:
            s.text(info_x + 18, y + 58, body, size=18, fill=COLORS["muted"], family="SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace")
        else:
            s.wrapped_text(info_x + 18, y + 54, info_w - 36, body, size=16, fill=COLORS["muted"], line_height=22)

    # Footer
    foot_y = sec3_y + 742 + 68
    take_x = content_x
    run_x = content_x + 820
    box_h = 332
    s.rect(take_x, foot_y, 760, box_h, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    takeaway = (
        "The framework's public value is not just speed. It is parallel research with visible "
        "contracts, reproducible outputs, and a built-in audit trail."
    )
    s.wrapped_text(take_x + 22, foot_y + 24, 710, takeaway, size=28, fill=COLORS["ink"], line_height=38, weight=700)
    take_note = (
        "If you hang one piece in a gallery, it should communicate that the hard problem is not "
        "\"how do we run more agents?\" but \"how do we keep many agents scientifically coherent?\""
    )
    s.wrapped_text(take_x + 22, foot_y + 184, 710, take_note, size=18, fill=COLORS["muted"], line_height=27)

    s.rect(run_x, foot_y, 710, box_h, rx=22, fill="rgba(255,255,255,0.44)", stroke="rgba(20,34,49,0.14)", stroke_width=1)
    s.text(run_x + 22, foot_y + 18, "Run the Workflow", size=40, fill=COLORS["ink"], weight=700, family=title_family)
    commands = [
        "make gate",
        "make test",
        "python scripts/swarm.py plan",
        "python scripts/swarm.py tick --runner local --max-workers 1 --dry-run",
        "python scripts/sweep_tasks.py --dry-run",
    ]
    cy = foot_y + 88
    for cmd in commands:
        s.text(run_x + 22, cy, "\u2022", size=18, fill=COLORS["ink"], weight=700)
        s.text(run_x + 40, cy - 1, cmd, size=15, fill=COLORS["ink"], family="SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace")
        cy += 36
    s.wrapped_text(run_x + 22, foot_y + 276, 650, "Open this file directly in a browser for display, or print to PDF for a wall poster.", size=18, fill=COLORS["muted"], line_height=27)

    s.add("</svg>")
    out_path.write_text(s.svg(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the swarm workflow poster as SVG.")
    parser.add_argument(
        "--out",
        default="docs/swarm_workflow_poster.svg",
        help="Output SVG path relative to repo root.",
    )
    args = parser.parse_args()
    render(Path(args.out))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
