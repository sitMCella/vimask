"""Output rendering (design §8). Fixed template, data from the KB only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from rich.console import Console
from rich.text import Text

from .composer import ComposedAnswer
from .kb import CURSOR, Entry, Example

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z0-9_-]+\}")
_SPECIAL_RE = re.compile(r"<[^<>\s]+>")

Primary = Union[Entry, ComposedAnswer]


@dataclass
class Answer:
    primary: Primary
    alternatives: list[Entry] = field(default_factory=list)
    more: list[Entry] = field(default_factory=list)  # shown by /more
    confidence: str = "high"  # high | medium
    session_mode: str | None = None
    source: str = "retrieval"  # reverse | alias | fast-path | selector | composed
    flavor: str | None = None


@dataclass
class NotFound:
    question: str
    excerpts: list  # list[HelpChunk]
    reason: str = ""


Response = Union[Answer, NotFound]


def keys_text(keys: str) -> Text:
    """Keys with special keys and placeholders highlighted."""
    t = Text()
    pos = 0
    spans = sorted(
        [(m.start(), m.end(), "placeholder") for m in _PLACEHOLDER_RE.finditer(keys)]
        + [(m.start(), m.end(), "special") for m in _SPECIAL_RE.finditer(keys)]
    )
    for start, end, kind in spans:
        if start < pos:
            continue
        t.append(keys[pos:start], style="bold cyan")
        t.append(keys[start:end], style="bold magenta" if kind == "placeholder" else "bold cyan")
        pos = end
    t.append(keys[pos:], style="bold cyan")
    return t


def example_line(label: str, line: str) -> Text:
    t = Text(f"  {label:<8} ", style="dim")
    idx = line.find(CURSOR)
    if idx < 0:
        t.append(line)
        return t
    t.append(line[:idx])
    t.append(CURSOR, style="bold yellow")
    t.append(line[idx + 1 :])
    return t


def render_example(console: Console, ex: Example) -> None:
    before = ex.before.split("\n")
    after = ex.after.split("\n")
    for i, line in enumerate(before[:5]):
        console.print(example_line("before:" if i == 0 else "", line))
    for i, line in enumerate(after[:5]):
        console.print(example_line("after:" if i == 0 else "", line))
    if len(before) > 5 or len(after) > 5:
        console.print(Text("  …", style="dim"))


def _short(desc: str, n: int = 60) -> str:
    desc = desc.rstrip(".")
    return desc if len(desc) <= n else desc[: n - 1].rstrip() + "…"


def _mode_label(mode: str) -> str:
    return {"normal": "", "insert": "insert mode", "visual": "visual mode", "cmdline": "command-line", "ex": ""}[mode]


def render_answer(console: Console, ans: Answer) -> None:
    p = ans.primary
    console.print()
    head = Text("  ")
    head.append_text(keys_text(p.keys))
    mode = p.mode if isinstance(p, Entry) else "normal"
    label = _mode_label(mode)
    if len(p.keys) > 12:
        if label:
            head.append(f"   [{label}]", style="yellow")
        console.print(head)
        head = Text("  " + " " * 10)
        head.append(p.description)
    else:
        head.append(" " * max(1, 11 - len(p.keys)))
        head.append(p.description)
        if label:
            head.append(f"   [{label}]", style="yellow")
    console.print(head)

    if ans.session_mode and ans.session_mode != mode and mode in ("normal", "ex"):
        console.print(Text(f"  press <Esc> first — you are in {ans.session_mode} mode", style="yellow"))
    elif ans.session_mode == "normal" and mode == "visual":
        console.print(Text("  select text first (v / V / <C-v>)", style="yellow"))

    if isinstance(p, Entry):
        if p.placeholders:
            legend = ", ".join(f"{{{n}}}" for n in p.placeholders)
            console.print(Text(f"  replace {legend} with your own text", style="magenta"))
        if p.setup_keys and p.example:
            note = p.setup_desc or f"example assumes {p.setup_keys} was run before"
            console.print(Text(f"  ({note})", style="dim"))
        if p.options:
            console.print(Text(f"  (with :set {' '.join(p.options)})", style="dim"))
        if p.example:
            console.print()
            if p.example.keys and p.example.keys != p.keys:
                console.print(Text(f"  example: {p.example.keys}", style="dim"))
            render_example(console, p.example)
        elif p.effect:
            console.print()
            console.print(Text(f"  effect:  {p.effect}", style="dim"))
        console.print()
        console.print(Text(f"  see:     :help {p.help_tag}", style="dim"))
    else:
        if p.example:
            console.print()
            render_example(console, p.example)
        elif p.note:
            console.print(Text(f"  ({p.note})", style="dim"))
        console.print()
        console.print(Text(f"  see:     :help {p.operator.help_tag}, :help {p.target.help_tag}", style="dim"))
        console.print(
            Text(
                f"  parts:   {p.operator.keys} ({p.operator.description.rstrip('.')}) + "
                f"{p.target.keys} ({p.target.description.rstrip('.')})",
                style="dim",
            )
        )

    for i, a in enumerate(ans.alternatives):
        alts = Text("  also:    " if i == 0 else "           ", style="dim")
        alts.append_text(keys_text(a.keys))
        alts.append(" " * max(1, 11 - len(a.keys)))
        alts.append(_short(a.description), style="dim")
        console.print(alts)
    if ans.confidence == "medium":
        console.print(Text("  (not certain this is what you meant — /more shows other matches)", style="dim italic"))
    console.print()


def render_not_found(console: Console, nf: NotFound) -> None:
    console.print()
    console.print(Text("  Not in knowledge base.", style="bold yellow"))
    if nf.reason:
        console.print(Text(f"  {nf.reason}", style="dim"))
    if nf.excerpts:
        console.print(Text("  Closest :help topics (verbatim):", style="dim"))
        for ch in nf.excerpts:
            console.print()
            console.print(Text(f"  :help {ch.tag}", style="bold"))
            for line in ch.excerpt_lines(4):
                console.print(Text(f"    {line}", style="dim"))
    console.print()


def render_more(console: Console, entries: list[Entry]) -> None:
    if not entries:
        console.print(Text("  nothing more", style="dim"))
        return
    console.print()
    for e in entries:
        line = Text("  ")
        line.append_text(keys_text(e.keys))
        line.append(" " * max(1, 10 - len(e.keys)))
        line.append(e.description)
        line.append(f"   :help {e.help_tag}", style="dim")
        console.print(line)
    console.print()


def render_response(console: Console, resp: Response) -> None:
    if isinstance(resp, Answer):
        render_answer(console, resp)
    else:
        render_not_found(console, resp)


def plain_response(resp: Response) -> str:
    """Render to a plain string (one-shot mode, tests)."""
    console = Console(record=True, width=100, color_system=None, force_terminal=False, highlight=False)
    with console.capture() as cap:
        render_response(console, resp)
    return cap.get()
