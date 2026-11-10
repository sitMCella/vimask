"""Importer: bootstrap draft entries from ``$VIMRUNTIME/doc/index.txt``.

``index.txt`` lists every default key with its help tag and a one-line
description, per mode. The importer produces *draft* YAML (``verified:
none``, no aliases, no example) in an output directory for a human to
review and move into ``kb/``. It also powers ``vimask kb check``, which
cross-checks hand-written ``keys``/``help_tag`` pairs against the index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..help_index import runtime_doc_dir

_SECTION_RE = re.compile(r"^\d+(?:\.\d+)?\.?\s+(.+?)\s+\*([^*]+)\*\s*$")
_ROW_RE = re.compile(r"^\|([^|]+)\|\s+(\S.*?)(?:\s{2,}|\t)(?:([12])\s+)?(\S.*)$")
_ROW_NOTE_RE = re.compile(r"^\|([^|]+)\|\s+(\S+)\s+([12])\s+(\S.*)$")

_SECTION_MODE = {
    "insert-index": "insert",
    "normal-index": "normal",
    "objects": "normal",
    "CTRL-W": "normal",
    "[": "normal",
    "g": "normal",
    "z": "normal",
    "operator-pending-index": "normal",
    "visual-index": "visual",
    "ex-edit-index": "cmdline",
    "terminal-job-index": "normal",
    "ex-cmd-index": "ex",
}


@dataclass
class IndexRow:
    tag: str
    keys: str
    description: str
    section: str
    mode: str
    flavor: list[str] = field(default_factory=list)


def ctrl_to_notation(keys: str) -> str:
    """``CTRL-W v`` -> ``<C-w>v``; ``CTRL-R {register}`` -> ``<C-r>{register}``."""
    out = re.sub(r"CTRL-(\S)", lambda m: f"<C-{m.group(1).lower()}>", keys)
    out = out.replace(" ", "")
    return out


def parse_index(path: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    section = ""
    mode = "normal"
    pending: IndexRow | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = raw.rstrip()
        m = _SECTION_RE.match(line)
        if m:
            section = m.group(2)
            mode = _SECTION_MODE.get(section, mode)
            pending = None
            continue
        if line.startswith("|"):
            m = _ROW_NOTE_RE.match(line) or _ROW_RE.match(line)
            if not m:
                pending = None
                continue
            groups = m.groups()
            if len(groups) == 4:
                tag, keys, _note, desc = groups
            else:
                tag, keys, desc = groups[0], groups[1], groups[-1]
            pending = IndexRow(tag.strip(), ctrl_to_notation(keys.strip()), desc.strip(), section, mode)
            rows.append(pending)
        elif pending and line.startswith("\t\t\t\t") and line.strip():
            pending.description += " " + line.strip()
        else:
            pending = None
    return rows


def import_all(out_dir: Path, flavors: dict[str, str | None]) -> dict[str, int]:
    """Parse index.txt of every available flavor and write draft YAML files."""
    merged: dict[tuple[str, str], IndexRow] = {}
    for flavor, exe in flavors.items():
        doc = runtime_doc_dir(flavor, exe) if exe else None
        if not doc or not (doc / "index.txt").is_file():
            continue
        for row in parse_index(doc / "index.txt"):
            key = (row.tag, row.keys)
            if key in merged:
                if flavor not in merged[key].flavor:
                    merged[key].flavor.append(flavor)
            else:
                row.flavor = [flavor]
                merged[key] = row
    out_dir.mkdir(parents=True, exist_ok=True)
    by_section: dict[str, list[dict]] = {}
    for row in merged.values():
        ident = re.sub(r"[^a-z0-9]+", "-", row.tag.lower()).strip("-") or "entry"
        kind = "ex" if row.mode == "ex" else "command"
        if row.section == "objects":
            kind = "textobject"
        entry = {
            "id": f"draft-{ident}",
            "keys": row.keys,
            "kind": kind,
            "mode": row.mode,
            "flavor": row.flavor,
            "category": f"import/{row.section}",
            "description": row.description,
            "aliases": [],
            "help_tag": row.tag,
            "verified": "none",
        }
        by_section.setdefault(row.section, []).append(entry)
    counts = {}
    for section, entries in by_section.items():
        name = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-") or "misc"
        path = out_dir / f"_draft-{name}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(entries, fh, allow_unicode=True, sort_keys=False)
        counts[section] = len(entries)
    return counts


def index_lookup(flavors: dict[str, str | None]) -> dict[str, set[str]]:
    """Map help tag -> set of keys as listed in index.txt (all flavors)."""
    out: dict[str, set[str]] = {}
    for flavor, exe in flavors.items():
        doc = runtime_doc_dir(flavor, exe) if exe else None
        if not doc or not (doc / "index.txt").is_file():
            continue
        for row in parse_index(doc / "index.txt"):
            out.setdefault(row.tag, set()).add(row.keys)
    return out
