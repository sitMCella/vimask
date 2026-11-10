"""Knowledge-base schema and loading.

One YAML file per category under ``kb/``; each file is a list of entries.
Validation is strict: an invalid entry fails the whole load, because the KB
is the only source of keystrokes the tool may ever print.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CURSOR = "┃"  # '┃' marks the cursor in examples

Kind = Literal["command", "operator", "motion", "textobject", "ex"]
Mode = Literal["normal", "insert", "visual", "cmdline", "ex"]
Flavor = Literal["vim", "nvim"]
Verified = Literal["none", "tag", "example"]

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_-]+)\}")


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: str
    after: str
    # Concrete keys to run for the example when the entry's ``keys`` cannot
    # run alone: ``f{char}`` -> ``f(``, text object ``iw`` -> ``diw``.
    keys: str | None = None

    @field_validator("before")
    @classmethod
    def _one_cursor_in_before(cls, v: str) -> str:
        if v.count(CURSOR) != 1:
            raise ValueError("example.before must contain exactly one cursor marker '┃'")
        return v

    @field_validator("after")
    @classmethod
    def _at_most_one_cursor_in_after(cls, v: str) -> str:
        if v.count(CURSOR) > 1:
            raise ValueError("example.after may contain at most one cursor marker '┃'")
        return v


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    keys: str
    kind: Kind = "command"
    mode: Mode = "normal"
    flavor: list[Flavor] = Field(default_factory=lambda: ["vim", "nvim"])
    category: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    help_tag: str
    alternatives: list[str] = Field(default_factory=list)
    setup_keys: str | None = None
    # Human wording for setup_keys shown in the answer, e.g. "after
    # recording a macro with qa...q". Defaults to showing setup_keys.
    setup_desc: str | None = None
    example: Example | None = None
    effect: str | None = None
    sample: str | None = None
    # Ex ``set`` arguments applied before running the example, e.g.
    # ["textwidth=30"]. Shown to the user as an assumption.
    options: list[str] = Field(default_factory=list)
    # Composition helpers (operators: verb, motions/text objects: noun).
    verb: str | None = None
    noun: str | None = None
    verified: Verified = "none"

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"invalid id {v!r}: use lowercase letters, digits and '-'")
        return v

    @field_validator("keys", "description", "help_tag", "category")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("flavor")
    @classmethod
    def _flavor_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("flavor must list at least one of vim, nvim")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _consistency(self) -> "Entry":
        if self.kind in ("motion", "textobject") and self.sample is not None:
            if self.sample.count(CURSOR) != 1:
                raise ValueError("sample must contain exactly one cursor marker '┃'")
        if self.kind == "ex" and not self.keys.startswith(":"):
            raise ValueError("ex entries must store the full command starting with ':'")
        if self.verified == "example" and self.example is None:
            raise ValueError("verified: example requires an example")
        if self.example is not None and self.example.keys is None:
            if _PLACEHOLDER_RE.search(self.keys) and self.kind != "ex":
                raise ValueError("entries with a {placeholder} in keys need example.keys")
            if self.kind in ("operator", "textobject"):
                raise ValueError(f"{self.kind} examples need example.keys (e.g. 'diw')")
        return self

    @property
    def example_keys(self) -> str | None:
        if self.example is None:
            return None
        return self.example.keys or self.keys

    # --- derived -----------------------------------------------------------

    @property
    def placeholders(self) -> list[str]:
        """Placeholder names in ``keys`` such as ``pattern`` in ``{pattern}``.

        ``{char}`` on motions is the composer ``arg`` slot, not a placeholder.
        """
        names = _PLACEHOLDER_RE.findall(self.keys)
        return [n for n in names if not (self.kind == "motion" and n == "char")]

    @property
    def needs_arg(self) -> bool:
        return self.kind == "motion" and "{char}" in self.keys

    @property
    def composable(self) -> bool:
        return self.kind in ("operator", "motion", "textobject")

    def retrieval_text(self) -> str:
        parts = [self.description, *self.aliases, self.category.replace("/", " ")]
        if self.effect:
            parts.append(self.effect)
        return " . ".join(parts)


class KnowledgeBase:
    def __init__(self, entries: Iterable[Entry], kb_hash: str = "") -> None:
        self.entries: list[Entry] = list(entries)
        self.by_id: dict[str, Entry] = {}
        for e in self.entries:
            if e.id in self.by_id:
                raise ValueError(f"duplicate KB id {e.id!r}")
            self.by_id[e.id] = e
        self.kb_hash = kb_hash
        self._by_keys: dict[str, list[Entry]] = {}
        for e in self.entries:
            self._by_keys.setdefault(e.keys, []).append(e)
        self._check_references()

    def _check_references(self) -> None:
        for e in self.entries:
            for alt in e.alternatives:
                if alt not in self.by_id:
                    raise ValueError(f"{e.id}: unknown alternative {alt!r}")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def get(self, entry_id: str) -> Entry | None:
        return self.by_id.get(entry_id)

    def by_keys(self, keys: str) -> list[Entry]:
        return list(self._by_keys.get(keys, []))

    def indexed(self) -> list[Entry]:
        """Entries the retrieval index may contain (verified tag or example)."""
        return [e for e in self.entries if e.verified != "none"]

    def catalogue(self, kind: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind and e.verified != "none"]


def kb_files(kb_dir: Path) -> list[Path]:
    return sorted(p for p in kb_dir.glob("*.yaml") if not p.name.startswith("_"))


def load_entries(path: Path) -> list[Entry]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of entries")
    entries = []
    for i, item in enumerate(raw):
        try:
            entries.append(Entry.model_validate(item))
        except Exception as exc:  # re-raise with location
            ident = item.get("id", f"#{i}") if isinstance(item, dict) else f"#{i}"
            raise ValueError(f"{path.name} entry {ident}: {exc}") from exc
    return entries


def compute_kb_hash(kb_dir: Path) -> str:
    h = hashlib.sha256()
    for path in kb_files(kb_dir):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_kb(kb_dir: Path, include_unverified: bool = False) -> KnowledgeBase:
    """Load every KB file. Unverified entries are kept in the object (tooling
    needs them) but ``indexed()`` and the engine only use verified ones."""
    entries: list[Entry] = []
    for path in kb_files(kb_dir):
        entries.extend(load_entries(path))
    kb = KnowledgeBase(entries, kb_hash=compute_kb_hash(kb_dir))
    if not include_unverified:
        pass  # filtering happens at use sites via indexed()
    return kb


# --- cursor marker helpers ------------------------------------------------


def split_cursor(text: str) -> tuple[list[str], int, int]:
    """Return (lines without marker, line (1-based), byte column (1-based)).

    Columns are byte offsets, which is what Vim's ``cursor()``/``getpos()``
    use. Example text is expected to be ASCII apart from the marker.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        idx = line.find(CURSOR)
        if idx >= 0:
            prefix = line[:idx]
            lines[i] = prefix + line[idx + 1 :]
            return lines, i + 1, len(prefix.encode("utf-8")) + 1
    return lines, 0, 0


def join_cursor(lines: list[str], lnum: int, col: int) -> str:
    """Inverse of :func:`split_cursor`. ``lnum``/``col`` 1-based; col is bytes."""
    out = list(lines)
    if 1 <= lnum <= len(out):
        raw = out[lnum - 1].encode("utf-8")
        byte_idx = max(0, min(col - 1, len(raw)))
        out[lnum - 1] = raw[:byte_idx].decode("utf-8", "replace") + CURSOR + raw[byte_idx:].decode(
            "utf-8", "replace"
        )
    return "\n".join(out)
