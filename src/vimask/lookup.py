"""Exact, reverse and alias lookup (pipeline step 2). No LLM involved."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .kb import Entry, KnowledgeBase

_FILLER = (
    "how do i",
    "how do you",
    "how can i",
    "how to",
    "how would i",
    "what is the command to",
    "what's the command to",
    "what is the key to",
    "which key",
    "what key",
    "in vim",
    "in neovim",
    "in nvim",
    "with vim",
    "please",
    "can i",
    "i want to",
    "i need to",
    "is there a way to",
)

_QUOTED_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'")
_REVERSE_RE = re.compile(
    r"^(?:what|which)\s+(?:does|do|is|are|will)\s+(?P<keys>\S+)\s+(?:do|mean|means|command)?\s*\??$"
    r"|^(?:explain|describe|meaning of|what about)\s+(?P<keys2>\S+)\s*\??$"
    r"|^(?P<keys3>\S+)\s+(?:does what|means what|meaning)\s*\??$"
)


def normalize(question: str) -> str:
    """Lowercase, trim, drop common filler phrases and trailing punctuation."""
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)
    for f in _FILLER:
        q = re.sub(r"\b" + re.escape(f) + r"\b", " ", q)
    q = re.sub(r"[?!.]+$", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


@dataclass
class LookupHit:
    entries: list[Entry]
    reason: str  # "reverse" | "alias"


def reverse_lookup(question: str, kb: KnowledgeBase) -> LookupHit | None:
    """Find an entry whose ``keys`` appear literally in the question.

    Matches quoted/backticked tokens first (``what does `ciw` do``), then a
    small set of "what does X do" phrasings, then any bare token equal to a
    KB ``keys`` value, provided the token is not an English word that merely
    coincides with a key (``w``, ``a`` …) — single-letter keys therefore need
    quoting or a reverse-lookup phrasing.
    """
    raw = question.strip()
    candidates: list[str] = []
    for m in _QUOTED_RE.finditer(raw):
        candidates.append(next(g for g in m.groups() if g is not None))
    if candidates:
        for tok in candidates:
            hits = _hits(tok, kb)
            if hits:
                return LookupHit(hits, "reverse")
        return None

    m = _REVERSE_RE.match(raw.lower().strip())
    if m:
        tok = m.group("keys") or m.group("keys2") or m.group("keys3")
        # the user typed the keys in their own case; recover it from raw
        for word in raw.split():
            if word.lower().rstrip("?") == tok.rstrip("?"):
                tok = word.rstrip("?")
                break
        hits = _hits(tok, kb)
        if hits:
            return LookupHit(hits, "reverse")

    for word in raw.split():
        stripped = word.strip(",.;:!?") if not word.startswith(":") else word.rstrip(",.;!?")
        for cand in (word, stripped):
            if len(cand) < 2:
                continue
            hits = _hits(cand, kb)
            if hits and not cand.isalpha():
                # tokens such as di( or <C-r> or :wq are unambiguous
                return LookupHit(hits, "reverse")
    return None


def _hits(token: str, kb: KnowledgeBase) -> list[Entry]:
    hits = [e for e in kb.by_keys(token) if e.verified != "none"]
    if not hits and token.startswith(":"):
        # ``:%s/foo/bar/g`` should find ``:%s/{pattern}/{replacement}/g``:
        # match the entry's keys as a template with placeholders filled in.
        hits = [e for e in kb.indexed() if e.kind == "ex" and _template_match(e.keys, token)]
    if not hits and token.startswith(":"):
        hits = [e for e in kb.indexed() if e.kind == "ex" and e.keys.split()[0] == token.split()[0]]
    return hits


def _template_match(keys: str, token: str) -> bool:
    if "{" not in keys:
        return False
    parts = re.split(r"\{[a-zA-Z0-9_-]+\}", keys)
    pattern = "[^/ ]+".join(re.escape(p) for p in parts)
    return re.fullmatch(pattern, token) is not None


def alias_lookup(normalized: str, kb: KnowledgeBase) -> LookupHit | None:
    if not normalized:
        return None
    for e in kb.indexed():
        if normalized == e.description.lower().rstrip(".") or any(
            normalized == a.lower() or normalized == normalize(a) for a in e.aliases
        ):
            return LookupHit([e], "alias")
    return None
