"""KB integrity: schema, references, verification tiers."""

from __future__ import annotations

import pytest

from vimask.kb import CURSOR, Entry, join_cursor, split_cursor


def test_kb_loads_and_is_non_trivial(kb):
    assert len(kb) >= 300
    assert len(kb.indexed()) >= 300


def test_every_entry_is_verified(kb):
    """`verified: none` entries are invisible to users; the shipped KB must have none."""
    unverified = [e.id for e in kb if e.verified == "none"]
    assert unverified == []


def test_examples_have_cursor(kb):
    for e in kb:
        if e.example:
            assert e.example.before.count(CURSOR) == 1, e.id


def test_composable_entries_have_samples_and_nouns(kb):
    for e in kb:
        if e.kind in ("motion", "textobject"):
            assert e.sample, f"{e.id} needs a sample buffer for the composer"
            assert e.noun, f"{e.id} needs a noun"
        if e.kind == "operator":
            assert e.verb, f"{e.id} needs a verb"


def test_alternatives_resolve(kb):
    for e in kb:
        for alt in e.alternatives:
            assert alt in kb.by_id, f"{e.id} -> {alt}"
            assert alt != e.id


def test_ex_entries_start_with_colon(kb):
    for e in kb:
        if e.kind == "ex":
            assert e.keys.startswith(":"), e.id


def test_aliases_are_lowercase_and_unique(kb):
    for e in kb:
        assert len(set(a.lower() for a in e.aliases)) == len(e.aliases), e.id


def test_schema_rejects_bad_entries():
    base = dict(id="x", keys="x", category="c", description="d", help_tag="x")
    with pytest.raises(ValueError):
        Entry.model_validate({**base, "id": "Bad-ID"})
    with pytest.raises(ValueError):
        Entry.model_validate({**base, "kind": "ex", "keys": "w"})
    with pytest.raises(ValueError):
        Entry.model_validate({**base, "example": {"before": "no cursor", "after": "x"}})
    with pytest.raises(ValueError):
        Entry.model_validate({**base, "keys": "f{char}", "kind": "motion", "example": {"before": "┃a", "after": "a┃"}})
    with pytest.raises(ValueError):
        Entry.model_validate({**base, "unknown_field": 1})


def test_cursor_roundtrip():
    lines, lnum, col = split_cursor("ab\ncd┃e\nf")
    assert (lines, lnum, col) == (["ab", "cde", "f"], 2, 3)
    assert join_cursor(lines, lnum, col) == "ab\ncd┃e\nf"
    assert join_cursor(["abc"], 1, 4) == "abc┃"
    assert join_cursor([""], 1, 1) == "┃"


def test_placeholders_and_arg():
    e = Entry.model_validate(dict(id="s", keys=":%s/{pattern}/{replacement}/g", kind="ex", category="c", description="d", help_tag=":s"))
    assert e.placeholders == ["pattern", "replacement"]
    m = Entry.model_validate(dict(id="f", keys="f{char}", kind="motion", category="c", description="d", help_tag="f", sample="┃a"))
    assert m.needs_arg and m.placeholders == []
