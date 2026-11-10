"""Headless runner and verifier (real vim/nvim when available)."""

from __future__ import annotations

import pytest

from vimask.vimrun import normalize_keys_for_run, run_keys, vim_string_literal

from .conftest import needs_nvim, needs_vim


def test_vim_string_literal_escapes_keys_and_quotes():
    assert vim_string_literal("ci\"x<Esc>") == 'ci\\"x\\<Esc>'
    assert vim_string_literal("<C-v>j$A;<Esc>") == "\\<C-v>j$A;\\<Esc>"
    assert vim_string_literal("<C-\\><C-n>") == "\\<C-\\>\\<C-n>"
    # regex atoms are not keys
    assert vim_string_literal(":%s/\\<cat\\>/dog/g") == ":%s/\\\\<cat\\\\>/dog/g"
    assert vim_string_literal("di<") == "di<"
    assert vim_string_literal("<C-w><") == "\\<C-w><"


def test_normalize_adds_cr_to_ex_commands():
    assert normalize_keys_for_run(":%s/a/b/g") == ":%s/a/b/g<CR>"
    assert normalize_keys_for_run(":%y<CR>Gp") == ":%y<CR>Gp"
    assert normalize_keys_for_run("dw") == "dw"


@needs_vim
def test_run_keys_vim():
    r = run_keys("print(foo, ba┃r)", "di(", flavor="vim")
    assert r.ok and r.text == "print(┃)"


@needs_nvim
def test_run_keys_nvim():
    r = run_keys("print(foo, ba┃r)", "di(", flavor="nvim")
    assert r.ok and r.text == "print(┃)"


@needs_vim
def test_run_keys_setup_and_insert_mode():
    r = run_keys("one two┃ three", "A end", flavor="vim")
    assert r.ok and r.text == "one two three end┃"
    r = run_keys("ab┃cd", "u", setup_keys="x", flavor="vim")
    assert r.ok and r.text == "ab┃cd"


@needs_vim
def test_run_keys_cannot_reach_shell():
    """Restricted mode: :! must fail, not execute."""
    r = run_keys("┃x", ":r !echo pwned<CR>", flavor="vim")
    assert not r.ok or "pwned" not in "\n".join(r.lines)


@needs_nvim
def test_run_keys_nvim_shell_is_disabled():
    r = run_keys("┃x", ":r !echo pwned<CR>", flavor="nvim")
    assert "pwned" not in "\n".join(r.lines)


@needs_vim
def test_verify_entry_detects_wrong_example(cfg, kb):
    from vimask.kb import Entry
    from vimask.tools.verify import Environment, verify_entry

    env = Environment.detect(cfg)
    good = kb.get("delete-inside-parens")
    rep = verify_entry(good, env, cfg)
    assert rep.ok and rep.tier == "example", rep.problems
    bad = Entry.model_validate({**good.model_dump(), "id": "bad", "example": {"before": "f(a┃b)", "after": "f(a┃)"}})
    rep = verify_entry(bad, env, cfg)
    assert not rep.ok
    missing_tag = Entry.model_validate({**good.model_dump(), "id": "bad2", "help_tag": "no-such-tag-xyz"})
    rep = verify_entry(missing_tag, env, cfg)
    assert not rep.ok and rep.tier == "none"
