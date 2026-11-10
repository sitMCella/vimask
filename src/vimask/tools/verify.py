"""Example verifier (design §7.2).

For every entry:
1. check ``help_tag`` exists in the ``doc/tags`` file of each listed flavor;
2. if an example is present, run ``setup_keys`` + ``keys`` in each flavor's
   headless editor and compare buffer + cursor with ``after``.

The resulting tier (none/tag/example) is written back into the YAML with a
round-trip loader so comments and formatting survive.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML

from ..config import Config
from ..help_index import load_tags, runtime_doc_dir
from ..kb import CURSOR, Entry, kb_files, load_entries
from ..vimrun import find_editor, run_keys


@dataclass
class VerifyReport:
    entry_id: str
    tier: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class Environment:
    editors: dict[str, str | None]
    tags: dict[str, set[str]]

    @classmethod
    def detect(cls, cfg: Config) -> "Environment":
        editors = {
            "vim": find_editor("vim", cfg.vim_path),
            "nvim": find_editor("nvim", cfg.nvim_path),
        }
        tags: dict[str, set[str]] = {}
        for fl, exe in editors.items():
            doc = runtime_doc_dir(fl, exe) if exe else None
            tags[fl] = load_tags(doc) if doc else set()
        return cls(editors, tags)


def _tag_candidates(entry: Entry) -> list[str]:
    tag = entry.help_tag
    out = [tag]
    if not tag.startswith(":") and entry.kind == "ex":
        out.append(":" + tag)
    return out


def verify_entry(entry: Entry, env: Environment, cfg: Config, strict_cursor: bool = True) -> VerifyReport:
    rep = VerifyReport(entry.id, "none")
    tag_ok = True
    for fl in entry.flavor:
        tagset = env.tags.get(fl) or set()
        if not tagset:
            rep.problems.append(f"{fl}: doc/tags not available (editor missing?)")
            tag_ok = False
            continue
        if not any(t in tagset for t in _tag_candidates(entry)):
            rep.problems.append(f"{fl}: help tag {entry.help_tag!r} not found")
            tag_ok = False
    if not tag_ok:
        return rep
    rep.tier = "tag"
    if entry.example is None:
        return rep

    for fl in entry.flavor:
        exe = env.editors.get(fl)
        if not exe:
            rep.problems.append(f"{fl}: editor not found; example not runnable")
            continue
        res = run_keys(
            entry.example.before,
            entry.example_keys or entry.keys,
            flavor=fl,
            setup_keys=entry.setup_keys,
            options=entry.options,
            editor=exe,
            timeout=cfg.editor_timeout,
        )
        if not res.ok:
            rep.problems.append(f"{fl}: run failed: {res.error}")
            continue
        expected = entry.example.after
        got = res.text
        if CURSOR not in expected or not strict_cursor:
            got_cmp = got.replace(CURSOR, "")
            exp_cmp = expected.replace(CURSOR, "")
        else:
            got_cmp, exp_cmp = got, expected
        if got_cmp != exp_cmp:
            rep.problems.append(f"{fl}: example mismatch\n      expected: {expected!r}\n      got:      {got!r}")
    if rep.problems:
        # tag is fine but the example is not: keep tier "tag" so the entry
        # stays usable without an example only if the caller decides so.
        return rep
    rep.tier = "example"
    return rep


def verify_kb(cfg: Config, *, only: set[str] | None = None, write: bool = True, verbose: bool = False) -> list[VerifyReport]:
    env = Environment.detect(cfg)
    reports: list[VerifyReport] = []
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    for path in kb_files(cfg.kb_dir):
        entries = load_entries(path)
        tiers: dict[str, str] = {}
        for e in entries:
            if only and e.id not in only:
                continue
            rep = verify_entry(e, env, cfg)
            reports.append(rep)
            # an entry with a failing example is demoted to none so the bad
            # example can never be shown
            tiers[e.id] = rep.tier if rep.ok else "none"
            if verbose or not rep.ok:
                status = "ok " if rep.ok else "FAIL"
                print(f"  {status} {e.id:<36} {rep.tier if rep.ok else 'none'}", file=sys.stderr)
                for p in rep.problems:
                    print(f"       - {p}", file=sys.stderr)
        if write and tiers:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.load(fh)
            changed = False
            for item in data:
                tier = tiers.get(item.get("id"))
                if tier is not None and item.get("verified") != tier:
                    item["verified"] = tier
                    changed = True
            if changed:
                with path.open("w", encoding="utf-8") as fh:
                    yaml.dump(data, fh)
    return reports


def summarize(reports: list[VerifyReport]) -> str:
    total = len(reports)
    ex = sum(1 for r in reports if r.ok and r.tier == "example")
    tag = sum(1 for r in reports if r.ok and r.tier == "tag")
    bad = sum(1 for r in reports if not r.ok)
    return f"{total} entries: {ex} example-verified, {tag} tag-verified, {bad} failed"
