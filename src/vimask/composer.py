"""Deterministic composer (design §5.1): operator + count + motion/text object.

Takes the validated Stage B parts, checks legality, builds the keystroke
string and produces the example by running the keys in a headless editor on
the target's ``sample`` buffer. Nothing here comes from model text except the
choice among verified catalogue IDs and a single printable ``arg`` character.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .kb import Entry, Example, KnowledgeBase
from .selector import StageBResult
from .vimrun import EditorNotFound, find_editor, run_keys

MAX_COUNT = 99


@dataclass
class ComposedAnswer:
    keys: str
    description: str
    operator: Entry
    target: Entry
    count: int = 1
    arg: str | None = None
    example: Example | None = None
    note: str = ""
    help_tags: list[str] = field(default_factory=list)
    flavor: str = "vim"

    @property
    def mode(self) -> str:
        return "normal"


class CompositionError(ValueError):
    pass


def _check_arg(target: Entry, arg: str | None) -> None:
    if target.needs_arg:
        if arg is None:
            raise CompositionError(f"{target.keys} needs a character argument")
        if len(arg) != 1 or not arg.isprintable() or arg.isspace():
            raise CompositionError("argument must be one printable character")
    elif arg is not None:
        raise CompositionError(f"{target.keys} takes no argument")


def build_keys(operator: Entry, target: Entry, count: int = 1, arg: str | None = None) -> str:
    if operator.kind != "operator":
        raise CompositionError(f"{operator.id} is not an operator")
    if target.kind not in ("motion", "textobject"):
        raise CompositionError(f"{target.id} is not a motion or text object")
    if not (1 <= count <= MAX_COUNT):
        raise CompositionError(f"count must be within 1 and {MAX_COUNT}")
    _check_arg(target, arg)
    target_keys = target.keys.replace("{char}", arg) if target.needs_arg else target.keys
    if target.kind == "textobject" and count > 1 and not target.id.endswith("-paragraph"):
        # counts on most text objects are legal but rarely what the user
        # means ("delete 3 words" is d3w, not d3iw); keep count for motions.
        pass
    # Design §5.1: the count goes between operator and target (d3w), which
    # is equivalent to 3dw and reads as "delete 3 words".
    middle = str(count) if count > 1 else ""
    return f"{operator.keys}{middle}{target_keys}"


def describe(operator: Entry, target: Entry, count: int = 1, arg: str | None = None) -> str:
    verb = operator.verb or operator.description.rstrip(".")
    noun = target.noun or target.description.rstrip(".")
    if arg is not None:
        noun = noun.replace("{char}", f"'{arg}'")
    if count > 1:
        return f"{verb} {count}× {noun}."
    return f"{verb} {noun}."


def compose(
    parts: StageBResult,
    kb: KnowledgeBase,
    cfg: Config,
    *,
    flavor: str = "vim",
    run_example: bool = True,
) -> ComposedAnswer:
    if parts.operator is None or parts.target is None:
        raise CompositionError(parts.error or "incomplete composition")
    op, tg = parts.operator, parts.target
    if flavor not in op.flavor or flavor not in tg.flavor:
        raise CompositionError(f"parts not available in {flavor}")
    keys = build_keys(op, tg, parts.count, parts.arg)
    answer = ComposedAnswer(
        keys=keys,
        description=describe(op, tg, parts.count, parts.arg),
        operator=op,
        target=tg,
        count=parts.count,
        arg=parts.arg,
        help_tags=[op.help_tag, tg.help_tag],
        flavor=flavor,
    )
    if run_example and tg.sample:
        exe = find_editor(flavor, cfg.vim_path if flavor == "vim" else cfg.nvim_path)
        if not exe:
            answer.note = f"example omitted: {flavor} not found on PATH"
            return answer
        try:
            res = run_keys(tg.sample, keys, flavor=flavor, editor=exe, timeout=cfg.editor_timeout)
        except EditorNotFound as exc:
            answer.note = f"example omitted: {exc}"
            return answer
        if res.ok and res.text != tg.sample:
            answer.example = Example(before=tg.sample, after=res.text)
        elif res.ok:
            answer.note = "no visible effect on the sample buffer"
        else:
            answer.note = f"example omitted: {res.error}"
    return answer
