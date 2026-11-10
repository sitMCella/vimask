"""LLM selector (pipeline step 4): Stage A picks entry IDs, Stage B picks
composition parts. The model only ever returns IDs from lists we gave it;
``validate_*`` drops anything else, so model output can never reach the
renderer as text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kb import Entry, KnowledgeBase
from .llm import OllamaClient

STAGE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "ids": {"type": "array", "items": {"type": "string"}},
        "compose": {"type": "boolean"},
    },
    "required": ["ids", "compose"],
}

STAGE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "operator": {"type": "string"},
        "target": {"type": "string"},
        "arg": {"type": ["string", "null"]},
    },
    "required": ["count", "operator", "target", "arg"],
}

STAGE_A_SYSTEM = (
    "You choose which of the listed Vim commands answers the user's question. "
    "Each candidate is given as ID | keys | description. "
    "Return a JSON object {\"ids\": [...], \"compose\": bool}. "
    "ids contains only IDs from the list, best first; return an empty list if none answers the question. "
    "Set compose to true only when the question combines an editing action (delete, change, yank, indent, "
    "case change) with a count, motion or text object that is NOT available as a single listed command. "
    "Never invent commands or IDs."
)

STAGE_B_SYSTEM = (
    "The user's question is a Vim operator + motion/text-object composition, e.g. 'delete the next 3 words' = "
    "count 3, operator d, target w. Pick the operator and the target from the catalogue below "
    "(given as ID | keys | description). Return JSON {\"count\": int, \"operator\": ID, \"target\": ID, \"arg\": "
    "string or null}. count is 1 when the question has no number. arg is the single character required by "
    "targets whose keys contain {char} (e.g. 'delete until the closing paren' = target t{char}, arg ')'); "
    "otherwise null. Use only IDs from the catalogue. Never invent anything."
)


@dataclass
class StageAResult:
    ids: list[str] = field(default_factory=list)  # validated, in model order
    compose: bool = False
    raw: dict | None = None
    called: bool = False


@dataclass
class StageBResult:
    count: int = 1
    operator: Entry | None = None
    target: Entry | None = None
    arg: str | None = None
    raw: dict | None = None
    error: str = ""


def _fmt(entries: list[Entry]) -> str:
    return "\n".join(f"{e.id} | {e.keys} | {e.description}" for e in entries)


def stage_a_prompt(question: str, candidates: list[Entry], previous: tuple[str, str] | None = None) -> str:
    parts = []
    if previous:
        parts.append(f"Previous question: {previous[0]}\nPrevious answer: {previous[1]}\n")
    parts.append(f"Question: {question}\n\nCandidates:\n{_fmt(candidates)}")
    return "\n".join(parts)


def validate_stage_a(data: dict | None, candidates: list[Entry]) -> StageAResult:
    res = StageAResult(raw=data, called=True)
    if not data:
        return res
    allowed = {e.id for e in candidates}
    ids = data.get("ids") or []
    if isinstance(ids, list):
        seen = set()
        for i in ids:
            if isinstance(i, str) and i in allowed and i not in seen:
                res.ids.append(i)
                seen.add(i)
    res.compose = bool(data.get("compose", False))
    return res


def select_stage_a(
    client: OllamaClient,
    question: str,
    candidates: list[Entry],
    previous: tuple[str, str] | None = None,
) -> StageAResult:
    data = client.chat_json(STAGE_A_SYSTEM, stage_a_prompt(question, candidates, previous), STAGE_A_SCHEMA)
    return validate_stage_a(data, candidates)


def stage_b_prompt(question: str, operators: list[Entry], targets: list[Entry]) -> str:
    return f"Question: {question}\n\nOperators:\n{_fmt(operators)}\n\nTargets (motions and text objects):\n{_fmt(targets)}"


def validate_stage_b(data: dict | None, kb: KnowledgeBase) -> StageBResult:
    res = StageBResult(raw=data)
    if not data:
        res.error = "no selector output"
        return res
    op = kb.get(str(data.get("operator", "")))
    tg = kb.get(str(data.get("target", "")))
    if op is None or op.kind != "operator" or op.verified == "none":
        res.error = "operator not in catalogue"
        return res
    if tg is None or tg.kind not in ("motion", "textobject") or tg.verified == "none":
        res.error = "target not in catalogue"
        return res
    try:
        count = int(data.get("count", 1))
    except (TypeError, ValueError):
        count = 1
    arg = data.get("arg")
    res.count = count
    res.operator = op
    res.target = tg
    res.arg = arg if isinstance(arg, str) and arg else None
    return res


def select_stage_b(client: OllamaClient, question: str, kb: KnowledgeBase) -> StageBResult:
    operators = kb.catalogue("operator")
    targets = kb.catalogue("motion") + kb.catalogue("textobject")
    data = client.chat_json(STAGE_B_SYSTEM, stage_b_prompt(question, operators, targets), STAGE_B_SCHEMA)
    return validate_stage_b(data, kb)
