"""Evaluation set runner (design §7.3).

``eval/questions.yaml`` holds ``{q: ..., expect: <id> | null, tags: [...]}``
items. ``expect: null`` means the correct behaviour is "not found".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .engine import Engine, Session
from .render import Answer, NotFound


@dataclass
class EvalItem:
    q: str
    expect: str | None
    tags: list[str] = field(default_factory=list)
    mode: str | None = None
    flavor: str = "vim"
    # for compositions: expected composed keys
    expect_keys: str | None = None


@dataclass
class EvalOutcome:
    item: EvalItem
    got: str | None
    got_keys: str | None
    confidence: str
    path: str
    correct: bool
    false_answer: bool
    latency_ms: float


@dataclass
class EvalMetrics:
    n: int
    top1_accuracy: float
    false_answer_rate: float
    not_found_precision: float
    selector_rate: float
    p50_ms: float
    p95_ms: float
    outcomes: list[EvalOutcome]

    def summary(self) -> str:
        return (
            f"n={self.n} top1={self.top1_accuracy:.1%} false-answer={self.false_answer_rate:.1%} "
            f"selector-rate={self.selector_rate:.1%} p50={self.p50_ms:.0f}ms p95={self.p95_ms:.0f}ms"
        )


def load_eval(path: Path) -> list[EvalItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    items = []
    for r in raw:
        items.append(
            EvalItem(
                q=str(r["q"]),
                expect=r.get("expect"),
                tags=list(r.get("tags", [])),
                mode=r.get("mode"),
                flavor=r.get("flavor", "vim"),
                expect_keys=r.get("expect_keys"),
            )
        )
    return items


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[idx]


def run_eval(engine: Engine, items: list[EvalItem]) -> EvalMetrics:
    outcomes: list[EvalOutcome] = []
    for it in items:
        session = Session(mode=it.mode, flavor=it.flavor)
        t0 = time.perf_counter()
        resp = engine.answer(it.q, session)
        ms = (time.perf_counter() - t0) * 1000
        trace = engine.last_trace
        got = None
        got_keys = None
        conf = "low"
        if isinstance(resp, Answer):
            conf = resp.confidence
            got_keys = resp.primary.keys
            got = resp.primary.id if hasattr(resp.primary, "id") else f"composed:{resp.primary.keys}"
        if it.expect_keys is not None:
            correct = got_keys == it.expect_keys
        elif it.expect is None:
            correct = isinstance(resp, NotFound)
        else:
            correct = got == it.expect
        false_answer = isinstance(resp, Answer) and conf == "high" and not correct
        outcomes.append(EvalOutcome(it, got, got_keys, conf, trace.path, correct, false_answer, ms))

    n = len(outcomes)
    answered = [o for o in outcomes if o.got is not None]
    nf_expected = [o for o in outcomes if o.item.expect is None and o.item.expect_keys is None]
    nf_correct = sum(1 for o in nf_expected if o.correct)
    lat = [o.latency_ms for o in outcomes]
    return EvalMetrics(
        n=n,
        top1_accuracy=(sum(o.correct for o in outcomes) / n) if n else 0.0,
        false_answer_rate=(sum(o.false_answer for o in outcomes) / n) if n else 0.0,
        not_found_precision=(nf_correct / len(nf_expected)) if nf_expected else 1.0,
        selector_rate=(sum(1 for o in outcomes if o.path in ("selector", "composed")) / n) if n else 0.0,
        p50_ms=_pct(lat, 0.5),
        p95_ms=_pct(lat, 0.95),
        outcomes=outcomes,
    )
