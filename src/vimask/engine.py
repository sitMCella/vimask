"""Query engine: orchestrates the pipeline of design §5."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from .composer import CompositionError, compose
from .config import Config
from .kb import Entry, KnowledgeBase
from .llm import OllamaClient
from .lookup import alias_lookup, normalize, reverse_lookup
from .render import Answer, NotFound, Response
from .retrieval import Candidate, RetrievalResult, Retriever, VectorIndex
from .selector import select_stage_a, select_stage_b


@dataclass
class Session:
    mode: str | None = None  # normal | insert | visual | None (unknown)
    flavor: str = "vim"
    last_question: str | None = None
    last_entry: Entry | None = None
    last_more: list[Entry] = field(default_factory=list)

    def previous(self) -> tuple[str, str] | None:
        if self.last_question and self.last_entry:
            return self.last_question, f"{self.last_entry.keys} — {self.last_entry.description}"
        return None


@dataclass
class Trace:
    """What happened for one query; shown by /debug and used by eval."""

    normalized: str = ""
    path: str = ""  # reverse | alias | fast-path | selector | composed | fallback
    used_embeddings: bool = False
    selector_called: bool = False
    top: list[tuple[str, float]] = field(default_factory=list)
    margin: float = 0.0
    confidence: str = ""
    note: str = ""


class Engine:
    def __init__(
        self,
        cfg: Config,
        kb: KnowledgeBase,
        *,
        client: OllamaClient | None = None,
        use_selector: bool = True,
        use_embeddings: bool = True,
        help_index_loader=None,
        log_missing: bool = True,
    ) -> None:
        self.cfg = cfg
        self.kb = kb
        self.client = client
        self.use_selector = use_selector and client is not None
        self.help_index_loader = help_index_loader
        self._help_index = None
        self.log_missing = log_missing
        self.last_trace = Trace()

        vector_index = None
        embed_fn = None
        self.index_warning = ""
        if use_embeddings and client is not None and client.available:
            vi = VectorIndex.load(cfg.index_dir)
            if vi is None:
                self.index_warning = "no vector index; run `vimask kb build` (lexical-only retrieval)"
            elif not vi.matches(cfg.embed_model, kb.kb_hash):
                self.index_warning = "vector index is stale (KB or embedding model changed); run `vimask kb build`"
            elif not client.status().has(cfg.embed_model):
                self.index_warning = f"embedding model {cfg.embed_model} not pulled (lexical-only retrieval)"
            else:
                vector_index = vi
                embed_fn = client.embed
        self.retriever = Retriever(
            kb,
            vector_index=vector_index,
            embed_fn=embed_fn,
            lexical_weight=cfg.lexical_weight,
            vector_weight=cfg.vector_weight,
            mode_boost=cfg.mode_boost,
        )
        if self.use_selector and not client.status().has(cfg.selector_model):
            self.use_selector = False
            self.selector_warning = f"selector model {cfg.selector_model} not pulled (retrieval-only)"
        else:
            self.selector_warning = ""

    # ------------------------------------------------------------------ info

    @property
    def mode_name(self) -> str:
        if self.use_selector and self.retriever.has_embeddings:
            return "full (hybrid retrieval + LLM selector)"
        if self.retriever.has_embeddings:
            return "retrieval-only (hybrid, no selector)"
        if self.use_selector:
            return "lexical retrieval + LLM selector"
        return "lexical-only (no Ollama)"

    # --------------------------------------------------------------- pipeline

    def answer(self, question: str, session: Session | None = None) -> Response:
        session = session or Session()
        trace = Trace()
        self.last_trace = trace
        flavor = session.flavor

        # 1. normalize
        norm = normalize(question)
        trace.normalized = norm
        if not norm and not question.strip():
            return NotFound(question, [], reason="empty question")

        # 2. exact / reverse lookup
        hit = reverse_lookup(question, self.kb)
        if hit:
            entries = [e for e in hit.entries if flavor in e.flavor] or hit.entries
            trace.path = "reverse"
            trace.confidence = "high"
            return self._answer_from(entries[0], entries[1:], [], session, "reverse", "high")
        hit = alias_lookup(norm, self.kb)
        if hit:
            trace.path = "alias"
            trace.confidence = "high"
            e = hit.entries[0]
            return self._answer_from(e, self._alts(e, []), [], session, "alias", "high")

        # 3. retrieve
        rr = self.retriever.retrieve(norm or question, k=self.cfg.top_k, flavor=flavor, mode=session.mode)
        trace.used_embeddings = rr.used_embeddings
        trace.top = [(c.entry.id, round(c.score, 3)) for c in rr.candidates]
        trace.margin = round(rr.margin, 3)
        if not rr.candidates or rr.top1.score < self.cfg.fallback_score:
            trace.path = "fallback"
            trace.confidence = "low"
            return self._fallback(question, "no close match")

        # 4. select (conditional)
        if rr.used_embeddings:
            fs, fm = self.cfg.fast_score, self.cfg.fast_margin
        else:
            fs, fm = self.cfg.fast_score_lexical_only, self.cfg.fast_margin_lexical_only
        fast = rr.top1.score >= fs and rr.margin >= fm
        if fast or not self.use_selector:
            if not fast and not self.use_selector:
                # Retrieval-only mode cannot disambiguate: show medium confidence
                # when the margin is small, fallback when top-1 is weak.
                if rr.top1.score < self.cfg.answer_score_no_selector:
                    trace.path = "fallback"
                    trace.confidence = "low"
                    return self._fallback(question, "match too weak without the selector")
                conf = "medium" if rr.margin < fm else "high"
            else:
                conf = "high"
            trace.path = "fast-path" if fast else "retrieval"
            trace.confidence = conf
            return self._answer_from_candidates(rr, [c.entry for c in rr.candidates], session, trace.path, conf)

        trace.selector_called = True
        cands = [c.entry for c in rr.candidates]
        sel = select_stage_a(self.client, question, cands, session.previous())
        if sel.compose and not sel.ids:
            resp = self._compose(question, session, trace)
            if resp is not None:
                return resp
        if not sel.ids:
            if rr.top1.score >= fs:
                # the model found nothing but retrieval is strong: trust retrieval, medium
                trace.path = "retrieval"
                trace.confidence = "medium"
                return self._answer_from_candidates(rr, cands, session, "retrieval", "medium")
            trace.path = "fallback"
            trace.confidence = "low"
            return self._fallback(question, "selector found no matching command")

        # 5. validate: confidence derived by code
        first = sel.ids[0]
        top_ids = [c.entry.id for c in rr.candidates]
        if first == top_ids[0]:
            conf = "high"
        elif first in top_ids[:3]:
            conf = "medium"
        else:
            conf = "low"
        trace.path = "selector"
        trace.confidence = conf
        if conf == "low":
            if sel.compose:
                resp = self._compose(question, session, trace)
                if resp is not None:
                    return resp
            return self._fallback(question, "selector and retrieval disagree")
        ordered = [self.kb.get(i) for i in sel.ids] + [e for e in cands if e.id not in sel.ids]
        return self._answer_from_candidates(rr, ordered, session, "selector", conf)

    # ---------------------------------------------------------------- helpers

    def _alts(self, primary: Entry, rest: list[Entry]) -> list[Entry]:
        out: list[Entry] = []
        for alt_id in primary.alternatives:
            e = self.kb.get(alt_id)
            if e and e.verified != "none" and e.id != primary.id and e not in out:
                out.append(e)
        for e in rest:
            if e.id != primary.id and e not in out:
                out.append(e)
        return out

    def _answer_from(
        self, primary: Entry, alts: list[Entry], more: list[Entry], session: Session, source: str, conf: str
    ) -> Answer:
        session.last_question = session.last_question  # unchanged until caller sets
        session.last_entry = primary
        session.last_more = alts[2:] + more
        return Answer(
            primary=primary,
            alternatives=alts[:2],
            more=alts[2:] + more,
            confidence=conf,
            session_mode=session.mode,
            source=source,
            flavor=session.flavor,
        )

    def _answer_from_candidates(
        self, rr: RetrievalResult, ordered: list[Entry], session: Session, source: str, conf: str
    ) -> Answer:
        primary = ordered[0]
        alts = self._alts(primary, ordered[1:])
        return self._answer_from(primary, alts, [], session, source, conf)

    def _compose(self, question: str, session: Session, trace: Trace) -> Response | None:
        parts = select_stage_b(self.client, question, self.kb)
        if parts.operator is None:
            trace.note = parts.error
            return None
        try:
            composed = compose(parts, self.kb, self.cfg, flavor=session.flavor)
        except CompositionError as exc:
            trace.note = str(exc)
            return None
        trace.path = "composed"
        trace.confidence = "medium"
        session.last_entry = None
        session.last_more = []
        return Answer(primary=composed, confidence="medium", session_mode=session.mode, source="composed")

    def _help_index_get(self):
        if self._help_index is None and self.help_index_loader is not None:
            try:
                self._help_index = self.help_index_loader() or False
            except Exception:
                self._help_index = False
        return self._help_index or None

    def _fallback(self, question: str, reason: str) -> NotFound:
        excerpts = []
        hi = self._help_index_get()
        if hi is not None:
            excerpts = hi.search(normalize(question) or question, k=3)
        if self.log_missing:
            self.log_missing_question(question)
        return NotFound(question, excerpts, reason=reason)

    def log_missing_question(self, question: str) -> None:
        try:
            self.cfg.missing_log.parent.mkdir(parents=True, exist_ok=True)
            stamp = _dt.datetime.now().isoformat(timespec="seconds")
            with self.cfg.missing_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp}\t{question.strip()}\n")
        except OSError:
            pass
