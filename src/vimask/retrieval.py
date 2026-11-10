"""Hybrid retrieval: BM25 (always) + embeddings (when Ollama is reachable).

Scores are fused as a weighted sum of *normalized* scores so the fused value
carries absolute meaning (needed for the fast-path and fallback thresholds):

* lexical: BM25 score divided by the IDF mass of the query, i.e. roughly the
  fraction of the query's information that the entry text matches; clipped
  to [0, 1].
* vector: cosine similarity rescaled from the typical [0.4, 1.0] band of
  short-text embedding models to [0, 1] and clipped.

Reciprocal rank fusion was considered (see design §4.1) but its scores are
rank-only, so a lexical-only top-1 would always score the same regardless of
how well it matched.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .kb import Entry, KnowledgeBase

_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "for", "my",
    "me", "it", "its", "this", "that", "with", "from", "is", "be", "do", "does",
    "i", "you", "vim", "command", "key", "keys", "keystroke", "keystrokes",
    "what", "them", "how",
}

# Small synonym map applied to query *and* document tokens so that common
# paraphrases meet. Kept tiny on purpose; aliases in the KB do the real work.
_SYNONYMS = {
    "remove": "delete",
    "erase": "delete",
    "kill": "delete",
    "copy": "yank",
    "duplicate": "yank",
    "bracket": "paren",
    "brackets": "paren",
    "parenthesis": "paren",
    "parentheses": "paren",
    "parens": "paren",
    "braces": "brace",
    "curly": "brace",
    "quotes": "quote",
    "quotation": "quote",
    "sentence": "sentence",
    "para": "paragraph",
    "lines": "line",
    "words": "word",
    "chars": "char",
    "character": "char",
    "characters": "char",
    "capitalize": "uppercase",
    "capitalise": "uppercase",
    "upper": "uppercase",
    "lower": "lowercase",
    "beginning": "start",
    "begin": "start",
    "first": "start",
    "finish": "end",
    "last": "end",
    "goto": "go",
    "jump": "go",
    "move": "go",
    "navigate": "go",
    "cursor": "go",
    "backwards": "backward",
    "back": "backward",
    "previous": "backward",
    "prev": "backward",
    "forwards": "forward",
    "next": "forward",
    "splits": "split",
    "pane": "window",
    "panes": "window",
    "windows": "window",
    "buffers": "buffer",
    "tabs": "tab",
    "search": "search",
    "find": "search",
    "look": "search",
    "replace": "substitute",
    "swap": "substitute",
    "exit": "quit",
    "close": "quit",
    "leave": "quit",
    "write": "save",
    "saving": "save",
    "indent": "indent",
    "indentation": "indent",
    "undo": "undo",
    "revert": "undo",
    "fold": "fold",
    "collapse": "fold",
    "registers": "register",
    "clipboard": "register",
    "macro": "macro",
    "macros": "macro",
    "repeat": "repeat",
    "redo": "redo",
    "again": "repeat",
    "whole": "entire",
    "all": "entire",
    "everything": "entire",
    "file": "buffer",
    "document": "buffer",
    # The KB writes numeric placeholders as both {N} and {count}; users write
    # "line n". Fold them together so "jump to line n" meets "{count}G".
    "n": "count",
    "access": "enter",
    "activate": "enter",
}


def raw_tokens(text: str) -> list[str]:
    """Lowercase tokens minus stop words, before synonym folding."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def tokenize(text: str) -> list[str]:
    return [_SYNONYMS.get(t, t) for t in raw_tokens(text)]


@dataclass
class Candidate:
    entry: Entry
    score: float
    lexical: float = 0.0
    vector: float = 0.0
    # Unclipped lexical strength; breaks ties between candidates whose
    # normalized score saturated at 1.0 (otherwise ties fell to id order).
    strength: float = 0.0


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    used_embeddings: bool
    query_tokens: list[str] = field(default_factory=list)

    @property
    def top1(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> float:
        if len(self.candidates) < 2:
            return self.candidates[0].score if self.candidates else 0.0
        return self.candidates[0].score - self.candidates[1].score


class LexicalIndex:
    # A query word this close (difflib ratio) to a known word is treated as
    # a typo of it: "delet" -> "delete", "charcter" -> "character".
    TYPO_CUTOFF = 0.8
    TYPO_MIN_LEN = 4

    def __init__(self, entries: Sequence[Entry]) -> None:
        self.entries = list(entries)
        self.docs = [tokenize(e.retrieval_text()) for e in self.entries]
        self.bm25 = BM25Okapi(self.docs) if self.docs else None
        vocab: set[str] = set(_SYNONYMS)
        for e in self.entries:
            vocab.update(raw_tokens(e.retrieval_text()))
        self.vocab = vocab
        self._vocab_list = sorted(vocab)

    def query_tokens(self, query: str) -> list[str]:
        """Tokenize a query, correcting obvious typos against the KB vocabulary.

        Only alphabetic words of at least ``TYPO_MIN_LEN`` letters that are
        unknown to the index are corrected, so key sequences and short words
        are left alone; words with no close match stay unknown and keep
        pulling the score down (see :meth:`scores`).
        """
        out = []
        for t in raw_tokens(query):
            if t not in self.vocab and t.isalpha() and len(t) >= self.TYPO_MIN_LEN:
                close = difflib.get_close_matches(t, self._vocab_list, n=1, cutoff=self.TYPO_CUTOFF)
                if close:
                    t = close[0]
            out.append(_SYNONYMS.get(t, t))
        return out

    def scores(self, query_tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(normalized, raw)`` lexical scores for every entry.

        ``normalized`` is in [0, 1] and carries absolute meaning for the
        thresholds; ``raw`` is the unclipped ratio used to rank entries whose
        normalized score saturated.
        """
        if not self.bm25 or not query_tokens:
            z = np.zeros(len(self.entries))
            return z, z
        raw = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        # Unknown tokens (typos, out-of-scope words such as "telescope") count
        # as the rarest possible term: they add to the query's IDF mass but
        # can never be matched, which pulls the score of every entry down.
        # Vim vocabulary is small and shared, so a question made mostly of
        # unknown words ends up below the fallback threshold.
        unknown_idf = max(self.bm25.idf.values(), default=1.0)
        idf_mass = sum(self.bm25.idf.get(t, unknown_idf) for t in query_tokens)
        if idf_mass <= 0:
            z = np.zeros(len(self.entries))
            return z, z
        ratio = raw / idf_mass
        return np.clip(ratio, 0.0, 1.0), ratio


class VectorIndex:
    """Brute-force cosine search over one ``.npy`` matrix."""

    META = "vectors.json"
    NPY = "vectors.npy"

    def __init__(self, ids: list[str], matrix: np.ndarray, model: str, kb_hash: str) -> None:
        self.ids = ids
        self.matrix = matrix
        self.model = model
        self.kb_hash = kb_hash
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.unit = matrix / norms

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex | None":
        meta_p, npy_p = index_dir / cls.META, index_dir / cls.NPY
        if not (meta_p.is_file() and npy_p.is_file()):
            return None
        meta = json.loads(meta_p.read_text())
        matrix = np.load(npy_p)
        if len(meta["ids"]) != matrix.shape[0]:
            return None
        return cls(meta["ids"], matrix, meta["model"], meta["kb_hash"])

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / self.NPY, self.matrix)
        (index_dir / self.META).write_text(
            json.dumps({"ids": self.ids, "model": self.model, "kb_hash": self.kb_hash, "dim": int(self.matrix.shape[1])})
        )

    def matches(self, model: str, kb_hash: str) -> bool:
        return self.model == model and self.kb_hash == kb_hash

    def similarities(self, query_vec: np.ndarray) -> dict[str, float]:
        q = np.asarray(query_vec, dtype=float)
        n = np.linalg.norm(q)
        if n == 0:
            return {}
        sims = self.unit @ (q / n)
        return {i: float(s) for i, s in zip(self.ids, sims)}


def rescale_cosine(cos: float, low: float = 0.4, high: float = 0.95) -> float:
    return float(np.clip((cos - low) / (high - low), 0.0, 1.0))


class Retriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        vector_index: VectorIndex | None = None,
        embed_fn=None,
        lexical_weight: float = 0.6,
        vector_weight: float = 0.4,
        mode_boost: float = 0.05,
    ) -> None:
        self.kb = kb
        self.entries = kb.indexed()
        self.lexical = LexicalIndex(self.entries)
        self.vector_index = vector_index
        self.embed_fn = embed_fn  # callable(str) -> np.ndarray | None
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.mode_boost = mode_boost

    @property
    def has_embeddings(self) -> bool:
        return self.vector_index is not None and self.embed_fn is not None

    def retrieve(
        self,
        query: str,
        *,
        k: int = 8,
        flavor: str | None = None,
        mode: str | None = None,
    ) -> RetrievalResult:
        qtoks = self.lexical.query_tokens(query)
        lex, lex_raw = self.lexical.scores(qtoks)

        vec_scores: dict[str, float] = {}
        used_embeddings = False
        if self.has_embeddings:
            qvec = None
            try:
                qvec = self.embed_fn(query)
            except Exception:
                qvec = None
            if qvec is not None:
                used_embeddings = True
                vec_scores = {i: rescale_cosine(s) for i, s in self.vector_index.similarities(qvec).items()}

        if used_embeddings:
            wl, wv = self.lexical_weight, self.vector_weight
        else:
            wl, wv = 1.0, 0.0

        cands: list[Candidate] = []
        for i, e in enumerate(self.entries):
            if flavor and flavor not in e.flavor:
                continue
            l = float(lex[i])
            v = vec_scores.get(e.id, 0.0)
            score = wl * l + wv * v
            if score <= 0:
                continue
            if mode and e.mode == mode:
                score += self.mode_boost
            strength = wl * float(lex_raw[i]) + wv * v
            cands.append(Candidate(e, min(score, 1.0), lexical=l, vector=v, strength=strength))
        cands.sort(key=lambda c: (-c.score, -c.strength, c.entry.id))
        return RetrievalResult(cands[:k], used_embeddings, qtoks)


def build_vector_index(kb: KnowledgeBase, embed_many, model: str) -> VectorIndex:
    """``embed_many(list[str]) -> list[list[float]]``."""
    entries = kb.indexed()
    texts = [e.retrieval_text() for e in entries]
    vectors = embed_many(texts)
    matrix = np.asarray(vectors, dtype=np.float32)
    return VectorIndex([e.id for e in entries], matrix, model, kb.kb_hash)
