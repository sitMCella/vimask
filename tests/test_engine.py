"""Pipeline tests without Ollama: lookup, retrieval, fast path, fallback,
selector validation (mocked), composer, rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from vimask.composer import CompositionError, build_keys, compose, describe
from vimask.engine import Engine, Session
from vimask.evaluate import load_eval, run_eval
from vimask.lookup import alias_lookup, normalize, reverse_lookup
from vimask.render import Answer, NotFound, plain_response
from vimask.retrieval import Retriever, tokenize
from vimask.selector import StageBResult, validate_stage_a, validate_stage_b

from .conftest import REPO, needs_vim


@pytest.fixture
def engine(cfg, kb):
    return Engine(cfg, kb, client=None, use_selector=False, use_embeddings=False, log_missing=False)


def test_normalize_strips_filler():
    assert normalize("How do I delete a word in Vim?") == "delete a word"
    assert normalize("  please  join   lines!! ") == "join lines"


def test_reverse_lookup_quoted_and_phrased(kb):
    assert reverse_lookup("what does `ciw` do", kb).entries[0].id == "change-inner-word"
    assert reverse_lookup("what does ciw do?", kb).entries[0].id == "change-inner-word"
    assert reverse_lookup("explain gUU", kb).entries[0].id == "uppercase-line"
    assert reverse_lookup("di(", kb).entries[0].id == "delete-inside-parens"
    assert reverse_lookup("what is :%s/foo/bar/g", kb) is not None
    # plain words that happen to be keys are not reverse lookups
    assert reverse_lookup("delete a word", kb) is None
    assert reverse_lookup("go to the end of the line", kb) is None


def test_reverse_lookup_ex_template(kb):
    hit = reverse_lookup("what is :%s/foo/bar/g", kb)
    assert hit.entries[0].id == "substitute-all"
    assert reverse_lookup(":s/x/y/", kb).entries[0].id == "substitute-line"


def test_alias_lookup(kb):
    assert alias_lookup("delete word forward", kb).entries[0].id == "delete-word"
    assert alias_lookup("no such alias at all", kb) is None
    # aliases written as full questions match after normalization
    assert alias_lookup(normalize("how do i exit vim"), kb).entries[0].id == "quit"


def test_query_typo_correction(kb):
    r = Retriever(kb)
    assert r.lexical.query_tokens("delet a charcter") == ["delete", "char"]
    # short tokens and key sequences are never "corrected"
    assert r.lexical.query_tokens("ciw dd") == ["ciw", "dd"]
    assert r.retrieve("delet a charcter").top1.entry.id == "delete-char"


def test_saturated_scores_rank_by_strength(kb):
    r = Retriever(kb)
    res = r.retrieve("delete the word under the cursor", k=3)
    assert res.top1.entry.id == "delete-inner-word"
    assert res.candidates[0].strength >= res.candidates[1].strength


def test_tokenize_synonyms():
    assert tokenize("Remove the parentheses") == ["delete", "paren"]
    assert "yank" in tokenize("copy a word")


def test_retrieval_scores_are_normalized(kb):
    r = Retriever(kb)
    res = r.retrieve("delete everything inside the parentheses", k=5)
    assert res.top1.entry.id in ("delete-inside-parens", "delete-around-parens")
    assert 0 < res.top1.score <= 1.0
    assert all(c.lexical <= 1.0 for c in res.candidates)


def test_flavor_filter(kb):
    r = Retriever(kb)
    res = r.retrieve("browse files file explorer", k=5, flavor="nvim")
    assert all("nvim" in c.entry.flavor for c in res.candidates)


def test_engine_answers_and_falls_back(engine):
    resp = engine.answer("delete everything inside the parentheses")
    assert isinstance(resp, Answer)
    assert resp.primary.id == "delete-inside-parens"
    assert resp.primary.keys == "di("
    assert engine.last_trace.path in ("alias", "fast-path", "retrieval")

    resp = engine.answer("what is the weather today")
    assert isinstance(resp, NotFound)
    assert engine.last_trace.path == "fallback"


def test_engine_mode_hint(engine):
    s = Session(mode="insert")
    resp = engine.answer("delete a word", s)
    assert isinstance(resp, Answer)
    assert resp.primary.mode == "normal"
    out = plain_response(resp)
    assert "press <Esc> first" in out


def test_render_contains_only_kb_data(engine, kb):
    resp = engine.answer("find and replace in whole file")
    out = plain_response(resp)
    e = kb.get("substitute-all")
    assert e.keys in out
    assert e.description in out
    assert f":help {e.help_tag}" in out
    assert "replace {pattern}, {replacement}" in out


def test_stage_a_validation_drops_unknown_ids(kb):
    cands = [kb.get("delete-word"), kb.get("delete-inner-word")]
    res = validate_stage_a({"ids": ["delete-inner-word", "made-up", "delete-word", "delete-inner-word"], "compose": False}, cands)
    assert res.ids == ["delete-inner-word", "delete-word"]
    assert validate_stage_a(None, cands).ids == []
    assert validate_stage_a({"ids": "not-a-list", "compose": True}, cands).compose is True


def test_stage_b_validation(kb):
    res = validate_stage_b({"count": 3, "operator": "op-delete", "target": "mo-word-forward", "arg": None}, kb)
    assert res.operator.id == "op-delete" and res.target.id == "mo-word-forward" and res.count == 3
    bad = validate_stage_b({"count": 1, "operator": "delete-line", "target": "mo-word-forward", "arg": None}, kb)
    assert bad.operator is None and bad.error
    bad = validate_stage_b({"count": 1, "operator": "op-delete", "target": "undo", "arg": None}, kb)
    assert bad.target is None


def test_build_keys_and_describe(kb):
    d, w, t = kb.get("op-delete"), kb.get("mo-word-forward"), kb.get("mo-till-char")
    assert build_keys(d, w, 3) == "d3w"
    assert build_keys(d, t, 1, ")") == "dt)"
    assert describe(d, w, 3) == "Delete 3× to the start of the next word."
    assert describe(d, t, 1, ")") == "Delete up to (not including) ')'."
    with pytest.raises(CompositionError):
        build_keys(d, t, 1, None)  # arg required
    with pytest.raises(CompositionError):
        build_keys(d, w, 1, "x")  # arg forbidden
    with pytest.raises(CompositionError):
        build_keys(d, w, 0)
    with pytest.raises(CompositionError):
        build_keys(d, t, 1, "ab")
    with pytest.raises(CompositionError):
        build_keys(kb.get("delete-line"), w, 1)


@needs_vim
def test_compose_runs_example_in_vim(cfg, kb):
    parts = StageBResult(count=2, operator=kb.get("op-delete"), target=kb.get("mo-word-forward"))
    ans = compose(parts, kb, cfg, flavor="vim")
    assert ans.keys == "d2w"
    assert ans.example is not None
    assert ans.example.after == "one ┃four five six seven eight"
    parts = StageBResult(count=1, operator=kb.get("op-change"), target=kb.get("to-inner-paren"))
    ans = compose(parts, kb, cfg, flavor="vim")
    assert ans.keys == "ci("
    assert ans.example.after == "print(┃) + g(x)"


class FakeClient:
    """Stands in for OllamaClient: returns canned JSON."""

    available = True

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def status(self, refresh=False):
        from vimask.llm import OllamaStatus

        return OllamaStatus(True, ["qwen2.5:3b-instruct", "nomic-embed-text"])

    def chat_json(self, system, user, schema, num_predict=None):
        self.calls.append(user)
        return self.answers.pop(0) if self.answers else None


def test_selector_path_with_mock(cfg, kb):
    client = FakeClient([{"ids": ["delete-inner-word"], "compose": False}])
    eng = Engine(cfg, kb, client=client, use_selector=True, use_embeddings=False, log_missing=False)
    # an ambiguous question: make the fast path impossible by lowering nothing;
    # force the selector by raising thresholds
    eng.cfg.fast_score_lexical_only = 2.0
    resp = eng.answer("get rid of the word i am on")
    assert isinstance(resp, Answer)
    assert eng.last_trace.selector_called
    assert resp.primary.id == "delete-inner-word"
    assert "delete-inner-word | diw |" in client.calls[0]


def test_selector_cannot_inject_ids(cfg, kb):
    client = FakeClient([{"ids": ["quit-force"], "compose": False}])  # not among candidates
    eng = Engine(cfg, kb, client=client, use_selector=True, use_embeddings=False, log_missing=False)
    eng.cfg.fast_score_lexical_only = 2.0
    resp = eng.answer("wipe out everything inside the parentheses")
    # the injected id is dropped: either "not found" or a retrieval answer, never quit-force
    if isinstance(resp, Answer):
        assert resp.primary.id != "quit-force"
        assert resp.confidence == "medium"
    else:
        assert isinstance(resp, NotFound)


@needs_vim
def test_compose_path_with_mock(cfg, kb):
    client = FakeClient(
        [
            {"ids": [], "compose": True},
            {"count": 3, "operator": "op-delete", "target": "mo-word-forward", "arg": None},
        ]
    )
    eng = Engine(cfg, kb, client=client, use_selector=True, use_embeddings=False, log_missing=False)
    eng.cfg.fast_score_lexical_only = 2.0
    resp = eng.answer("delete the next three words")
    assert isinstance(resp, Answer)
    assert resp.primary.keys == "d3w"
    assert eng.last_trace.path == "composed"
    out = plain_response(resp)
    assert "d3w" in out and "before:" in out


def test_eval_set_lexical_only(cfg, kb):
    """Aggregate metrics in the no-Ollama configuration (design §7.3)."""
    items = load_eval(REPO / "eval" / "questions.yaml")
    eng = Engine(cfg, kb, client=None, use_selector=False, use_embeddings=False, log_missing=False)
    m = run_eval(eng, items)
    wrong = [o for o in m.outcomes if not o.correct]
    detail = "\n".join(f"{o.item.q!r}: expected {o.item.expect} got {o.got} ({o.confidence}, {o.path})" for o in wrong)
    assert m.top1_accuracy >= 0.85, f"top1={m.top1_accuracy:.2%}\n{detail}"
    assert m.false_answer_rate <= 0.02, f"false answers={m.false_answer_rate:.2%}\n{detail}"
    assert m.not_found_precision == 1.0, detail
