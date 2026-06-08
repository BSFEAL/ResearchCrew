"""Tests for researchcrew.llm — all run without real API keys."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchcrew.llm.judge import LLMDebateJudge
from researchcrew.models.debate_record import DebateVerdict
from researchcrew.models.hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hypothesis(id: str, statement: str = "Test hypothesis") -> Hypothesis:
    return Hypothesis(id=id, statement=statement)


def _make_complete(response: str):
    """Return a synchronous complete() stub that always returns response."""
    return lambda prompt, **kwargs: response


# ---------------------------------------------------------------------------
# LLMDebateJudge
# ---------------------------------------------------------------------------

class TestLLMDebateJudge:
    def test_good_json_verdict(self):
        ha = _make_hypothesis("A")
        hb = _make_hypothesis("B")
        payload = json.dumps(
            {"winner_id": "B", "loser_id": "A", "rationale": "B is stronger", "confidence": 0.85}
        )
        judge = LLMDebateJudge(_make_complete(payload))
        verdict = judge.judge(ha, hb, "test context")
        assert verdict.winner_id == "B"
        assert verdict.loser_id == "A"
        assert verdict.confidence == pytest.approx(0.85)
        assert "stronger" in verdict.rationale

    def test_bad_json_falls_back_to_a(self):
        ha = _make_hypothesis("X")
        hb = _make_hypothesis("Y")
        judge = LLMDebateJudge(_make_complete("not json at all"))
        verdict = judge.judge(ha, hb, "context")
        assert verdict.winner_id == "X"
        assert verdict.loser_id == "Y"
        assert verdict.confidence == pytest.approx(0.5)

    def test_invalid_winner_id_is_corrected(self):
        ha = _make_hypothesis("A")
        hb = _make_hypothesis("B")
        payload = json.dumps(
            {"winner_id": "UNKNOWN", "loser_id": "B", "rationale": "x", "confidence": 0.6}
        )
        judge = LLMDebateJudge(_make_complete(payload))
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.winner_id in {"A", "B"}

    def test_winner_equals_loser_is_corrected(self):
        ha = _make_hypothesis("A")
        hb = _make_hypothesis("B")
        payload = json.dumps(
            {"winner_id": "A", "loser_id": "A", "rationale": "oops", "confidence": 0.5}
        )
        judge = LLMDebateJudge(_make_complete(payload))
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.winner_id != verdict.loser_id

    def test_confidence_clamped(self):
        ha = _make_hypothesis("A")
        hb = _make_hypothesis("B")
        payload = json.dumps(
            {"winner_id": "A", "loser_id": "B", "rationale": "x", "confidence": 99.0}
        )
        judge = LLMDebateJudge(_make_complete(payload))
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.confidence <= 1.0


# ---------------------------------------------------------------------------
# ResearchLLM factory methods (using a concrete stub subclass)
# ---------------------------------------------------------------------------

class _StubLLM:
    """Minimal stub that satisfies the ResearchLLM interface without ABC machinery."""

    def __init__(self, response: str = "stub response") -> None:
        self._response = response

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        return self._response

    # Borrow factory methods from ResearchLLM without subclassing ABC
    section_writer_fn = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.section_writer_fn
    outline_fn = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.outline_fn
    review_fn = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.review_fn
    rewrite_fn = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.rewrite_fn
    screener_fn = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.screener_fn
    debate_judge = __import__(
        "researchcrew.llm.base", fromlist=["ResearchLLM"]
    ).ResearchLLM.debate_judge


class TestResearchLLMFactories:
    def test_section_writer_returns_string(self):
        llm = _StubLLM("Here is the introduction.")
        fn = llm.section_writer_fn(llm)
        result = fn("Introduction", {"topic": "AI"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_outline_fn_parses_json(self):
        payload = json.dumps(
            {
                "section_plan": [{"name": "Intro", "description": "Overview", "estimated_words": 300}],
                "visualization_plan": [],
                "citation_strategy": "IEEE",
            }
        )
        llm = _StubLLM(payload)
        fn = llm.outline_fn(llm)
        result = fn("Write an outline about X")
        assert "section_plan" in result

    def test_outline_fn_bad_json_returns_empty_dict(self):
        llm = _StubLLM("not json")
        fn = llm.outline_fn(llm)
        result = fn("prompt")
        assert result == {}

    def test_review_fn_returns_four_dims(self):
        payload = json.dumps(
            {"clarity": 8.0, "correctness": 7.5, "novelty": 6.0, "citation_quality": 9.0}
        )
        llm = _StubLLM(payload)
        fn = llm.review_fn(llm)
        section = MagicMock()
        section.name = "Methods"
        section.content = "We used X."
        scores = fn(section)
        assert set(scores.keys()) == {"clarity", "correctness", "novelty", "citation_quality"}
        assert scores["clarity"] == pytest.approx(8.0)

    def test_review_fn_bad_json_returns_defaults(self):
        llm = _StubLLM("bad")
        fn = llm.review_fn(llm)
        section = MagicMock()
        section.name = "X"
        section.content = "y"
        scores = fn(section)
        assert all(v == pytest.approx(7.0) for v in scores.values())

    def test_rewrite_fn_returns_string(self):
        llm = _StubLLM("Improved section text.")
        fn = llm.rewrite_fn(llm)
        section = MagicMock()
        section.name = "Results"
        section.content = "Old text."
        score = MagicMock()
        score.scores = {"clarity": 5.0, "correctness": 8.0}
        score.weakest_dimension = "clarity"
        result = fn(section, score)
        assert isinstance(result, str)

    def test_screener_fn_returns_float_in_range(self):
        llm = _StubLLM("0.73")
        fn = llm.screener_fn(llm)
        paper = MagicMock()
        paper.title = "A study on X"
        paper.abstract = "We investigate X."
        score = fn(paper, "Understanding X")
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.73)

    def test_screener_fn_clamps_out_of_range(self):
        llm = _StubLLM("5.0")
        fn = llm.screener_fn(llm)
        paper = MagicMock()
        paper.title = "T"
        paper.abstract = ""
        assert fn(paper, "goal") == pytest.approx(1.0)

    def test_screener_fn_bad_response_returns_half(self):
        llm = _StubLLM("not a number")
        fn = llm.screener_fn(llm)
        paper = MagicMock()
        paper.title = "T"
        paper.abstract = ""
        assert fn(paper, "goal") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Provider constructors — ImportError when SDK not installed
# ---------------------------------------------------------------------------

class TestProviderConstructors:
    def test_anthropic_llm_raises_import_error_when_missing(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            from importlib import reload
            import researchcrew.llm.anthropic_llm as mod
            reload(mod)
            with pytest.raises(ImportError, match="researchcrew\\[llm\\]"):
                mod.AnthropicResearchLLM(api_key="fake")

    def test_openai_llm_raises_import_error_when_missing(self):
        with patch.dict("sys.modules", {"openai": None}):
            from importlib import reload
            import researchcrew.llm.openai_llm as mod
            reload(mod)
            with pytest.raises(ImportError, match="researchcrew\\[llm\\]"):
                mod.OpenAIResearchLLM(api_key="fake")
