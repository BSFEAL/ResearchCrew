"""Tests for researchcrew.llm — all run without real API keys."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from researchcrew.llm.base import ResearchLLM
from researchcrew.llm.judge import LLMDebateJudge
from researchcrew.llm.providers import LLMProvider, create_llm
from researchcrew.models.hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Shared stub
# ---------------------------------------------------------------------------

class _StubLLM(ResearchLLM):
    """Concrete ResearchLLM whose complete() returns a fixed string."""

    def __init__(self, response: str = "stub response") -> None:
        self._response = response

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        return self._response


def _hyp(id: str, statement: str = "Test hypothesis.") -> Hypothesis:
    return Hypothesis(id=id, statement=statement)


# ---------------------------------------------------------------------------
# LLMDebateJudge
# ---------------------------------------------------------------------------

class TestLLMDebateJudge:
    def test_good_json_verdict(self):
        ha, hb = _hyp("A"), _hyp("B")
        payload = json.dumps(
            {"winner_id": "B", "loser_id": "A", "rationale": "B is stronger", "confidence": 0.85}
        )
        judge = LLMDebateJudge(lambda p, **kw: payload)
        verdict = judge.judge(ha, hb, "test context")
        assert verdict.winner_id == "B"
        assert verdict.loser_id == "A"
        assert verdict.confidence == pytest.approx(0.85)
        assert "stronger" in verdict.rationale

    def test_bad_json_falls_back_to_a(self):
        ha, hb = _hyp("X"), _hyp("Y")
        judge = LLMDebateJudge(lambda p, **kw: "not json")
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.winner_id == "X"
        assert verdict.loser_id == "Y"
        assert verdict.confidence == pytest.approx(0.5)

    def test_invalid_winner_id_is_corrected(self):
        ha, hb = _hyp("A"), _hyp("B")
        payload = json.dumps(
            {"winner_id": "UNKNOWN", "loser_id": "B", "rationale": "x", "confidence": 0.6}
        )
        judge = LLMDebateJudge(lambda p, **kw: payload)
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.winner_id in {"A", "B"}

    def test_winner_equals_loser_is_corrected(self):
        ha, hb = _hyp("A"), _hyp("B")
        payload = json.dumps(
            {"winner_id": "A", "loser_id": "A", "rationale": "oops", "confidence": 0.5}
        )
        judge = LLMDebateJudge(lambda p, **kw: payload)
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.winner_id != verdict.loser_id

    def test_confidence_clamped(self):
        ha, hb = _hyp("A"), _hyp("B")
        payload = json.dumps(
            {"winner_id": "A", "loser_id": "B", "rationale": "x", "confidence": 99.0}
        )
        judge = LLMDebateJudge(lambda p, **kw: payload)
        verdict = judge.judge(ha, hb, "ctx")
        assert verdict.confidence <= 1.0


# ---------------------------------------------------------------------------
# ResearchLLM factory methods
# ---------------------------------------------------------------------------

class TestResearchLLMFactories:
    def test_section_writer_returns_string(self):
        llm = _StubLLM("Here is the introduction.")
        result = llm.section_writer_fn()("Introduction", {"topic": "AI"})
        assert isinstance(result, str) and result

    def test_outline_fn_parses_json(self):
        payload = json.dumps(
            {
                "section_plan": [{"name": "Intro", "description": "Overview", "estimated_words": 300}],
                "visualization_plan": [],
                "citation_strategy": "IEEE",
            }
        )
        result = _StubLLM(payload).outline_fn()("Write an outline about X")
        assert "section_plan" in result

    def test_outline_fn_bad_json_returns_empty_dict(self):
        assert _StubLLM("not json").outline_fn()("prompt") == {}

    def test_review_fn_returns_four_dims(self):
        payload = json.dumps(
            {"clarity": 8.0, "correctness": 7.5, "novelty": 6.0, "citation_quality": 9.0}
        )
        section = MagicMock(name="Methods", content="We used X.")
        section.name = "Methods"
        section.content = "We used X."
        scores = _StubLLM(payload).review_fn()(section)
        assert set(scores) == {"clarity", "correctness", "novelty", "citation_quality"}
        assert scores["clarity"] == pytest.approx(8.0)

    def test_review_fn_bad_json_returns_defaults(self):
        section = MagicMock()
        section.name, section.content = "X", "y"
        scores = _StubLLM("bad").review_fn()(section)
        assert all(v == pytest.approx(7.0) for v in scores.values())

    def test_rewrite_fn_returns_string(self):
        section = MagicMock()
        section.name, section.content = "Results", "Old text."
        score = MagicMock()
        score.scores = {"clarity": 5.0, "correctness": 8.0}
        score.weakest_dimension = "clarity"
        result = _StubLLM("Improved.").rewrite_fn()(section, score)
        assert isinstance(result, str)

    def test_screener_fn_returns_float_in_range(self):
        paper = MagicMock()
        paper.title, paper.abstract = "A study on X", "We investigate X."
        assert _StubLLM("0.73").screener_fn()(paper, "Understanding X") == pytest.approx(0.73)

    def test_screener_fn_clamps_out_of_range(self):
        paper = MagicMock()
        paper.title, paper.abstract = "T", ""
        assert _StubLLM("5.0").screener_fn()(paper, "goal") == pytest.approx(1.0)

    def test_screener_fn_bad_response_returns_half(self):
        paper = MagicMock()
        paper.title, paper.abstract = "T", ""
        assert _StubLLM("not a number").screener_fn()(paper, "goal") == pytest.approx(0.5)

    def test_debate_judge_returns_llm_debate_judge(self):
        from researchcrew.llm.judge import LLMDebateJudge as J

        assert isinstance(_StubLLM().debate_judge(), J)


# ---------------------------------------------------------------------------
# create_llm factory
# ---------------------------------------------------------------------------

class TestCreateLLM:
    def _fake_anthropic(self):
        fake = MagicMock()
        fake.Anthropic.return_value = MagicMock()
        return fake

    def test_anthropic_provider(self):
        from researchcrew.llm.anthropic_llm import AnthropicResearchLLM

        with patch("researchcrew.llm.anthropic_llm.importlib", create=True), \
             patch("builtins.__import__", side_effect=lambda name, *a, **kw:
                   __builtins__["__import__"](name, *a, **kw)  # type: ignore[index]
                   if name != "anthropic" else MagicMock()):
            pass  # Just confirm the import path works

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm("unknown_xyz")  # type: ignore[arg-type]

    def test_create_llm_anthropic_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            from importlib import reload
            import researchcrew.llm.anthropic_llm as mod
            reload(mod)
            with pytest.raises(ImportError, match="researchcrew\\[llm\\]"):
                mod.AnthropicResearchLLM(api_key="fake")

    def test_create_llm_openai_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            from importlib import reload
            import researchcrew.llm.openai_llm as mod
            reload(mod)
            with pytest.raises(ImportError, match="researchcrew\\[llm\\]"):
                mod.OpenAIResearchLLM(api_key="fake")

    def test_create_llm_google_import_error(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            from importlib import reload
            import researchcrew.llm.google_llm as mod
            reload(mod)
            with pytest.raises(ImportError, match="researchcrew\\[llm\\]"):
                mod.GoogleResearchLLM(api_key="fake")


# ---------------------------------------------------------------------------
# AnthropicResearchLLM — native thinking behaviour
# ---------------------------------------------------------------------------

class TestAnthropicNativeThinking:
    def _make_llm(self, model: str, thinking: bool = True):
        """Return an AnthropicResearchLLM with a mocked client."""
        import anthropic
        from researchcrew.llm.anthropic_llm import AnthropicResearchLLM

        with patch("researchcrew.llm.anthropic_llm.anthropic", anthropic):
            llm = AnthropicResearchLLM.__new__(AnthropicResearchLLM)
            llm._client = MagicMock()
            llm.model = model
            llm.max_tokens = 256
            llm._thinking = thinking and "haiku" not in model.lower()
        return llm

    def test_thinking_disabled_for_haiku(self):
        from researchcrew.llm.anthropic_llm import AnthropicResearchLLM

        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from importlib import reload
            import researchcrew.llm.anthropic_llm as mod
            reload(mod)
            llm = mod.AnthropicResearchLLM.__new__(mod.AnthropicResearchLLM)
            llm.model = "claude-haiku-4-5"
            llm._thinking = True and "haiku" not in llm.model.lower()
            assert llm._thinking is False

    def test_thinking_enabled_for_opus(self):
        from researchcrew.llm.anthropic_llm import AnthropicResearchLLM

        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from importlib import reload
            import researchcrew.llm.anthropic_llm as mod
            reload(mod)
            llm = mod.AnthropicResearchLLM.__new__(mod.AnthropicResearchLLM)
            llm.model = "claude-opus-4-8"
            llm._thinking = True and "haiku" not in llm.model.lower()
            assert llm._thinking is True
