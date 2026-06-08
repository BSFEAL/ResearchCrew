from __future__ import annotations

import abc
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchcrew.literature.corpus import Paper
    from researchcrew.writing.models import Section
    from researchcrew.writing.refinement import SectionScore


class ResearchLLM(abc.ABC):
    """Abstract base: implement complete() to get factory methods for every injection point."""

    @abc.abstractmethod
    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        """Send a single-turn prompt and return the text response."""
        ...

    # ------------------------------------------------------------------
    # Factory methods — return correctly-shaped callables for each
    # injection point in the ResearchCrew component classes.
    # ------------------------------------------------------------------

    def debate_judge(self) -> "LLMDebateJudge":
        from researchcrew.llm.judge import LLMDebateJudge

        return LLMDebateJudge(self.complete)

    def section_writer_fn(self):
        """Returns callable matching SectionWriter(llm_write) signature."""

        def _write(section_name: str, context: dict[str, Any]) -> str:
            context_text = "\n".join(
                f"{k}: {v}" for k, v in context.items() if k != "full_outline"
            )
            outline_hint = ""
            if "full_outline" in context:
                outline_hint = f"\n\nFull outline context:\n{context['full_outline']}"
            prompt = (
                f"You are writing the '{section_name}' section of a scientific research paper.\n\n"
                f"Context:\n{context_text}{outline_hint}\n\n"
                "Write a complete, well-structured section using precise academic language. "
                "Include relevant citations as [Author, Year] placeholders where appropriate. "
                "Do not include the section heading in your response — only the body text."
            )
            return self.complete(prompt)

        return _write

    def outline_fn(self):
        """Returns callable matching OutlineBuilder(llm_outline) signature."""

        def _outline(prompt: str) -> dict[str, Any]:
            full_prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON — no markdown fences, no commentary. "
                "The JSON must contain: section_plan (list of {name, description, estimated_words}), "
                "visualization_plan (list of {figure_type, caption}), "
                "citation_strategy (string)."
            )
            raw = self.complete(full_prompt, json_mode=True)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}

        return _outline

    def review_fn(self):
        """Returns callable matching PeerReviewSimulator(llm_review) signature."""

        def _review(section: "Section") -> dict[str, float]:
            prompt = (
                f"Review the following '{section.name}' section of a research paper.\n\n"
                f"{section.content}\n\n"
                "Score each dimension from 0 to 10. Return ONLY valid JSON with keys: "
                "clarity, correctness, novelty, citation_quality. "
                "No markdown, no commentary."
            )
            raw = self.complete(prompt, json_mode=True)
            try:
                data = json.loads(raw)
                dims = ("clarity", "correctness", "novelty", "citation_quality")
                return {k: max(0.0, min(10.0, float(data.get(k, 7.0)))) for k in dims}
            except Exception:
                return {"clarity": 7.0, "correctness": 7.0, "novelty": 7.0, "citation_quality": 7.0}

        return _review

    def rewrite_fn(self):
        """Returns callable matching PeerReviewSimulator(llm_rewrite) signature."""

        def _rewrite(section: "Section", score: "SectionScore") -> str:
            scores_text = ", ".join(
                f"{k}={v:.1f}" for k, v in score.scores.items()
            )
            prompt = (
                f"Improve the following '{section.name}' section of a research paper.\n\n"
                f"Current peer-review scores: {scores_text}\n"
                f"Primary weakness to address: {score.weakest_dimension}\n\n"
                f"{section.content}\n\n"
                "Rewrite the section, directly addressing the identified weakness. "
                "Preserve the academic tone. Return only the rewritten body text — no heading."
            )
            return self.complete(prompt)

        return _rewrite

    def screener_fn(self):
        """Returns callable matching TwoPhaseScreener(llm_scorer) signature."""

        def _score(paper: "Paper", research_goal: str) -> float:
            abstract = getattr(paper, "abstract", "") or ""
            prompt = (
                f"Research goal: {research_goal}\n\n"
                f"Paper title: {paper.title}\n"
                f"Abstract: {abstract[:1000]}\n\n"
                "Rate the relevance of this paper to the research goal on a scale from 0.0 (irrelevant) "
                "to 1.0 (highly relevant). Respond with a single floating-point number only."
            )
            raw = self.complete(prompt).strip()
            try:
                return max(0.0, min(1.0, float(raw)))
            except (ValueError, TypeError):
                return 0.5

        return _score
