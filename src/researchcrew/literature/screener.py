from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from researchcrew.literature.corpus import Paper


class ScreeningCriteria(BaseModel):
    """Inclusion/exclusion criteria for two-phase screening."""

    include_keywords: list[str] = field(default_factory=list)  # type: ignore[assignment]
    exclude_keywords: list[str] = field(default_factory=list)  # type: ignore[assignment]
    min_year: int | None = None
    max_year: int | None = None
    require_abstract: bool = True
    min_relevance_score: float = 0.3


@dataclass
class ScreeningResult:
    paper_id: str
    passed_phase1: bool
    passed_phase2: bool
    relevance_score: float
    reason: str = ""


class TwoPhaseScreener:
    """AgentSLR-inspired two-phase screener.

    Phase 1: fast keyword + year filter on title + abstract (no LLM).
    Phase 2: relevance score against research goal (LLM-based when scorer is wired).
    """

    def __init__(
        self,
        criteria: ScreeningCriteria | None = None,
        llm_scorer: Any | None = None,
    ) -> None:
        self.criteria = criteria or ScreeningCriteria()
        self.llm_scorer = llm_scorer

    def screen(
        self,
        papers: list[Paper],
        research_goal: str = "",
    ) -> list[ScreeningResult]:
        results = []
        phase1_survivors = []

        for paper in papers:
            r = self._phase1(paper)
            results.append(r)
            if r.passed_phase1:
                phase1_survivors.append(paper)

        if self.llm_scorer is not None and phase1_survivors:
            for paper in phase1_survivors:
                score = self._phase2_score(paper, research_goal)
                for r in results:
                    if r.paper_id == paper.id:
                        r.passed_phase2 = score >= self.criteria.min_relevance_score
                        r.relevance_score = score
                        break

        return results

    def passed(self, papers: list[Paper], research_goal: str = "") -> list[Paper]:
        results = {r.paper_id: r for r in self.screen(papers, research_goal)}
        return [
            p for p in papers
            if results.get(p.id, ScreeningResult(p.id, False, False, 0.0)).passed_phase1
        ]

    def _phase1(self, paper: Paper) -> ScreeningResult:
        c = self.criteria
        text = f"{paper.title} {paper.abstract}".lower()

        if c.require_abstract and not paper.abstract.strip():
            return ScreeningResult(paper.id, False, False, 0.0, "no abstract")
        if c.min_year and paper.year and paper.year < c.min_year:
            return ScreeningResult(paper.id, False, False, 0.0, "too old")
        if c.max_year and paper.year and paper.year > c.max_year:
            return ScreeningResult(paper.id, False, False, 0.0, "too recent")
        if c.exclude_keywords and any(kw.lower() in text for kw in c.exclude_keywords):
            return ScreeningResult(paper.id, False, False, 0.0, "excluded keyword")
        if c.include_keywords and not any(kw.lower() in text for kw in c.include_keywords):
            return ScreeningResult(paper.id, False, False, 0.0, "no include keyword")

        return ScreeningResult(paper.id, True, False, 0.0)

    def _phase2_score(self, paper: Paper, research_goal: str) -> float:
        if self.llm_scorer is None:
            return 1.0
        try:
            return float(self.llm_scorer(paper, research_goal))
        except Exception:
            return 0.0
