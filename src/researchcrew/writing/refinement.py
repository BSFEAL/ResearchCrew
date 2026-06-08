from __future__ import annotations

from dataclasses import dataclass

from researchcrew.writing.models import PaperDraft, Section, SectionStatus


@dataclass
class SectionScore:
    section_name: str
    clarity: float
    correctness: float
    novelty: float
    citation_quality: float

    @property
    def mean(self) -> float:
        return (self.clarity + self.correctness + self.novelty + self.citation_quality) / 4.0

    @property
    def weakest_dimension(self) -> str:
        dims = {
            "clarity": self.clarity,
            "correctness": self.correctness,
            "novelty": self.novelty,
            "citation_quality": self.citation_quality,
        }
        return min(dims, key=lambda k: dims[k])


class PeerReviewSimulator:
    """PaperOrchestra-style iterative peer-review refinement loop.

    Wire ``llm_review`` and ``llm_rewrite`` to real LLM callables for production.
    Both default to no-ops so the class is usable without an LLM in tests.
    """

    DEFAULT_THRESHOLD = 7.0
    DEFAULT_MAX_ROUNDS = 3

    def __init__(
        self,
        llm_review: object | None = None,
        llm_rewrite: object | None = None,
        acceptance_threshold: float = DEFAULT_THRESHOLD,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self._llm_review = llm_review
        self._llm_rewrite = llm_rewrite
        self.acceptance_threshold = acceptance_threshold
        self.max_rounds = max_rounds

    def refine(self, draft: PaperDraft) -> dict[str, list[SectionScore]]:
        """Run up to max_rounds of review→rewrite on every drafted section.

        Returns history of SectionScore lists per section name.
        """
        history: dict[str, list[SectionScore]] = {}

        for section_name, section in draft.sections.items():
            if section.status == SectionStatus.PLANNED:
                continue
            section_history: list[SectionScore] = []

            for _ in range(self.max_rounds):
                score = self._review(section)
                section_history.append(score)
                section.scores = {
                    "clarity": score.clarity,
                    "correctness": score.correctness,
                    "novelty": score.novelty,
                    "citation_quality": score.citation_quality,
                }
                if score.mean >= self.acceptance_threshold:
                    break
                new_content = self._rewrite(section, score)
                section.bump_version(new_content)

            section.status = SectionStatus.REFINED
            history[section_name] = section_history

        draft.touch()
        return history

    def _review(self, section: Section) -> SectionScore:
        if self._llm_review is not None and callable(self._llm_review):
            raw = self._llm_review(section)  # type: ignore[call-arg]
            if isinstance(raw, dict):
                return SectionScore(
                    section_name=section.name,
                    clarity=float(raw.get("clarity", 7.0)),
                    correctness=float(raw.get("correctness", 7.0)),
                    novelty=float(raw.get("novelty", 7.0)),
                    citation_quality=float(raw.get("citation_quality", 7.0)),
                )
        return SectionScore(
            section_name=section.name,
            clarity=8.0, correctness=8.0, novelty=8.0, citation_quality=8.0,
        )

    def _rewrite(self, section: Section, score: SectionScore) -> str:
        if self._llm_rewrite is not None and callable(self._llm_rewrite):
            result = self._llm_rewrite(section, score)  # type: ignore[call-arg]
            return str(result)
        return section.content
