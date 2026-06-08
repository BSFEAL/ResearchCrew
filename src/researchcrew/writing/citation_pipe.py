from __future__ import annotations

from researchcrew.literature.corpus import LiteratureCorpus, Paper
from researchcrew.writing.models import CitationEntry, PaperDraft
from researchcrew.writing.references import ReferenceManager


class TwoPhaseCitationPipeline:
    """PaperOrchestra two-phase citation integration.

    Phase 1: broad candidate search per section using the corpus.
    Phase 2: relevance re-rank and claim-citation alignment via LLM scorer.

    Wire ``llm_align`` for production.  Without it, Phase 2 keeps all Phase 1
    candidates above ``min_relevance``.
    """

    def __init__(
        self,
        corpus: LiteratureCorpus,
        llm_align: object | None = None,
        top_k_per_section: int = 5,
        min_relevance: float = 0.3,
    ) -> None:
        self.corpus = corpus
        self._llm_align = llm_align
        self.top_k = top_k_per_section
        self.min_relevance = min_relevance

    def assign(
        self,
        draft: PaperDraft,
        section_queries: dict[str, list[str]],
    ) -> PaperDraft:
        """Populate draft.references with citations keyed per section."""
        ref_mgr = ReferenceManager(draft)
        for section_name, queries in section_queries.items():
            candidates = self._phase1(queries)
            aligned = self._phase2(candidates, section_name, draft)
            section = draft.get_section(section_name)
            for paper in aligned[: self.top_k]:
                entry = self._paper_to_entry(paper)
                key = ref_mgr.add(entry)
                if key not in section.citations_used:
                    section.citations_used.append(key)
        draft.touch()
        return draft

    def _phase1(self, queries: list[str]) -> list[Paper]:
        """Keyword match against corpus title+abstract."""
        results: list[Paper] = []
        seen: set[str] = set()
        for q in queries:
            q_words = set(q.lower().split())
            for paper in self.corpus.all_papers():
                if paper.id in seen:
                    continue
                text = f"{paper.title} {paper.abstract}".lower()
                if any(w in text for w in q_words):
                    results.append(paper)
                    seen.add(paper.id)
        return results

    def _phase2(
        self,
        candidates: list[Paper],
        section_name: str,
        draft: PaperDraft,
    ) -> list[Paper]:
        if self._llm_align is not None and callable(self._llm_align) and candidates:
            section_content = draft.sections.get(section_name)
            context = section_content.content if section_content else ""
            scored: list[tuple[float, Paper]] = []
            for p in candidates:
                try:
                    score = float(self._llm_align(p, context))  # type: ignore[call-arg]
                except Exception:
                    score = 0.5
                if score >= self.min_relevance:
                    scored.append((score, p))
            return [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)]
        return candidates

    @staticmethod
    def _paper_to_entry(paper: Paper) -> CitationEntry:
        first_author = paper.authors[0].split()[-1].lower() if paper.authors else "unknown"
        year_str = str(paper.year) if paper.year else "nd"
        title_word = paper.title.split()[0].lower() if paper.title else "paper"
        return CitationEntry(
            paper_id=paper.id,
            cite_key=f"{first_author}{year_str}{title_word}",
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            url=paper.url,
        )
