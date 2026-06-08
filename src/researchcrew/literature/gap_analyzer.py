from __future__ import annotations

from dataclasses import dataclass

from researchcrew.literature.corpus import LiteratureCorpus, Paper
from researchcrew.models.hypothesis import Hypothesis


@dataclass
class ResearchGap:
    hypothesis: Hypothesis
    min_corpus_similarity: float
    nearest_paper_id: str | None
    nearest_paper_title: str | None

    def __str__(self) -> str:
        return (
            f"Gap: '{self.hypothesis.title}' "
            f"(nearest paper similarity: {self.min_corpus_similarity:.2f})"
        )


class GapAnalyzer:
    """Identifies hypothesis vectors with no nearby corpus paper (under-explored areas).

    Requires ``researchcrew[memory]`` for embedding support.
    Falls back to keyword-overlap scoring when embeddings are unavailable.
    """

    def __init__(
        self,
        gap_threshold: float = 0.4,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.gap_threshold = gap_threshold
        self.model_name = model_name

    def find_gaps(
        self,
        hypotheses: list[Hypothesis],
        corpus: LiteratureCorpus,
    ) -> list[ResearchGap]:
        papers = corpus.all_papers()
        if not papers:
            return [
                ResearchGap(
                    hypothesis=h,
                    min_corpus_similarity=0.0,
                    nearest_paper_id=None,
                    nearest_paper_title=None,
                )
                for h in hypotheses
            ]

        try:
            return self._find_gaps_embeddings(hypotheses, papers)
        except ImportError:
            return self._find_gaps_jaccard(hypotheses, papers)

    def _find_gaps_embeddings(
        self, hypotheses: list[Hypothesis], papers: list[Paper]
    ) -> list[ResearchGap]:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        h_texts = [h.body for h in hypotheses]
        p_texts = [f"{p.title} {p.abstract}" for p in papers]

        h_vecs = model.encode(h_texts, normalize_embeddings=True)
        p_vecs = model.encode(p_texts, normalize_embeddings=True)
        sim_matrix = h_vecs @ p_vecs.T  # (n_hyp, n_papers)

        gaps = []
        for i, h in enumerate(hypotheses):
            sims = sim_matrix[i]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < self.gap_threshold:
                gaps.append(ResearchGap(
                    hypothesis=h,
                    min_corpus_similarity=best_sim,
                    nearest_paper_id=papers[best_idx].id,
                    nearest_paper_title=papers[best_idx].title,
                ))
        return gaps

    def _find_gaps_jaccard(
        self, hypotheses: list[Hypothesis], papers: list[Paper]
    ) -> list[ResearchGap]:
        gaps = []
        for h in hypotheses:
            h_words = set(h.body.lower().split())
            best_sim, best_paper = 0.0, papers[0]
            for p in papers:
                p_words = set(f"{p.title} {p.abstract}".lower().split())
                union = h_words | p_words
                sim = len(h_words & p_words) / len(union) if union else 0.0
                if sim > best_sim:
                    best_sim, best_paper = sim, p
            if best_sim < self.gap_threshold:
                gaps.append(ResearchGap(
                    hypothesis=h,
                    min_corpus_similarity=best_sim,
                    nearest_paper_id=best_paper.id,
                    nearest_paper_title=best_paper.title,
                ))
        return gaps
