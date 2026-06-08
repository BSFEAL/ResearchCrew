from __future__ import annotations

from researchcrew.literature.citation_graph import CitationGraph
from researchcrew.literature.corpus import LiteratureCorpus, Paper
from researchcrew.literature.gap_analyzer import GapAnalyzer, ResearchGap
from researchcrew.literature.screener import ScreeningCriteria, TwoPhaseScreener
from researchcrew.literature.searcher import MultiSourceSearcher


__all__ = [
    "CitationGraph",
    "GapAnalyzer",
    "LiteratureCorpus",
    "MultiSourceSearcher",
    "Paper",
    "ResearchGap",
    "ScreeningCriteria",
    "TwoPhaseScreener",
]
