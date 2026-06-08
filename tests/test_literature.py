from __future__ import annotations

from researchcrew.literature.citation_graph import CitationGraph
from researchcrew.literature.corpus import LiteratureCorpus, Paper
from researchcrew.literature.gap_analyzer import GapAnalyzer
from researchcrew.literature.screener import ScreeningCriteria, TwoPhaseScreener
from researchcrew.models.hypothesis import Hypothesis


def _make_paper(
    title: str,
    abstract: str = "",
    year: int = 2024,
    source: str = "arxiv",
) -> Paper:
    return Paper(title=title, abstract=abstract, year=year, source=source)  # type: ignore[arg-type]


# ── LiteratureCorpus ──────────────────────────────────────────────────

def test_corpus_add_and_size() -> None:
    corpus = LiteratureCorpus(project_id="test")
    corpus.add(_make_paper("Paper A"))
    corpus.add(_make_paper("Paper B"))
    assert corpus.size == 2


def test_corpus_deduplication_by_doi() -> None:
    corpus = LiteratureCorpus(project_id="test")
    p1 = Paper(title="A", doi="10.1000/test", source="arxiv")
    p2 = Paper(title="A duplicate", doi="10.1000/test", source="semantic_scholar")
    corpus.add(p1)
    corpus.add(p2)
    assert corpus.size == 1


def test_corpus_all_papers() -> None:
    corpus = LiteratureCorpus(project_id="test")
    for i in range(4):
        corpus.add(_make_paper(f"Paper {i}"))
    assert len(corpus.all_papers()) == 4


# ── TwoPhaseScreener ───────────────────────────────────────────────

def test_screener_include_keyword() -> None:
    papers = [
        _make_paper("CRISPR gene editing", abstract="CRISPR method"),
        _make_paper("Neural networks", abstract="Deep learning"),
    ]
    screener = TwoPhaseScreener(
        ScreeningCriteria(include_keywords=["crispr"])
    )
    passed = screener.passed(papers)
    assert len(passed) == 1
    assert passed[0].title == "CRISPR gene editing"


def test_screener_exclude_keyword() -> None:
    papers = [
        _make_paper("Good paper", abstract="relevant content"),
        _make_paper("Retracted paper", abstract="retracted content"),
    ]
    screener = TwoPhaseScreener(
        ScreeningCriteria(exclude_keywords=["retracted"])
    )
    passed = screener.passed(papers)
    assert all("retracted" not in p.title.lower() for p in passed)


def test_screener_year_filter() -> None:
    papers = [
        _make_paper("Old paper", year=2015),
        _make_paper("New paper", year=2023),
    ]
    screener = TwoPhaseScreener(ScreeningCriteria(min_year=2020))
    passed = screener.passed(papers)
    assert len(passed) == 1
    assert passed[0].title == "New paper"


def test_screener_no_abstract_filtered() -> None:
    papers = [
        Paper(title="Has abstract", abstract="some text", source="arxiv"),
        Paper(title="No abstract", abstract="", source="arxiv"),
    ]
    screener = TwoPhaseScreener(ScreeningCriteria(require_abstract=True))
    passed = screener.passed(papers)
    assert len(passed) == 1


# ── CitationGraph ────────────────────────────────────────────────────

def test_citation_graph_add_edge() -> None:
    g = CitationGraph()
    g.add_edge("p1", "p2")
    assert "p2" in g.references("p1")
    assert "p1" in g.cited_by("p2")


def test_citation_graph_from_corpus() -> None:
    corpus = LiteratureCorpus(project_id="test")
    p1 = Paper(title="A", source="arxiv", cited_by=["p2_fake"])
    p2 = Paper(title="B", source="arxiv", references=[p1.id])
    corpus.add(p1)
    corpus.add(p2)
    g = CitationGraph.from_corpus(corpus)
    assert g.num_edges >= 1


def test_citation_graph_hubs() -> None:
    g = CitationGraph()
    for i in range(5):
        g.add_edge(f"paper_{i}", "hub")
    hubs = g.hub_ids(top_k=1)
    assert hubs[0] == "hub"


# ── GapAnalyzer (Jaccard fallback, no embeddings needed) ───────────

def test_gap_analyzer_finds_gaps() -> None:
    corpus = LiteratureCorpus(project_id="test")
    corpus.add(_make_paper(
        "Protein folding using AlphaFold",
        abstract="AlphaFold predicts protein structure accurately",
    ))
    hypotheses = [
        Hypothesis(
            title="Quantum computing for chemistry",
            body="Quantum algorithms can simulate molecular dynamics efficiently",
            domain="chemistry",
        )
    ]
    gaps = GapAnalyzer(gap_threshold=0.05).find_gaps(hypotheses, corpus)
    assert len(gaps) == 1
    assert gaps[0].hypothesis.title == "Quantum computing for chemistry"


def test_gap_analyzer_empty_corpus() -> None:
    corpus = LiteratureCorpus(project_id="test")
    hypotheses = [Hypothesis(title="Any", body="Any body text", domain="test")]
    gaps = GapAnalyzer().find_gaps(hypotheses, corpus)
    assert len(gaps) == 1
    assert gaps[0].nearest_paper_id is None
