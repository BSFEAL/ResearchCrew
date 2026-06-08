from __future__ import annotations

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.research_goal import Domain, ResearchGoal
from researchcrew.writing.exporter import PaperExporter
from researchcrew.writing.models import (
    CitationEntry,
    Outline,
    PaperDraft,
    Section,
    SectionStatus,
)
from researchcrew.writing.outline import OutlineBuilder
from researchcrew.writing.references import ReferenceManager
from researchcrew.writing.refinement import PeerReviewSimulator, SectionScore
from researchcrew.writing.section_writer import SECTION_ORDER, SectionWriter


def _make_draft(title: str = "Test Paper") -> PaperDraft:
    return PaperDraft(project_id="test", title=title)


def _make_goal() -> ResearchGoal:
    return ResearchGoal(
        title="Test Goal",
        description="Understanding neural network generalisation.",
        domain=Domain.ML,
        keywords=["neural networks", "generalisation"],
    )


def _make_hypotheses(n: int = 3) -> list[Hypothesis]:
    return [
        Hypothesis(
            title=f"Hypothesis {i}",
            body=f"Networks generalise via mechanism {i}",
            domain="ml",
            elo_score=1200.0 + i * 50,
        )
        for i in range(n)
    ]


# ── Models ───────────────────────────────────────────────────────────

def test_section_bump_version() -> None:
    s = Section(name="intro")
    s.bump_version("First draft.")
    assert s.content == "First draft."
    assert s.version == 1
    s.bump_version("Second draft.")
    assert s.content == "Second draft."
    assert len(s.revision_history) == 1  # first draft stored


def test_draft_get_section_creates_if_missing() -> None:
    draft = _make_draft()
    s = draft.get_section("introduction")
    assert s.name == "introduction"
    assert "introduction" in draft.sections


# ── OutlineBuilder ──────────────────────────────────────────────────────

def test_outline_builder_default() -> None:
    outline = OutlineBuilder().build(_make_goal(), _make_hypotheses())
    section_names = [s["name"] for s in outline.section_plan]
    for expected in ("related_work", "methods", "abstract"):
        assert expected in section_names


def test_outline_builder_has_citation_strategy() -> None:
    outline = OutlineBuilder().build(_make_goal(), _make_hypotheses())
    assert outline.citation_strategy
    for section in SECTION_ORDER:
        assert section in outline.citation_strategy


def test_outline_builder_figures() -> None:
    outline = OutlineBuilder().build(_make_goal(), _make_hypotheses(), conference_target="NeurIPS")
    assert len(outline.visualization_plan) >= 1
    assert outline.conference_target == "NeurIPS"


# ── SectionWriter ──────────────────────────────────────────────────────

def test_section_writer_placeholder_without_llm() -> None:
    draft = _make_draft()
    writer = SectionWriter()
    section = writer.draft_section(draft, "introduction", "hypotheses", "bibliography")
    assert "PLACEHOLDER" in section.content
    assert section.status == SectionStatus.DRAFTED


def test_section_writer_calls_llm() -> None:
    calls: list[str] = []

    def fake_llm(section_name: str, ctx: dict) -> str:  # type: ignore[return]
        calls.append(section_name)
        return f"Content for {section_name}"

    draft = _make_draft()
    writer = SectionWriter(llm_write=fake_llm)
    writer.draft_all(draft, "hypotheses summary", "bibliography summary")
    assert len(calls) == len(SECTION_ORDER)


# ── ReferenceManager ──────────────────────────────────────────────────

def test_reference_manager_deduplication() -> None:
    draft = _make_draft()
    mgr = ReferenceManager(draft)
    e1 = CitationEntry(paper_id="p1", cite_key="smith2024test", title="Test", year=2024)
    e2 = CitationEntry(paper_id="p1", cite_key="smith2024test", title="Test", year=2024)
    k1 = mgr.add(e1)
    k2 = mgr.add(e2)
    assert k1 == k2
    assert len(draft.references) == 1


def test_reference_manager_bibtex() -> None:
    draft = _make_draft()
    mgr = ReferenceManager(draft)
    mgr.add(CitationEntry(
        paper_id="p2",
        cite_key="jones2025attention",
        title="Attention Is All You Need",
        authors=["Vaswani A"],
        year=2017,
        arxiv_id="1706.03762",
    ))
    bib = mgr.to_bibtex()
    assert "jones2025attention" in bib
    assert "Attention Is All You Need" in bib


# ── PeerReviewSimulator ───────────────────────────────────────────────

def test_simulator_passes_high_scoring_sections() -> None:
    draft = _make_draft()
    section = draft.get_section("methods")
    section.bump_version("Detailed methods section.")

    sim = PeerReviewSimulator(acceptance_threshold=7.0)
    history = sim.refine(draft)
    assert "methods" in history
    assert history["methods"][0].mean >= 7.0
    assert draft.sections["methods"].status == SectionStatus.REFINED


def test_simulator_iterates_on_low_score() -> None:
    round_counter = [0]

    def reviewer(section: Section) -> dict:
        round_counter[0] += 1
        score = 5.0 if round_counter[0] < 3 else 9.0
        return {"clarity": score, "correctness": score, "novelty": score, "citation_quality": score}

    def rewriter(section: Section, score: SectionScore) -> str:
        return f"Improved v{section.version + 1}"

    draft = _make_draft()
    section = draft.get_section("introduction")
    section.bump_version("First draft.")

    sim = PeerReviewSimulator(
        llm_review=reviewer, llm_rewrite=rewriter,
        acceptance_threshold=7.0, max_rounds=5,
    )
    sim.refine(draft)
    assert round_counter[0] >= 3


# ── Exporter ────────────────────────────────────────────────────────────

def test_exporter_markdown_has_title() -> None:
    draft = _make_draft("My Paper")
    md = PaperExporter().to_markdown(draft)
    assert "# My Paper" in md


def test_exporter_latex_has_documentclass() -> None:
    draft = _make_draft()
    latex = PaperExporter().to_latex(draft)
    assert "\\documentclass" in latex


def test_exporter_markdown_includes_sections() -> None:
    draft = _make_draft("Paper With Sections")
    section = draft.get_section("methods")
    section.bump_version("We used neural networks.")
    md = PaperExporter().to_markdown(draft)
    assert "We used neural networks." in md
