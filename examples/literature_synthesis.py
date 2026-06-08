"""Systematic literature review + paper writing example.

Requires:
    pip install researchcrew[literature,llm]
    export OPENAI_API_KEY=...

Usage::

    python examples/literature_synthesis.py
"""
from __future__ import annotations

from researchcrew.crews.paper_writing import make_paper_writing_crew
from researchcrew.literature.corpus import LiteratureCorpus
from researchcrew.literature.searcher import MultiSourceSearcher
from researchcrew.models.research_goal import Domain, ResearchGoal
from researchcrew.project.project import ResearchProject
from researchcrew.project.store import ProjectStore
from researchcrew.writing.models import PaperDraft
from researchcrew.writing.outline import OutlineBuilder
from researchcrew.writing.exporter import PaperExporter


TOPIC = "transformer attention mechanisms for long-context document understanding"


def run() -> None:
    store = ProjectStore()
    project = store.create(
        ResearchProject(
            name="Attention Mechanisms Survey",
            domain="machine_learning",
            description=TOPIC,
        )
    )

    goal = ResearchGoal(
        title="Long-Context Attention Survey",
        description=TOPIC,
        domain=Domain.ML,
        keywords=["attention", "transformer", "long-context", "document"],
    )

    corpus = LiteratureCorpus(project_id=project.id)
    print(f"Searching literature for: {TOPIC}")
    try:
        searcher = MultiSourceSearcher()
        n = searcher.search(TOPIC, corpus, sources=["arxiv"], max_results=15)
        print(f"Found {n} papers from ArXiv.")
    except ImportError:
        print("httpx not installed — skipping live search. Install researchcrew[literature].")

    print(f"\nCorpus size: {corpus.size} papers")
    print("Building paper outline...")

    from researchcrew.models.hypothesis import Hypothesis
    placeholder_hypotheses = [
        Hypothesis(
            title="Sparse Attention Outperforms Full Attention for Long Documents",
            body="Sparse attention patterns reduce complexity from O(n²) to O(n log n) without accuracy loss.",
            domain="machine_learning",
            elo_score=1350.0,
        )
    ]

    outline = OutlineBuilder().build(goal, placeholder_hypotheses, conference_target="ACL")
    draft = PaperDraft(project_id=project.id, title=f"Survey: {TOPIC.title()}")
    draft.outline = outline

    print("\nOutline sections:", [s["name"] for s in outline.section_plan])
    print("\nExporting skeleton Markdown...")
    md = PaperExporter().to_markdown(draft)
    print(md[:800])

    print("\nRunning paper writing crew (requires LLM)...")
    crew = make_paper_writing_crew(verbose=False, conference_target="ACL")
    result = crew.kickoff(
        inputs={
            "research_goal": TOPIC,
            "top_hypotheses": "Sparse attention for long-context documents.",
            "corpus_summary": f"{corpus.size} papers retrieved.",
        }
    )
    print(result)


if __name__ == "__main__":
    run()
