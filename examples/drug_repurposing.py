"""Drug repurposing example: Co-Scientist crew on AML / liver fibrosis.

Requires:
    pip install researchcrew[llm]   # or [all]
    export OPENAI_API_KEY=...

Usage::

    python examples/drug_repurposing.py
"""
from __future__ import annotations

from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.project.lifecycle import ProjectLifecycle
from researchcrew.project.project import ResearchProject
from researchcrew.project.store import ProjectStore


RESEARCH_GOAL = (
    "Identify existing approved drugs that could be repurposed to treat "
    "acute myeloid leukaemia (AML) by targeting epigenetic regulators, "
    "focusing on mechanisms validated in liver fibrosis models."
)


def run() -> None:
    store = ProjectStore()
    project = store.create(
        ResearchProject(
            name="AML Drug Repurposing",
            domain="biomedicine",
            description=RESEARCH_GOAL,
            author="researchcrew-example",
        )
    )

    lc = ProjectLifecycle(project)
    lc.advance()  # IDEATION → LITERATURE_REVIEW
    lc.advance()  # LITERATURE_REVIEW → HYPOTHESIS_GENERATION
    lc.advance()  # HYPOTHESIS_GENERATION → TOURNAMENT
    store.save(project)

    crew = make_co_scientist_crew(verbose=True)
    result = crew.kickoff(
        inputs={
            "research_goal": RESEARCH_GOAL,
            "domain": "biomedicine",
            "max_hypotheses": "10",
            "tournament_rounds": "2",
        }
    )

    print("\n" + "=" * 60)
    print("Top hypotheses from Co-Scientist crew:")
    print("=" * 60)
    print(result)

    lc.advance()  # TOURNAMENT → WRITING
    store.save(project)
    print(f"\nProject status: {project.status.value}")
    print(f"Workspace: {project.workspace_dir}")


if __name__ == "__main__":
    run()
