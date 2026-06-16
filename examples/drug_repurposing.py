"""Drug repurposing example: Co-Scientist crew on AML / liver fibrosis.

The Co-Scientist crew generates 10 hypotheses targeting epigenetic regulators
in AML, runs a 2-round Elo tournament, and produces a ranked leaderboard with
mechanistic justifications.

Requires::

    pip install 'researchcrew[llm]'
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GOOGLE_API_KEY

Usage::

    python examples/drug_repurposing.py
    python examples/drug_repurposing.py --provider openai --rounds 3
    python examples/drug_repurposing.py --colab   # validate top drugs in Colab
"""
from __future__ import annotations

import argparse

from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.llm import create_llm
from researchcrew.project.lifecycle import ProjectLifecycle
from researchcrew.project.project import ResearchProject
from researchcrew.project.store import ProjectStore


RESEARCH_GOAL = (
    "Identify existing approved drugs that could be repurposed to treat "
    "acute myeloid leukaemia (AML) by targeting epigenetic regulators, "
    "focusing on mechanisms validated in liver fibrosis models."
)


def run(provider: str = "anthropic", rounds: int = 2, use_colab: bool = False) -> None:
    print("ResearchCrew — Biomedical Drug Repurposing")
    print(f"Provider: {provider}  |  Rounds: {rounds}  |  Colab: {use_colab}\n")

    llm   = create_llm(provider)
    store = ProjectStore()
    project = store.create(
        ResearchProject(
            name="AML Drug Repurposing",
            domain="biomedicine",
            description=RESEARCH_GOAL,
        )
    )

    lc = ProjectLifecycle(project)
    lc.advance()  # IDEATION → LITERATURE_REVIEW
    lc.advance()  # LITERATURE_REVIEW → HYPOTHESIS_GENERATION
    lc.advance()  # HYPOTHESIS_GENERATION → TOURNAMENT
    store.save(project)

    crew = make_co_scientist_crew(llm=llm, verbose=True, use_colab=use_colab)
    result = crew.kickoff(
        inputs={
            "research_goal":     RESEARCH_GOAL,
            "domain":            "biomedicine",
            "max_hypotheses":    "10",
            "tournament_rounds": str(rounds),
        }
    )

    print("\n" + "=" * 65)
    print("Top hypotheses from Co-Scientist crew:")
    print("=" * 65)
    print(result)

    lc.advance()  # TOURNAMENT → WRITING
    store.save(project)
    print(f"\nProject status : {project.status.value}")
    print(f"Project ID     : {project.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AML drug repurposing via Co-Scientist crew")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google"])
    parser.add_argument("--rounds",   type=int, default=2, help="Tournament rounds")
    parser.add_argument("--colab",    action="store_true", help="Enable Colab experiment tools")
    args = parser.parse_args()
    run(provider=args.provider, rounds=args.rounds, use_colab=args.colab)


if __name__ == "__main__":
    main()
