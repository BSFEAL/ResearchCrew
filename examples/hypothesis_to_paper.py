"""Full end-to-end example: hypothesis tournament → paper draft → Markdown export.

This is the flagship ResearchCrew demo. It chains both crews:

  1. Co-Scientist Crew — 7 agents run an Elo tournament to surface the best
     hypotheses for a given research goal.
  2. Paper Writing Crew — 4 agents take the top hypotheses and produce a
     complete publication-ready paper draft.
  3. Export — the draft is written to a Markdown file you can open immediately.

Estimated runtime: 3–8 minutes depending on the LLM provider and rate limits.

Requirements::

    pip install 'researchcrew[llm]'
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GOOGLE_API_KEY

Usage::

    python examples/hypothesis_to_paper.py

    # Specify a different research topic
    python examples/hypothesis_to_paper.py --topic "CRISPR off-target effects in gene therapy"

    # Use a different LLM provider
    python examples/hypothesis_to_paper.py --provider openai

    # Enable live Colab experiments during hypothesis evolution
    python examples/hypothesis_to_paper.py --colab
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TOPIC = (
    "Sparse and efficient attention mechanisms for processing "
    "long-context documents with transformer models"
)
DEFAULT_DOMAIN = "machine_learning"
DEFAULT_VENUE  = "NeurIPS"


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def _section(text: str) -> None:
    print(f"\n── {text} {'─' * max(0, 65 - len(text))}")


# ── main flow ─────────────────────────────────────────────────────────────────

def run(
    topic: str = DEFAULT_TOPIC,
    domain: str = DEFAULT_DOMAIN,
    venue: str = DEFAULT_VENUE,
    provider: str = "anthropic",
    use_colab: bool = False,
    max_hypotheses: int = 8,
    tournament_rounds: int = 2,
    verbose: bool = False,
) -> Path:
    """Run the full hypothesis-to-paper pipeline and return the output Markdown path."""
    from researchcrew.crews.co_scientist import make_co_scientist_crew
    from researchcrew.crews.paper_writing import make_paper_writing_crew
    from researchcrew.llm import create_llm
    from researchcrew.project.lifecycle import ProjectLifecycle
    from researchcrew.project.project import ResearchProject
    from researchcrew.project.store import ProjectStore

    _banner("ResearchCrew — Hypothesis → Paper Pipeline")
    print(f"  Topic    : {textwrap.shorten(topic, 65)}")
    print(f"  Domain   : {domain}")
    print(f"  Venue    : {venue}")
    print(f"  Provider : {provider}")
    print(f"  Colab    : {'enabled' if use_colab else 'disabled'}")
    print(f"  Hypotheses / Rounds: {max_hypotheses} / {tournament_rounds}")

    # ── LLM setup ─────────────────────────────────────────────────────────────
    _section("Initialising LLM")
    llm = create_llm(provider)
    print(f"  {type(llm).__name__} ready.")

    # ── Project ───────────────────────────────────────────────────────────────
    _section("Creating project")
    store   = ProjectStore()
    project = store.create(
        ResearchProject(
            name=f"H→P: {topic[:50]}",
            domain=domain,
            description=topic,
        )
    )
    lc = ProjectLifecycle(project)
    lc.advance()   # IDEATION → LITERATURE_REVIEW
    lc.advance()   # LITERATURE_REVIEW → HYPOTHESIS_GENERATION
    lc.advance()   # HYPOTHESIS_GENERATION → TOURNAMENT
    store.save(project)
    print(f"  Project ID : {project.id}")
    print(f"  Status     : {project.status.value}")

    # ── Phase 1: Co-Scientist hypothesis tournament ────────────────────────────
    _banner("Phase 1 / 2 — Co-Scientist Hypothesis Tournament")
    print("  Agents: Generation · Reflection · Proximity · Ranking · Evolution · Meta-review")
    if use_colab:
        print("  Colab tools: ColabExecuteTool · ColabInstallTool · ColabRuntimeTool ENABLED")
    print()

    scientist_crew = make_co_scientist_crew(
        llm=llm,
        verbose=verbose,
        max_tournament_rounds=tournament_rounds,
        use_colab=use_colab,
    )
    tournament_result = scientist_crew.kickoff(
        inputs={
            "research_goal":     topic,
            "domain":            domain,
            "max_hypotheses":    str(max_hypotheses),
            "tournament_rounds": str(tournament_rounds),
        }
    )

    top_hypotheses_str = str(tournament_result)
    lc.advance()   # TOURNAMENT → WRITING
    store.save(project)

    _section("Tournament complete")
    preview = top_hypotheses_str[:400].replace("\n", " ")
    print(f"  {textwrap.shorten(preview, 400)}")
    print(f"  [full output: {len(top_hypotheses_str)} chars]")

    # ── Phase 2: Paper Writing ─────────────────────────────────────────────────
    _banner("Phase 2 / 2 — Paper Writing Crew")
    print("  Agents: Literature · Outline · Writer · Editor")
    print()

    writing_crew = make_paper_writing_crew(
        llm=llm,
        verbose=verbose,
        conference_target=venue,
    )
    paper_result = writing_crew.kickoff(
        inputs={
            "research_goal":   topic,
            "top_hypotheses":  top_hypotheses_str,
            "corpus_summary":  f"Literature retrieved for: {topic}",
        }
    )

    lc.advance()   # WRITING → SUBMITTED
    store.save(project)

    # ── Export ────────────────────────────────────────────────────────────────
    _section("Exporting paper draft")
    slug = "".join(c if c.isalnum() else "_" for c in topic[:40]).lower().strip("_")
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{slug}_{ts}.md"

    metadata = (
        f"---\n"
        f"title: \"{topic}\"\n"
        f"domain: {domain}\n"
        f"venue: {venue}\n"
        f"provider: {provider}\n"
        f"project_id: {project.id}\n"
        f"generated: {datetime.now().isoformat()}\n"
        f"---\n\n"
    )
    out_path.write_text(metadata + str(paper_result), encoding="utf-8")

    _banner("Done")
    print(f"  Project status : {project.status.value}")
    print(f"  Paper draft    : {out_path}")
    print(f"  Word count     : ~{len(str(paper_result).split()):,} words")
    print()

    return out_path


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResearchCrew end-to-end: hypothesis tournament → paper draft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Research topic / goal (default: sparse attention for long-context docs)",
    )
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help="Research domain (default: machine_learning)",
    )
    parser.add_argument(
        "--venue",
        default=DEFAULT_VENUE,
        help="Target conference/venue (default: NeurIPS)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai", "google"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Enable Google Colab experiments during hypothesis evolution",
    )
    parser.add_argument(
        "--hypotheses",
        type=int,
        default=8,
        metavar="N",
        help="Initial hypothesis population size (default: 8)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        metavar="N",
        help="Tournament rounds (default: 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show CrewAI agent verbose output",
    )
    args = parser.parse_args()

    out = run(
        topic=args.topic,
        domain=args.domain,
        venue=args.venue,
        provider=args.provider,
        use_colab=args.colab,
        max_hypotheses=args.hypotheses,
        tournament_rounds=args.rounds,
        verbose=args.verbose,
    )
    print(f"Paper saved to: {out}")


if __name__ == "__main__":
    main()
