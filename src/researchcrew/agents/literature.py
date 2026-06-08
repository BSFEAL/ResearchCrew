from __future__ import annotations

from typing import Any

from crewai import Agent


def make_literature_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Systematic literature reviewer that searches across ArXiv, PubMed, and Semantic Scholar."""
    return Agent(
        role="Systematic Literature Reviewer",
        goal=(
            "Search and critically analyse scientific literature across multiple databases. "
            "Screen papers for relevance, extract key findings, map the citation graph, "
            "and identify research gaps that hypothesis generation should target."
        ),
        backstory=(
            "You are an expert in systematic reviews with deep knowledge of academic "
            "databases. You apply rigorous inclusion/exclusion criteria, synthesise "
            "conflicting evidence, and produce structured summaries that fast-track "
            "hypothesis generation."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 30),
    )
