from __future__ import annotations

from typing import Any

from crewai import Agent


def make_outline_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Converts raw research materials into a structured paper outline (JSON)."""
    return Agent(
        role="Paper Structure Architect",
        goal=(
            "Analyse the research goal, top hypotheses, and literature corpus to produce "
            "a structured JSON paper outline. The outline must include a section plan, "
            "a visualisation plan, per-section citation search strategies, and "
            "scope descriptions for each section."
        ),
        backstory=(
            "You have published in Nature, Science, and NeurIPS and know exactly what "
            "top-tier venues expect. You are exceptional at translating raw research "
            "findings into a compelling narrative arc before a single word is written."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 20),
    )
