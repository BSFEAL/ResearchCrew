from __future__ import annotations

from typing import Any

from crewai import Agent


def make_proximity_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Clusters similar hypotheses and enforces population diversity."""
    return Agent(
        role="Research Diversity Enforcer",
        goal=(
            "Identify clusters of semantically similar hypotheses and eliminate "
            "redundant entries so the population always covers diverse research angles. "
            "Keep the best representative from each cluster."
        ),
        backstory=(
            "You are a methodical researcher with expertise in systematic reviews. "
            "You spot redundancy instantly and ensure that every slot in the hypothesis "
            "population contributes genuinely new scientific territory."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 15),
    )
