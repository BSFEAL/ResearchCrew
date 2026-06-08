from __future__ import annotations

from typing import Any

from crewai import Agent


def make_evolution_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Refines top-ranked hypotheses and generates improved variants for the next round."""
    return Agent(
        role="Hypothesis Evolver",
        goal=(
            "Take the top-ranked hypotheses from the current tournament generation, "
            "synthesize their strengths, address their weaknesses, and produce "
            "refined variants that re-enter the tournament at the next generation."
        ),
        backstory=(
            "You are an inventive scientist who specialises in iterative idea refinement. "
            "You see how to combine the best aspects of competing hypotheses and push "
            "quality higher through targeted improvements."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 25),
    )
