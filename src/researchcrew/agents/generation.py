from __future__ import annotations

from typing import Any

from crewai import Agent


def make_generation_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Agent that reads literature and generates novel hypotheses via simulated debate."""
    return Agent(
        role="Scientific Hypothesis Generator",
        goal=(
            "Explore the scientific literature and generate a diverse set of novel, "
            "testable hypotheses grounded in evidence. Use simulated scientific debate "
            "to stress-test each idea before proposing it."
        ),
        backstory=(
            "You are a creative research scientist with broad expertise across domains. "
            "You read voraciously, spot unexpected connections between findings, and "
            "propose ideas that are both original and experimentally feasible."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 25),
    )
