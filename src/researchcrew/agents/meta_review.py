from __future__ import annotations

from typing import Any

from crewai import Agent


def make_meta_review_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Synthesises tournament feedback and refines agent prompts for the next cycle."""
    return Agent(
        role="Research Meta-Analyst",
        goal=(
            "Review all tournament outcomes and reflection feedback from the current "
            "cycle. Identify systematic weaknesses in the hypothesis generation strategy "
            "and produce a refined research brief for the next generation."
        ),
        backstory=(
            "You are a science administrator and methodologist who studies how research "
            "teams improve over time. You find patterns in what worked and what failed, "
            "and translate those patterns into actionable process improvements."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 20),
    )
