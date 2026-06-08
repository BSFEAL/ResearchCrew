from __future__ import annotations

from typing import Any

from crewai import Agent


def make_supervisor_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Research director that coordinates all specialist agents and decides convergence."""
    return Agent(
        role="Research Director",
        goal=(
            "Coordinate the full research pipeline: direct specialist agents, "
            "manage the hypothesis budget, monitor tournament convergence, and "
            "decide when the population has reached sufficient quality to proceed "
            "to paper writing."
        ),
        backstory=(
            "You are a senior principal investigator who has led major research "
            "programmes. You are decisive, strategic, and know when to push for "
            "more exploration versus when to converge on the best ideas."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", True),
        max_iter=kwargs.get("max_iter", 40),
    )
