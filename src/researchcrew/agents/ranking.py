from __future__ import annotations

from typing import Any

from crewai import Agent


def make_ranking_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Runs the Elo hypothesis tournament via pairwise debate."""
    return Agent(
        role="Tournament Organizer and Judge",
        goal=(
            "Rank the hypothesis population by running structured pairwise debates. "
            "For each pair, adjudicate which hypothesis is more scientifically "
            "promising and update the Elo leaderboard accordingly."
        ),
        backstory=(
            "You are a seasoned scientific editor who has evaluated thousands of "
            "research proposals. You excel at comparative assessment and can clearly "
            "articulate why one research direction is stronger than another."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 30),
    )
