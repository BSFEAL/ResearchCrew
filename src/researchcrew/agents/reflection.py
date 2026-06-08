from __future__ import annotations

from typing import Any

from crewai import Agent


def make_reflection_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Virtual peer reviewer that challenges hypotheses for correctness and novelty."""
    return Agent(
        role="Critical Peer Reviewer",
        goal=(
            "Rigorously evaluate each hypothesis for scientific correctness, "
            "genuine novelty, logical consistency, and experimental feasibility. "
            "Assign novelty and feasibility scores and provide actionable feedback."
        ),
        backstory=(
            "You are a strict but constructive academic reviewer with decades of "
            "experience in grant review panels. You have high standards, catch "
            "flawed reasoning quickly, and your feedback always improves the work."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 25),
    )
