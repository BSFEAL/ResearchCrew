from __future__ import annotations

from typing import Any

from crewai import Agent


def make_editor_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Simulates peer review to iteratively improve draft sections."""
    return Agent(
        role="Peer Review Simulator",
        goal=(
            "Score each draft section on clarity, correctness, novelty, and citation "
            "quality using structured JSON feedback. Apply targeted rewrites until all "
            "scores exceed the acceptance threshold or max_rounds is reached."
        ),
        backstory=(
            "You are a rigorous academic editor who has reviewed for top journals for "
            "twenty years. Your feedback is specific, actionable, and always improves "
            "the paper without changing the authors' core message."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 30),
    )
