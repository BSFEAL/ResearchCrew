from __future__ import annotations

from typing import Any

from crewai import Agent


def make_writer_agent(
    llm: Any | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Agent:
    """Drafts paper sections from hypotheses, citations, and the outline."""
    return Agent(
        role="Scientific Paper Writer",
        goal=(
            "Draft each paper section in dependency order (related work → methods → "
            "results → discussion → introduction → abstract → conclusion), integrating "
            "resolved citations and figures. Produce clear, publication-ready prose."
        ),
        backstory=(
            "You are an experienced scientific author who has written dozens of "
            "peer-reviewed papers. You write precisely, cite correctly, and follow "
            "venue style guides automatically."
        ),
        tools=tools or [],
        llm=llm,
        verbose=kwargs.get("verbose", False),
        max_iter=kwargs.get("max_iter", 35),
    )
