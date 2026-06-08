from __future__ import annotations

from typing import Any

from crewai import Crew, Process, Task

from researchcrew.agents.editor import make_editor_agent
from researchcrew.agents.literature import make_literature_agent
from researchcrew.agents.outline import make_outline_agent
from researchcrew.agents.writer import make_writer_agent
from researchcrew.tools.literature_search import LiteratureSearchTool
from researchcrew.tools.memory_query import MemoryQueryTool


def make_paper_writing_crew(
    llm: Any | None = None,
    verbose: bool = False,
    conference_target: str = "general audience",
) -> Crew:
    """Assemble a PaperOrchestra-style writing crew (4 agents, sequential).

    Run with::

        crew = make_paper_writing_crew(llm=my_llm, conference_target="NeurIPS")
        result = crew.kickoff(inputs={
            "research_goal": "...",
            "top_hypotheses": "...",  # JSON string from Co-Scientist crew
            "corpus_summary": "...",   # literature summary
        })
    """
    search_tool = LiteratureSearchTool()
    memory_tool = MemoryQueryTool()

    literature = make_literature_agent(llm=llm, tools=[search_tool], verbose=verbose)
    outline = make_outline_agent(llm=llm, tools=[memory_tool], verbose=verbose)
    writer = make_writer_agent(llm=llm, verbose=verbose)
    editor = make_editor_agent(llm=llm, verbose=verbose)

    tasks = [
        Task(
            description=(
                "Perform a two-phase citation search for the paper sections described in the "
                "research goal: {research_goal}. Phase 1: broad keyword search across databases. "
                "Phase 2: re-rank by relevance and verify claim-citation alignment."
            ),
            expected_output=(
                "A structured bibliography with per-section citation assignments "
                "(title, year, DOI, relevance note)."
            ),
            agent=literature,
        ),
        Task(
            description=(
                "Create a structured JSON paper outline for target venue: {conference_target}. "
                "Input: research goal, top hypotheses, and resolved citations. "
                "Output must include section_plan, visualization_plan, and citation_strategy."
            ),
            expected_output=(
                "A JSON outline with ordered sections, per-section scope, citation hints, "
                "and figure descriptions."
            ),
            agent=outline,
        ),
        Task(
            description=(
                "Draft all paper sections in dependency order: related_work → methods → results "
                "→ discussion → introduction → abstract → conclusion. "
                "Integrate citations from the bibliography and follow the outline's scope for each section."
            ),
            expected_output=(
                "A complete paper draft in Markdown with inline citation keys and "
                "figure placeholders."
            ),
            agent=writer,
        ),
        Task(
            description=(
                "Conduct a simulated peer review of the full draft. Score each section on "
                "clarity (0-10), correctness (0-10), novelty (0-10), and citation quality (0-10). "
                "Apply targeted rewrites to any section scoring below 7 in any dimension."
            ),
            expected_output=(
                "A refined paper draft with per-section scores and a review summary "
                "explaining every change made."
            ),
            agent=editor,
        ),
    ]

    return Crew(
        agents=[literature, outline, writer, editor],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
    )
