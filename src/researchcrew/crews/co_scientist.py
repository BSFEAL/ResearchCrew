from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task

from researchcrew.agents.evolution import make_evolution_agent
from researchcrew.agents.generation import make_generation_agent
from researchcrew.agents.meta_review import make_meta_review_agent
from researchcrew.agents.proximity import make_proximity_agent
from researchcrew.agents.ranking import make_ranking_agent
from researchcrew.agents.reflection import make_reflection_agent
from researchcrew.agents.supervisor import make_supervisor_agent
from researchcrew.tools.debate_tool import DebateTool
from researchcrew.tools.hypothesis_store import HypothesisStoreTool
from researchcrew.tools.literature_search import LiteratureSearchTool
from researchcrew.tools.memory_query import MemoryQueryTool


def make_co_scientist_crew(
    llm: Any | None = None,
    verbose: bool = False,
    max_tournament_rounds: int = 3,
) -> Crew:
    """Assemble a full Co-Scientist crew (7 agents, hierarchical process).

    Run with::

        crew = make_co_scientist_crew(llm=my_llm)
        result = crew.kickoff(inputs={"research_goal": "...", "domain": "biomedicine"})
    """
    search_tool = LiteratureSearchTool()
    store_tool = HypothesisStoreTool()
    debate_tool = DebateTool()
    memory_tool = MemoryQueryTool()

    generation = make_generation_agent(
        llm=llm, tools=[search_tool, store_tool], verbose=verbose
    )
    reflection = make_reflection_agent(llm=llm, tools=[store_tool], verbose=verbose)
    ranking = make_ranking_agent(llm=llm, tools=[debate_tool, store_tool], verbose=verbose)
    evolution = make_evolution_agent(
        llm=llm, tools=[store_tool, memory_tool], verbose=verbose
    )
    proximity = make_proximity_agent(llm=llm, tools=[store_tool], verbose=verbose)
    meta_review = make_meta_review_agent(llm=llm, tools=[memory_tool], verbose=verbose)
    supervisor = make_supervisor_agent(llm=llm, verbose=verbose)

    tasks = [
        Task(
            description=(
                "Search the scientific literature and generate an initial population of "
                "{max_hypotheses} diverse, testable hypotheses for the research goal: "
                "{research_goal}. Domain: {domain}."
            ),
            expected_output=(
                "A structured list of {max_hypotheses} hypothesis objects, each with "
                "title, body, domain, and novelty justification."
            ),
            agent=generation,
        ),
        Task(
            description=(
                "Review all generated hypotheses. For each, assign a novelty_score "
                "(0-1) and feasibility_score (0-1) and flag any that violate the "
                "research constraints."
            ),
            expected_output=(
                "Annotated hypothesis list with novelty_score, feasibility_score, and "
                "per-hypothesis critique."
            ),
            agent=reflection,
        ),
        Task(
            description=(
                "Cluster the hypothesis population to remove redundant ideas. "
                "Keep the best representative from each semantic cluster."
            ),
            expected_output=(
                "A de-duplicated hypothesis list with cluster assignments."
            ),
            agent=proximity,
        ),
        Task(
            description=(
                "Run {tournament_rounds} rounds of pairwise Elo debates on the "
                "de-duplicated population. After each round, eliminate the bottom "
                "30%% and record the updated leaderboard."
            ),
            expected_output=(
                "Ranked hypothesis leaderboard with Elo scores and debate records."
            ),
            agent=ranking,
        ),
        Task(
            description=(
                "Take the top 5 hypotheses from the tournament. For each, generate "
                "an improved variant that addresses the reflection feedback and "
                "incorporates the strongest elements of adjacent hypotheses."
            ),
            expected_output=(
                "5 evolved hypothesis variants ready to re-enter the tournament."
            ),
            agent=evolution,
        ),
        Task(
            description=(
                "Synthesise all tournament outcomes, reflection scores, and evolution "
                "results. Produce a refined research brief that describes what worked, "
                "what failed, and what the next generation should focus on."
            ),
            expected_output=(
                "A concise research brief (max 500 words) and a final ranked list of "
                "the top 10 hypotheses with scientific justifications."
            ),
            agent=meta_review,
        ),
    ]

    return Crew(
        agents=[generation, reflection, proximity, ranking, evolution, meta_review],
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=supervisor,
        verbose=verbose,
    )
