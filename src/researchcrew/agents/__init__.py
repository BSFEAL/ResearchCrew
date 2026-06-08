from __future__ import annotations

from researchcrew.agents.editor import make_editor_agent
from researchcrew.agents.evolution import make_evolution_agent
from researchcrew.agents.generation import make_generation_agent
from researchcrew.agents.literature import make_literature_agent
from researchcrew.agents.meta_review import make_meta_review_agent
from researchcrew.agents.outline import make_outline_agent
from researchcrew.agents.proximity import make_proximity_agent
from researchcrew.agents.ranking import make_ranking_agent
from researchcrew.agents.reflection import make_reflection_agent
from researchcrew.agents.supervisor import make_supervisor_agent
from researchcrew.agents.writer import make_writer_agent


__all__ = [
    "make_editor_agent",
    "make_evolution_agent",
    "make_generation_agent",
    "make_literature_agent",
    "make_meta_review_agent",
    "make_outline_agent",
    "make_proximity_agent",
    "make_ranking_agent",
    "make_reflection_agent",
    "make_supervisor_agent",
    "make_writer_agent",
]
