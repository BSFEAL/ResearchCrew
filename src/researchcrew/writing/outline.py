from __future__ import annotations

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.research_goal import ResearchGoal
from researchcrew.writing.models import FigurePlan, Outline
from researchcrew.writing.section_writer import SECTION_ORDER


class OutlineBuilder:
    """Builds a structured paper Outline from research materials.

    When ``llm_outline`` is provided it is called with a prompt string and must
    return a dict conforming to the Outline schema.  Without it a sensible
    default structure is generated from the inputs.
    """

    def __init__(self, llm_outline: object | None = None) -> None:
        self._llm = llm_outline

    def build(
        self,
        research_goal: ResearchGoal,
        top_hypotheses: list[Hypothesis],
        conference_target: str | None = None,
    ) -> Outline:
        if self._llm is not None and callable(self._llm):
            prompt = self._build_prompt(research_goal, top_hypotheses, conference_target)
            raw = self._llm(prompt)  # type: ignore[call-arg]
            if isinstance(raw, dict):
                return Outline(**raw)

        return self._default_outline(research_goal, top_hypotheses, conference_target)

    @staticmethod
    def _default_outline(
        goal: ResearchGoal,
        hypotheses: list[Hypothesis],
        conference_target: str | None,
    ) -> Outline:
        section_plan = [
            {
                "name": s,
                "scope": f"Standard {s.replace('_', ' ')} for a paper on: {goal.title}",
                "citation_hints": [],
            }
            for s in SECTION_ORDER
        ]
        hyp_titles = ", ".join(h.title for h in hypotheses[:5])
        section_plan[0]["scope"] = f"Related work relevant to: {hyp_titles}"

        figures = [
            FigurePlan(
                caption=f"Figure 1: Overview of {goal.title}",
                description="High-level system or conceptual overview diagram",
                section="methods",
            )
        ]
        citation_strategy = {
            s: [f"{goal.title} {s.replace('_', ' ')}"]
            for s in SECTION_ORDER
        }
        return Outline(
            section_plan=section_plan,
            visualization_plan=figures,
            citation_strategy=citation_strategy,
            conference_target=conference_target,
        )

    @staticmethod
    def _build_prompt(
        goal: ResearchGoal,
        hypotheses: list[Hypothesis],
        conference_target: str | None,
    ) -> str:
        hyp_block = "\n".join(
            f"- [{h.elo_score:.0f} Elo] {h.title}: {h.body[:200]}"
            for h in hypotheses[:10]
        )
        return (
            f"Research goal: {goal.description}\n\n"
            f"Top hypotheses:\n{hyp_block}\n\n"
            f"Target venue: {conference_target or 'general audience'}\n\n"
            "Produce a JSON paper outline with keys: section_plan (list of "
            "{name, scope, citation_hints}), visualization_plan (list of "
            "{caption, description, section}), citation_strategy "
            "({section: [queries]}), conference_target."
        )
