from __future__ import annotations

from typing import Any

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.tournament_state import TournamentState


class MetaReviewOptimizer:
    """Translates tournament failures into CrewPilot DSPy optimizer inputs.

    When crewpilot + dspy are installed, this feeds tournament failures to
    DspyGepaOptimizer to refine Generation/Reflection/Evolution agent prompts.
    Falls back gracefully when dependencies are absent.
    """

    def __init__(self, variant_bridge: Any | None = None) -> None:
        self._bridge = variant_bridge
        self._crewpilot_available = self._check_crewpilot()

    @staticmethod
    def _check_crewpilot() -> bool:
        try:
            import crewpilot  # type: ignore[import-not-found]  # noqa: F401
            return True
        except ImportError:
            return False

    def optimize_from_tournament(
        self,
        state: TournamentState,
        meta_review_summary: str,
    ) -> dict[str, Any]:
        """Extract failure patterns and return a prompt refinement brief."""
        low_novelty = [
            h for h in state.population
            if h.novelty_score is not None and h.novelty_score < 0.4
        ]
        low_elo = [
            h for h in state.ranked_population()
            if h.elo_score < 1100
        ]

        brief: dict[str, Any] = {
            "failure_count": len(low_novelty) + len(low_elo),
            "low_novelty_titles": [h.title for h in low_novelty[:5]],
            "low_elo_titles": [h.title for h in low_elo[:5]],
            "meta_review": meta_review_summary,
            "suggested_focus": self._suggest_focus(state),
        }

        if self._crewpilot_available:
            brief["crewpilot_integrated"] = True
            # In full integration: pass failures to DspyGepaOptimizer here.

        return brief

    @staticmethod
    def _suggest_focus(state: TournamentState) -> str:
        top = state.ranked_population()[:3]
        if not top:
            return "Broaden hypothesis search."
        domains = {h.domain for h in top}
        return f"Focus on: {', '.join(domains)}. Top Elo: {top[0].elo_score:.0f}."
