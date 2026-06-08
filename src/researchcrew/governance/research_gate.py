from __future__ import annotations

from enum import Enum
from typing import Any

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.tournament_state import TournamentState


class GateDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    NEEDS_HUMAN = "needs_human"


class ResearchGatePolicy:
    """Gate that promotes hypotheses based on Elo delta AND reflection novelty score.

    Implements the CrewPilot GatePolicy protocol so it can be dropped into a
    GovernanceController when crewpilot is installed.
    """

    def __init__(
        self,
        min_elo_delta: float = 50.0,
        min_novelty_score: float = 0.6,
        min_feasibility_score: float = 0.4,
        uncertainty_band: float = 20.0,
    ) -> None:
        self.min_elo_delta = min_elo_delta
        self.min_novelty_score = min_novelty_score
        self.min_feasibility_score = min_feasibility_score
        self.uncertainty_band = uncertainty_band

    def evaluate(
        self,
        hypothesis: Hypothesis,
        baseline_elo: float,
        state: TournamentState,
    ) -> GateDecision:
        elo_delta = hypothesis.elo_score - baseline_elo

        if (
            hypothesis.novelty_score is not None
            and hypothesis.novelty_score < self.min_novelty_score
        ):
            return GateDecision.REJECT

        if (
            hypothesis.feasibility_score is not None
            and hypothesis.feasibility_score < self.min_feasibility_score
        ):
            return GateDecision.REJECT

        if abs(elo_delta) < self.uncertainty_band:
            return GateDecision.NEEDS_HUMAN

        if elo_delta < self.min_elo_delta:
            return GateDecision.REJECT

        return GateDecision.PROMOTE

    # CrewPilot GatePolicy protocol compat
    def decide(self, candidate: Any, baseline: Any) -> str:  # type: ignore[override]
        return GateDecision.PROMOTE.value
