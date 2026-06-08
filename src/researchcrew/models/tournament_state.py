from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from researchcrew.models.debate_record import DebateRecord
from researchcrew.models.hypothesis import Hypothesis


class TournamentPhase(str, Enum):
    INITIALIZED = "initialized"
    ROUND_IN_PROGRESS = "round_in_progress"
    ROUND_COMPLETE = "round_complete"
    CONVERGED = "converged"


class TournamentState(BaseModel):
    project_id: str
    generation: int = 0
    phase: TournamentPhase = TournamentPhase.INITIALIZED
    population: list[Hypothesis] = Field(default_factory=list)
    debate_history: list[DebateRecord] = Field(default_factory=list)
    min_matches_per_hypothesis: int = 3
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        for h in self.population:
            if h.id == hypothesis_id:
                return h
        return None

    def ranked_population(self) -> list[Hypothesis]:
        return sorted(self.population, key=lambda h: h.elo_score, reverse=True)

    def match_count(self, hypothesis_id: str) -> int:
        return sum(
            1
            for d in self.debate_history
            if hypothesis_id in (d.hypothesis_a_id, d.hypothesis_b_id)
        )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
