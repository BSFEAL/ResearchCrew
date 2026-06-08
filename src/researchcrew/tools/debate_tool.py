from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import Field


class DebateTool(BaseTool):
    """Run a structured pairwise debate between two hypotheses.

    In production, inject a ``DebateJudge`` by subclassing and overriding
    ``_run``.  The default implementation returns a structured stub.
    """

    name: str = "Pairwise Debate"
    description: str = (
        "Run a 3-turn structured debate between hypothesis_a_id and hypothesis_b_id. "
        "Returns a JSON verdict with winner_id, loser_id, rationale, and confidence."
    )

    def _run(  # type: ignore[override]
        self,
        hypothesis_a_id: str,
        hypothesis_b_id: str,
        context: str = "",
    ) -> str:
        return (
            f'{{"winner_id": "{hypothesis_a_id}", '
            f'"loser_id": "{hypothesis_b_id}", '
            f'"rationale": "stub — wire to PairwiseDebate", '
            f'"confidence": 0.5}}'
        )
