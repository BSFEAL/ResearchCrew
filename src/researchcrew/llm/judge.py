from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

from researchcrew.models.debate_record import DebateVerdict

if TYPE_CHECKING:
    from researchcrew.models.hypothesis import Hypothesis


class LLMDebateJudge:
    """Implements the DebateJudge Protocol using a complete() callable."""

    def __init__(self, complete: Callable[[str], str]) -> None:
        self._complete = complete

    def judge(
        self,
        hypothesis_a: "Hypothesis",
        hypothesis_b: "Hypothesis",
        context: str,
    ) -> DebateVerdict:
        prompt = (
            f"You are judging a scientific debate between two hypotheses.\n\n"
            f"Context: {context}\n\n"
            f"Hypothesis A (id={hypothesis_a.id}):\n{hypothesis_a.statement}\n\n"
            f"Hypothesis B (id={hypothesis_b.id}):\n{hypothesis_b.statement}\n\n"
            "Decide which hypothesis is better supported by evidence and reasoning. "
            "Return ONLY valid JSON with keys: winner_id (string), loser_id (string), "
            "rationale (string, max 200 words), confidence (float 0.0-1.0). "
            "winner_id and loser_id must be exactly the id values given above."
        )
        raw = self._complete(prompt)
        valid_ids = {hypothesis_a.id, hypothesis_b.id}
        try:
            data = json.loads(raw)
            winner_id = str(data.get("winner_id", hypothesis_a.id))
            loser_id = str(data.get("loser_id", hypothesis_b.id))
            if winner_id not in valid_ids:
                winner_id = hypothesis_a.id
            if loser_id not in valid_ids:
                loser_id = hypothesis_b.id
            if winner_id == loser_id:
                loser_id = hypothesis_b.id if winner_id == hypothesis_a.id else hypothesis_a.id
            return DebateVerdict(
                winner_id=winner_id,
                loser_id=loser_id,
                rationale=str(data.get("rationale", "No rationale provided.")),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.7)))),
            )
        except Exception:
            return DebateVerdict(
                winner_id=hypothesis_a.id,
                loser_id=hypothesis_b.id,
                rationale="Failed to parse LLM response — defaulting to hypothesis A.",
                confidence=0.5,
            )
