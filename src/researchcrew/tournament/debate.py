from __future__ import annotations

from typing import Protocol, runtime_checkable

from researchcrew.models.debate_record import DebateRecord, DebateTurn, DebateVerdict
from researchcrew.models.hypothesis import Hypothesis


@runtime_checkable
class DebateJudge(Protocol):
    """Anything that can adjudicate a pairwise hypothesis debate."""

    def judge(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        context: str,
    ) -> DebateVerdict: ...


class PairwiseDebate:
    """Structured 3-turn debate: A argues → B argues → judge decides."""

    def __init__(self, judge: DebateJudge) -> None:
        self.judge = judge

    def run(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        context: str = "",
    ) -> DebateRecord:
        turns: list[DebateTurn] = [
            DebateTurn(
                speaker="hypothesis_a",
                content=f"[{hypothesis_a.title}] {hypothesis_a.body}",
            ),
            DebateTurn(
                speaker="hypothesis_b",
                content=f"[{hypothesis_b.title}] {hypothesis_b.body}",
            ),
        ]
        verdict = self.judge.judge(hypothesis_a, hypothesis_b, context)
        turns.append(DebateTurn(speaker="judge", content=verdict.rationale))
        return DebateRecord(
            hypothesis_a_id=hypothesis_a.id,
            hypothesis_b_id=hypothesis_b.id,
            turns=turns,
            verdict=verdict,
        )
