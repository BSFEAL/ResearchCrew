from __future__ import annotations

import pytest

from researchcrew.models.debate_record import DebateVerdict
from researchcrew.models.hypothesis import Hypothesis, HypothesisStatus
from researchcrew.models.tournament_state import TournamentPhase, TournamentState
from researchcrew.tournament.debate import DebateJudge, PairwiseDebate
from researchcrew.tournament.proximity import ClusterMap
from researchcrew.tournament.tournament import Tournament


# ── Fixtures / helpers ────────────────────────────────────────────────────

class AlwaysAWinsJudge:
    """Deterministic judge: hypothesis_a always wins."""

    def judge(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        context: str = "",
    ) -> DebateVerdict:
        return DebateVerdict(
            winner_id=hypothesis_a.id,
            loser_id=hypothesis_b.id,
            rationale="A always wins (test stub)",
            confidence=1.0,
        )


class HigherEloWinsJudge:
    """Deterministic judge: whichever hypothesis has higher Elo wins."""

    def judge(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        context: str = "",
    ) -> DebateVerdict:
        if hypothesis_a.elo_score >= hypothesis_b.elo_score:
            winner, loser = hypothesis_a, hypothesis_b
        else:
            winner, loser = hypothesis_b, hypothesis_a
        return DebateVerdict(
            winner_id=winner.id,
            loser_id=loser.id,
            rationale="Higher Elo wins (test stub)",
            confidence=0.9,
        )


def _make_hypotheses(n: int, domain: str = "test") -> list[Hypothesis]:
    return [
        Hypothesis(title=f"H{i}", body=f"This hypothesis proposes mechanism {i} for disease X", domain=domain)
        for i in range(n)
    ]


def _make_state(n: int = 4) -> TournamentState:
    return TournamentState(project_id="test-project", population=_make_hypotheses(n))


# ── DebateJudge protocol ──────────────────────────────────────────────────

def test_judge_protocol_satisfied() -> None:
    assert isinstance(AlwaysAWinsJudge(), DebateJudge)


# ── PairwiseDebate ────────────────────────────────────────────────────────

def test_debate_produces_record() -> None:
    debate = PairwiseDebate(judge=AlwaysAWinsJudge())
    h_a, h_b = _make_hypotheses(2)
    record = debate.run(h_a, h_b)
    assert record.verdict.winner_id == h_a.id
    assert record.verdict.loser_id == h_b.id
    assert len(record.turns) == 3  # a argues, b argues, judge decides


def test_debate_turns_have_correct_speakers() -> None:
    debate = PairwiseDebate(judge=AlwaysAWinsJudge())
    h_a, h_b = _make_hypotheses(2)
    record = debate.run(h_a, h_b)
    speakers = [t.speaker for t in record.turns]
    assert speakers == ["hypothesis_a", "hypothesis_b", "judge"]


def test_debate_sets_hypothesis_ids() -> None:
    debate = PairwiseDebate(judge=AlwaysAWinsJudge())
    h_a, h_b = _make_hypotheses(2)
    record = debate.run(h_a, h_b)
    assert record.hypothesis_a_id == h_a.id
    assert record.hypothesis_b_id == h_b.id


# ── ClusterMap ────────────────────────────────────────────────────────────

def test_cluster_all_unique() -> None:
    hypotheses = [
        Hypothesis(title="A", body="alpha beta gamma", domain="test"),
        Hypothesis(title="B", body="delta epsilon zeta", domain="test"),
        Hypothesis(title="C", body="eta theta iota", domain="test"),
    ]
    cm = ClusterMap.from_hypotheses(hypotheses, similarity_threshold=0.5)
    assert cm.num_clusters == 3


def test_cluster_identical_bodies() -> None:
    hypotheses = [
        Hypothesis(title="A", body="exact same words here", domain="test"),
        Hypothesis(title="B", body="exact same words here", domain="test"),
        Hypothesis(title="C", body="completely different topic", domain="test"),
    ]
    cm = ClusterMap.from_hypotheses(hypotheses, similarity_threshold=0.9)
    assert cm.num_clusters == 2


def test_representative_ids_one_per_cluster() -> None:
    hypotheses = _make_hypotheses(4)
    cm = ClusterMap.from_hypotheses(hypotheses, similarity_threshold=0.99)
    reps = cm.representative_ids()
    assert len(reps) == cm.num_clusters
    assert len(set(reps)) == len(reps)  # all unique


def test_cluster_of_returns_index() -> None:
    hypotheses = _make_hypotheses(3)
    cm = ClusterMap.from_hypotheses(hypotheses, similarity_threshold=0.99)
    for h in hypotheses:
        idx = cm.cluster_of(h.id)
        assert idx is not None


# ── Tournament ────────────────────────────────────────────────────────────

def test_add_hypotheses_marks_in_tournament() -> None:
    state = TournamentState(project_id="proj")
    t = Tournament(state=state, judge=AlwaysAWinsJudge(), seed=42)
    hypotheses = _make_hypotheses(3)
    t.add_hypotheses(hypotheses)
    assert all(h.status == HypothesisStatus.IN_TOURNAMENT for h in state.population)


def test_run_round_produces_records() -> None:
    state = _make_state(4)
    for h in state.population:
        h.status = HypothesisStatus.IN_TOURNAMENT
    t = Tournament(state=state, judge=AlwaysAWinsJudge(), seed=0)
    records = t.run_round(enforce_diversity=False)
    assert len(records) > 0
    assert state.generation == 1
    assert state.phase == TournamentPhase.ROUND_COMPLETE


def test_run_round_updates_debate_history() -> None:
    state = _make_state(4)
    for h in state.population:
        h.status = HypothesisStatus.IN_TOURNAMENT
    t = Tournament(state=state, judge=AlwaysAWinsJudge(), seed=0)
    records = t.run_round(enforce_diversity=False)
    assert len(state.debate_history) == len(records)


def test_eliminate_bottom_marks_eliminated() -> None:
    state = _make_state(6)
    for h in state.population:
        h.status = HypothesisStatus.IN_TOURNAMENT
    t = Tournament(state=state, judge=HigherEloWinsJudge(), seed=0)
    t.run_round(enforce_diversity=False)
    eliminated = t.eliminate_bottom(keep_fraction=0.5)
    assert all(h.status == HypothesisStatus.ELIMINATED for h in eliminated)
    assert len(eliminated) == 3


def test_top_k_returns_sorted() -> None:
    state = _make_state(5)
    state.population[0].elo_score = 1500.0
    state.population[1].elo_score = 1400.0
    state.population[2].elo_score = 1300.0
    top = Tournament(
        state=state, judge=AlwaysAWinsJudge()
    ).top_k(3)
    assert top[0].elo_score == 1500.0
    assert len(top) == 3


def test_insufficient_active_returns_empty() -> None:
    state = _make_state(1)
    state.population[0].status = HypothesisStatus.IN_TOURNAMENT
    t = Tournament(state=state, judge=AlwaysAWinsJudge())
    assert t.run_round() == []
