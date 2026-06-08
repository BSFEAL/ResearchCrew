from __future__ import annotations

import pytest

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.tournament_state import TournamentState
from researchcrew.tournament.elo import EloRating


def _make_state(n: int = 2) -> TournamentState:
    population = [
        Hypothesis(title=f"H{i}", body=f"Body of hypothesis {i}", domain="test")
        for i in range(n)
    ]
    return TournamentState(project_id="test-project", population=population)


def test_expected_score_equal_ratings() -> None:
    elo = EloRating()
    assert abs(elo.expected_score(1200, 1200) - 0.5) < 1e-9


def test_expected_score_stronger_player() -> None:
    elo = EloRating()
    assert elo.expected_score(1400, 1200) > 0.5
    assert elo.expected_score(1200, 1400) < 0.5


def test_expected_scores_sum_to_one() -> None:
    elo = EloRating()
    ea = elo.expected_score(1300, 1100)
    eb = elo.expected_score(1100, 1300)
    assert abs(ea + eb - 1.0) < 1e-9


def test_update_winner_gains_loser_loses() -> None:
    elo = EloRating()
    state = _make_state(2)
    a_id = state.population[0].id
    b_id = state.population[1].id
    before_a = state.population[0].elo_score
    before_b = state.population[1].elo_score

    elo.update(winner_id=a_id, loser_id=b_id, state=state)

    assert state.population[0].elo_score > before_a
    assert state.population[1].elo_score < before_b


def test_update_zero_sum_equal_k() -> None:
    """When both players share the same K-factor, Elo is zero-sum."""
    elo = EloRating()
    state = _make_state(2)
    total_before = sum(h.elo_score for h in state.population)
    elo.update(state.population[0].id, state.population[1].id, state)
    total_after = sum(h.elo_score for h in state.population)
    assert abs(total_after - total_before) < 1e-3


def test_update_unknown_id_raises() -> None:
    elo = EloRating()
    state = _make_state(2)
    with pytest.raises(ValueError, match="not found"):
        elo.update("no-such-id", state.population[0].id, state)


def test_k_factor_provisional() -> None:
    elo = EloRating(provisional_k=48.0, default_k=32.0, provisional_threshold=5)
    assert elo.k_factor(0) == 48.0
    assert elo.k_factor(4) == 48.0
    assert elo.k_factor(5) == 32.0
    assert elo.k_factor(10) == 32.0


def test_ranked_population_descending() -> None:
    state = _make_state(3)
    state.population[0].elo_score = 1100.0
    state.population[1].elo_score = 1300.0
    state.population[2].elo_score = 1200.0
    ranked = state.ranked_population()
    assert ranked[0].elo_score == 1300.0
    assert ranked[1].elo_score == 1200.0
    assert ranked[2].elo_score == 1100.0


def test_match_count_increments() -> None:
    state = _make_state(2)
    from researchcrew.models.debate_record import DebateRecord, DebateVerdict
    a_id = state.population[0].id
    b_id = state.population[1].id
    verdict = DebateVerdict(winner_id=a_id, loser_id=b_id, rationale="A was better", confidence=0.8)
    record = DebateRecord(hypothesis_a_id=a_id, hypothesis_b_id=b_id, verdict=verdict)
    state.debate_history.append(record)
    assert state.match_count(a_id) == 1
    assert state.match_count(b_id) == 1
