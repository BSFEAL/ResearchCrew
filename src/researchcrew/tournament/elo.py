from __future__ import annotations

import math

from researchcrew.models.tournament_state import TournamentState


DEFAULT_K: float = 32.0
PROVISIONAL_K: float = 48.0
PROVISIONAL_THRESHOLD: int = 5


class EloRating:
    """Elo rating engine for the hypothesis tournament.

    Call ``update`` *before* appending the resulting DebateRecord to
    ``state.debate_history`` so that match_count reflects prior matches only.
    """

    def __init__(
        self,
        default_k: float = DEFAULT_K,
        provisional_k: float = PROVISIONAL_K,
        provisional_threshold: int = PROVISIONAL_THRESHOLD,
    ) -> None:
        self.default_k = default_k
        self.provisional_k = provisional_k
        self.provisional_threshold = provisional_threshold

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def k_factor(self, match_count: int) -> float:
        return (
            self.provisional_k
            if match_count < self.provisional_threshold
            else self.default_k
        )

    def update(
        self,
        winner_id: str,
        loser_id: str,
        state: TournamentState,
    ) -> TournamentState:
        winner = state.get_hypothesis(winner_id)
        loser = state.get_hypothesis(loser_id)
        if winner is None or loser is None:
            raise ValueError(
                f"Hypothesis not found: {winner_id!r} or {loser_id!r}"
            )

        k_winner = self.k_factor(state.match_count(winner_id))
        k_loser = self.k_factor(state.match_count(loser_id))

        expected_winner = self.expected_score(winner.elo_score, loser.elo_score)
        expected_loser = 1.0 - expected_winner

        winner.elo_score = round(
            winner.elo_score + k_winner * (1.0 - expected_winner), 2
        )
        loser.elo_score = round(
            loser.elo_score + k_loser * (0.0 - expected_loser), 2
        )
        return state
