from __future__ import annotations

import itertools
import random

from researchcrew.models.debate_record import DebateRecord
from researchcrew.models.hypothesis import Hypothesis, HypothesisStatus
from researchcrew.models.tournament_state import TournamentPhase, TournamentState
from researchcrew.tournament.debate import DebateJudge, PairwiseDebate
from researchcrew.tournament.elo import EloRating
from researchcrew.tournament.proximity import ClusterMap


class Tournament:
    """Orchestrates hypothesis population through Elo-rated pairwise debate rounds."""

    def __init__(
        self,
        state: TournamentState,
        judge: DebateJudge,
        elo: EloRating | None = None,
        seed: int | None = None,
    ) -> None:
        self.state = state
        self.debate = PairwiseDebate(judge=judge)
        self.elo = elo or EloRating()
        self._rng = random.Random(seed)

    def add_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        for h in hypotheses:
            h.status = HypothesisStatus.IN_TOURNAMENT
        self.state.population.extend(hypotheses)
        self.state.touch()

    def run_round(self, enforce_diversity: bool = True) -> list[DebateRecord]:
        """Debate all active pairs once, update Elo, and advance the generation counter."""
        active = [
            h for h in self.state.population
            if h.status == HypothesisStatus.IN_TOURNAMENT
        ]
        if len(active) < 2:
            return []

        pairs = self._select_pairs(active, enforce_diversity)
        records: list[DebateRecord] = []

        self.state.phase = TournamentPhase.ROUND_IN_PROGRESS
        for h_a, h_b in pairs:
            record = self.debate.run(h_a, h_b)
            # Update Elo before appending — match_count uses history, so counts stay consistent.
            self.elo.update(
                winner_id=record.verdict.winner_id,
                loser_id=record.verdict.loser_id,
                state=self.state,
            )
            self.state.debate_history.append(record)
            records.append(record)

        self.state.generation += 1
        self.state.phase = TournamentPhase.ROUND_COMPLETE
        self.state.touch()
        return records

    def eliminate_bottom(self, keep_fraction: float = 0.7) -> list[Hypothesis]:
        ranked = self.state.ranked_population()
        keep_n = max(2, int(len(ranked) * keep_fraction))
        eliminated = ranked[keep_n:]
        for h in eliminated:
            h.status = HypothesisStatus.ELIMINATED
        self.state.touch()
        return eliminated

    def top_k(self, k: int) -> list[Hypothesis]:
        return self.state.ranked_population()[:k]

    def is_converged(self, min_spread: float = 100.0) -> bool:
        ranked = self.state.ranked_population()
        if len(ranked) < 2:
            return True
        return (ranked[0].elo_score - ranked[-1].elo_score) >= min_spread

    def _select_pairs(
        self,
        active: list[Hypothesis],
        enforce_diversity: bool,
    ) -> list[tuple[Hypothesis, Hypothesis]]:
        candidates = active
        if enforce_diversity and len(active) > 3:
            cluster_map = ClusterMap.from_hypotheses(active)
            rep_ids = set(cluster_map.representative_ids())
            candidates = [h for h in active if h.id in rep_ids]
            if len(candidates) < 2:
                candidates = active

        pairs = list(itertools.combinations(candidates, 2))
        self._rng.shuffle(pairs)
        return pairs
