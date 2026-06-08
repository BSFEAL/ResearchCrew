"""MLEvolve-style algorithm discovery on a Kaggle-style benchmark.

Demonstrates the Progressive MCGS loop with the Co-Scientist crew
acting as the expansion oracle.

Requires:
    pip install researchcrew[llm]
    export OPENAI_API_KEY=...

Usage::

    python examples/ml_algo_discovery.py
"""
from __future__ import annotations

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.models.tournament_state import TournamentState
from researchcrew.search.mcgs import ProgressiveMCGS
from researchcrew.search.scheduler import ExplorationScheduler
from researchcrew.tournament.elo import EloRating
from researchcrew.tournament.tournament import Tournament


ROOT_IDEAS = [
    Hypothesis(
        title="Gradient Boosting with Focal Loss",
        body="Adapt focal loss from object detection to tabular gradient boosting to address class imbalance.",
        domain="machine_learning",
        elo_score=1200.0,
    ),
    Hypothesis(
        title="TabPFN with Test-Time Augmentation",
        body="Apply test-time augmentation (feature permutations) to TabPFN to reduce variance on small datasets.",
        domain="machine_learning",
        elo_score=1200.0,
    ),
    Hypothesis(
        title="Stacked Ensemble with OOF Meta-features",
        body="Build a 3-layer stacking ensemble using out-of-fold predictions as meta-features for the blender.",
        domain="machine_learning",
        elo_score=1200.0,
    ),
]


def run(n_iterations: int = 5) -> None:
    scheduler = ExplorationScheduler(c_max=2.0, c_min=0.5, explore_until=2, decay_until=8)
    graph = ProgressiveMCGS(scheduler=scheduler)

    for idea in ROOT_IDEAS:
        graph.add_root(idea)

    state = TournamentState(project_id="ml-discovery")
    for node in graph.nodes.values():
        state.population.append(node.hypothesis)

    print(f"Starting MCGS with {graph.size} root nodes.")
    print(f"Initial population: {len(state.population)} hypotheses\n")

    for i in range(n_iterations):
        if graph.root_ids:
            node = graph.select(graph.root_ids[i % len(graph.root_ids)])
            simulated_score = node.hypothesis.elo_score / 1200.0 + (i * 0.05)
            graph.backpropagate(node.id, simulated_score)
            best_q = graph.best_q()
            c = scheduler.step(best_q)
            print(f"Iter {i+1}: node='{node.hypothesis.title[:40]}' score={simulated_score:.3f} C={c:.2f}")

    print("\nTop 3 nodes by Q-value:")
    for node in graph.best_nodes(3):
        print(f"  [{node.q_value:.3f}] {node.hypothesis.title}")


if __name__ == "__main__":
    run()
