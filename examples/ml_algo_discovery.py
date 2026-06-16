"""MLEvolve-style algorithm discovery using Progressive MCGS.

Seeds a Monte Carlo Graph Search with three root hypotheses for tabular
classification, then runs iterative UCB-guided exploration + backpropagation.
The ExplorationScheduler decays the UCB constant from C_max → C_min,
transitioning from exploration to exploitation automatically.

No LLM required for this example — scores are simulated to demonstrate
the MCGS mechanics. Plug in a real evaluator (e.g., via ColabExecuteTool)
for production use.

Requires::

    pip install researchcrew   # core only — no LLM needed

Usage::

    python examples/ml_algo_discovery.py
    python examples/ml_algo_discovery.py --iterations 15 --top 5
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


def run(n_iterations: int = 10, top_n: int = 3) -> None:
    print("ResearchCrew — Progressive MCGS Algorithm Discovery")
    print(f"Iterations: {n_iterations}  |  Show top: {top_n}\n")

    scheduler = ExplorationScheduler(c_max=2.0, c_min=0.5, explore_until=2, decay_until=8)
    graph     = ProgressiveMCGS(scheduler=scheduler)

    for idea in ROOT_IDEAS:
        graph.add_root(idea)

    state = TournamentState(project_id="ml-discovery")
    for node in graph.nodes.values():
        state.population.append(node.hypothesis)

    print(f"Seeded {graph.size} root nodes into the MCGS graph.")
    print(f"Initial population: {len(state.population)} hypotheses")
    print("─" * 65)
    print(f"{'Iter':>4}  {'Node (truncated)':<42}  {'Score':>6}  {'C':>5}")
    print("─" * 65)

    for i in range(n_iterations):
        if graph.root_ids:
            node = graph.select(graph.root_ids[i % len(graph.root_ids)])
            simulated_score = min(1.0, node.hypothesis.elo_score / 1200.0 + (i * 0.05))
            graph.backpropagate(node.id, simulated_score)
            best_q = graph.best_q()
            c      = scheduler.step(best_q)
            title  = node.hypothesis.title[:40]
            print(f"{i+1:>4}  {title:<42}  {simulated_score:.3f}  {c:.3f}")

    print("─" * 65)
    print(f"\nTop {top_n} nodes by Q-value:")
    for rank, node in enumerate(graph.best_nodes(top_n), 1):
        print(f"  {rank}. [{node.q_value:.4f}] {node.hypothesis.title}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Progressive MCGS algorithm discovery")
    parser.add_argument("--iterations", type=int, default=10, help="MCGS iterations")
    parser.add_argument("--top",        type=int, default=3,  help="Show top-N nodes")
    args = parser.parse_args()
    run(n_iterations=args.iterations, top_n=args.top)


if __name__ == "__main__":
    main()
