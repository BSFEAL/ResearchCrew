from __future__ import annotations

from typing import Callable

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.search.mcgs import ProgressiveMCGS
from researchcrew.search.node import SearchNode


def fuse_branches(
    graph: ProgressiveMCGS,
    node_a_id: str,
    node_b_id: str,
    factory: Callable[[Hypothesis, Hypothesis], Hypothesis],
) -> SearchNode | None:
    """Merge two stagnating branches into a new node and cross-link both sources.

    ``factory`` receives the two source hypotheses and must return a new fused
    Hypothesis.  In production this is typically an Evolution agent call.
    """
    node_a = graph.nodes.get(node_a_id)
    node_b = graph.nodes.get(node_b_id)
    if node_a is None or node_b is None:
        return None

    fused_hyp = factory(node_a.hypothesis, node_b.hypothesis)
    parent_id = node_a_id if node_a.q_value >= node_b.q_value else node_b_id
    fused = graph.expand(parent_id, fused_hyp)

    graph.add_reference_edge(node_a_id, fused.id)
    graph.add_reference_edge(node_b_id, fused.id)
    return fused
