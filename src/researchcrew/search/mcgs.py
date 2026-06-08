from __future__ import annotations

import math

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.search.node import SearchNode
from researchcrew.search.scheduler import ExplorationScheduler


class ProgressiveMCGS:
    """Monte Carlo Graph Search for hypothesis space exploration.

    Extends UCT tree search with:
    - Cross-branch reference edges for knowledge transfer
    - Entropy-inspired ExplorationScheduler
    - Partial-credit backpropagation along reference edges
    """

    _REF_CREDIT: float = 0.3

    def __init__(self, scheduler: ExplorationScheduler | None = None) -> None:
        self.nodes: dict[str, SearchNode] = {}
        self.root_ids: list[str] = []
        self.scheduler = scheduler or ExplorationScheduler()

    # ── Graph construction ──────────────────────────────────────────────

    def add_root(self, hypothesis: Hypothesis) -> SearchNode:
        node = SearchNode(hypothesis=hypothesis, depth=0)
        self.nodes[node.id] = node
        self.root_ids.append(node.id)
        return node

    def expand(self, parent_id: str, child_hypothesis: Hypothesis) -> SearchNode:
        parent = self.nodes[parent_id]
        child = SearchNode(
            hypothesis=child_hypothesis,
            parent_id=parent_id,
            depth=parent.depth + 1,
        )
        self.nodes[child.id] = child
        parent.children_ids.append(child.id)
        return child

    def add_reference_edge(self, from_id: str, to_id: str) -> None:
        """Cross-branch edge: value at ``to_id`` partially credits ``from_id``."""
        if from_id in self.nodes and to_id in self.nodes:
            if to_id not in self.nodes[from_id].reference_ids:
                self.nodes[from_id].reference_ids.append(to_id)

    # ── MCTS operators ────────────────────────────────────────────────

    def select(self, root_id: str) -> SearchNode:
        """UCT traversal from root to a leaf or unvisited child."""
        node = self.nodes[root_id]
        c = self.scheduler.c(self.scheduler.iteration)
        while node.children_ids:
            children = [self.nodes[cid] for cid in node.children_ids]
            unvisited = [n for n in children if n.visit_count == 0]
            if unvisited:
                return unvisited[0]
            node = max(children, key=lambda n: self._uct(n, c))
        return node

    def backpropagate(self, node_id: str, value: float) -> None:
        """Update path to root and push partial credit along reference edges."""
        node: SearchNode | None = self.nodes.get(node_id)
        while node is not None:
            node.update(value)
            for ref_id in node.reference_ids:
                ref = self.nodes.get(ref_id)
                if ref is not None:
                    ref.update(value * self._REF_CREDIT)
            node = self.nodes.get(node.parent_id or "")

    # ── Queries ───────────────────────────────────────────────────────

    def best_nodes(self, k: int = 5) -> list[SearchNode]:
        visited = [n for n in self.nodes.values() if n.visit_count > 0]
        return sorted(visited, key=lambda n: n.q_value, reverse=True)[:k]

    def best_q(self) -> float:
        nodes = [n for n in self.nodes.values() if n.visit_count > 0]
        return max((n.q_value for n in nodes), default=0.0)

    @property
    def size(self) -> int:
        return len(self.nodes)

    # ── Internal ───────────────────────────────────────────────────────

    def _uct(self, node: SearchNode, c: float) -> float:
        parent = self.nodes.get(node.parent_id or "")
        n_parent = parent.visit_count if parent and parent.visit_count > 0 else 1
        return node.q_value + c * math.sqrt(math.log(n_parent) / node.visit_count)
