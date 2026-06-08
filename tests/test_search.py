from __future__ import annotations

import pytest

from researchcrew.models.hypothesis import Hypothesis
from researchcrew.search.fusion import fuse_branches
from researchcrew.search.mcgs import ProgressiveMCGS
from researchcrew.search.node import SearchNode
from researchcrew.search.scheduler import ExplorationScheduler


def _make_hypothesis(title: str, body: str = "test") -> Hypothesis:
    return Hypothesis(title=title, body=body, domain="test")


# ── ExplorationScheduler ───────────────────────────────────────────

def test_scheduler_starts_at_c_max() -> None:
    s = ExplorationScheduler(c_max=2.0, c_min=0.5, explore_until=5, decay_until=15)
    assert s.c(1) == 2.0
    assert s.c(5) == 2.0


def test_scheduler_ends_at_c_min() -> None:
    s = ExplorationScheduler(c_max=2.0, c_min=0.5, explore_until=5, decay_until=15)
    assert s.c(15) == 0.5
    assert s.c(100) == 0.5


def test_scheduler_linear_decay() -> None:
    s = ExplorationScheduler(c_max=2.0, c_min=0.0, explore_until=0, decay_until=10)
    mid = s.c(5)
    assert 0.0 < mid < 2.0


def test_scheduler_stagnation_resets_c() -> None:
    s = ExplorationScheduler(
        c_max=2.0, c_min=0.5, explore_until=0, decay_until=5, stagnation_window=3
    )
    for _ in range(10):
        s.step(1.0)  # constant Q → stagnation
    assert s.is_stagnating
    assert s.c(s.iteration) == 2.0


def test_scheduler_no_stagnation_when_improving() -> None:
    s = ExplorationScheduler(stagnation_window=3)
    for i in range(10):
        s.step(float(i))
    assert not s.is_stagnating


# ── ProgressiveMCGS ────────────────────────────────────────────

def test_add_root_registers_node() -> None:
    g = ProgressiveMCGS()
    node = g.add_root(_make_hypothesis("Root"))
    assert node.id in g.nodes
    assert node.id in g.root_ids
    assert node.depth == 0


def test_expand_links_parent_child() -> None:
    g = ProgressiveMCGS()
    root = g.add_root(_make_hypothesis("Root"))
    child = g.expand(root.id, _make_hypothesis("Child"))
    assert child.parent_id == root.id
    assert child.id in root.children_ids
    assert child.depth == 1


def test_backpropagate_updates_path() -> None:
    g = ProgressiveMCGS()
    root = g.add_root(_make_hypothesis("Root"))
    child = g.expand(root.id, _make_hypothesis("Child"))
    g.backpropagate(child.id, 1.0)
    assert child.visit_count == 1
    assert root.visit_count == 1


def test_reference_edge_receives_partial_credit() -> None:
    g = ProgressiveMCGS()
    a = g.add_root(_make_hypothesis("A"))
    b = g.add_root(_make_hypothesis("B"))
    g.add_reference_edge(a.id, b.id)
    # backprop through b: a should get partial credit
    g.backpropagate(b.id, 1.0)
    assert b.visit_count == 1
    # a is referenced BY b's path, but the edge goes a->b, so when b is updated
    # a's reference_ids contains b; but b isn't traversed via a's ancestry.
    # Let's instead check the direct reference from a to b:
    g.backpropagate(a.id, 1.0)  # a updates and pushes credit to b
    assert b.total_value > 1.0  # b received partial credit


def test_best_nodes_sorted() -> None:
    g = ProgressiveMCGS()
    for i in range(3):
        node = g.add_root(_make_hypothesis(f"H{i}"))
        g.backpropagate(node.id, float(i + 1))
    best = g.best_nodes(2)
    assert best[0].q_value >= best[1].q_value


def test_select_prefers_unvisited() -> None:
    g = ProgressiveMCGS()
    root = g.add_root(_make_hypothesis("Root"))
    g.backpropagate(root.id, 0.5)
    child = g.expand(root.id, _make_hypothesis("Child"))
    selected = g.select(root.id)
    assert selected.id == child.id  # unvisited child preferred


# ── Branch fusion ───────────────────────────────────────────────

def test_fuse_creates_node_and_links() -> None:
    g = ProgressiveMCGS()
    a = g.add_root(_make_hypothesis("A"))
    b = g.add_root(_make_hypothesis("B"))
    g.backpropagate(a.id, 1.0)
    g.backpropagate(b.id, 0.5)

    def factory(ha: Hypothesis, hb: Hypothesis) -> Hypothesis:
        return Hypothesis(title="A+B", body=f"{ha.body} + {hb.body}", domain="test")

    fused = fuse_branches(g, a.id, b.id, factory)
    assert fused is not None
    assert fused.id in g.nodes
    assert fused.id in g.nodes[a.id].reference_ids
    assert fused.id in g.nodes[b.id].reference_ids


def test_fuse_unknown_node_returns_none() -> None:
    g = ProgressiveMCGS()
    a = g.add_root(_make_hypothesis("A"))
    fused = fuse_branches(g, a.id, "nonexistent", lambda x, y: x)
    assert fused is None
