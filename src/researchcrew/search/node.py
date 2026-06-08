from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from researchcrew.models.hypothesis import Hypothesis


@dataclass
class SearchNode:
    """A node in the Progressive MCGS graph.

    Extends a UCT tree node with cross-branch reference edges that allow
    value to flow between branches during backpropagation.
    """

    hypothesis: Hypothesis
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    depth: int = 0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.total_value += value
