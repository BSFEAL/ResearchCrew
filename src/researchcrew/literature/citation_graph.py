from __future__ import annotations

from dataclasses import dataclass, field

from researchcrew.literature.corpus import LiteratureCorpus, Paper


@dataclass
class CitationGraph:
    """Directed graph of paper citations stored as adjacency lists.

    Edges: ``references`` (this paper cites) and ``cited_by`` (cites this paper).
    """

    _out: dict[str, set[str]] = field(default_factory=dict)  # paper → papers it cites
    _in: dict[str, set[str]] = field(default_factory=dict)   # paper → papers that cite it

    @classmethod
    def from_corpus(cls, corpus: LiteratureCorpus) -> CitationGraph:
        g = cls()
        for paper in corpus.all_papers():
            for ref_id in paper.references:
                g.add_edge(paper.id, ref_id)
            for citer_id in paper.cited_by:
                g.add_edge(citer_id, paper.id)
        return g

    def add_edge(self, from_id: str, to_id: str) -> None:
        self._out.setdefault(from_id, set()).add(to_id)
        self._in.setdefault(to_id, set()).add(from_id)

    def references(self, paper_id: str) -> set[str]:
        return self._out.get(paper_id, set())

    def cited_by(self, paper_id: str) -> set[str]:
        return self._in.get(paper_id, set())

    def expand_forward(
        self, paper_id: str, depth: int = 1
    ) -> set[str]:
        """Papers cited by descendants of ``paper_id`` up to ``depth`` hops."""
        frontier = {paper_id}
        visited: set[str] = set()
        for _ in range(depth):
            next_frontier: set[str] = set()
            for pid in frontier:
                for cid in self._in.get(pid, set()):
                    if cid not in visited:
                        next_frontier.add(cid)
            visited |= frontier
            frontier = next_frontier - visited
        return visited - {paper_id}

    def expand_backward(
        self, paper_id: str, depth: int = 1
    ) -> set[str]:
        """Papers that cite ancestors of ``paper_id`` up to ``depth`` hops."""
        frontier = {paper_id}
        visited: set[str] = set()
        for _ in range(depth):
            next_frontier: set[str] = set()
            for pid in frontier:
                for ref in self._out.get(pid, set()):
                    if ref not in visited:
                        next_frontier.add(ref)
            visited |= frontier
            frontier = next_frontier - visited
        return visited - {paper_id}

    def hub_ids(self, top_k: int = 10) -> list[str]:
        """Most-cited papers (highest in-degree)."""
        in_deg = {pid: len(citers) for pid, citers in self._in.items()}
        return sorted(in_deg, key=lambda x: in_deg[x], reverse=True)[:top_k]

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._out.values())
