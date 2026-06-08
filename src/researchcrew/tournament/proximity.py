from __future__ import annotations

from dataclasses import dataclass, field

from researchcrew.models.hypothesis import Hypothesis


@dataclass
class ClusterMap:
    """Groups hypotheses by textual similarity to enforce population diversity."""

    clusters: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_hypotheses(
        cls,
        hypotheses: list[Hypothesis],
        similarity_threshold: float = 0.7,
        use_embeddings: bool = False,
    ) -> ClusterMap:
        if use_embeddings:
            return cls._cluster_by_embeddings(hypotheses, similarity_threshold)
        return cls._cluster_by_jaccard(hypotheses, similarity_threshold)

    @classmethod
    def _cluster_by_jaccard(
        cls,
        hypotheses: list[Hypothesis],
        threshold: float,
    ) -> ClusterMap:
        clusters: list[list[str]] = []
        h_words: dict[str, set[str]] = {
            h.id: set(h.body.lower().split()) for h in hypotheses
        }

        for h in hypotheses:
            placed = False
            for cluster in clusters:
                rep_id = cluster[0]
                rep_words = h_words[rep_id]
                mine = h_words[h.id]
                union = mine | rep_words
                if union and len(mine & rep_words) / len(union) >= threshold:
                    cluster.append(h.id)
                    placed = True
                    break
            if not placed:
                clusters.append([h.id])

        return cls(clusters=clusters)

    @classmethod
    def _cluster_by_embeddings(
        cls,
        hypotheses: list[Hypothesis],
        threshold: float,
    ) -> ClusterMap:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install researchcrew[memory] for embedding-based clustering."
            ) from e

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [h.body for h in hypotheses]
        embeddings = model.encode(texts, normalize_embeddings=True)

        assigned = [False] * len(hypotheses)
        clusters: list[list[str]] = []

        for i, h in enumerate(hypotheses):
            if assigned[i]:
                continue
            cluster = [h.id]
            assigned[i] = True
            for j in range(i + 1, len(hypotheses)):
                if assigned[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    cluster.append(hypotheses[j].id)
                    assigned[j] = True
            clusters.append(cluster)

        return cls(clusters=clusters)

    def representative_ids(self) -> list[str]:
        return [c[0] for c in self.clusters if c]

    def cluster_of(self, hypothesis_id: str) -> int | None:
        for idx, cluster in enumerate(self.clusters):
            if hypothesis_id in cluster:
                return idx
        return None

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)
