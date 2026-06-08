from __future__ import annotations

from researchcrew.memory.entry import MemoryEntry
from researchcrew.memory.store import MemoryStore


class BM25Retriever:
    """Keyword BM25 retrieval over plan + content text."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def query(
        self, query: str, project_id: str = "", top_k: int = 5
    ) -> list[MemoryEntry]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "Install researchcrew[memory] for BM25 retrieval."
            ) from exc

        entries = (
            self.store.load_by_project(project_id)
            if project_id
            else self.store.load_all()
        )
        if not entries:
            return []

        corpus = [f"{e.plan} {e.content}".lower().split() for e in entries]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, entries), key=lambda x: x[0], reverse=True)
        return [e for _, e in ranked[:top_k]]


class FAISSRetriever:
    """Embedding cosine-similarity retrieval via FAISS inner-product index."""

    def __init__(
        self, store: MemoryStore, model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.store = store
        self.model_name = model_name
        self._model: object | None = None
        self._index: object | None = None
        self._indexed: list[MemoryEntry] = []

    def query(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        try:
            import faiss  # type: ignore[import-untyped]
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install researchcrew[memory] for FAISS retrieval."
            ) from exc

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        if self._index is None:
            entries = self.store.load_all()
            if not entries:
                return []
            texts = [f"{e.plan} {e.content}" for e in entries]
            vecs = self._model.encode(texts, normalize_embeddings=True).astype("float32")  # type: ignore[union-attr]
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            self._index = idx
            self._indexed = entries

        model = self._model
        import numpy as np
        q_vec = model.encode([query], normalize_embeddings=True).astype("float32")  # type: ignore[union-attr]
        k = min(top_k, len(self._indexed))
        _, idxs = self._index.search(q_vec, k)  # type: ignore[union-attr]
        return [self._indexed[i] for i in idxs[0] if 0 <= i < len(self._indexed)]


class EnsembleRetriever:
    """Reciprocal Rank Fusion of BM25 and FAISS results."""

    def __init__(
        self,
        store: MemoryStore,
        model_name: str = "all-MiniLM-L6-v2",
        k_rrf: int = 60,
    ) -> None:
        self.bm25 = BM25Retriever(store)
        self.faiss = FAISSRetriever(store, model_name)
        self.k_rrf = k_rrf

    def query(
        self,
        query: str,
        project_id: str = "",
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[MemoryEntry]:
        try:
            bm25_results = self.bm25.query(query, project_id=project_id, top_k=candidate_k)
        except ImportError:
            bm25_results = []

        try:
            faiss_results = self.faiss.query(query, top_k=candidate_k)
        except ImportError:
            faiss_results = []

        if not bm25_results and not faiss_results:
            return []

        scores: dict[str, float] = {}
        for rank, entry in enumerate(bm25_results):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        for rank, entry in enumerate(faiss_results):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)

        pool = {e.id: e for e in bm25_results + faiss_results}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [pool[eid] for eid, _ in ranked[:top_k] if eid in pool]
