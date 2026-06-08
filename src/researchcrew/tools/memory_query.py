from __future__ import annotations

from crewai.tools import BaseTool


class MemoryQueryTool(BaseTool):
    """Query the Retrospective Memory store for relevant past experiences.

    Subclass and override ``_run`` to inject a live ``MemoryStore`` with
    BM25+FAISS retrieval.  The default returns a structured stub.
    """

    name: str = "Memory Query"
    description: str = (
        "Search the Retrospective Memory for experiences relevant to a query. "
        "Returns the top-k matching MemoryEntry records as a JSON list."
    )

    def _run(  # type: ignore[override]
        self,
        query: str,
        project_id: str = "",
        top_k: int = 5,
    ) -> str:
        return (
            f'{{"query": "{query}", "project_id": "{project_id}", '
            f'"results": [], "note": "stub — install researchcrew[memory] and wire MemoryStore"}}'
        )
