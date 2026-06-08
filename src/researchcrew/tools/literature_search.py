from __future__ import annotations

from crewai.tools import BaseTool


class LiteratureSearchTool(BaseTool):
    """Search scientific literature across ArXiv, PubMed, and Semantic Scholar.

    Subclass and override ``_run`` to inject a live ``MultiSourceSearcher``.
    The default implementation returns a structured stub so agents can run
    end-to-end without API keys during development.
    """

    name: str = "Literature Search"
    description: str = (
        "Search scientific databases (ArXiv, PubMed, Semantic Scholar) for papers "
        "matching a query. Returns a JSON list of papers with title, abstract, year, "
        "and source."
    )

    def _run(  # type: ignore[override]
        self,
        query: str,
        sources: str = "arxiv,pubmed,semantic_scholar",
        max_results: int = 20,
        year_from: int | None = None,
    ) -> str:
        return (
            f'{{"query": "{query}", "sources": "{sources}", '
            f'"papers": [], "note": "stub — install researchcrew[literature] and wire MultiSourceSearcher"}}'
        )
