from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: Literal["arxiv", "pubmed", "semantic_scholar", "openalex", "manual"] = "manual"
    doi: str | None = None
    arxiv_id: str | None = None
    pubmed_id: str | None = None
    full_text: str | None = None
    embedding: list[float] | None = None
    citation_count: int = 0
    references: list[str] = Field(default_factory=list)
    cited_by: list[str] = Field(default_factory=list)
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def canonical_id(self) -> str:
        """Stable external ID (DOI preferred, fallback to arxiv_id, then internal id)."""
        return self.doi or self.arxiv_id or self.pubmed_id or self.id


class SearchQuery(BaseModel):
    query: str
    sources: list[str]
    max_results: int
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result_count: int = 0


class LiteratureCorpus(BaseModel):
    project_id: str
    papers: dict[str, Paper] = Field(default_factory=dict)
    query_log: list[SearchQuery] = Field(default_factory=list)

    def add(self, paper: Paper) -> str:
        """Add paper, deduplicating by canonical_id. Returns the stored id."""
        for existing in self.papers.values():
            if existing.canonical_id == paper.canonical_id:
                return existing.id
        self.papers[paper.id] = paper
        return paper.id

    def get_by_canonical(self, cid: str) -> Paper | None:
        return next(
            (p for p in self.papers.values() if p.canonical_id == cid), None
        )

    @property
    def size(self) -> int:
        return len(self.papers)

    def all_papers(self) -> list[Paper]:
        return list(self.papers.values())
