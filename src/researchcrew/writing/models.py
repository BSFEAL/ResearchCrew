from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SectionStatus(str, Enum):
    PLANNED = "planned"
    DRAFTED = "drafted"
    REFINED = "refined"
    FINAL = "final"


class Section(BaseModel):
    name: str
    content: str = ""
    status: SectionStatus = SectionStatus.PLANNED
    version: int = 0
    citations_used: list[str] = Field(default_factory=list)
    agent_id: str = ""
    revision_history: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)

    def bump_version(self, new_content: str) -> None:
        self.revision_history.append(self.content)
        self.content = new_content
        self.version += 1
        self.status = SectionStatus.DRAFTED


class FigurePlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    caption: str
    description: str
    section: str
    generated: bool = False


class CitationEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str
    cite_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    journal: str | None = None


class Outline(BaseModel):
    section_plan: list[dict[str, Any]] = Field(default_factory=list)
    visualization_plan: list[FigurePlan] = Field(default_factory=list)
    citation_strategy: dict[str, list[str]] = Field(default_factory=dict)
    conference_target: str | None = None
    notes: str = ""


class PaperDraft(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    title: str = ""
    outline: Outline = Field(default_factory=Outline)
    sections: dict[str, Section] = Field(default_factory=dict)
    references: list[CitationEntry] = Field(default_factory=list)
    figures: list[FigurePlan] = Field(default_factory=list)
    status: SectionStatus = SectionStatus.PLANNED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_section(self, name: str) -> Section:
        if name not in self.sections:
            self.sections[name] = Section(name=name)
        return self.sections[name]

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
