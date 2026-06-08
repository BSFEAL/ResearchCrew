from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from researchcrew.models.research_goal import ResearchGoal


class ProjectStatus(str, Enum):
    IDEATION = "ideation"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    TOURNAMENT = "tournament"
    WRITING = "writing"
    PEER_REVIEW = "peer_review"
    FINAL = "final"


class ProjectConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_hypotheses: int = 50
    min_hypotheses_to_write: int = 10
    min_elo_spread: float = 100.0
    tournament_rounds: int = 3
    max_search_results: int = 100
    paper_conference_target: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ResearchProject(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: str
    description: str = ""
    research_goals: list[ResearchGoal] = Field(default_factory=list)
    status: ProjectStatus = ProjectStatus.IDEATION
    author: str | None = None
    tags: list[str] = Field(default_factory=list)
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    workspace_dir: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
