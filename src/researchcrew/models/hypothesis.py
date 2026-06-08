from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    IN_TOURNAMENT = "in_tournament"
    PROMOTED = "promoted"
    ELIMINATED = "eliminated"
    EVOLVED = "evolved"


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    body: str
    domain: str
    elo_score: float = 1200.0
    status: HypothesisStatus = HypothesisStatus.DRAFT
    parent_ids: list[str] = Field(default_factory=list)
    generation: int = 0
    novelty_score: float | None = None
    feasibility_score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
