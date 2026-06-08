from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class DebateTurn(BaseModel):
    speaker: Literal["hypothesis_a", "hypothesis_b", "judge"]
    content: str


class DebateVerdict(BaseModel):
    winner_id: str
    loser_id: str
    rationale: str
    confidence: float


class DebateRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_a_id: str
    hypothesis_b_id: str
    turns: list[DebateTurn] = Field(default_factory=list)
    verdict: DebateVerdict
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
