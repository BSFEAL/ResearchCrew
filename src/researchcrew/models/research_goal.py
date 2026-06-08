from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Domain(str, Enum):
    BIOMEDICINE = "biomedicine"
    CHEMISTRY = "chemistry"
    ML = "machine_learning"
    PHYSICS = "physics"
    GENERAL = "general"


class Constraint(BaseModel):
    description: str
    hard: bool = True


class ResearchGoal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    domain: Domain = Domain.GENERAL
    constraints: list[Constraint] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
