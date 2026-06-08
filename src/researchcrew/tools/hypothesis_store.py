from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class _ReadInput(BaseModel):
    hypothesis_id: str = Field(description="UUID of the hypothesis to retrieve.")


class _WriteInput(BaseModel):
    title: str = Field(description="Short title for the hypothesis.")
    body: str = Field(description="Full hypothesis statement.")
    domain: str = Field(description="Research domain (e.g. biomedicine, ml).")
    parent_ids: list[str] = Field(default_factory=list, description="IDs of parent hypotheses.")


class HypothesisStoreTool(BaseTool):
    """Read and write Hypothesis objects from/to the active tournament state.

    In production, wire this to a ``TournamentState`` instance via subclassing
    or dependency injection.  The default implementation returns stubs so that
    agents can be tested without a live tournament.
    """

    name: str = "Hypothesis Store"
    description: str = (
        "Read an existing hypothesis by ID, or create a new one with title/body/domain. "
        "Returns hypothesis details as a JSON string."
    )

    def _run(self, action: str = "read", **kwargs: object) -> str:  # type: ignore[override]
        if action == "write":
            data = _WriteInput(**kwargs)  # type: ignore[arg-type]
            return (
                f'{{"status": "created", "title": "{data.title}", '
                f'"domain": "{data.domain}"}}'
            )
        hypothesis_id = str(kwargs.get("hypothesis_id", ""))
        return f'{{"id": "{hypothesis_id}", "status": "stub — wire to TournamentState"}}'
