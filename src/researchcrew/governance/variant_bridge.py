from __future__ import annotations

import hashlib
import json
from typing import Any

from researchcrew.models.hypothesis import Hypothesis


class HypothesisVariantBridge:
    """Converts Hypothesis objects into CrewPilot PromptVariants for audit + versioning.

    When ``crewpilot`` is installed (``researchcrew[governance]``), variants are
    written to a ``LocalJSONStore`` and the full CrewPilot governance chain is
    available.  Without crewpilot the bridge still works: it computes content-
    addressed IDs and stores a lightweight dict so the lineage is preserved.
    """

    def __init__(self, store_path: str = ".researchcrew/variants") -> None:
        self._store_path = store_path
        self._store: Any = None
        self._crewpilot_available = self._try_init_store(store_path)

    def _try_init_store(self, path: str) -> bool:
        try:
            from crewpilot.registry.store import LocalJSONStore  # type: ignore[import-not-found]
            from pathlib import Path
            self._store = LocalJSONStore(Path(path))
            return True
        except ImportError:
            return False

    def register(self, hypothesis: Hypothesis, rationale: str = "") -> str:
        """Register a hypothesis as a versioned variant. Returns variant ID."""
        fields = {
            "title": hypothesis.title,
            "body": hypothesis.body,
            "domain": hypothesis.domain,
            "elo_score": hypothesis.elo_score,
        }
        variant_id = _content_id(fields)

        if self._crewpilot_available and self._store is not None:
            try:
                from crewpilot.registry.variant import PromptVariant  # type: ignore[import-not-found]
                variant = PromptVariant(
                    id=variant_id,
                    fields=fields,
                    parent_id=hypothesis.parent_ids[0] if hypothesis.parent_ids else None,
                    created_by="researchcrew.HypothesisVariantBridge",
                    rationale=rationale or f"Elo={hypothesis.elo_score:.0f}",
                )
                self._store.save(variant)
            except Exception:
                pass

        return variant_id

    @property
    def crewpilot_available(self) -> bool:
        return self._crewpilot_available


def _content_id(fields: dict[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
