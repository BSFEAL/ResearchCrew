from __future__ import annotations

import json
from pathlib import Path

from researchcrew.memory.entry import MemoryEntry, OutcomeLabel
from researchcrew.memory.store import MemoryStore


_KB_DIR = Path(__file__).parent / "knowledge_bases"


class DomainKB:
    """Cold-start domain knowledge base for Retrospective Memory seeding.

    Loads static JSONL seed files (one per domain) into a MemoryStore so
    that retrieval works from the very first iteration of a new project.
    """

    SUPPORTED_DOMAINS = (
        "biomedicine",
        "machine_learning",
        "chemistry",
        "physics",
        "general",
    )

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def seed(self, domain: str, project_id: str) -> int:
        """Seed from the built-in KB for ``domain``. Returns entries added."""
        key = domain.lower().replace(" ", "_").replace("-", "_")
        path = _KB_DIR / f"{key}.jsonl"
        if not path.exists():
            return 0
        return self._load_jsonl(path, project_id, key)

    def seed_from_file(self, path: Path, project_id: str, domain: str) -> int:
        """Seed from a user-supplied JSONL file."""
        return self._load_jsonl(path, project_id, domain)

    def _load_jsonl(self, path: Path, project_id: str, domain: str) -> int:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            self.store.append(
                MemoryEntry(
                    project_id=project_id,
                    plan=data.get("plan", "domain knowledge"),
                    content=data.get("content", ""),
                    outcome=OutcomeLabel(data.get("outcome", "success")),
                    domain=domain,
                    tags=data.get("tags", []),
                )
            )
            count += 1
        return count
