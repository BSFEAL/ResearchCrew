from __future__ import annotations

import os
import tempfile
from pathlib import Path

from researchcrew.memory.entry import MemoryEntry


class MemoryStore:
    """Append-only JSONL store for Retrospective Memory entries.

    Writes are atomic (tmp-file + os.replace) so a crash mid-write never
    produces a truncated or partially-written record.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: MemoryEntry) -> None:
        existing = self._read_raw()
        self._write_atomic(existing + entry.model_dump_json() + "\n")

    def load_all(self) -> list[MemoryEntry]:
        raw = self._read_raw()
        if not raw.strip():
            return []
        return [
            MemoryEntry.model_validate_json(line)
            for line in raw.splitlines()
            if line.strip()
        ]

    def load_by_project(self, project_id: str) -> list[MemoryEntry]:
        return [e for e in self.load_all() if e.project_id == project_id]

    def __len__(self) -> int:
        return len(self.load_all())

    def _read_raw(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def _write_atomic(self, content: str) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=".memstore_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
