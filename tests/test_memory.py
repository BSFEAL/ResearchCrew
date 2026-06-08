from __future__ import annotations

from pathlib import Path

import pytest

from researchcrew.memory.entry import MemoryEntry, OutcomeLabel
from researchcrew.memory.store import MemoryStore


def _make_entry(
    project_id: str = "proj-1",
    outcome: OutcomeLabel = OutcomeLabel.SUCCESS,
) -> MemoryEntry:
    return MemoryEntry(
        project_id=project_id,
        plan="Test the novel approach using transformer architecture.",
        content="The hypothesis shows strong preliminary support.",
        outcome=outcome,
        domain="test",
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.jsonl")


def test_empty_store(store: MemoryStore) -> None:
    assert store.load_all() == []
    assert len(store) == 0


def test_append_and_load(store: MemoryStore) -> None:
    entry = _make_entry()
    store.append(entry)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].id == entry.id


def test_multiple_entries(store: MemoryStore) -> None:
    for i in range(5):
        store.append(_make_entry(project_id=f"proj-{i}"))
    assert len(store.load_all()) == 5
    assert len(store) == 5


def test_load_by_project(store: MemoryStore) -> None:
    store.append(_make_entry(project_id="alpha"))
    store.append(_make_entry(project_id="beta"))
    store.append(_make_entry(project_id="alpha"))
    result = store.load_by_project("alpha")
    assert len(result) == 2
    assert all(e.project_id == "alpha" for e in result)


def test_entry_fields_roundtrip(store: MemoryStore) -> None:
    entry = _make_entry(outcome=OutcomeLabel.FAILURE)
    store.append(entry)
    loaded = store.load_all()[0]
    assert loaded.id == entry.id
    assert loaded.outcome == OutcomeLabel.FAILURE
    assert loaded.project_id == entry.project_id
    assert loaded.created_at == entry.created_at
    assert loaded.plan == entry.plan


def test_append_is_cumulative(store: MemoryStore) -> None:
    for _ in range(3):
        store.append(_make_entry(project_id="same"))
    assert len(store.load_all()) == 3


def test_missing_project_returns_empty(store: MemoryStore) -> None:
    store.append(_make_entry(project_id="exists"))
    assert store.load_by_project("does-not-exist") == []
