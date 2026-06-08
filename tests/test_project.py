from __future__ import annotations

from pathlib import Path

import pytest

from researchcrew.project.lifecycle import LifecycleError, ProjectLifecycle
from researchcrew.project.project import ProjectStatus, ResearchProject
from researchcrew.project.store import ProjectStore


def _make_project(**kwargs: str) -> ResearchProject:
    return ResearchProject(
        name=kwargs.get("name", "Test Project"),
        domain=kwargs.get("domain", "test"),
        description=kwargs.get("description", "A test research project"),
    )


# ── Lifecycle ─────────────────────────────────────────────────────────────

def test_default_status() -> None:
    p = _make_project()
    assert p.status == ProjectStatus.IDEATION


def test_advance_from_ideation() -> None:
    p = _make_project()
    lc = ProjectLifecycle(p)
    assert lc.advance() == ProjectStatus.LITERATURE_REVIEW
    assert p.status == ProjectStatus.LITERATURE_REVIEW


def test_full_advance_sequence() -> None:
    p = _make_project()
    lc = ProjectLifecycle(p)
    expected = [
        ProjectStatus.LITERATURE_REVIEW,
        ProjectStatus.HYPOTHESIS_GENERATION,
        ProjectStatus.TOURNAMENT,
        ProjectStatus.WRITING,
        ProjectStatus.PEER_REVIEW,
        ProjectStatus.FINAL,
    ]
    for expected_status in expected:
        assert lc.advance() == expected_status
    assert p.status == ProjectStatus.FINAL


def test_advance_from_final_raises() -> None:
    p = _make_project()
    p.status = ProjectStatus.FINAL
    lc = ProjectLifecycle(p)
    with pytest.raises(LifecycleError, match="terminal"):
        lc.advance()


def test_can_advance_false_at_final() -> None:
    p = _make_project()
    p.status = ProjectStatus.FINAL
    assert ProjectLifecycle(p).can_advance() is False


def test_rollback_succeeds() -> None:
    p = _make_project()
    p.status = ProjectStatus.TOURNAMENT
    lc = ProjectLifecycle(p)
    lc.rollback(ProjectStatus.LITERATURE_REVIEW)
    assert p.status == ProjectStatus.LITERATURE_REVIEW


def test_rollback_to_same_or_later_raises() -> None:
    p = _make_project()
    p.status = ProjectStatus.TOURNAMENT
    lc = ProjectLifecycle(p)
    with pytest.raises(LifecycleError):
        lc.rollback(ProjectStatus.TOURNAMENT)
    with pytest.raises(LifecycleError):
        lc.rollback(ProjectStatus.WRITING)


def test_advance_updates_timestamp() -> None:
    p = _make_project()
    before = p.updated_at
    lc = ProjectLifecycle(p)
    lc.advance()
    assert p.updated_at >= before


# ── ProjectStore ──────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(base_dir=tmp_path / "projects")


def test_create_and_load(store: ProjectStore) -> None:
    p = _make_project(name="Created Project")
    created = store.create(p)
    loaded = store.load(created.id)
    assert loaded.id == created.id
    assert loaded.name == "Created Project"
    assert loaded.status == ProjectStatus.IDEATION


def test_workspace_dir_set_on_create(store: ProjectStore) -> None:
    p = store.create(_make_project())
    assert p.workspace_dir != ""
    assert Path(p.workspace_dir).exists()


def test_subdirs_created(store: ProjectStore) -> None:
    p = store.create(_make_project())
    workspace = Path(p.workspace_dir)
    for subdir in ("corpus", "hypotheses", "tournament", "memory", "search", "governance"):
        assert (workspace / subdir).exists(), f"Missing subdir: {subdir}"


def test_save_persists_status_change(store: ProjectStore) -> None:
    p = store.create(_make_project())
    p.status = ProjectStatus.LITERATURE_REVIEW
    store.save(p)
    assert store.load(p.id).status == ProjectStatus.LITERATURE_REVIEW


def test_list_projects(store: ProjectStore) -> None:
    for i in range(3):
        store.create(_make_project(name=f"Project {i}"))
    assert len(store.list_projects()) == 3


def test_delete_removes_project(store: ProjectStore) -> None:
    p = store.create(_make_project())
    store.delete(p.id)
    with pytest.raises(FileNotFoundError):
        store.load(p.id)


def test_load_nonexistent_raises(store: ProjectStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load("no-such-id")
