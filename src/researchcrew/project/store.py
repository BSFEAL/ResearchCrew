from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from researchcrew.project.project import ResearchProject


_DEFAULT_BASE = Path(".researchcrew") / "projects"

_SUBDIRS: tuple[str, ...] = (
    "corpus",
    "hypotheses",
    "tournament",
    "memory",
    "search",
    "paper/sections",
    "governance",
)


class ProjectStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _DEFAULT_BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, project: ResearchProject) -> ResearchProject:
        project_dir = self.base_dir / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        for subdir in _SUBDIRS:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)
        project.workspace_dir = str(project_dir)
        self._save(project)
        return project

    def save(self, project: ResearchProject) -> None:
        project.touch()
        self._save(project)

    def load(self, project_id: str) -> ResearchProject:
        path = self._project_file(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id!r}")
        return ResearchProject.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> list[ResearchProject]:
        projects: list[ResearchProject] = []
        if not self.base_dir.exists():
            return projects
        for project_dir in self.base_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_file = project_dir / "project.json"
            if not project_file.exists():
                continue
            try:
                projects.append(
                    ResearchProject.model_validate_json(
                        project_file.read_text(encoding="utf-8")
                    )
                )
            except Exception:  # skip corrupt project files
                pass
        return sorted(projects, key=lambda p: p.created_at, reverse=True)

    def delete(self, project_id: str) -> None:
        project_dir = self.base_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)

    def _project_file(self, project_id: str) -> Path:
        return self.base_dir / project_id / "project.json"

    def _save(self, project: ResearchProject) -> None:
        path = self._project_file(project.id)
        content = project.model_dump_json(indent=2)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".proj_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
