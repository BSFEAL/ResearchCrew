from __future__ import annotations

from researchcrew.project.project import ProjectStatus, ResearchProject


_TRANSITIONS: dict[ProjectStatus, list[ProjectStatus]] = {
    ProjectStatus.IDEATION: [ProjectStatus.LITERATURE_REVIEW],
    ProjectStatus.LITERATURE_REVIEW: [ProjectStatus.HYPOTHESIS_GENERATION],
    ProjectStatus.HYPOTHESIS_GENERATION: [ProjectStatus.TOURNAMENT],
    ProjectStatus.TOURNAMENT: [ProjectStatus.WRITING],
    ProjectStatus.WRITING: [ProjectStatus.PEER_REVIEW],
    ProjectStatus.PEER_REVIEW: [ProjectStatus.FINAL],
    ProjectStatus.FINAL: [],
}

_STATUS_ORDER: list[ProjectStatus] = [
    ProjectStatus.IDEATION,
    ProjectStatus.LITERATURE_REVIEW,
    ProjectStatus.HYPOTHESIS_GENERATION,
    ProjectStatus.TOURNAMENT,
    ProjectStatus.WRITING,
    ProjectStatus.PEER_REVIEW,
    ProjectStatus.FINAL,
]


class LifecycleError(Exception):
    pass


class ProjectLifecycle:
    def __init__(self, project: ResearchProject) -> None:
        self.project = project

    def can_advance(self) -> bool:
        return bool(_TRANSITIONS.get(self.project.status))

    def next_status(self) -> ProjectStatus | None:
        transitions = _TRANSITIONS.get(self.project.status, [])
        return transitions[0] if transitions else None

    def advance(self) -> ProjectStatus:
        next_s = self.next_status()
        if next_s is None:
            raise LifecycleError(
                f"Project is already in terminal state: {self.project.status.value!r}"
            )
        self.project.status = next_s
        self.project.touch()
        return next_s

    def rollback(self, target: ProjectStatus) -> None:
        current_idx = _STATUS_ORDER.index(self.project.status)
        target_idx = _STATUS_ORDER.index(target)
        if target_idx >= current_idx:
            raise LifecycleError(
                f"Cannot roll back from {self.project.status.value!r} to "
                f"{target.value!r}: target must precede current status."
            )
        self.project.status = target
        self.project.touch()
