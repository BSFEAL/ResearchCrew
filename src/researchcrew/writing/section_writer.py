from __future__ import annotations

from researchcrew.writing.models import PaperDraft, Section, SectionStatus


# PaperOrchestra dependency order: write what you know first, frame last.
SECTION_ORDER: list[str] = [
    "related_work",
    "methods",
    "results",
    "discussion",
    "introduction",
    "abstract",
    "conclusion",
]


class SectionWriter:
    """Drafts paper sections in dependency order using an LLM callable.

    ``llm_write(section_name, context_dict)`` must return a string.
    Inject a real LLM call by subclassing or passing a callable at construction.
    """

    def __init__(self, llm_write: object | None = None) -> None:
        self._llm_write = llm_write

    def draft_all(
        self,
        draft: PaperDraft,
        hypotheses_summary: str,
        bibliography_summary: str,
    ) -> PaperDraft:
        for section_name in SECTION_ORDER:
            self.draft_section(
                draft, section_name, hypotheses_summary, bibliography_summary
            )
        return draft

    def draft_section(
        self,
        draft: PaperDraft,
        section_name: str,
        hypotheses_summary: str,
        bibliography_summary: str,
    ) -> Section:
        section = draft.get_section(section_name)
        scope = self._scope_from_outline(draft.outline, section_name)

        if self._llm_write is not None and callable(self._llm_write):
            context = {
                "section": section_name,
                "scope": scope,
                "hypotheses": hypotheses_summary,
                "bibliography": bibliography_summary,
                "prior_sections": {
                    k: v.content[:500]
                    for k, v in draft.sections.items()
                    if v.status != SectionStatus.PLANNED
                },
            }
            content = self._llm_write(section_name, context)  # type: ignore[call-arg]
        else:
            content = f"[{section_name.upper()} PLACEHOLDER — wire SectionWriter.llm_write]"

        section.bump_version(str(content))
        draft.touch()
        return section

    @staticmethod
    def _scope_from_outline(outline: object, section_name: str) -> str:
        from researchcrew.writing.models import Outline
        if isinstance(outline, Outline):
            for item in outline.section_plan:
                if item.get("name") == section_name:
                    return str(item.get("scope", ""))
        return ""
