from __future__ import annotations

from researchcrew.writing.models import PaperDraft
from researchcrew.writing.references import ReferenceManager
from researchcrew.writing.section_writer import SECTION_ORDER


class PaperExporter:
    """Exports PaperDraft to Markdown, LaTeX, or BibTeX."""

    def to_markdown(self, draft: PaperDraft) -> str:
        lines = [f"# {draft.title or 'Untitled'}", ""]
        for name in SECTION_ORDER:
            section = draft.sections.get(name)
            if section and section.content:
                heading = name.replace("_", " ").title()
                lines += [f"## {heading}", "", section.content, ""]
        if draft.references:
            lines += ["## References", ""]
            for ref in draft.references:
                lines.append(
                    f"- [{ref.cite_key}] {', '.join(ref.authors[:3])} "
                    f"({ref.year}). *{ref.title}*."
                )
        return "\n".join(lines)

    def to_latex(self, draft: PaperDraft, template: str | None = None) -> str:
        body_parts: list[str] = []
        for name in SECTION_ORDER:
            section = draft.sections.get(name)
            if section and section.content:
                heading = name.replace("_", " ").title()
                body_parts.append(f"\\section{{{heading}}}")
                body_parts.append(section.content)
                body_parts.append("")

        bibtex = ReferenceManager(draft).to_bibtex()
        body = "\n".join(body_parts)
        title = draft.title or "Untitled"

        if template:
            return template.replace("{{TITLE}}", title).replace("{{BODY}}", body)

        return (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            f"\\title{{{title}}}\n"
            "\\maketitle\n\n"
            f"{body}\n"
            "\\end{document}"
        )

    def to_bibtex(self, draft: PaperDraft) -> str:
        return ReferenceManager(draft).to_bibtex()
