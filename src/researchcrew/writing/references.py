from __future__ import annotations

from researchcrew.writing.models import CitationEntry, PaperDraft


class ReferenceManager:
    """Manages citations and exports BibTeX."""

    def __init__(self, draft: PaperDraft) -> None:
        self.draft = draft

    def add(self, entry: CitationEntry) -> str:
        """Add citation, returning cite_key. Deduplicates by paper_id."""
        for existing in self.draft.references:
            if existing.paper_id == entry.paper_id:
                return existing.cite_key
        if not entry.cite_key:
            entry.cite_key = self._make_key(entry)
        self.draft.references.append(entry)
        return entry.cite_key

    def to_bibtex(self) -> str:
        lines: list[str] = []
        for ref in self.draft.references:
            authors = " and ".join(ref.authors) if ref.authors else "Unknown"
            entry = [
                f"@article{{{ref.cite_key},",
                f"  title = {{{ref.title}}},",
                f"  author = {{{authors}}},",
            ]
            if ref.year:
                entry.append(f"  year = {{{ref.year}}},")
            if ref.journal:
                entry.append(f"  journal = {{{ref.journal}}},")
            if ref.doi:
                entry.append(f"  doi = {{{ref.doi}}},")
            if ref.arxiv_id:
                entry.append(f"  eprint = {{{ref.arxiv_id}}},")
            if ref.url:
                entry.append(f"  url = {{{ref.url}}},")
            entry.append("}")
            lines.append("\n".join(entry))
        return "\n\n".join(lines)

    @staticmethod
    def _make_key(entry: CitationEntry) -> str:
        first_author = (
            entry.authors[0].split()[-1].lower() if entry.authors else "unknown"
        )
        year = str(entry.year) if entry.year else "nd"
        title_word = entry.title.split()[0].lower() if entry.title else "paper"
        return f"{first_author}{year}{title_word}"
