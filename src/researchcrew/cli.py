from __future__ import annotations

from pathlib import Path

import click

from researchcrew.project.lifecycle import LifecycleError, ProjectLifecycle
from researchcrew.project.project import ProjectStatus, ResearchProject
from researchcrew.project.store import ProjectStore


@click.group()
def app() -> None:
    """researchcrew — multi-agent scientific research library."""


# ── project ────────────────────────────────────────────────────────────

@app.group()
def project() -> None:
    """Manage research projects."""


@project.command("create")
@click.option("--name", required=True)
@click.option("--domain", default="general", show_default=True)
@click.option("--description", default="")
@click.option("--author", default=None)
@click.option("--tag", "tags", multiple=True, help="Repeatable tags.")
def project_create(
    name: str, domain: str, description: str, author: str | None, tags: tuple[str, ...]
) -> None:
    """Create a new research project."""
    store = ProjectStore()
    p = ResearchProject(
        name=name, domain=domain, description=description,
        author=author, tags=list(tags),
    )
    created = store.create(p)
    click.echo(f"Created: {created.id}")
    click.echo(f"  Name  : {created.name}")
    click.echo(f"  Domain: {created.domain}")
    click.echo(f"  Path  : {created.workspace_dir}")


@project.command("list")
def project_list() -> None:
    """List all research projects."""
    projects = ProjectStore().list_projects()
    if not projects:
        click.echo("No projects found.")
        return
    for p in projects:
        click.echo(f"  {p.id[:8]}  [{p.status.value:22}]  {p.name}")


@project.command("status")
@click.argument("project_id")
def project_status(project_id: str) -> None:
    """Show project metadata."""
    p = ProjectStore().load(project_id)
    click.echo(f"Project : {p.name}  ({p.id})")
    click.echo(f"  Status  : {p.status.value}")
    click.echo(f"  Domain  : {p.domain}")
    click.echo(f"  Author  : {p.author or '—'}")
    click.echo(f"  Tags    : {', '.join(p.tags) or '—'}")
    click.echo(f"  Goals   : {len(p.research_goals)}")
    click.echo(f"  Created : {p.created_at.isoformat()}")
    click.echo(f"  Path    : {p.workspace_dir or '—'}")


@project.command("advance")
@click.argument("project_id")
def project_advance(project_id: str) -> None:
    """Advance to the next lifecycle phase."""
    store = ProjectStore()
    p = store.load(project_id)
    lc = ProjectLifecycle(p)
    try:
        new_status = lc.advance()
        store.save(p)
        click.echo(f"Advanced to: {new_status.value}")
    except LifecycleError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@project.command("rollback")
@click.argument("project_id")
@click.option("--to", "target", required=True)
def project_rollback(project_id: str, target: str) -> None:
    """Roll back to an earlier lifecycle phase."""
    store = ProjectStore()
    p = store.load(project_id)
    lc = ProjectLifecycle(p)
    try:
        lc.rollback(ProjectStatus(target))
        store.save(p)
        click.echo(f"Rolled back to: {target}")
    except (LifecycleError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# ── literature ───────────────────────────────────────────────────────

@app.group()
def literature() -> None:
    """Literature search and analysis."""


@literature.command("search")
@click.option("--query", required=True)
@click.option("--sources", default="arxiv", show_default=True)
@click.option("--max-results", default=20, show_default=True, type=int)
def literature_search(query: str, sources: str, max_results: int) -> None:
    """Search scientific databases and print titles."""
    from researchcrew.literature.corpus import LiteratureCorpus
    from researchcrew.literature.searcher import MultiSourceSearcher

    corpus = LiteratureCorpus(project_id="cli")
    try:
        searcher = MultiSourceSearcher()
        n = searcher.search(
            query, corpus,
            sources=sources.split(","),
            max_results=max_results,
        )
        click.echo(f"Found {n} new papers:")
        for p in corpus.all_papers():
            click.echo(f"  [{p.year or '?'}] {p.title[:80]}")
    except ImportError:
        click.echo("Install researchcrew[literature] for live search.", err=True)
        raise SystemExit(1)


# ── paper ────────────────────────────────────────────────────────────

@app.group()
def paper() -> None:
    """Paper drafting and export."""


@paper.command("export")
@click.option("--title", default="Research Paper", show_default=True)
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "latex", "bibtex"]), show_default=True)
@click.option("--output", default=None, help="Output file path (stdout if omitted).")
def paper_export(title: str, fmt: str, output: str | None) -> None:
    """Export a paper draft (scaffold with no content, for testing)."""
    from researchcrew.writing.exporter import PaperExporter
    from researchcrew.writing.models import PaperDraft

    draft = PaperDraft(project_id="cli", title=title)
    exporter = PaperExporter()
    if fmt == "markdown":
        content = exporter.to_markdown(draft)
    elif fmt == "latex":
        content = exporter.to_latex(draft)
    else:
        content = exporter.to_bibtex(draft)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Written to {output}")
    else:
        click.echo(content)


# ── memory ────────────────────────────────────────────────────────────

@app.group()
def memory() -> None:
    """Inspect and query Retrospective Memory."""


@memory.command("query")
@click.option("--query", "q", required=True)
@click.option("--store", "store_path",
              default=".researchcrew/memory.jsonl", show_default=True)
@click.option("--top-k", default=5, show_default=True, type=int)
def memory_query(q: str, store_path: str, top_k: int) -> None:
    """Query Retrospective Memory (BM25 keyword search)."""
    from researchcrew.memory.store import MemoryStore

    store = MemoryStore(Path(store_path))
    try:
        from researchcrew.memory.retriever import BM25Retriever
        results = BM25Retriever(store).query(q, top_k=top_k)
    except ImportError:
        results = store.load_all()[:top_k]
        click.echo("(BM25 unavailable — showing recent entries; install researchcrew[memory])", err=True)

    if not results:
        click.echo("No entries found.")
        return
    for e in results:
        click.echo(f"  [{e.outcome.value}] {e.plan[:60]}")
