from __future__ import annotations

import click

from researchcrew.project.lifecycle import LifecycleError, ProjectLifecycle
from researchcrew.project.project import ProjectStatus, ResearchProject
from researchcrew.project.store import ProjectStore


@click.group()
def app() -> None:
    """researchcrew — multi-agent scientific research library."""


@app.group()
def project() -> None:
    """Manage research projects."""


@project.command("create")
@click.option("--name", required=True, help="Project name.")
@click.option("--domain", default="general", show_default=True, help="Research domain.")
@click.option("--description", default="", help="Short description.")
@click.option("--author", default=None, help="Author name.")
def project_create(
    name: str, domain: str, description: str, author: str | None
) -> None:
    """Create a new research project."""
    store = ProjectStore()
    p = ResearchProject(
        name=name, domain=domain, description=description, author=author
    )
    created = store.create(p)
    click.echo(f"Created project: {created.id}")
    click.echo(f"  Name:   {created.name}")
    click.echo(f"  Domain: {created.domain}")
    click.echo(f"  Status: {created.status.value}")
    click.echo(f"  Path:   {created.workspace_dir}")


@project.command("list")
def project_list() -> None:
    """List all research projects."""
    store = ProjectStore()
    projects = store.list_projects()
    if not projects:
        click.echo("No projects found.")
        return
    for p in projects:
        click.echo(f"  {p.id[:8]}  [{p.status.value:22}]  {p.name}")


@project.command("status")
@click.argument("project_id")
def project_status(project_id: str) -> None:
    """Show project status and metadata."""
    store = ProjectStore()
    p = store.load(project_id)
    click.echo(f"Project : {p.name}  ({p.id})")
    click.echo(f"  Status  : {p.status.value}")
    click.echo(f"  Domain  : {p.domain}")
    click.echo(f"  Author  : {p.author or '—'}")
    click.echo(f"  Created : {p.created_at.isoformat()}")
    click.echo(f"  Updated : {p.updated_at.isoformat()}")
    click.echo(f"  Path    : {p.workspace_dir or '—'}")


@project.command("advance")
@click.argument("project_id")
def project_advance(project_id: str) -> None:
    """Advance project to the next lifecycle phase."""
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
@click.option("--to", "target", required=True, help="Target status (e.g. tournament).")
def project_rollback(project_id: str, target: str) -> None:
    """Roll back project to an earlier lifecycle phase."""
    store = ProjectStore()
    p = store.load(project_id)
    lc = ProjectLifecycle(p)
    try:
        target_status = ProjectStatus(target)
        lc.rollback(target_status)
        store.save(p)
        click.echo(f"Rolled back to: {target_status.value}")
    except (LifecycleError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e
