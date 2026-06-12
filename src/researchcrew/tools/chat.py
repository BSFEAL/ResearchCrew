#!/usr/bin/env python3
"""
researchcrew chat — natural-language research project interface.

Usage
-----
    python chat.py                 # start / resume last project
    python chat.py new             # create new project
    python chat.py open <id>       # open project by id
    python chat.py list            # list all projects

In-session slash commands
-------------------------
    /new  [title]     create a new project
    /open <id>        open existing project
    /list             list all projects
    /status           show project overview
    /papers           list collected related works
    /outline          show paper outline
    /draft [section]  show draft (all or one section)
    /code             list saved code artifacts
    /save             force-save current state
    /export [path]    write publication-ready markdown
    /help             show this help
    /quit  /exit      exit
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ── optional deps ──────────────────────────────────────────────────────────────

for pkg, install in [("rich", "rich"), ("prompt_toolkit", "prompt_toolkit"), ("anthropic", "anthropic")]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Missing dependency — install with:  pip install {install}", file=sys.stderr)
        sys.exit(1)

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table, Column
from rich.text import Text
from rich.theme import Theme
from rich.syntax import Syntax
from rich import box as rbox
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
import anthropic


# ── configuration ──────────────────────────────────────────────────────────────

APP_DIR      = Path.home() / ".researchcrew"
PROJECTS_DIR = APP_DIR / "projects"
HISTORY_FILE = APP_DIR / ".input_history"
MODEL        = "claude-opus-4-8"

THEME = Theme({
    "user":      "bold #5fd7ff",
    "assistant": "bold #87d787",
    "system":    "dim white",
    "error":     "bold red",
    "warning":   "yellow",
    "success":   "bold green",
    "info":      "cyan",
    "muted":     "dim",
    "title":     "bold #d787ff",
    "cmd":       "bold cyan",
    "label":     "bold white",
})

CONSOLE = Console(theme=THEME, highlight=False)


# ── helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", text.lower())[:40]


# ── project data model ─────────────────────────────────────────────────────────

class Project:
    """All state for one research project."""

    def __init__(self, project_id: str, title: str, research_goal: str = "") -> None:
        self.project_id   = project_id
        self.title        = title
        self.research_goal = research_goal
        self.status: str  = "exploring"   # exploring | writing | review | submitted
        self.related_works: list[dict[str, Any]] = []
        self.code_artifacts: list[dict[str, Any]] = []
        self.sections: dict[str, str] = {}   # section_name → markdown text
        self.outline: dict[str, Any] | None = None
        self.notes: str  = ""
        self.keywords: list[str] = []
        self.created_at  = _now()
        self.updated_at  = _now()
        self.messages: list[dict[str, str]] = []   # anthropic wire format

    # ── dirs ─────────────────────────────────────────────────────────────────

    @property
    def dir(self) -> Path:        return PROJECTS_DIR / self.project_id
    @property
    def meta_path(self) -> Path:  return self.dir / "meta.json"
    @property
    def chat_path(self) -> Path:  return self.dir / "chat.jsonl"
    @property
    def draft_dir(self) -> Path:  return self.dir / "draft"
    @property
    def code_dir(self) -> Path:   return self.dir / "code"

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.updated_at = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.draft_dir.mkdir(exist_ok=True)
        self.code_dir.mkdir(exist_ok=True)

        self.meta_path.write_text(json.dumps({
            "project_id":   self.project_id,
            "title":        self.title,
            "research_goal": self.research_goal,
            "status":       self.status,
            "related_works": self.related_works,
            "code_artifacts": self.code_artifacts,
            "outline":      self.outline,
            "section_order": list(self.sections.keys()),
            "notes":        self.notes,
            "keywords":     self.keywords,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
        }, indent=2))

        for name, content in self.sections.items():
            (self.draft_dir / f"{_slug(name)}.md").write_text(content)

        _LANG_EXT = {"python":"py","javascript":"js","typescript":"ts",
                     "bash":"sh","r":"r","julia":"jl"}
        for art in self.code_artifacts:
            ext  = _LANG_EXT.get(art.get("language","").lower(), "txt")
            fname = _slug(art["name"]) + f".{ext}"
            (self.code_dir / fname).write_text(art["content"])

        with self.chat_path.open("w") as f:
            for msg in self.messages:
                f.write(json.dumps(msg) + "\n")

    @classmethod
    def load(cls, project_id: str) -> "Project":
        meta = json.loads((PROJECTS_DIR / project_id / "meta.json").read_text())
        p = cls(meta["project_id"], meta["title"], meta.get("research_goal",""))
        p.status         = meta.get("status","exploring")
        p.related_works  = meta.get("related_works",[])
        p.code_artifacts = meta.get("code_artifacts",[])
        p.outline        = meta.get("outline")
        p.notes          = meta.get("notes","")
        p.keywords       = meta.get("keywords",[])
        p.created_at     = meta.get("created_at", _now())
        p.updated_at     = meta.get("updated_at", _now())
        draft_dir = PROJECTS_DIR / project_id / "draft"
        for name in meta.get("section_order", []):
            fpath = draft_dir / f"{_slug(name)}.md"
            if fpath.exists():
                p.sections[name] = fpath.read_text()
        chat_path = PROJECTS_DIR / project_id / "chat.jsonl"
        if chat_path.exists():
            for line in chat_path.read_text().splitlines():
                if line.strip():
                    p.messages.append(json.loads(line))
        return p

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        if not PROJECTS_DIR.exists():
            return []
        out = []
        for d in sorted(PROJECTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            mp = d / "meta.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    out.append({
                        "id":       m["project_id"],
                        "title":    m["title"],
                        "status":   m.get("status","exploring"),
                        "updated":  m.get("updated_at","")[:10],
                        "sections": len(m.get("section_order",[])),
                        "papers":   len(m.get("related_works",[])),
                    })
                except Exception:
                    pass
        return out

    # ── context for Claude ────────────────────────────────────────────────────

    def context_block(self) -> str:
        parts = [
            f"Title: {self.title}",
            f"Research goal: {self.research_goal or '(not yet defined)'}",
            f"Status: {self.status}",
        ]
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        if self.related_works:
            sample = "; ".join(f"\"{w.get('title','')}\"" for w in self.related_works[:6])
            parts.append(f"Related works ({len(self.related_works)}): {sample}")
        if self.outline:
            sec_names = [s.get("name","") for s in self.outline.get("section_plan",[])]
            parts.append(f"Outline: {', '.join(sec_names)}")
        if self.sections:
            parts.append(f"Draft sections written: {', '.join(self.sections.keys())}")
        if self.code_artifacts:
            parts.append(f"Code artifacts: {', '.join(a['name'] for a in self.code_artifacts)}")
        return "\n".join(parts)

    # ── export ─────────────────────────────────────────────────────────────────

    def export_markdown(self) -> str:
        lines: list[str] = [f"# {self.title}", ""]
        if self.research_goal:
            lines += [f"> **Research Goal:** {self.research_goal}", ""]

        STANDARD_ORDER = ["Abstract","Introduction","Related Work","Background",
                          "Methodology","Methods","Experimental Setup","Results",
                          "Discussion","Conclusion","Future Work","References"]
        ordered   = [s for s in STANDARD_ORDER if s in self.sections]
        remainder = [s for s in self.sections   if s not in set(STANDARD_ORDER)]

        for name in ordered + remainder:
            lines += [f"## {name}", "", self.sections[name], ""]

        if self.related_works and "References" not in self.sections:
            lines += ["## References", ""]
            for i, w in enumerate(self.related_works, 1):
                authors = ", ".join(w.get("authors",[])[:3])
                if len(w.get("authors",[])) > 3:
                    authors += " et al."
                year  = w.get("year","n.d.")
                title = w.get("title","Untitled")
                url   = w.get("url","")
                ref   = f"{i}. {authors} ({year}). *{title}*."
                if url:
                    ref += f" <{url}>"
                lines.append(ref)
            lines.append("")

        lines += ["---", f"*Generated by researchcrew chat · {_now()[:10]}*"]
        return "\n".join(lines)


# ── system prompt ──────────────────────────────────────────────────────────────

_SYSTEM_BASE = """\
You are a scientific research assistant embedded in the researchcrew CLI.
You help researchers through the full lifecycle: literature review, hypothesis
generation, paper outlining, section writing, and publication preparation.

Current project state:
{context}

Behavioral guidelines:
- Keep responses focused and actionable.
- When writing a paper section, use proper academic prose with citation placeholders like [Author, Year].
- When asked to search for papers, describe what you would search for and suggest using the
  `search_papers` tool (available separately) with those keywords.
- After writing a section say: "Type /save-section <name> to save this to your draft."
- After generating a code snippet say: "Type /save-code <name> to add this to your project."
- After listing papers say: "Type /add-paper <title> to add one to your related works."
- Be concise in conversational turns; be thorough when writing academic content.
"""

def _system_prompt(project: Project) -> str:
    return _SYSTEM_BASE.format(context=project.context_block())


# ── display helpers ────────────────────────────────────────────────────────────

def _print_header(project: Project | None = None) -> None:
    CONSOLE.print()
    title = Text("researchcrew chat", style="title")
    subtitle = Text(" — AI research companion", style="muted")
    CONSOLE.print(title + subtitle)
    CONSOLE.print(Rule(style="muted"))
    if project:
        status_color = {
            "exploring": "cyan", "writing": "yellow",
            "review": "blue", "submitted": "green",
        }.get(project.status, "white")
        CONSOLE.print(
            f"  [label]Project:[/]  {project.title}  "
            f"[{status_color}][{project.status}][/{status_color}]"
        )
        if project.research_goal:
            goal = project.research_goal[:80] + ("…" if len(project.research_goal) > 80 else "")
            CONSOLE.print(f"  [label]Goal:[/]     [muted]{goal}[/muted]")
        CONSOLE.print(Rule(style="muted"))
    CONSOLE.print(
        "  Type your message or [cmd]/help[/] for commands. "
        "[muted]Ctrl-C to exit.[/muted]"
    )
    CONSOLE.print()


def _print_help() -> None:
    t = Table(box=None, show_header=False, padding=(0,2))
    t.add_column("cmd", style="cmd",   no_wrap=True)
    t.add_column("desc", style="muted")
    rows = [
        ("/new [title]",    "Create a new research project"),
        ("/open <id>",      "Open an existing project"),
        ("/list",           "List all projects"),
        ("/status",         "Show project overview"),
        ("/papers",         "List collected related works"),
        ("/outline",        "Show paper outline"),
        ("/draft [section]","Show full draft or one section"),
        ("/code",           "List saved code artifacts"),
        ("/save",           "Force-save current state"),
        ("/export [path]",  "Write publication-ready Markdown file"),
        ("/save-section <name>", "Save last assistant response as a paper section"),
        ("/add-paper <title>",   "Add a paper to related works (interactive)"),
        ("/save-code <name>",    "Save last code block as a code artifact"),
        ("/help",           "Show this help"),
        ("/quit  /exit",    "Exit"),
    ]
    for cmd, desc in rows:
        t.add_row(cmd, desc)
    CONSOLE.print(Panel(t, title="[label]Commands[/]", border_style="muted", padding=(1,2)))


def _print_status(p: Project) -> None:
    CONSOLE.print()
    CONSOLE.print(f"[label]Title:[/]       {p.title}")
    CONSOLE.print(f"[label]ID:[/]          [muted]{p.project_id}[/muted]")
    CONSOLE.print(f"[label]Status:[/]      {p.status}")
    CONSOLE.print(f"[label]Goal:[/]        {p.research_goal or '(not set)'}")
    CONSOLE.print(f"[label]Keywords:[/]    {', '.join(p.keywords) or '(none)'}")
    CONSOLE.print(f"[label]Papers:[/]      {len(p.related_works)}")
    CONSOLE.print(f"[label]Sections:[/]    {', '.join(p.sections.keys()) or '(none)'}")
    CONSOLE.print(f"[label]Code:[/]        {len(p.code_artifacts)} artifact(s)")
    CONSOLE.print(f"[label]Messages:[/]    {len(p.messages)}")
    CONSOLE.print(f"[label]Updated:[/]     {p.updated_at[:19]} UTC")
    CONSOLE.print()


def _print_projects(projects: list[dict]) -> None:
    if not projects:
        CONSOLE.print("[muted]No projects found. Use /new to create one.[/muted]")
        return
    t = Table(box=rbox.SIMPLE, show_header=True, header_style="label")
    t.add_column("#",        width=3,  justify="right")
    t.add_column("ID",       style="muted", no_wrap=True)
    t.add_column("Title",    style="white")
    t.add_column("Status",   width=12)
    t.add_column("Updated",  width=12)
    t.add_column("§ Sects",  width=7, justify="right")
    t.add_column("§ Papers", width=8, justify="right")
    for i, p in enumerate(projects, 1):
        sc = {"exploring":"cyan","writing":"yellow","review":"blue","submitted":"green"}.get(p["status"],"white")
        t.add_row(
            str(i), p["id"][:20], p["title"][:40],
            f"[{sc}]{p['status']}[/{sc}]",
            p["updated"], str(p["sections"]), str(p["papers"]),
        )
    CONSOLE.print(t)


def _print_papers(p: Project) -> None:
    if not p.related_works:
        CONSOLE.print("[muted]No related works yet.[/muted]")
        return
    t = Table(box=rbox.SIMPLE, show_header=True, header_style="label")
    t.add_column("#", width=3, justify="right")
    t.add_column("Title")
    t.add_column("Authors", style="muted")
    t.add_column("Year", width=6)
    t.add_column("Cites", width=7, justify="right")
    for i, w in enumerate(p.related_works, 1):
        authors = ", ".join(w.get("authors",[])[:2])
        if len(w.get("authors",[])) > 2: authors += " et al."
        t.add_row(str(i), w.get("title","")[:60], authors[:30],
                  str(w.get("year","")), str(w.get("citation_count","")))
    CONSOLE.print(t)


def _print_code(p: Project) -> None:
    if not p.code_artifacts:
        CONSOLE.print("[muted]No code artifacts yet.[/muted]")
        return
    for art in p.code_artifacts:
        lang = art.get("language","text")
        CONSOLE.print(Panel(
            Syntax(art["content"], lang, theme="monokai", line_numbers=True),
            title=f"[cmd]{art['name']}[/cmd]  [muted]({lang})[/muted]",
            border_style="muted",
        ))


def _print_draft(p: Project, section: str | None = None) -> None:
    if not p.sections:
        CONSOLE.print("[muted]No draft sections yet.[/muted]")
        return
    names = [section] if section else list(p.sections.keys())
    for name in names:
        if name not in p.sections:
            CONSOLE.print(f"[warning]Section '{name}' not found.[/warning]")
            continue
        CONSOLE.print(Panel(
            Markdown(p.sections[name]),
            title=f"[label]{name}[/label]",
            border_style="muted",
            padding=(1, 2),
        ))


# ── Claude streaming ───────────────────────────────────────────────────────────

def _stream_response(client: anthropic.Anthropic, project: Project) -> str:
    """Stream the next assistant turn; returns the full text."""
    full_text = ""

    with Live(console=CONSOLE, refresh_per_second=15) as live:
        live.update(Text(""))
        with client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=_system_prompt(project),
            messages=project.messages,
        ) as stream:
            for delta in stream.text_stream:
                full_text += delta
                # render visible text only (no thinking blocks in stream)
                live.update(
                    Panel(
                        Markdown(full_text),
                        title="[assistant]Claude[/assistant]",
                        border_style="#87d787",
                        padding=(0, 1),
                    )
                )

    CONSOLE.print()
    return full_text


# ── post-response extraction ───────────────────────────────────────────────────

_CODE_FENCE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)

def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(language, code), …] for all fenced blocks with ≥5 lines."""
    blocks = []
    for m in _CODE_FENCE.finditer(text):
        lang, code = m.group(1) or "text", m.group(2).strip()
        if code.count("\n") >= 4:
            blocks.append((lang, code))
    return blocks


def _looks_like_section(text: str) -> bool:
    """Heuristic: long response with paragraph prose."""
    return len(text) > 400 and text.count("\n\n") >= 2


# ── input session ──────────────────────────────────────────────────────────────

def _make_prompt_session() -> PromptSession:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=False,
    )


def _prompt_user(session: PromptSession, project: Project | None) -> str:
    name = project.title[:20] if project else "no project"
    try:
        return session.prompt(
            HTML(f'<ansibrightcyan>You</ansibrightcyan> '
                 f'<ansigray>[{name}]</ansigray> '
                 f'<ansibrightcyan>❯</ansibrightcyan> '),
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return "/quit"


# ── project wizard ─────────────────────────────────────────────────────────────

def _create_project_interactive(session: PromptSession, title: str = "") -> Project:
    if not title:
        CONSOLE.print("[info]New project[/info]")
        title = session.prompt(HTML('<ansiwhite>Title: </ansiwhite>')).strip()
        if not title:
            title = "Untitled Research"
    goal = session.prompt(HTML('<ansiwhite>Research goal (1-2 sentences): </ansiwhite>')).strip()
    keywords_raw = session.prompt(HTML('<ansiwhite>Keywords (comma-separated, optional): </ansiwhite>')).strip()
    pid = str(uuid.uuid4())[:8] + "-" + _slug(title)
    p = Project(pid, title, goal)
    p.keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    p.save()
    CONSOLE.print(f"[success]Created project [bold]{pid}[/bold][/success]")
    return p


# ── slash command handler ──────────────────────────────────────────────────────

def _handle_slash(
    cmd: str,
    args: str,
    project: Project | None,
    last_response: str,
    session: PromptSession,
    client: anthropic.Anthropic,
) -> tuple[Project | None, bool]:
    """
    Returns (updated_project, should_quit).
    """
    cmd = cmd.lower()

    if cmd in ("/quit", "/exit"):
        return project, True

    if cmd == "/help":
        _print_help()

    elif cmd == "/new":
        project = _create_project_interactive(session, args)
        _print_header(project)

    elif cmd == "/open":
        pid = args.strip()
        if not pid:
            all_p = Project.list_all()
            _print_projects(all_p)
            pid = session.prompt(HTML('<ansiwhite>Enter project id: </ansiwhite>')).strip()
        try:
            project = Project.load(pid)
            CONSOLE.print(f"[success]Opened: {project.title}[/success]")
            _print_header(project)
        except Exception as e:
            CONSOLE.print(f"[error]Could not open '{pid}': {e}[/error]")

    elif cmd == "/list":
        _print_projects(Project.list_all())

    elif cmd == "/status":
        if project:
            _print_status(project)
        else:
            CONSOLE.print("[warning]No active project. Use /new to create one.[/warning]")

    elif cmd == "/papers":
        if project:
            _print_papers(project)
        else:
            CONSOLE.print("[warning]No active project.[/warning]")

    elif cmd == "/code":
        if project:
            _print_code(project)
        else:
            CONSOLE.print("[warning]No active project.[/warning]")

    elif cmd == "/outline":
        if project and project.outline:
            CONSOLE.print(Markdown(json.dumps(project.outline, indent=2)))
        elif project:
            CONSOLE.print("[muted]No outline yet. Ask Claude to create one.[/muted]")
        else:
            CONSOLE.print("[warning]No active project.[/warning]")

    elif cmd == "/draft":
        if project:
            _print_draft(project, args.strip() or None)
        else:
            CONSOLE.print("[warning]No active project.[/warning]")

    elif cmd == "/save":
        if project:
            project.save()
            CONSOLE.print("[success]Saved.[/success]")
        else:
            CONSOLE.print("[warning]Nothing to save.[/warning]")

    elif cmd == "/save-section":
        if not project:
            CONSOLE.print("[warning]No active project.[/warning]")
        elif not last_response:
            CONSOLE.print("[warning]No assistant response to save yet.[/warning]")
        else:
            name = args.strip() or session.prompt(HTML('<ansiwhite>Section name: </ansiwhite>')).strip()
            if name:
                project.sections[name] = last_response
                project.save()
                CONSOLE.print(f"[success]Saved section '{name}'.[/success]")

    elif cmd == "/save-code":
        if not project:
            CONSOLE.print("[warning]No active project.[/warning]")
        else:
            blocks = _extract_code_blocks(last_response)
            if not blocks:
                CONSOLE.print("[warning]No code blocks found in last response.[/warning]")
            else:
                lang, code = blocks[0]
                name = args.strip() or session.prompt(HTML('<ansiwhite>Artifact name: </ansiwhite>')).strip()
                if name:
                    # remove existing artifact with same name
                    project.code_artifacts = [a for a in project.code_artifacts if a["name"] != name]
                    project.code_artifacts.append({"name": name, "language": lang,
                                                   "content": code, "created_at": _now()})
                    project.save()
                    CONSOLE.print(f"[success]Saved code artifact '{name}' ({lang}).[/success]")

    elif cmd == "/add-paper":
        if not project:
            CONSOLE.print("[warning]No active project.[/warning]")
        else:
            title = args.strip() or session.prompt(HTML('<ansiwhite>Paper title: </ansiwhite>')).strip()
            authors_raw = session.prompt(HTML('<ansiwhite>Authors (comma-separated): </ansiwhite>')).strip()
            year_raw    = session.prompt(HTML('<ansiwhite>Year: </ansiwhite>')).strip()
            url_raw     = session.prompt(HTML('<ansiwhite>URL or DOI (optional): </ansiwhite>')).strip()
            project.related_works.append({
                "title":   title,
                "authors": [a.strip() for a in authors_raw.split(",") if a.strip()],
                "year":    int(year_raw) if year_raw.isdigit() else None,
                "url":     url_raw or None,
                "added_at": _now(),
            })
            project.save()
            CONSOLE.print(f"[success]Added '{title}' to related works.[/success]")

    elif cmd == "/export":
        if not project:
            CONSOLE.print("[warning]No active project.[/warning]")
        elif not project.sections:
            CONSOLE.print("[warning]No sections to export yet. Write some draft sections first.[/warning]")
        else:
            default_path = Path.cwd() / f"{_slug(project.title)}.md"
            path_str = args.strip() or str(default_path)
            out_path = Path(path_str)
            out_path.write_text(project.export_markdown())
            CONSOLE.print(f"[success]Exported to {out_path}[/success]")

    else:
        CONSOLE.print(f"[warning]Unknown command: {cmd}. Type /help for available commands.[/warning]")

    return project, False


# ── main chat loop ─────────────────────────────────────────────────────────────

def _chat_loop(project: Project | None, client: anthropic.Anthropic) -> None:
    session     = _make_prompt_session()
    last_response = ""

    _print_header(project)

    while True:
        try:
            user_input = _prompt_user(session, project)
        except KeyboardInterrupt:
            CONSOLE.print("\n[muted]Interrupted. Type /quit to exit.[/muted]")
            continue

        if not user_input:
            continue

        # ── slash command ─────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd   = parts[0]
            args  = parts[1] if len(parts) > 1 else ""
            project, should_quit = _handle_slash(
                cmd, args, project, last_response, session, client
            )
            if should_quit:
                if project:
                    project.save()
                CONSOLE.print("[muted]Goodbye.[/muted]")
                break
            continue

        # ── need a project for chat ───────────────────────────────────────────
        if project is None:
            CONSOLE.print(
                "[info]No active project. Creating one automatically…[/info]"
            )
            project = _create_project_interactive(session, "")
            _print_header(project)

        # ── add user message ──────────────────────────────────────────────────
        project.messages.append({"role": "user", "content": user_input})

        # ── stream response ───────────────────────────────────────────────────
        try:
            response_text = _stream_response(client, project)
        except anthropic.AuthenticationError:
            CONSOLE.print("[error]Invalid API key. Set ANTHROPIC_API_KEY and restart.[/error]")
            project.messages.pop()
            continue
        except Exception as exc:
            CONSOLE.print(f"[error]API error: {exc}[/error]")
            project.messages.pop()
            continue

        last_response = response_text
        project.messages.append({"role": "assistant", "content": response_text})

        # ── auto-save ─────────────────────────────────────────────────────────
        project.save()

        # ── gentle hints about detected content ──────────────────────────────
        code_blocks = _extract_code_blocks(response_text)
        if code_blocks:
            langs = ", ".join(set(b[0] for b in code_blocks))
            CONSOLE.print(
                f"  [muted]↳ {len(code_blocks)} code block(s) detected [{langs}]. "
                "Use [cmd]/save-code <name>[/cmd] to save.[/muted]"
            )
        if _looks_like_section(response_text) and not code_blocks:
            CONSOLE.print(
                "  [muted]↳ This looks like a paper section. "
                "Use [cmd]/save-section <name>[/cmd] to add to your draft.[/muted]"
            )


# ── entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = (argv or sys.argv)[1:]

    # ensure dirs exist
    APP_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(exist_ok=True)

    # get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        CONSOLE.print("[error]ANTHROPIC_API_KEY is not set.[/error]")
        CONSOLE.print("[muted]Export it and re-run:  export ANTHROPIC_API_KEY=sk-ant-…[/muted]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    project: Project | None = None

    subcmd = args[0].lower() if args else ""

    if subcmd == "list":
        _print_projects(Project.list_all())
        return

    if subcmd == "new":
        title = " ".join(args[1:])
        session = _make_prompt_session()
        project = _create_project_interactive(session, title)

    elif subcmd == "open" and len(args) >= 2:
        try:
            project = Project.load(args[1])
            CONSOLE.print(f"[success]Opened: {project.title}[/success]")
        except Exception as e:
            CONSOLE.print(f"[error]Could not open '{args[1]}': {e}[/error]")
            sys.exit(1)

    else:
        # resume last project if one exists
        all_p = Project.list_all()
        if all_p:
            try:
                project = Project.load(all_p[0]["id"])
                CONSOLE.print(f"[info]Resuming: {project.title}[/info]")
            except Exception:
                pass

    _chat_loop(project, client)


if __name__ == "__main__":
    main()
