"""
Unit tests for researchcrew.tools.chat — no Anthropic API required.
Tests cover Project persistence, export, slash command routing,
and code/section auto-detection helpers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── bootstrap: stub heavy optional imports before the module loads ────────────

_stub_anthropic = MagicMock()
_stub_anthropic.Anthropic = MagicMock
_stub_anthropic.AuthenticationError = Exception
sys.modules.setdefault("anthropic", _stub_anthropic)

for _mod in [
    "rich", "rich.console", "rich.live", "rich.markdown", "rich.panel",
    "rich.rule", "rich.syntax", "rich.table", "rich.text", "rich.theme",
    "rich.box", "prompt_toolkit", "prompt_toolkit.history",
    "prompt_toolkit.auto_suggest", "prompt_toolkit.key_binding",
    "prompt_toolkit.formatted_text",
]:
    sys.modules.setdefault(_mod, MagicMock())

from researchcrew.tools.chat import (  # noqa: E402
    Project,
    _extract_code_blocks,
    _looks_like_section,
    _slug,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_projects(tmp_path: Path, monkeypatch):
    """Redirect PROJECTS_DIR and APP_DIR to a temporary directory."""
    import researchcrew.tools.chat as chat_mod

    monkeypatch.setattr(chat_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(chat_mod, "APP_DIR", tmp_path.parent)
    return tmp_path


# ── _slug ─────────────────────────────────────────────────────────────────────

class TestSlug:
    def test_lowercases(self):
        assert _slug("Hello World") == "hello_world"

    def test_replaces_special_chars(self):
        assert _slug("A/B:C") == "a_b_c"

    def test_truncates_at_40(self):
        assert len(_slug("x" * 100)) == 40

    def test_keeps_hyphens_and_underscores(self):
        assert _slug("my-paper_v2") == "my-paper_v2"


# ── Project creation & save/load ─────────────────────────────────────────────

class TestProjectPersistence:
    def test_save_creates_meta_json(self, tmp_projects):
        p = Project("test-id", "My Paper", "Test goal")
        p.save()
        meta_path = tmp_projects / "test-id" / "meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["title"] == "My Paper"
        assert data["research_goal"] == "Test goal"
        assert data["status"] == "exploring"

    def test_save_creates_directories(self, tmp_projects):
        p = Project("test-id2", "Title")
        p.save()
        assert (tmp_projects / "test-id2" / "draft").is_dir()
        assert (tmp_projects / "test-id2" / "code").is_dir()

    def test_save_and_load_roundtrip(self, tmp_projects):
        p = Project("rt-id", "Roundtrip", "A roundtrip goal")
        p.status   = "writing"
        p.keywords = ["nlp", "transformers"]
        p.sections["Abstract"] = "This is the abstract."
        p.messages.append({"role": "user", "content": "Hello"})
        p.save()

        loaded = Project.load("rt-id")
        assert loaded.title         == "Roundtrip"
        assert loaded.research_goal == "A roundtrip goal"
        assert loaded.status        == "writing"
        assert loaded.keywords      == ["nlp", "transformers"]
        assert loaded.sections["Abstract"] == "This is the abstract."
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Hello"

    def test_save_persists_code_artifacts(self, tmp_projects):
        p = Project("code-id", "Code Test")
        p.code_artifacts.append({
            "name": "model", "language": "python",
            "content": "import torch\nmodel = torch.nn.Linear(10, 2)",
            "created_at": "2026-01-01",
        })
        p.save()
        assert (tmp_projects / "code-id" / "code" / "model.py").exists()

    def test_list_all_returns_projects(self, tmp_projects):
        Project("p1", "Paper One").save()
        Project("p2", "Paper Two").save()
        listing = Project.list_all()
        ids = [p["id"] for p in listing]
        assert "p1" in ids
        assert "p2" in ids

    def test_list_all_empty(self, tmp_projects):
        assert Project.list_all() == []


# ── Project.context_block ─────────────────────────────────────────────────────

class TestContextBlock:
    def test_includes_title_and_goal(self):
        p = Project("x", "My Title", "My Goal")
        ctx = p.context_block()
        assert "My Title" in ctx
        assert "My Goal" in ctx

    def test_includes_keywords(self):
        p = Project("x", "T")
        p.keywords = ["llm", "rl"]
        assert "llm" in p.context_block()

    def test_includes_section_names(self):
        p = Project("x", "T")
        p.sections["Introduction"] = "text"
        assert "Introduction" in p.context_block()

    def test_includes_related_works_sample(self):
        p = Project("x", "T")
        p.related_works = [{"title": "Famous Paper", "authors": [], "year": 2020}]
        assert "Famous Paper" in p.context_block()


# ── Project.export_markdown ───────────────────────────────────────────────────

class TestExportMarkdown:
    def test_title_in_output(self):
        p = Project("x", "Great Research")
        md = p.export_markdown()
        assert "# Great Research" in md

    def test_sections_in_standard_order(self):
        p = Project("x", "T")
        p.sections["Conclusion"]   = "We conclude."
        p.sections["Introduction"] = "We introduce."
        md = p.export_markdown()
        intro_pos = md.index("## Introduction")
        concl_pos = md.index("## Conclusion")
        assert intro_pos < concl_pos

    def test_auto_references_from_related_works(self):
        p = Project("x", "T")
        p.sections["Abstract"] = "text"
        p.related_works = [{
            "title":   "Awesome Paper",
            "authors": ["Smith, J.", "Doe, A."],
            "year":    2021,
            "url":     "https://example.com",
        }]
        md = p.export_markdown()
        assert "## References" in md
        assert "Awesome Paper" in md
        assert "Smith, J." in md

    def test_no_duplicate_references_if_section_exists(self):
        p = Project("x", "T")
        p.sections["References"] = "Custom refs."
        p.related_works = [{"title": "A", "authors": [], "year": 2020}]
        md = p.export_markdown()
        assert md.count("## References") == 1


# ── _extract_code_blocks ──────────────────────────────────────────────────────

class TestExtractCodeBlocks:
    def test_extracts_python_block(self):
        text = "```python\nimport os\nprint(1)\nx = 2\ny = 3\nz = 4\n```"
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        lang, code = blocks[0]
        assert lang == "python"
        assert "import os" in code

    def test_ignores_short_blocks(self):
        text = "```python\nprint(1)\n```"
        assert _extract_code_blocks(text) == []

    def test_multiple_blocks(self):
        snippet = "line1\nline2\nline3\nline4\nline5"
        text    = f"```python\n{snippet}\n```\n```bash\n{snippet}\n```"
        blocks  = _extract_code_blocks(text)
        assert len(blocks) == 2
        assert {b[0] for b in blocks} == {"python", "bash"}

    def test_no_language_tag_defaults_to_text(self):
        snippet = "\n".join([f"line{i}" for i in range(6)])
        text    = f"```\n{snippet}\n```"
        blocks  = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "text"


# ── _looks_like_section ───────────────────────────────────────────────────────

class TestLooksLikeSection:
    def test_long_prose_is_section(self):
        text = "This is a long paragraph.\n\n" * 30
        assert _looks_like_section(text) is True

    def test_short_text_is_not_section(self):
        assert _looks_like_section("Short reply.") is False

    def test_long_but_no_paragraphs_not_section(self):
        assert _looks_like_section("x" * 500) is False
