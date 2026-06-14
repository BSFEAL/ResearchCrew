"""
Unit tests for researchcrew.tools.colab_tool.
No real Colab connection required — fastmcp and mcp are fully stubbed.
"""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── stub fastmcp and mcp before import ───────────────────────────────────────

_stub_content = MagicMock()
_stub_content.text = '{"stdout": "42\\n", "stderr": ""}'

_stub_result = [_stub_content]

_stub_client_instance = MagicMock()
_stub_client_instance.__aenter__ = AsyncMock(return_value=_stub_client_instance)
_stub_client_instance.__aexit__ = AsyncMock(return_value=False)
_stub_client_instance.call_tool = AsyncMock(return_value=_stub_result)

_stub_fastmcp = MagicMock()
_stub_fastmcp.Client = MagicMock(return_value=_stub_client_instance)
sys.modules.setdefault("fastmcp", _stub_fastmcp)

_stub_mcp = MagicMock()
_stub_mcp.StdioServerParameters = MagicMock(return_value=MagicMock())
sys.modules.setdefault("mcp", _stub_mcp)

from researchcrew.tools.colab_tool import (  # noqa: E402
    ColabExecuteTool,
    ColabInstallTool,
    ColabRuntimeTool,
    _call_colab,
    _run_async,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_client_mock() -> None:
    _stub_client_instance.call_tool.reset_mock()
    _stub_client_instance.call_tool.return_value = _stub_result


# ── _run_async ────────────────────────────────────────────────────────────────

class TestRunAsync:
    def test_runs_coroutine_without_event_loop(self):
        async def _coro() -> int:
            return 99

        assert _run_async(_coro()) == 99

    def test_runs_coroutine_inside_running_loop(self):
        """When called from inside a running loop, should use a thread."""
        async def _outer():
            async def _inner() -> str:
                return "ok"
            return _run_async(_inner())

        result = asyncio.run(_outer())
        assert result == "ok"


# ── _call_colab (async) ───────────────────────────────────────────────────────

class TestCallColab:
    def setup_method(self):
        _reset_client_mock()

    def test_returns_text_content(self):
        result = asyncio.run(_call_colab("execute_code", {"code": "print(42)"}))
        assert "42" in result

    def test_passes_tool_name_and_args(self):
        asyncio.run(_call_colab("install_package", {"package": "numpy"}))
        _stub_client_instance.call_tool.assert_called_once_with(
            "install_package", {"package": "numpy"}
        )

    def test_empty_result_returns_json(self):
        _stub_client_instance.call_tool.return_value = []
        result = asyncio.run(_call_colab("get_runtime_info", {}))
        data = json.loads(result)
        assert "result" in data

    def test_exception_returns_error_json(self):
        _stub_client_instance.call_tool.side_effect = RuntimeError("connection refused")
        result = asyncio.run(_call_colab("execute_code", {"code": "x"}))
        data = json.loads(result)
        assert data["error"] == "RuntimeError"
        assert "connection refused" in data["detail"]

    def test_uses_url_when_env_set(self, monkeypatch):
        monkeypatch.setenv("COLAB_MCP_URL", "http://localhost:8765")
        _stub_fastmcp.Client.reset_mock()
        asyncio.run(_call_colab("get_runtime_info", {}))
        _stub_fastmcp.Client.assert_called_once_with("http://localhost:8765")

    def test_uses_stdio_when_no_env(self, monkeypatch):
        monkeypatch.delenv("COLAB_MCP_URL", raising=False)
        _stub_mcp.StdioServerParameters.reset_mock()
        asyncio.run(_call_colab("get_runtime_info", {}))
        _stub_mcp.StdioServerParameters.assert_called_once()

    def test_import_error_returns_hint(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "fastmcp":
                raise ImportError("no module named fastmcp")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        result = asyncio.run(_call_colab("execute_code", {"code": "x"}))
        data = json.loads(result)
        assert "hint" in data
        assert "researchcrew[colab]" in data["hint"]


# ── ColabExecuteTool ──────────────────────────────────────────────────────────

class TestColabExecuteTool:
    def setup_method(self):
        _reset_client_mock()

    def test_run_returns_output(self):
        tool = ColabExecuteTool()
        out = tool._run("print('hello')")
        assert "42" in out or "hello" in out or len(out) > 0

    def test_passes_code_to_call_tool(self):
        tool = ColabExecuteTool()
        tool._run("import os")
        call_args = _stub_client_instance.call_tool.call_args
        assert call_args[0][0] == "execute_code"
        assert call_args[0][1]["code"] == "import os"

    def test_default_timeout(self):
        tool = ColabExecuteTool()
        tool._run("x = 1")
        args = _stub_client_instance.call_tool.call_args[0][1]
        assert args["timeout"] == 60

    def test_custom_timeout(self):
        tool = ColabExecuteTool()
        tool._run("x = 1", timeout=120)
        args = _stub_client_instance.call_tool.call_args[0][1]
        assert args["timeout"] == 120

    def test_has_required_crewai_fields(self):
        tool = ColabExecuteTool()
        assert tool.name == "Colab Execute"
        assert len(tool.description) > 20


# ── ColabInstallTool ──────────────────────────────────────────────────────────

class TestColabInstallTool:
    def setup_method(self):
        _reset_client_mock()

    def test_calls_install_package(self):
        tool = ColabInstallTool()
        tool._run("torch")
        call_args = _stub_client_instance.call_tool.call_args[0]
        assert call_args[0] == "install_package"
        assert call_args[1]["package"] == "torch"

    def test_returns_string(self):
        tool = ColabInstallTool()
        result = tool._run("numpy")
        assert isinstance(result, str)


# ── ColabRuntimeTool ──────────────────────────────────────────────────────────

class TestColabRuntimeTool:
    def setup_method(self):
        _reset_client_mock()

    def test_calls_get_runtime_info(self):
        tool = ColabRuntimeTool()
        tool._run()
        call_args = _stub_client_instance.call_tool.call_args[0]
        assert call_args[0] == "get_runtime_info"
        assert call_args[1] == {}

    def test_accepts_ignored_string_input(self):
        tool = ColabRuntimeTool()
        result = tool._run("anything")
        assert isinstance(result, str)
