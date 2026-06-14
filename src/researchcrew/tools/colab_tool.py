from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from typing import Any

from crewai.tools import BaseTool


_COLAB_MCP_URL_ENV = "COLAB_MCP_URL"
_COLAB_MCP_CMD = "uvx"
_COLAB_MCP_ARGS = ["git+https://github.com/googlecolab/colab-mcp"]


def _run_async(coro: Any) -> Any:
    """Run a coroutine safely from any sync context (with or without a running loop)."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


async def _call_colab(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        from fastmcp import Client  # noqa: PLC0415
    except ImportError:
        return json.dumps({
            "error": "fastmcp not installed",
            "hint": "pip install researchcrew[colab]",
        })

    url = os.environ.get(_COLAB_MCP_URL_ENV, "")
    try:
        if url:
            source: Any = url
        else:
            from mcp import StdioServerParameters  # noqa: PLC0415
            source = StdioServerParameters(command=_COLAB_MCP_CMD, args=_COLAB_MCP_ARGS)

        async with Client(source) as client:
            result = await client.call_tool(tool_name, arguments)
            if result and hasattr(result[0], "text"):
                return str(result[0].text)
            return json.dumps({"result": str(result)})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": type(exc).__name__, "detail": str(exc)})


class ColabExecuteTool(BaseTool):
    """Execute Python code in a Google Colab runtime via colab-mcp.

    Requires the colab-mcp server to be running (see .mcp.json) or
    set COLAB_MCP_URL to an HTTP/SSE endpoint.
    """

    name: str = "Colab Execute"
    description: str = (
        "Execute Python code in a Google Colab runtime with GPU/CPU access. "
        "Returns stdout, stderr, and any cell output as JSON. "
        "Use this to run experiments, validate hypotheses with code, or benchmark algorithms. "
        "Input: Python source code as a string."
    )

    def _run(self, code: str, timeout: int = 60) -> str:  # type: ignore[override]
        return _run_async(_call_colab("execute_code", {"code": code, "timeout": timeout}))


class ColabInstallTool(BaseTool):
    """Install a Python package into the active Colab runtime."""

    name: str = "Colab Install Package"
    description: str = (
        "Install a Python package into the active Google Colab runtime via pip. "
        "Call this before running code that requires non-standard libraries. "
        "Input: package name (e.g. 'torch', 'scikit-learn==1.5.0')."
    )

    def _run(self, package: str) -> str:  # type: ignore[override]
        return _run_async(_call_colab("install_package", {"package": package}))


class ColabRuntimeTool(BaseTool):
    """Query the status of the connected Google Colab runtime."""

    name: str = "Colab Runtime Info"
    description: str = (
        "Get information about the current Google Colab runtime: "
        "accelerator type (GPU/TPU/CPU), available RAM, disk usage, and connection status. "
        "Input: any string (ignored)."
    )

    def _run(self, input: str = "") -> str:  # type: ignore[override]
        return _run_async(_call_colab("get_runtime_info", {}))
