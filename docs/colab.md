# Google Colab Integration

ResearchCrew connects to Google Colab runtimes through the
[colab-mcp](https://github.com/googlecolab/colab-mcp) Model Context Protocol server.
This lets agents execute Python code with GPU/TPU access directly from a crew run,
closing the loop between hypothesis generation and empirical validation.

---

## How It Works

Three CrewAI tools in `researchcrew.tools.colab_tool` communicate with the colab-mcp
server over the MCP protocol via `fastmcp.Client`:

| Tool | MCP call | Purpose |
|---|---|---|
| `ColabExecuteTool` | `execute_code` | Run Python code, capture stdout/stderr/output |
| `ColabInstallTool` | `install_package` | `pip install` a package into the runtime |
| `ColabRuntimeTool` | `get_runtime_info` | Query GPU type, RAM, connection status |

The tools connect to the server in two modes:

- **Subprocess (default):** `uvx git+https://github.com/googlecolab/colab-mcp` is launched
  automatically as a stdio MCP server — no manual start required.
- **HTTP/SSE:** set `COLAB_MCP_URL=http://host:port` to point at a server that is already
  running (e.g. behind a reverse proxy or on a remote machine).

---

## Quick Setup

### 1. Install the extra

```bash
pip install 'researchcrew[colab]'
```

This adds `fastmcp>=3.0` (the MCP client). The colab-mcp server itself is fetched
automatically via `uvx` at connection time, so no separate install step is needed.

### 2. Ensure `uvx` is available

```bash
pip install uv          # installs uvx
uvx --version           # verify
```

### 3. Authenticate with Google Colab

The colab-mcp server requires a live Colab session. Open a browser tab to
[colab.research.google.com](https://colab.research.google.com), start a runtime, then
follow the colab-mcp authentication flow (see its README for details).

---

## Auto-connect via `.mcp.json`

The repository ships `.mcp.json` at its root. Claude Code and Claude Desktop read this
file and connect to both servers automatically:

```json
{
  "mcpServers": {
    "researchcrew-papers": {
      "command": "python",
      "args": ["-m", "researchcrew.mcp.papers"]
    },
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

No further configuration is needed for Claude Code users — the tools appear in the
MCP tool list automatically once the servers start.

---

## Using Colab Tools Directly

```python
from researchcrew.tools import ColabExecuteTool, ColabInstallTool, ColabRuntimeTool

execute = ColabExecuteTool()
install = ColabInstallTool()
runtime = ColabRuntimeTool()

# Check what accelerator is available
print(runtime._run())

# Install a package
print(install._run("torch"))

# Run an experiment
code = """
import torch
x = torch.randn(1000, 1000).cuda()
print(f"Device: {x.device}, Shape: {x.shape}")
"""
print(execute._run(code, timeout=30))
```

---

## Using Colab in the Co-Scientist Crew

Pass `use_colab=True` to `make_co_scientist_crew`. This wires all three Colab tools
into the **Evolution** agent, allowing it to run code experiments while refining
top-ranked hypotheses:

```python
from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.llm import create_llm

llm  = create_llm("anthropic")
crew = make_co_scientist_crew(
    llm=llm,
    verbose=True,
    max_tournament_rounds=3,
    use_colab=True,
)

result = crew.kickoff(inputs={
    "research_goal": "Discover novel training objectives for reasoning LLMs.",
    "domain":            "machine_learning",
    "max_hypotheses":    "8",
    "tournament_rounds": "3",
})
```

With Colab enabled, the Evolution agent can:
- Install evaluation libraries (`ColabInstallTool`)
- Run benchmark scripts to score evolved hypotheses (`ColabExecuteTool`)
- Verify GPU availability before scheduling heavy runs (`ColabRuntimeTool`)

---

## Using Colab in a Custom Crew

```python
from crewai import Agent, Crew, Process, Task
from researchcrew.tools import ColabExecuteTool, ColabInstallTool

execute = ColabExecuteTool()
install = ColabInstallTool()

experimenter = Agent(
    role="ML Experimenter",
    goal=(
        "Run benchmark experiments in Google Colab to validate ML hypotheses. "
        "Install required libraries, execute scripts, and report results."
    ),
    backstory="You are a hands-on ML engineer who validates ideas with code.",
    tools=[install, execute],
    llm=my_llm,
)

task = Task(
    description=(
        "Implement and benchmark the following algorithm on CIFAR-10: {hypothesis}. "
        "Report top-1 accuracy and training time."
    ),
    expected_output="Accuracy and timing results with a brief interpretation.",
    agent=experimenter,
)

crew = Crew(agents=[experimenter], tasks=[task], process=Process.sequential)
crew.kickoff(inputs={"hypothesis": "Focal loss with class-balanced sampling"})
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COLAB_MCP_URL` | _(empty)_ | If set, connect to this HTTP/SSE URL instead of launching a subprocess |

---

## Error Handling

If the colab-mcp server is unreachable, the tools return a JSON error dict rather
than raising an exception:

```json
{"error": "ConnectionRefusedError", "detail": "[Errno 111] Connection refused"}
```

If `fastmcp` is not installed, the tools return:

```json
{"error": "fastmcp not installed", "hint": "pip install researchcrew[colab]"}
```

Both cases allow the crew to continue running without crashing — the agent sees the
error message and can decide whether to retry, skip, or report back to the supervisor.

---

## Running Without a Live Runtime

In development or CI, set `COLAB_MCP_URL` to a mock server, or simply omit
`use_colab=True`. The tools have full unit-test coverage using mocked fastmcp
clients — no real Colab session is needed to run the test suite:

```bash
pytest tests/test_colab_tool.py   # 18 tests, no network
```
