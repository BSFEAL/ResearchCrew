<p align="center">
  <img src="assets/logo.svg" alt="ResearchCrew" width="480" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python&logoColor=white" alt="Python Versions"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status"/>
</p>

<p align="center">
  <strong>A multi-agent scientific research library built on CrewAI.</strong><br/>
  Hypothesis generation · Literature review · Paper writing · MCP paper search · TUI chat interface.
</p>

---

## Overview

ResearchCrew is a Python library that automates the full scientific research lifecycle using coordinated AI agent crews. It implements two core research patterns:

- **Co-Scientist Crew** — a 7-agent tournament system that generates, debates, ranks, and evolves scientific hypotheses through Elo-rated pairwise competition.
- **Paper Writing Crew** — a 4-agent sequential pipeline that searches literature, outlines, drafts, and peer-reviews a full research paper.

Both crews accept any LLM through a unified `ResearchLLM` interface supporting Anthropic Claude, OpenAI GPT, and Google Gemini — all with a single `create_llm()` call.

Additional tools include a **FastMCP paper-search server** (arXiv, Semantic Scholar, PubMed, OpenAlex) and a **Rich/prompt_toolkit TUI chat interface** (`researchcrew-chat`) for managing research projects through natural language.

---

## Architecture

```
researchcrew/
│
├── crews/                     # Pre-built CrewAI crew assemblies
│   ├── co_scientist.py        # 7-agent hypothesis tournament (Co-Scientist)
│   └── paper_writing.py       # 4-agent literature → outline → draft → edit
│
├── agents/                    # Individual CrewAI agent factories
│   ├── generation.py          # Hypothesis ideation
│   ├── reflection.py          # Novelty & feasibility scoring
│   ├── proximity.py           # Semantic deduplication
│   ├── ranking.py             # Elo tournament runner
│   ├── evolution.py           # Hypothesis improvement
│   ├── meta_review.py         # Synthesis & research brief
│   ├── supervisor.py          # Hierarchical manager
│   ├── literature.py          # Citation search
│   ├── outline.py             # Paper structuring
│   ├── writer.py              # Section drafting
│   └── editor.py              # Peer review & rewriting
│
├── llm/                       # Provider-agnostic LLM layer
│   ├── base.py                # ResearchLLM ABC + injection-point factories
│   ├── anthropic_llm.py       # Claude (adaptive thinking, streaming)
│   ├── openai_llm.py          # GPT-4o
│   ├── google_llm.py          # Gemini (google-genai SDK)
│   ├── providers.py           # create_llm() factory
│   └── judge.py               # LLMDebateJudge for tournament arbitration
│
├── tournament/                # Elo-based hypothesis ranking
│   ├── elo.py                 # Elo rating calculator
│   ├── debate.py              # Pairwise debate engine
│   ├── proximity.py           # Embedding-based clustering
│   └── tournament.py          # Full tournament orchestrator
│
├── search/                    # Progressive MCGS exploration
│   ├── mcgs.py                # Monte Carlo Graph Search
│   ├── node.py                # SearchNode with Q-values
│   ├── scheduler.py           # UCB exploration scheduler
│   └── fusion.py              # Multi-strategy search fusion
│
├── literature/                # Literature review pipeline
│   ├── corpus.py              # Paper collection & storage
│   ├── searcher.py            # Multi-source search (arXiv, S2, PubMed)
│   ├── screener.py            # Two-phase relevance screening
│   ├── citation_graph.py      # Citation network analysis
│   └── gap_analyzer.py        # Research gap identification
│
├── writing/                   # Paper generation pipeline
│   ├── outline.py             # Structured outline builder
│   ├── section_writer.py      # Section-by-section drafting
│   ├── refinement.py          # Peer review simulation & rewriting
│   ├── citation_pipe.py       # Citation insertion pipeline
│   ├── references.py          # Reference formatting
│   ├── exporter.py            # Markdown / LaTeX export
│   └── models.py              # PaperDraft, Outline, Section, SectionStatus
│
├── memory/                    # Persistent knowledge store
│   ├── store.py               # In-process MemoryStore
│   ├── entry.py               # MemoryEntry with outcome labels
│   ├── retriever.py           # BM25 / FAISS / ensemble retrieval
│   └── knowledge_base.py      # High-level KnowledgeBase API
│
├── models/                    # Core domain models (Pydantic)
│   ├── hypothesis.py          # Hypothesis + HypothesisStatus
│   ├── research_goal.py       # ResearchGoal + Domain + Constraint
│   ├── tournament_state.py    # TournamentState + TournamentPhase
│   └── debate_record.py       # DebateRecord + DebateTurn + DebateVerdict
│
├── project/                   # Project lifecycle management
│   ├── project.py             # ResearchProject + ProjectStatus
│   ├── lifecycle.py           # State-machine transitions
│   └── store.py               # JSON-backed project store
│
├── governance/                # CrewPilot integration hooks
│   ├── research_gate.py       # Gate policy for stage transitions
│   ├── meta_optimizer.py      # Meta-review based prompt optimization
│   └── variant_bridge.py      # Hypothesis ↔ CrewPilot variant bridge
│
├── mcp/
│   └── papers.py              # FastMCP server — 5 paper-search tools
│
└── tools/
    ├── chat.py                # researchcrew-chat TUI
    ├── literature_search.py   # CrewAI LiteratureSearchTool
    ├── hypothesis_store.py    # CrewAI HypothesisStoreTool
    ├── debate_tool.py         # CrewAI DebateTool
    └── memory_query.py        # CrewAI MemoryQueryTool
```

---

## Installation

```bash
# Core only (no LLM providers, no network deps)
pip install researchcrew

# With Anthropic Claude (recommended)
pip install 'researchcrew[llm]'

# With MCP paper search server
pip install 'researchcrew[mcp]'

# With the TUI chat interface
pip install 'researchcrew[chat]'

# Everything
pip install 'researchcrew[all]'
```

**Python 3.10 – 3.13** is supported.

---

## Quick Start

### 1. Run the Co-Scientist Crew

```python
from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.llm import create_llm

llm = create_llm("anthropic")  # uses ANTHROPIC_API_KEY from env

crew = make_co_scientist_crew(llm=llm, verbose=True)
result = crew.kickoff(inputs={
    "research_goal": (
        "Identify drug repurposing candidates for AML by targeting "
        "epigenetic regulators validated in liver fibrosis models."
    ),
    "domain": "biomedicine",
    "max_hypotheses": "10",
    "tournament_rounds": "3",
})
print(result)
```

### 2. Run the Paper Writing Crew

```python
from researchcrew.crews.paper_writing import make_paper_writing_crew
from researchcrew.llm import create_llm

llm = create_llm("openai", model="gpt-4o")

crew = make_paper_writing_crew(llm=llm, conference_target="NeurIPS")
result = crew.kickoff(inputs={
    "research_goal": "Sparse attention mechanisms for long-context transformers",
    "top_hypotheses": "Sliding-window + global attention reduces O(n²) to O(n).",
    "corpus_summary": "15 papers retrieved from arXiv.",
})
print(result)
```

### 3. Start the TUI Chat

```bash
export ANTHROPIC_API_KEY=sk-ant-...

researchcrew-chat              # resume last project or start fresh
researchcrew-chat new          # guided new-project wizard
researchcrew-chat open <id>    # open a specific project
researchcrew-chat list         # list all projects
```

### 4. Use the MCP Paper Search Server

```bash
# Start the stdio MCP server (connect via any MCP client)
researchcrew-papers

# Smoke-test against live APIs
python -m researchcrew.mcp.papers --test
```

Or call the tools directly from Python:

```python
from researchcrew.mcp.papers import search_papers, fetch_paper, get_citations

papers = search_papers("attention transformer", max_results=10, year_from=2020)
paper  = fetch_paper("1706.03762")          # arXiv ID
cites  = get_citations("1706.03762", limit=20)
```

---

## LLM Providers

All components accept an optional `llm` argument. Pass `None` to use the CrewAI default, or supply a `ResearchLLM` instance for full control.

```python
from researchcrew.llm import create_llm

# Anthropic Claude — adaptive thinking enabled by default on Opus/Sonnet
anthropic_llm = create_llm("anthropic",
    model="claude-opus-4-8",     # default
    max_tokens=4096,
    thinking=True,               # adaptive thinking (auto-disabled for Haiku)
)

# OpenAI GPT
openai_llm = create_llm("openai",
    model="gpt-4o",              # default
    max_tokens=4096,
)

# Google Gemini
gemini_llm = create_llm("google",
    model="gemini-2.0-flash",    # default
    max_tokens=4096,
)
```

### Custom Provider

Subclass `ResearchLLM` and implement `complete()` — all injection-point factories are inherited automatically:

```python
from researchcrew.llm.base import ResearchLLM

class MyLLM(ResearchLLM):
    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        # call your API here
        return my_api.generate(prompt)

crew = make_co_scientist_crew(llm=MyLLM())
```

---

## MCP Paper Search Tools

The `researchcrew-papers` MCP server exposes five tools:

| Tool | Description |
|---|---|
| `search_papers` | Search arXiv, Semantic Scholar, PubMed, OpenAlex with dedup |
| `fetch_paper` | Full metadata by arXiv ID, DOI, or Semantic Scholar ID |
| `get_citations` | Forward citations (papers that cite a given paper) |
| `get_references` | Backward citations (papers referenced by a given paper) |
| `find_related` | Semantically related papers via S2 recommendations |

All tools return a list of paper dicts with: `title`, `abstract`, `authors`, `year`, `doi`, `arxiv_id`, `s2_id`, `citation_count`, `url`, `source`.

**Deduplication** is performed across sources using a dual-key index (canonical DOI/arXiv/S2 ID + version-stripped arXiv ID) to prevent the same paper appearing twice from different sources.

---

## TUI Chat Interface

`researchcrew-chat` is a Rich/prompt_toolkit terminal app for managing research projects through natural language conversation.

```
researchcrew chat — AI research companion
────────────────────────────────────────
Project:  Attention Mechanisms Survey  [exploring]
Goal:     Long-context transformer attention…
────────────────────────────────────────
Type your message or /help for commands.
```

### Slash Commands

| Command | Description |
|---|---|
| `/new [title]` | Create a new research project |
| `/open <id>` | Open an existing project |
| `/list` | List all projects |
| `/status` | Project overview (title, status, sections, papers) |
| `/papers` | List collected related works |
| `/outline` | Show the paper outline |
| `/draft [section]` | Show full draft or one named section |
| `/code` | List saved code artifacts |
| `/save` | Force-save current state |
| `/export [path]` | Write publication-ready Markdown |
| `/save-section <name>` | Save last response as a paper section |
| `/add-paper <title>` | Add a paper to related works (interactive) |
| `/save-code <name>` | Save last code block as a code artifact |
| `/help` | Show command reference |
| `/quit` or `/exit` | Exit (auto-saves) |

### Project Persistence

Each project is stored under `~/.researchcrew/projects/<id>/`:

```
<id>/
├── meta.json        # Title, goal, status, keywords, related works, outline
├── chat.jsonl       # Full conversation history (one message per line)
├── draft/           # Markdown files — one per named section
│   ├── abstract.md
│   ├── introduction.md
│   └── ...
└── code/            # Saved code artifacts
    ├── model.py
    └── experiment.sh
```

Run `/export` at any time to assemble a submission-ready Markdown document with sections in standard academic order (Abstract → Introduction → Related Work → Methodology → Results → Discussion → Conclusion → References).

---

## Project Lifecycle

ResearchCrew tracks a project through six phases:

```
IDEATION → LITERATURE_REVIEW → HYPOTHESIS_GENERATION → TOURNAMENT → WRITING → SUBMITTED
```

```python
from researchcrew.project.lifecycle import ProjectLifecycle
from researchcrew.project.project import ResearchProject
from researchcrew.project.store import ProjectStore

store = ProjectStore()
project = store.create(ResearchProject(
    name="My Study",
    domain="machine_learning",
    description="...",
))

lc = ProjectLifecycle(project)
lc.advance()   # IDEATION → LITERATURE_REVIEW
lc.advance()   # LITERATURE_REVIEW → HYPOTHESIS_GENERATION
store.save(project)
print(project.status)
```

---

## Tournament Engine

The Elo-based tournament ranks hypotheses through structured pairwise debates:

```python
from researchcrew.tournament.tournament import Tournament
from researchcrew.tournament.elo import EloRating
from researchcrew.models.hypothesis import Hypothesis

hypotheses = [
    Hypothesis(title="Idea A", body="...", domain="ml", elo_score=1200.0),
    Hypothesis(title="Idea B", body="...", domain="ml", elo_score=1200.0),
    Hypothesis(title="Idea C", body="...", domain="ml", elo_score=1200.0),
]

elo = EloRating(k_factor=32.0)
```

The `DebateTool` wraps this for use inside CrewAI agents. The `LLMDebateJudge` evaluates each debate turn and returns a verdict with reasoning.

---

## Memory & Retrieval

```python
from researchcrew.memory.store import MemoryStore
from researchcrew.memory.entry import MemoryEntry, OutcomeLabel
from researchcrew.memory.retriever import EnsembleRetriever

store = MemoryStore()
store.add(MemoryEntry(
    content="Sparse attention reduces complexity to O(n log n).",
    tags=["attention", "complexity"],
    outcome=OutcomeLabel.POSITIVE,
))

retriever = EnsembleRetriever(store)
results = retriever.retrieve("efficient attention mechanisms", top_k=5)
```

The `EnsembleRetriever` combines BM25 (lexical) and FAISS (semantic) retrieval with configurable weights. Requires `pip install 'researchcrew[memory]'`.

---

## Progressive MCGS Search

The `ProgressiveMCGS` implements Monte Carlo Graph Search for algorithm/hypothesis discovery, modelled on the MLEvolve paradigm:

```python
from researchcrew.search.mcgs import ProgressiveMCGS
from researchcrew.search.scheduler import ExplorationScheduler

scheduler = ExplorationScheduler(c_max=2.0, c_min=0.5, explore_until=2, decay_until=8)
graph = ProgressiveMCGS(scheduler=scheduler)

graph.add_root(root_hypothesis)

node = graph.select(graph.root_ids[0])
graph.backpropagate(node.id, score=0.85)

for node in graph.best_nodes(3):
    print(f"[{node.q_value:.3f}] {node.hypothesis.title}")
```

---

## Examples

| File | Description |
|---|---|
| `examples/drug_repurposing.py` | Co-Scientist crew on AML / liver fibrosis |
| `examples/literature_synthesis.py` | Literature survey + paper writing crew |
| `examples/ml_algo_discovery.py` | Progressive MCGS on a Kaggle-style benchmark |

Run any example (requires `pip install 'researchcrew[llm]'` and an API key):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/drug_repurposing.py
```

---

## Development

```bash
git clone https://github.com/bsfeal/researchcrew
cd researchcrew

pip install -e '.[all]'
pip install pytest respx

pytest tests/ -v
```

### Running Specific Test Suites

```bash
pytest tests/test_papers_mcp.py   # MCP paper search (30 tests, no network)
pytest tests/test_chat.py         # TUI chat helpers (25 tests)
pytest tests/test_elo.py          # Elo rating engine
pytest tests/test_tournament.py   # Debate & tournament
pytest tests/test_literature.py   # Literature pipeline
pytest tests/test_writing.py      # Paper writing pipeline
pytest tests/test_memory.py       # Memory & retrieval
pytest tests/test_search.py       # MCGS search
pytest tests/test_project.py      # Project lifecycle
pytest tests/test_llm.py          # LLM providers (unit-level)
```

---

## Optional Dependencies

| Extra | Packages | Purpose |
|---|---|---|
| `llm` | `anthropic`, `openai`, `google-genai` | LLM providers |
| `mcp` | `fastmcp`, `httpx` | MCP paper search server |
| `chat` | `anthropic`, `rich`, `prompt_toolkit` | TUI chat interface |
| `literature` | `httpx` | Live literature search |
| `memory` | `rank-bm25`, `faiss-cpu`, `sentence-transformers` | Semantic memory |
| `writing` | `jinja2` | Paper template rendering |
| `governance` | `crewpilot` | CrewPilot integration |
| `all` | all of the above | Full install |

---

## License

MIT © ResearchCrew Contributors
