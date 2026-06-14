# Agent Crews

ResearchCrew provides two pre-assembled CrewAI crews that cover the full research lifecycle.

---

## Co-Scientist Crew

A 7-agent hierarchical crew that implements the [AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) pattern: generate → reflect → deduplicate → rank → evolve → synthesise.

### Assembly

```python
from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.llm import create_llm

llm  = create_llm("anthropic")
crew = make_co_scientist_crew(
    llm=llm,
    verbose=True,
    max_tournament_rounds=3,
    use_colab=False,   # set True to give Evolution agent a live Colab runtime
)
```

### Running

```python
result = crew.kickoff(inputs={
    "research_goal":      "Identify drug repurposing candidates for AML.",
    "domain":             "biomedicine",
    "max_hypotheses":     "10",
    "tournament_rounds":  "3",
})
print(result)
```

### Agents

| Agent | Role | Tools |
|---|---|---|
| **Generation** | Search literature, ideate hypotheses | `LiteratureSearchTool`, `HypothesisStoreTool` |
| **Reflection** | Score novelty & feasibility | `HypothesisStoreTool` |
| **Proximity** | Semantic deduplication / clustering | `HypothesisStoreTool` |
| **Ranking** | Run Elo tournament debates | `DebateTool`, `HypothesisStoreTool` |
| **Evolution** | Improve top hypotheses | `HypothesisStoreTool`, `MemoryQueryTool` · `ColabExecuteTool`, `ColabInstallTool`, `ColabRuntimeTool` ¹ |
| **Meta-review** | Synthesise results, produce research brief | `MemoryQueryTool` |
| **Supervisor** | Hierarchical manager (not a task agent) | — |

¹ Colab tools are added only when `use_colab=True` is passed to `make_co_scientist_crew()`. See [colab.md](colab.md).

### Task Pipeline

```
Generation → Reflection → Proximity → Ranking → Evolution → Meta-review
```

Orchestrated with `Process.hierarchical` — the Supervisor agent routes tasks and resolves inter-agent disagreements.

---

## Paper Writing Crew

A 4-agent sequential crew that takes a research goal and hypothesis summary and produces a full draft paper.

### Assembly

```python
from researchcrew.crews.paper_writing import make_paper_writing_crew
from researchcrew.llm import create_llm

llm  = create_llm("anthropic")
crew = make_paper_writing_crew(
    llm=llm,
    verbose=True,
    conference_target="NeurIPS",
)
```

### Running

```python
result = crew.kickoff(inputs={
    "research_goal":   "Sparse attention for long-context document understanding.",
    "top_hypotheses":  "Sliding-window + global attention reduces O(n²) to O(n).",
    "corpus_summary":  "15 papers retrieved from arXiv.",
})
```

### Agents

| Agent | Role | Tools |
|---|---|---|
| **Literature** | Two-phase citation search | `LiteratureSearchTool` |
| **Outline** | JSON paper structure + section plan | `MemoryQueryTool` |
| **Writer** | Section-by-section drafting | — |
| **Editor** | Peer review simulation + targeted rewrites | — |

### Task Pipeline

```
Literature → Outline → Writer → Editor
```

Uses `Process.sequential`. The editor scores each section on clarity, correctness, novelty, and citation quality (0–10), then rewrites any section scoring below 7.

---

## Combining Both Crews

A common pattern is to run Co-Scientist first, then feed its output into the Paper Writing crew:

```python
from researchcrew.crews.co_scientist import make_co_scientist_crew
from researchcrew.crews.paper_writing import make_paper_writing_crew
from researchcrew.llm import create_llm

llm = create_llm("anthropic")

# Phase 1: hypothesis tournament
scientist_crew = make_co_scientist_crew(llm=llm)
hypotheses = scientist_crew.kickoff(inputs={
    "research_goal":     "Novel training objectives for reasoning LLMs.",
    "domain":            "machine_learning",
    "max_hypotheses":    "8",
    "tournament_rounds": "2",
})

# Phase 2: write the paper
writing_crew = make_paper_writing_crew(llm=llm, conference_target="ICLR")
paper = writing_crew.kickoff(inputs={
    "research_goal":   "Novel training objectives for reasoning LLMs.",
    "top_hypotheses":  str(hypotheses),
    "corpus_summary":  "20 papers from arXiv and Semantic Scholar.",
})

print(paper)
```

---

## CrewAI Tools

The crews use seven CrewAI-native tool wrappers:

| Tool | Class | Description |
|---|---|---|
| `LiteratureSearchTool` | `researchcrew.tools.literature_search` | Searches arXiv via the literature module |
| `HypothesisStoreTool` | `researchcrew.tools.hypothesis_store` | In-memory hypothesis CRUD |
| `DebateTool` | `researchcrew.tools.debate_tool` | Runs pairwise Elo debates |
| `MemoryQueryTool` | `researchcrew.tools.memory_query` | Retrieves from the knowledge store |
| `ColabExecuteTool` | `researchcrew.tools.colab_tool` | Execute Python code in a Colab runtime |
| `ColabInstallTool` | `researchcrew.tools.colab_tool` | Install packages into the Colab runtime |
| `ColabRuntimeTool` | `researchcrew.tools.colab_tool` | Query Colab runtime status (GPU/RAM) |

These can be used individually in custom crews:

```python
from researchcrew.tools import LiteratureSearchTool, HypothesisStoreTool, ColabExecuteTool

search  = LiteratureSearchTool()
store   = HypothesisStoreTool()
execute = ColabExecuteTool()

my_agent = Agent(
    role="Hypothesis Generator",
    goal="Generate and immediately test novel hypotheses with code.",
    tools=[search, store, execute],
    llm=my_llm,
)
```

The three Colab tools require the `colab-mcp` server to be running. See [colab.md](colab.md) for setup instructions.
