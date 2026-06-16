# Changelog

All notable changes to ResearchCrew are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-06-16

First public alpha release.

### Added

#### Core Framework
- `ResearchLLM` ABC — provider-agnostic LLM interface with `complete(prompt, *, json_mode)` and injection-point factory methods
- `AnthropicResearchLLM` — Claude integration with native adaptive thinking (`thinking: {type: "adaptive"}`), auto-disabled for Haiku
- `OpenAIResearchLLM` — GPT integration with JSON mode support
- `GoogleResearchLLM` — Gemini integration via `google-genai` SDK with `response_mime_type` JSON control
- `create_llm(provider, **kwargs)` factory — single entry point for all three providers
- `LLMDebateJudge` — LLM-powered pairwise hypothesis evaluator with confidence scoring

#### Co-Scientist Crew
- `make_co_scientist_crew(llm, verbose, max_tournament_rounds, use_colab)` — 7-agent hierarchical CrewAI crew
- Generation agent — literature-grounded hypothesis ideation
- Reflection agent — novelty (0–1) and feasibility (0–1) scoring
- Proximity agent — embedding-based semantic deduplication and clustering
- Ranking agent — Elo tournament orchestration with `DebateTool`
- Evolution agent — top-5 hypothesis refinement with memory-guided improvement; Colab tools when `use_colab=True`
- Meta-review agent — research brief synthesis and final ranked hypothesis list
- Supervisor agent — hierarchical process manager

#### Paper Writing Crew
- `make_paper_writing_crew(llm, verbose, conference_target)` — 4-agent sequential CrewAI crew
- Literature agent — two-phase citation search (broad → rerank → assign to sections)
- Outline agent — JSON paper plan with `section_plan`, `visualization_plan`, `citation_strategy`
- Writer agent — dependency-ordered section drafting (Related Work → Methods → Results → Discussion → Introduction → Abstract → Conclusion)
- Editor agent — simulated peer review with per-section scores (clarity, correctness, novelty, citation quality 0–10); targeted rewrites for sections below 7

#### Tournament Engine
- `EloRating` — Elo update with `K_eff = K_base × confidence`
- `DebateEngine` — pairwise debate orchestrator
- `ProximityFilter` — cosine-similarity clustering
- `Tournament` — full tournament with progressive bottom-30% elimination

#### Search (Progressive MCGS)
- `ProgressiveMCGS` — Monte Carlo Graph Search over hypothesis DAG
- `SearchNode` — node with visit count, cumulative reward, Q-value
- `ExplorationScheduler` — UCB constant decay from `C_max` → `C_min` over configurable window
- `HypothesisFusion` — multi-strategy search fusion

#### Literature Pipeline
- `MultiSourceSearcher` — concurrent search across arXiv, Semantic Scholar, PubMed, OpenAlex
- `LiteratureCorpus` — paper collection with deduplication
- `TwoPhaseScreener` — LLM-based relevance scoring and section assignment
- `CitationGraph` — citation network analysis
- `ResearchGapAnalyzer` — automatic gap identification from corpus

#### Writing Pipeline
- `OutlineBuilder` — venue-aware JSON paper outline (NeurIPS, ACL, Nature, etc.)
- `SectionWriter` — per-section draft generation
- `RefinementEngine` — recursive criticism and improvement (RCI) loop
- `CitationPipeline` — inline citation placeholder insertion
- `ReferenceFormatter` — bibliography formatting
- `PaperExporter` — Markdown and LaTeX export

#### Memory & Retrieval
- `MemoryStore` — persistent `MemoryEntry` store with `OutcomeLabel` (POSITIVE/NEGATIVE/NEUTRAL)
- `EnsembleRetriever` — BM25 (lexical) + FAISS (semantic) hybrid retrieval with configurable α weight
- `KnowledgeBase` — high-level retrieval API

#### MCP Paper Search Server
- `researchcrew-papers` FastMCP stdio server with five tools:
  - `search_papers` — multi-source search with cross-source deduplication
  - `fetch_paper` — full metadata by arXiv ID, DOI, or S2 ID
  - `get_citations` — forward citation lookup
  - `get_references` — backward reference lookup
  - `find_related` — Semantic Scholar recommendation-based related paper discovery
- Dual-key deduplication: canonical ID dict + version-stripped arXiv ID set
- OpenAlex inverted-index abstract reconstruction

#### Google Colab Integration
- `ColabExecuteTool` — execute Python code in a Colab GPU/TPU runtime via `colab-mcp`
- `ColabInstallTool` — `pip install` packages into the live runtime
- `ColabRuntimeTool` — query accelerator type, RAM, and connection status
- `.mcp.json` — project-level MCP config auto-connecting `researchcrew-papers` + `colab-mcp`
- Supports stdio subprocess (via `uvx`) and HTTP/SSE (`COLAB_MCP_URL` env var)

#### TUI Chat Interface (`researchcrew-chat`)
- `Project` — persistent research project (meta.json, chat.jsonl, draft/, code/)
- Rich/prompt_toolkit REPL with streaming Claude responses
- 16 slash commands: `/new`, `/open`, `/list`, `/status`, `/papers`, `/outline`, `/draft`, `/code`, `/save`, `/export`, `/save-section`, `/add-paper`, `/save-code`, `/help`, `/quit`, `/exit`
- Auto-detection: code blocks (≥5 lines) saved as artifacts; long prose saved as sections

#### Project Lifecycle & Governance
- `ProjectLifecycle` — state-machine: IDEATION → LITERATURE_REVIEW → HYPOTHESIS_GENERATION → TOURNAMENT → WRITING → SUBMITTED
- `ProjectStore` — JSON-backed persistence
- `ResearchGate` — configurable governance gate for stage transitions
- `MetaReviewOptimizer` — CrewPilot meta-review prompt optimiser
- `HypothesisVariantBridge` — hypothesis ↔ CrewPilot variant bridge

#### Infrastructure
- `pyproject.toml` optional extras: `llm`, `mcp`, `colab`, `chat`, `literature`, `memory`, `writing`, `governance`, `all`
- CLI entry points: `researchcrew`, `researchcrew-chat`, `researchcrew-papers`
- 73+ unit tests across 6 test modules (0 network calls required)
- Ruff linting + mypy strict type checking

### Examples
- `examples/drug_repurposing.py` — Co-Scientist on AML epigenetic drug repurposing
- `examples/literature_synthesis.py` — Literature survey + paper writing crew on long-context attention
- `examples/ml_algo_discovery.py` — Progressive MCGS for tabular ML algorithm discovery
- `examples/hypothesis_to_paper.py` — Full end-to-end: hypothesis tournament → paper draft → Markdown export

---

[Unreleased]: https://github.com/bsfeal/researchcrew/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bsfeal/researchcrew/releases/tag/v0.1.0
