# ResearchCrew: A Multi-Agent Framework for Autonomous Scientific Research

**Alan Ferrari** · K-Lab · alan.ferrari@k-lab.ch

**Version 0.1 · June 2026**

---

## Abstract

Scientific discovery increasingly depends on researchers' ability to synthesise vast bodies of literature, generate and evaluate hypotheses at scale, and produce high-quality written outputs — tasks that stretch human cognitive capacity. We introduce **ResearchCrew**, an open-source Python library that orchestrates multi-agent AI crews to automate the full scientific research lifecycle: literature search, hypothesis generation, competitive hypothesis ranking via Elo-rated tournament debate, hypothesis evolution, paper drafting, and simulated peer review. ResearchCrew is built on top of CrewAI and supports three major LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini) through a unified provider-agnostic interface with adaptive thinking. The system additionally exposes a FastMCP paper-search server covering arXiv, Semantic Scholar, PubMed, and OpenAlex, and provides a Rich/prompt\_toolkit terminal chat interface for interactive, project-aware research management. We describe the system architecture, the tournament engine, the Progressive Monte Carlo Graph Search mechanism, the memory subsystem, and the writing pipeline. We evaluate the design against three use cases — biomedical drug repurposing, ML algorithm discovery, and long-context attention mechanisms — and discuss limitations and future directions.

---

## 1. Introduction

The scientific enterprise has always been limited by the bandwidth of human researchers. A practitioner must simultaneously track thousands of papers across subfields, generate and critically assess hypotheses against existing evidence, and produce written communications that satisfy rigorous peer-review standards. The emergence of large language models (LLMs) capable of sophisticated reasoning, long-context synthesis, and structured generation has created a unique opportunity: to partially delegate these cognitive tasks to AI systems, accelerating the pace of discovery.

Prior work has explored LLMs in narrow research-adjacent roles: abstractive summarisation [CITATION], question answering over scientific corpora [CITATION], and automated paper generation [CITATION]. More recently, Google's AI Co-Scientist [CITATION] demonstrated a full hypothesis-generation pipeline with tournament-style ranking. AlphaFold [CITATION] showed that deep learning can solve problems considered intractable by classical methods, while MLEvolve-class systems [CITATION] apply Monte Carlo search to algorithm discovery.

ResearchCrew synthesises these advances into a single, modular, open-source library. Its core contributions are:

1. **A provider-agnostic LLM abstraction** with injection-point factory methods, supporting Anthropic Claude (with native adaptive thinking), OpenAI GPT, and Google Gemini.
2. **A Co-Scientist crew** implementing a 7-agent hierarchical pipeline: generation → reflection → semantic deduplication → Elo tournament → hypothesis evolution → meta-review synthesis.
3. **A Paper Writing crew** implementing a 4-agent sequential pipeline: literature search → outline generation → multi-section drafting → simulated peer review with targeted rewriting.
4. **A Progressive MCGS module** that applies Monte Carlo Graph Search to navigate hypothesis space, balancing exploration and exploitation via a UCB-derived scheduler.
5. **A FastMCP paper-search server** with cross-source deduplication across arXiv, Semantic Scholar, PubMed, and OpenAlex.
6. **An interactive TUI chat interface** (`researchcrew-chat`) for project-aware research management with persistent state.

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 describes the overall system architecture. Sections 4–8 detail each major subsystem. Section 9 presents three use cases. Section 10 discusses limitations and future work. Section 11 concludes.

---

## 2. Related Work

### 2.1 AI Co-Scientist Systems

The concept of an AI system that autonomously generates, evaluates, and refines scientific hypotheses was formalised by Gottweis et al. [CITATION] in the context of Google's AI Co-Scientist. Their system uses an LLM as an oracle for hypothesis scoring and employs tournament-style competition to surface high-quality ideas. ResearchCrew operationalises a closely related architecture within an open, composable framework, extending it with explicit Elo rating mechanics, embedding-based proximity filtering, and memory-augmented evolution.

AlphaFold 2 [CITATION] demonstrated that AI systems can solve grand-challenge scientific problems (protein structure prediction) with superhuman accuracy. ResearchCrew does not target a specific domain's prediction problem; instead, it targets the upstream ideation and writing processes that precede empirical validation.

### 2.2 Multi-Agent Research Frameworks

CrewAI [CITATION] and AutoGen [CITATION] established multi-agent frameworks where specialised LLM agents collaborate on complex tasks through structured role delegation and tool use. ResearchCrew builds directly on CrewAI's `Agent`, `Task`, and `Crew` primitives, contributing domain-specific agent factories and tool wrappers tuned for academic research workflows.

AgentLaboratory [CITATION] explored a multi-agent system for autonomous ML experiments, including data analysis and code generation. ResearchCrew complements this by focusing on the earlier stages (literature synthesis, hypothesis generation) and the later stage (paper writing), rather than experiment execution.

### 2.3 Elo Rating in AI Systems

Elo ratings originated in chess [CITATION] and have been applied to evaluate LLM outputs via preference learning [CITATION] and to rank model generations in RLHF pipelines [CITATION]. In ResearchCrew, Elo is applied differently: it ranks *hypotheses*, not model outputs. Each hypothesis accumulates rating changes through pairwise debate judgements, and the tournament produces a ranked leaderboard that guides which hypotheses to evolve and which to discard.

### 2.4 Monte Carlo Tree/Graph Search in Research

Monte Carlo Tree Search (MCTS) has been applied to mathematical proof search [CITATION], code synthesis [CITATION], and scientific hypothesis search [CITATION]. MLEvolve [CITATION] uses a graph variant (MCGS) where nodes represent algorithm variants and edges represent evolutionary steps. ResearchCrew's `ProgressiveMCGS` adapts this paradigm: nodes are hypotheses, expansion is driven by the evolution agent, and backpropagation reflects Elo score changes. A configurable UCB scheduler modulates the exploration constant over time.

### 2.5 Literature Retrieval and Screening

Automated literature review has a long history in biomedical NLP [CITATION]. Recent approaches combine dense retrieval (FAISS, bi-encoder models) with lexical search (BM25) in hybrid pipelines [CITATION]. ResearchCrew's `MultiSourceSearcher` queries four academic databases concurrently and applies a two-phase screening pipeline: broad keyword search followed by LLM-based relevance scoring.

### 2.6 Automated Paper Writing

Early work on automated scientific writing focused on data-to-text generation [CITATION]. More recent systems use LLMs to draft entire papers from structured outlines [CITATION]. ResearchCrew's Paper Writing crew follows a produce-then-review pattern: a Writer agent drafts all sections before an Editor agent conducts simulated peer review, scoring each section and applying targeted rewrites — an approach inspired by recursive criticism and improvement (RCI) [CITATION].

---

## 3. System Architecture

ResearchCrew is structured as a layered library. Figure 1 shows the high-level component diagram.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│   researchcrew-chat (TUI)   ·   researchcrew-papers (MCP)       │
│   researchcrew CLI          ·   Python API                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Crew Assemblies                               │
│        Co-Scientist Crew         Paper Writing Crew             │
│       (7 agents, hierarchical)  (4 agents, sequential)          │
└─────┬──────────────────────────────────────────────┬────────────┘
      │                                              │
┌─────▼──────────────┐              ┌───────────────▼────────────┐
│   Agent Layer      │              │    Writing Pipeline        │
│  generation        │              │  outline · section_writer  │
│  reflection        │              │  refinement · exporter     │
│  proximity         │              │  citation_pipe · references│
│  ranking           │              └───────────────────────────-┘
│  evolution         │
│  meta_review       │
│  supervisor        │
└─────┬──────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│                   Core Subsystems                               │
│  Tournament (Elo + Debate)   ·  MCGS Search                    │
│  Literature (Corpus + Screen) · Memory (BM25 + FAISS)          │
│  Project Lifecycle           ·  Governance (Gate + Optimizer)  │
└─────┬──────────────────────────────────────────────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────────┐
│                     LLM Provider Layer                          │
│   AnthropicResearchLLM   OpenAIResearchLLM   GoogleResearchLLM │
│   (adaptive thinking)    (json_mode)         (response_mime)    │
│              └───────────────────────────────┘                  │
│                      ResearchLLM (ABC)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Figure 1.** ResearchCrew component diagram.

The **LLM Provider Layer** abstracts provider-specific APIs behind a single `complete(prompt, *, json_mode)` method. Factory methods on `ResearchLLM` return callables shaped to match each component's injection point, decoupling crew components from provider implementation.

The **Core Subsystems** implement the domain logic: Elo-based tournament arbitration, Monte Carlo graph search, multi-source literature retrieval, BM25/FAISS memory retrieval, project state persistence, and CrewPilot governance hooks.

The **Agent Layer** wraps core subsystem operations in CrewAI agent factories. Each factory function returns a configured `crewai.Agent` with a role, goal, backstory, and pre-wired tools.

The **Crew Assemblies** compose agents and tasks into `crewai.Crew` objects with explicit process orchestration (hierarchical for Co-Scientist, sequential for Paper Writing).

**User interfaces** interact with the library through the Python API, the `researchcrew` CLI (Click-based), the `researchcrew-chat` TUI, and the `researchcrew-papers` MCP server.

---

## 4. Co-Scientist Crew

The Co-Scientist crew implements a six-stage pipeline inspired by Gottweis et al. [CITATION], extended with explicit Elo mechanics, semantic proximity filtering, and memory-augmented evolution.

### 4.1 Hypothesis Generation

The **Generation** agent takes a `research_goal` and `domain` as inputs, searches the literature using `LiteratureSearchTool`, and proposes an initial population of hypotheses. Each hypothesis is stored via `HypothesisStoreTool` as a `Hypothesis` object with fields: `title`, `body`, `domain`, `elo_score` (initialised to 1200), `novelty_score`, `feasibility_score`, and `status`.

The agent is prompted to generate maximally diverse hypotheses, drawing on different mechanistic pathways, experimental paradigms, and theoretical frameworks. Diversity at initialisation is critical: a homogeneous population leads to low-quality tournaments with little discriminative power.

### 4.2 Reflection

The **Reflection** agent reviews each hypothesis independently and assigns:
- `novelty_score ∈ [0, 1]` — how much the hypothesis diverges from prior art
- `feasibility_score ∈ [0, 1]` — whether the hypothesis can be tested with realistic resources and timelines

Reflection scores do not directly affect Elo rankings but are used downstream by the Evolution agent to target improvement efforts at the weakest dimensions.

### 4.3 Proximity-Based Deduplication

Semantic redundancy in the hypothesis population reduces tournament discriminability and wastes compute. The **Proximity** agent clusters the population using cosine similarity over LLM-generated embeddings. Within each cluster, only the highest-scoring representative (by novelty × feasibility) is retained. This step reduces a typical population of 10 hypotheses to 6–8 unique directions before the tournament begins.

### 4.4 Elo Tournament

The tournament is the core contribution of the Co-Scientist pipeline. The **Ranking** agent orchestrates `tournament_rounds` rounds of pairwise Elo debates using the `DebateTool`.

For each pair (H₁, H₂), the `LLMDebateJudge` evaluates the matchup on four criteria:
1. Scientific novelty relative to the research goal
2. Methodological feasibility
3. Potential impact if validated
4. Clarity of the testable prediction

The judge outputs a verdict (`H1_WINS`, `H2_WINS`, or `DRAW`) with a confidence score. The confidence modulates the K-factor applied in the Elo update:

```
K_eff = K_base × confidence                where K_base = 32
E₁ = 1 / (1 + 10^((R₂ − R₁) / 400))
R₁' = R₁ + K_eff × (S₁ − E₁)
```

After each round, the bottom 30% of hypotheses by Elo score are eliminated, and matchups for the next round are drawn from the survivors. This progressive elimination improves tournament efficiency: later rounds feature stronger hypotheses with more discriminative matchups.

**Table 1** shows the schedule for a three-round tournament with an initial population of 8 hypotheses:

| Round | Population | Matchups | Eliminated |
|-------|-----------|----------|------------|
| 1     | 8         | 8        | 2–3        |
| 2     | 5–6       | 5–6      | 1–2        |
| 3     | 4–5       | 4–5      | 1–2        |
| Final | 3         | —        | —          |

### 4.5 Evolution

The **Evolution** agent takes the top 5 hypotheses from the tournament leaderboard and generates an improved variant of each. Improvement is guided by:
- The reflection scores (target the weakest dimension: novelty or feasibility)
- Adjacent hypotheses (incorporate the strongest elements of related ideas)
- Historical memory (retrieve similar hypotheses from past projects via `MemoryQueryTool`)

Evolution produces `Hypothesis` objects with `parent_id` set to the original, enabling lineage tracking through the `ProgressiveMCGS` graph.

### 4.6 Meta-Review Synthesis

The **Meta-review** agent synthesises all tournament results, reflection scores, and evolved variants into a concise research brief (≤ 500 words) and a final ranked hypothesis list. The brief captures: what worked, what failed, what the next iteration should focus on, and how the current top hypotheses relate to each other mechanistically.

---

## 5. Paper Writing Crew

The Paper Writing crew takes the output of the Co-Scientist (or any hypothesis list) and produces a complete draft paper through four sequential agents.

### 5.1 Literature Agent

The **Literature** agent performs a two-phase citation search:

**Phase 1 (broad):** Query `LiteratureSearchTool` with multiple keyword permutations derived from the research goal. Collect papers across sources, deduplicating by DOI/arXiv ID. Target: 20–50 papers.

**Phase 2 (rerank):** Score each paper for relevance to the research goal using the LLM screener. Filter to the top 15–25 papers. Assign each paper to one or more draft sections (introduction, related work, methods, etc.) based on its abstract.

The output is a structured bibliography: a JSON list of `{title, year, doi, authors, relevance_note, section_assignments}` objects.

### 5.2 Outline Agent

The **Outline** agent produces a JSON outline with three components:

```json
{
  "section_plan": [
    {"name": "Abstract",      "description": "...", "estimated_words": 250},
    {"name": "Introduction",  "description": "...", "estimated_words": 800},
    ...
  ],
  "visualization_plan": [
    {"figure_type": "architecture_diagram", "caption": "..."},
    ...
  ],
  "citation_strategy": "Use [Author, Year] placeholders; dense in Related Work, selective in Methods."
}
```

The outline is tailored to the `conference_target` venue (e.g., NeurIPS, ACL, Nature) which governs expected section lengths, writing style, and figure conventions.

### 5.3 Writer Agent

The **Writer** agent drafts all sections in dependency order:

```
Related Work → Methods → Results → Discussion → Introduction → Abstract → Conclusion
```

This ordering reflects information dependencies: the introduction and abstract are most effectively written after the results and discussion are clear. Each section receives a context dict containing: the outline entry, assigned citations, the full outline (for cross-referencing), and any relevant hypotheses.

Sections are stored in the `PaperDraft` object as `Section` instances with `name`, `content`, `word_count`, and `status` (DRAFT).

### 5.4 Editor Agent (Simulated Peer Review)

The **Editor** agent conducts simulated peer review. For each section, it produces a `SectionScore` with four dimensions scored 0–10:

| Dimension | What is assessed |
|---|---|
| `clarity` | Readability, structure, precision of language |
| `correctness` | Factual accuracy, logical consistency |
| `novelty` | Originality of contribution relative to cited work |
| `citation_quality` | Appropriateness and completeness of citations |

Any section scoring below 7 on any dimension is rewritten. The rewrite prompt specifies the section's `weakest_dimension` as the primary target, preventing diffuse rewrites that improve all dimensions superficially. After rewriting, the section is re-scored to verify improvement. Sections that still score below threshold after two rewrite passes are flagged for human review.

---

## 6. Progressive MCGS

The `ProgressiveMCGS` module implements Monte Carlo Graph Search over the hypothesis space, enabling systematic exploration of the idea landscape beyond what a single tournament cycle covers.

### 6.1 Graph Structure

The MCGS graph is a directed acyclic graph (DAG) where:
- **Nodes** (`SearchNode`) represent individual hypotheses, annotated with visit count `n`, cumulative reward `Q`, and Q-value `q = Q / n`.
- **Edges** represent evolutionary derivation (child hypothesis evolved from parent).
- **Root nodes** are seed hypotheses provided by the user or the generation agent.

### 6.2 Selection

Selection uses Upper Confidence Bound (UCB1) adapted for graph search:

```
UCB(v) = q(v) + C × sqrt(ln(N) / n(v))
```

where `N` is the total number of visits across all nodes at the same depth, `n(v)` is the visit count of node `v`, and `C` is the exploration constant controlled by the `ExplorationScheduler`.

### 6.3 Exploration Scheduling

The `ExplorationScheduler` implements a decay schedule for `C`:

```
C(t) = C_max      if t ≤ explore_until
C(t) = C_max × exp(−λ(t − explore_until))   if explore_until < t ≤ decay_until
C(t) = C_min      if t > decay_until
```

Early iterations (high `C`) favour exploration: selecting novel, less-visited hypotheses. Later iterations (low `C`) favour exploitation: refining the best-performing branches. Parameters `C_max`, `C_min`, `explore_until`, `decay_until` and `λ` are configurable.

### 6.4 Expansion and Backpropagation

When a node is selected:
1. The **Evolution** agent generates a child hypothesis.
2. The child is scored (via Elo tournament or heuristic) to produce a `reward ∈ [0, 1]`.
3. Backpropagation updates `Q` and `n` along the path from the child to all ancestor roots.

The graph retains the full derivation history, allowing researchers to inspect the evolutionary trajectory of any top hypothesis.

---

## 7. Memory and Retrieval

ResearchCrew includes a persistent `MemoryStore` that accumulates `MemoryEntry` objects across research sessions. Each entry stores:

| Field | Description |
|---|---|
| `content` | The text to be indexed (hypothesis body, section text, etc.) |
| `tags` | Keyword tags for BM25 augmentation |
| `outcome` | `OutcomeLabel`: POSITIVE, NEGATIVE, NEUTRAL |
| `metadata` | Arbitrary JSON |
| `timestamp` | UTC creation time |

The `EnsembleRetriever` combines two retrieval strategies:

**BM25 (lexical):** Uses the `rank-bm25` library. Tokenises entries on whitespace and punctuation. Scores queries by TF-IDF-weighted term overlap. Fast, interpretable, no GPU required.

**FAISS (semantic):** Encodes entries with a sentence-transformer model. Computes cosine similarity in the embedding space. Captures semantic relationships invisible to lexical search (synonymy, paraphrase).

The ensemble score is a weighted average:

```
score(e, q) = α × score_BM25(e, q) + (1−α) × score_FAISS(e, q)
```

where `α ∈ [0, 1]` is configurable (default 0.5). The `MemoryQueryTool` exposes this retriever to CrewAI agents, enabling evolution and meta-review agents to learn from past project cycles.

---

## 8. MCP Paper Search Server

The `researchcrew-papers` MCP server provides five tools over the Model Context Protocol (MCP), enabling any MCP-compatible client (Claude Desktop, Cursor, Zed, or custom) to perform academic paper operations.

### 8.1 Source Coverage

| Source | API | Strengths |
|---|---|---|
| arXiv | Atom feed (no key) | CS, math, physics; preprints |
| Semantic Scholar | REST (S2 Graph API) | Citation counts, semantic similarity, recommendations |
| PubMed | NCBI eUtils | Biomedical literature; MeSH indexing |
| OpenAlex | REST (polite pool) | Open access; inverted-index abstracts |

### 8.2 Cross-Source Deduplication

A recurring challenge in multi-source search is that the same paper can appear under different identifiers on different platforms. For example, "Attention Is All You Need" (arXiv:1706.03762) may appear on:
- arXiv as `1706.03762v5` (with version suffix)
- Semantic Scholar with DOI `10.48550/arXiv.1706.03762`
- OpenAlex with its own work ID

ResearchCrew uses a two-key deduplication scheme:

1. **Canonical key** `= DOI ?? arXiv_ID ?? S2_ID ?? title`. The first non-null value is used as the dict key.
2. **arXiv secondary index**: a `set` of version-stripped arXiv IDs. When a paper's `arxiv_id` is already in the set, the paper is skipped — regardless of its canonical key. Version stripping (`1706.03762v5` → `1706.03762`) is applied at parse time during arXiv Atom feed ingestion.

This scheme correctly handles the case where S2's canonical key is a DOI but arXiv's canonical key is the bare arXiv ID for the same paper.

### 8.3 OpenAlex Inverted-Index Reconstruction

OpenAlex stores paper abstracts as *inverted indices*: a mapping from word to a list of positions. Reconstruction is `O(W log W)` where `W` is the vocabulary size:

```python
def _reconstruct_abstract(inv: dict[str, list[int]]) -> str:
    max_pos = max(pos for positions in inv.values() for pos in positions)
    words   = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            words[pos] = word
    return " ".join(w for w in words if w)
```

---

## 9. Use Cases

### 9.1 Biomedical Drug Repurposing

**Goal:** Identify existing approved drugs for acute myeloid leukaemia (AML) by targeting epigenetic regulators validated in liver fibrosis models.

**Workflow:**
1. The Generation agent searches PubMed and Semantic Scholar for AML + epigenetics + liver fibrosis, retrieving 24 papers.
2. 10 initial hypotheses are generated, covering: HDAC inhibitors, BET bromodomain inhibitors, DNMT inhibitors, EZH2 inhibitors, and LSD1/KDM1A inhibitors.
3. Reflection scores novelty 0.6–0.85 (most hypotheses have partial prior evidence) and feasibility 0.5–0.9 (some target tissue-specificity concerns).
4. Proximity deduplication removes 2 redundant HDAC-focused hypotheses.
5. A 3-round tournament produces a leaderboard with BET bromodomain inhibitors (JQ1 / iBET-762) at the top, followed by LSD1/KDM1A and EZH2 inhibitors.
6. Evolution generates 5 refined variants incorporating cross-domain mechanistic insights.
7. The meta-review brief recommends prioritising JQ1-derived analogues targeting BRD4, given highest combined novelty × feasibility × Elo, and flags the need for AML-specific xenograft validation data.

**Output:** A ranked hypothesis list with 8 unique drug targets, each with mechanistic justification, reflection scores, Elo rating, and a meta-review summary.

### 9.2 ML Algorithm Discovery

**Goal:** Discover novel algorithms for tabular data classification using Progressive MCGS.

**Workflow:**
1. Three root hypotheses are seeded: gradient boosting with focal loss, TabPFN with test-time augmentation, and stacked ensembles with OOF meta-features.
2. 10 MCGS iterations are run with `C_max=2.0`, `C_min=0.5`, `explore_until=2`, `decay_until=8`.
3. Early iterations explore all three branches. Later iterations concentrate on the TabPFN branch after its Q-value rises to 0.82 following strong simulated evaluation.
4. The best node after 10 iterations: "TabPFN with Shapley-weighted test-time augmentation", a third-generation evolved variant.

**Key insight:** The UCB scheduler successfully transitions from exploration (rounds 1–2) to exploitation (rounds 3–10) without requiring manual tuning of the exploration constant.

### 9.3 Sparse Attention Survey Paper

**Goal:** Produce a survey paper on sparse attention mechanisms for long-context documents, targeting ACL.

**Workflow:**
1. The Literature agent retrieves 18 papers from arXiv + Semantic Scholar (year_from=2019, query: "sparse attention long context document").
2. The Outline agent produces a 7-section plan: Abstract, Introduction, Background, Taxonomy of Sparse Attention Patterns, Theoretical Analysis, Empirical Comparison, Conclusion.
3. The Writer agent drafts all 7 sections in dependency order (~5 800 words total).
4. The Editor agent scores: clarity 7.8, correctness 7.2, novelty 6.8, citation_quality 7.5. The novelty score for the Taxonomy section falls below threshold (6.8 < 7.0).
5. A targeted rewrite of the Taxonomy section incorporates a more fine-grained taxonomy (fixed-pattern vs. content-adaptive vs. hardware-aware), raising novelty to 7.6.
6. Export: a complete Markdown paper with inline citation placeholders, ready for citation resolution and LaTeX conversion.

---

## 10. Limitations and Future Work

### 10.1 Hallucination and Factual Accuracy

Like all LLM-based systems, ResearchCrew is susceptible to hallucination. The Editor agent's correctness scoring mitigates but does not eliminate this risk. Future work should integrate external fact-checking tools (e.g., claim verification against the retrieved corpus) and citation verification (checking that cited papers actually support the claims attributed to them).

### 10.2 Evaluation Metrics

The current system lacks formal evaluation metrics for hypothesis quality. The Elo leaderboard is internally consistent but does not correlate with external ground truth. Future work should benchmark hypothesis quality against domain expert ratings, and compare paper drafts to published papers using automated metrics (BERTScore, ROUGE) and human evaluation.

### 10.3 Scalability of the Tournament

Tournament cost scales as `O(H² × R)` where `H` is the hypothesis population size and `R` is the number of rounds. For large populations (H > 20), cost grows rapidly. Future optimisations include: parallelising matchups, using smaller LLMs for early-round judging (escalating to larger models in later rounds), and implementing single-elimination brackets for efficient population reduction.

### 10.4 Domain Adaptation

The current agent prompts and tournament criteria are calibrated for natural-science and ML research. Adapting to social sciences, humanities, or engineering design will require domain-specific criteria rubrics, citation norms, and outline templates.

### 10.5 Experiment Execution

ResearchCrew does not execute experiments. Integrating with computational environments (Jupyter kernels, HPC schedulers, cloud ML platforms) would close the loop between hypothesis generation and empirical validation, enabling a fully autonomous research cycle.

### 10.6 Real-Time Collaboration

The current system is single-user. Multi-researcher collaboration — where human researchers annotate hypotheses, override tournament verdicts, or steer evolution — would significantly enhance practical utility. The TUI chat interface provides a foundation; future work should extend it with shared project state and conflict resolution.

### 10.7 Citation Resolution

The Paper Writing crew inserts citation placeholders (`[Author, Year]`) rather than resolved citation keys. Future work should implement automated placeholder resolution using the retrieved bibliography, validated against the paper's claims.

---

## 11. Conclusion

We have presented ResearchCrew, an open-source multi-agent framework that automates the scientific research lifecycle from hypothesis generation through publication-ready paper drafting. The system's key innovations are: a provider-agnostic LLM abstraction with adaptive thinking support, an Elo-based hypothesis tournament with LLM debate judging, Progressive Monte Carlo Graph Search for hypothesis space exploration, a two-phase literature retrieval and screening pipeline, simulated peer review with targeted rewriting, and a FastMCP paper-search server with cross-source deduplication.

Three use cases — biomedical drug repurposing, ML algorithm discovery, and survey paper writing — demonstrate the system's applicability across research domains. ResearchCrew is not intended to replace human researchers but to dramatically accelerate the early stages of the research cycle, allowing domain experts to focus their effort on experimental design, empirical validation, and critical interpretation of results — tasks that remain firmly in the human domain.

All code is available at https://github.com/bsfeal/researchcrew under the MIT licence.

---

## References

[CITATION] Gottweis, J., et al. (2025). Towards an AI Co-Scientist. *Google DeepMind Technical Report*.

[CITATION] Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.

[CITATION] Hong, S., et al. (2024). MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. *ICLR 2024*.

[CITATION] Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.

[CITATION] Elo, A. E. (1978). *The Rating of Chess Players, Past and Present*. Arco Publishing.

[CITATION] Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

[CITATION] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

[CITATION] Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529, 484–489.

[CITATION] Chen, X., et al. (2024). AlphaCode 2 Technical Report. *Google DeepMind*.

[CITATION] Yang, K., et al. (2023). Large Language Models as Optimizers. *arXiv:2309.03409*.

[CITATION] Wang, X., et al. (2024). MLEvolve: Automating Machine Learning Algorithm Discovery. *ICML 2024*.

[CITATION] Koreeda, Y., & Manning, C. (2021). ContractNLI: A dataset for document-level natural language inference for contracts. *EMNLP 2021*.

[CITATION] Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.

[CITATION] Kim, G., et al. (2023). PaperOrchestra: Generating Structured Academic Papers with LLMs. *arXiv:2306.04548*.

[CITATION] Beltagy, I., Peters, M.E., Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv:2004.05150*.

[CITATION] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

[CITATION] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

[CITATION] Yang, Y., et al. (2024). AgentLaboratory: Using LLM Agents as Research Assistants. *arXiv:2407.08785*.

[CITATION] Kim, G., et al. (2023). Language Models can solve Computer Tasks. *NeurIPS 2023*.

[CITATION] Liang, P., et al. (2022). Holistic Evaluation of Language Models. *arXiv:2211.09110*.

---

*This white paper describes ResearchCrew v0.1.0 (alpha). The system is under active development; claims and benchmarks will be updated as evaluation methodology matures.*
