# researchcrew-chat

`researchcrew-chat` is a Rich/prompt_toolkit terminal chat interface for managing research projects through natural language. It stores full project state locally and streams Claude responses in real time.

## Installation

```bash
pip install 'researchcrew[chat]'
export ANTHROPIC_API_KEY=sk-ant-...
```

## Starting the Chat

```bash
researchcrew-chat              # resume last project (or start fresh)
researchcrew-chat new          # guided new-project wizard
researchcrew-chat new "My Topic" # skip the title prompt
researchcrew-chat open <id>    # open a specific project by ID
researchcrew-chat list         # list all projects and exit
```

---

## In-Session Slash Commands

| Command | Description |
|---|---|
| `/new [title]` | Create a new project (interactive wizard if title omitted) |
| `/open <id>` | Open an existing project |
| `/list` | List all projects in a table |
| `/status` | Full project overview |
| `/papers` | Table of collected related works |
| `/outline` | Show paper outline (if generated) |
| `/draft [section]` | Rendered Markdown draft — all sections or one named section |
| `/code` | Syntax-highlighted code artifacts |
| `/save` | Force-save current state to disk |
| `/export [path]` | Write a publication-ready Markdown file |
| `/save-section <name>` | Save the last assistant response as a named paper section |
| `/add-paper <title>` | Add a paper to related works (interactive: title, authors, year, URL) |
| `/save-code <name>` | Save the first code block from the last response as an artifact |
| `/help` | Show command reference |
| `/quit` or `/exit` | Exit and auto-save |

---

## Typical Research Workflow

### 1. Create a project

```
You [no project] ❯ /new Sparse Attention Survey
Title: Sparse Attention Survey
Research goal (1-2 sentences): Survey efficient attention mechanisms that scale to long documents.
Keywords (comma-separated, optional): attention, transformer, long-context, efficiency
✓ Created project a1b2c3d4-sparse_attention_survey
```

### 2. Explore the literature

```
You [Sparse Attention...] ❯ What are the most important papers on sparse attention from 2020 onwards?
```

Claude will describe key papers. When you find one worth saving:

```
You [Sparse Attention...] ❯ /add-paper Longformer: The Long-Document Transformer
Paper title: Longformer: The Long-Document Transformer
Authors (comma-separated): Beltagy, I., Peters, M.E., Cohan, A.
Year: 2020
URL or DOI (optional): https://arxiv.org/abs/2004.05150
✓ Added 'Longformer: The Long-Document Transformer' to related works.
```

### 3. Generate an outline

```
You [Sparse Attention...] ❯ Create a structured outline for a survey paper targeting ACL.
```

Claude returns a section plan. Save it:

```
You [Sparse Attention...] ❯ /save-section Outline
✓ Saved section 'Outline'.
```

### 4. Write sections

```
You [Sparse Attention...] ❯ Write the Related Work section covering sliding-window,
global-local, and linear attention families.
```

Claude writes the section. The TUI detects it's a long prose response and hints:

```
  ↳ This looks like a paper section. Use /save-section <name> to add to your draft.
```

```
You [Sparse Attention...] ❯ /save-section Related Work
✓ Saved section 'Related Work'.
```

### 5. Write code

```
You [Sparse Attention...] ❯ Show a minimal PyTorch implementation of sliding-window attention.
```

Claude returns code. The TUI detects the code block:

```
  ↳ 1 code block(s) detected [python]. Use /save-code <name> to save.
```

```
You [Sparse Attention...] ❯ /save-code sliding_window_attention
✓ Saved code artifact 'sliding_window_attention' (python).
```

### 6. Export for submission

```
You [Sparse Attention...] ❯ /export
✓ Exported to /home/user/sparse_attention_survey.md
```

The exported file assembles sections in standard academic order:
Abstract → Introduction → Related Work → Background → Methodology → Results → Discussion → Conclusion → References

---

## Project Storage

All data is stored under `~/.researchcrew/projects/<project-id>/`:

```
<id>/
├── meta.json        # All metadata (title, goal, status, keywords, related works, outline)
├── chat.jsonl       # Conversation history (one JSON message per line)
├── draft/           # One .md file per named section
│   ├── abstract.md
│   ├── related_work.md
│   └── ...
└── code/            # Code artifacts (filename = slugified name + language extension)
    ├── sliding_window_attention.py
    └── benchmark_runner.sh
```

State is auto-saved after every assistant response. Manual `/save` is available for an explicit checkpoint.

---

## Auto-Detection Hints

After each response, the TUI analyses the content and shows context-aware hints:

- **Code blocks with ≥ 5 lines** → `↳ N code block(s) detected [lang]. Use /save-code <name> to save.`
- **Long prose (> 400 chars, ≥ 2 paragraphs)** → `↳ This looks like a paper section. Use /save-section <name> to add to your draft.`

---

## System Prompt

The assistant is briefed with the current project state on every turn:

```
You are a scientific research assistant embedded in the researchcrew CLI.
Current project state:
  Title: Sparse Attention Survey
  Research goal: Survey efficient attention mechanisms...
  Status: writing
  Keywords: attention, transformer, long-context, efficiency
  Related works (3): "Longformer: ...", "BigBird: ...", "Reformer: ..."
  Draft sections written: Outline, Related Work
```

This context is regenerated from the live project state, so it always reflects the latest saves.

---

## Model

The chat uses `claude-opus-4-8` with `thinking: {type: "adaptive"}` and `max_tokens: 8192`. To change the model, edit `MODEL` at the top of `src/researchcrew/tools/chat.py`.
