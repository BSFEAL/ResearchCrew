# Contributing

Thank you for your interest in ResearchCrew!

## Setup

```bash
git clone https://github.com/bsfeal/researchcrew
cd researchcrew

# Install in editable mode with all extras
pip install -e '.[all]'

# Install dev tools
pip install pytest respx ruff mypy
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific suite
pytest tests/test_papers_mcp.py -v      # MCP server (30 tests, no network)
pytest tests/test_chat.py -v            # Chat TUI (25 tests)

# With coverage
pip install pytest-cov
pytest tests/ --cov=src/researchcrew --cov-report=html
```

## Code Style

ResearchCrew uses **Ruff** for linting and formatting:

```bash
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
```

**Type checking** with mypy (strict mode):

```bash
mypy src/
```

## Commit Convention

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(llm): add Mistral provider
fix(mcp): strip arXiv version suffix before dedup
test(tournament): add Elo edge-case for identical scores
docs: update MCP tool reference
chore: bump fastmcp to 3.1
```

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`, `style`, `revert`

## Adding a New LLM Provider

1. Create `src/researchcrew/llm/<name>_llm.py` subclassing `ResearchLLM`
2. Implement `complete(prompt, *, json_mode=False) -> str`
3. Add the provider to `LLMProvider` literal in `providers.py`
4. Add the import branch in `create_llm()`
5. Add unit tests in `tests/test_llm.py`

## Adding an MCP Tool

1. Add a new `@mcp.tool()` function in `src/researchcrew/mcp/papers.py`
2. Follow the `_paper_dict()` return schema for consistency
3. Add tests in `tests/test_papers_mcp.py` using `@respx.mock`

## Pull Requests

- Branch from `main`
- One feature / fix per PR
- All tests must pass
- Update the relevant doc page in `docs/`
