# MCP Paper Search Server

ResearchCrew ships a [FastMCP](https://github.com/jlowin/fastmcp) server that exposes five academic paper-search tools. It can be used standalone as a stdio MCP server, or imported directly into Python.

## Installation

```bash
pip install 'researchcrew[mcp]'
```

## Starting the Server

```bash
# Start stdio MCP server (connect via any MCP client)
researchcrew-papers

# Or run the module directly
python -m researchcrew.mcp.papers

# Smoke-test against live APIs
python -m researchcrew.mcp.papers --test
```

---

## Tools

### `search_papers`

Search for academic papers across up to four sources with automatic cross-source deduplication.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Free-text search (title, keywords, authors) |
| `sources` | `list[str]` | `["arxiv","semantic_scholar"]` | Sources to query |
| `max_results` | `int` | `10` | Max papers per source (cap 50) |
| `year_from` | `int` | `None` | Exclude papers before this year |

**Sources**: `"arxiv"`, `"semantic_scholar"`, `"pubmed"`, `"openalex"`

**Example**

```python
from researchcrew.mcp.papers import search_papers

papers = search_papers(
    "sparse attention transformer long context",
    sources=["arxiv", "semantic_scholar", "openalex"],
    max_results=15,
    year_from=2020,
)

for p in papers[:3]:
    print(f"{p['title']} ({p['year']}) — {p['citation_count']} citations")
```

---

### `fetch_paper`

Fetch full metadata for a single paper by ID.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `paper_id` | `str` | arXiv ID (`"1706.03762"`), DOI (`"10.1145/..."`), or Semantic Scholar ID |

**Example**

```python
from researchcrew.mcp.papers import fetch_paper

paper = fetch_paper("1706.03762")   # "Attention Is All You Need"
print(paper["title"])
print(paper["abstract"][:200])
print(f"Citations: {paper['citation_count']}")
print(f"Authors: {', '.join(paper['authors'])}")
```

---

### `get_citations`

Return papers that **cite** the given paper (forward citations).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `paper_id` | `str` | required | arXiv ID, DOI, or S2 ID |
| `limit` | `int` | `20` | Max citing papers (cap 100) |

**Example**

```python
from researchcrew.mcp.papers import get_citations

citing = get_citations("1706.03762", limit=50)
print(f"Found {len(citing)} papers that cite Attention Is All You Need")
```

---

### `get_references`

Return papers **referenced by** the given paper (backward citations).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `paper_id` | `str` | required | arXiv ID, DOI, or S2 ID |
| `limit` | `int` | `50` | Max reference papers (cap 500) |

**Example**

```python
from researchcrew.mcp.papers import get_references

refs = get_references("1706.03762", limit=100)
top_cited = refs[:5]   # already sorted by citation count desc
```

---

### `find_related`

Find papers semantically related to a given title or abstract snippet using Semantic Scholar recommendations.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `title_or_abstract` | `str` | required | Title or abstract to find related work for |
| `limit` | `int` | `10` | Max results (cap 50) |

**Example**

```python
from researchcrew.mcp.papers import find_related

related = find_related(
    "Efficient attention mechanisms for processing long document sequences",
    limit=10,
)
for p in related:
    print(f"• {p['title']} ({p['year']})")
```

---

## Paper Dict Schema

All tools return paper dicts with this structure:

```python
{
    "title":          str,           # Paper title
    "abstract":       str,           # Full abstract text
    "authors":        list[str],     # Author names
    "year":           int | None,    # Publication year
    "doi":            str | None,    # DOI (e.g. "10.1145/1234")
    "arxiv_id":       str | None,    # arXiv ID (version-stripped, e.g. "1706.03762")
    "s2_id":          str | None,    # Semantic Scholar paper ID
    "pubmed_id":      str | None,    # PubMed ID
    "openalex_id":    str | None,    # OpenAlex work ID (e.g. "W123")
    "url":            str | None,    # Landing page URL
    "citation_count": int,           # Number of citations
    "source":         str,           # "arxiv" | "semantic_scholar" | "pubmed" | "openalex"
}
```

---

## Deduplication Logic

Papers from different sources that refer to the same work are merged using a two-key scheme:

1. **Canonical key**: DOI → arXiv ID → Semantic Scholar ID → title (first non-null)
2. **arXiv secondary index**: a `set` of version-stripped arXiv IDs (e.g. `1706.03762v5` → `1706.03762`) prevents double-counting when a paper appears in both S2 and arXiv with different canonical keys.

Results are sorted by `citation_count` descending, then truncated to `max_results`.

---

## Using as MCP Server

The server runs over stdio and is compatible with any MCP client (Claude Desktop, Cursor, Zed, etc.).

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "researchcrew-papers": {
      "command": "researchcrew-papers"
    }
  }
}
```

---

## API Rate Limits

| Source | Rate limit | Notes |
|---|---|---|
| arXiv | ~3 req/s | No key required |
| Semantic Scholar | 100 req/5 min (unauthenticated) | Add `X-API-KEY` header for higher limits |
| PubMed | 3 req/s without key | Set `NCBI_API_KEY` env var for 10 req/s |
| OpenAlex | Polite pool: 10 req/s | Set `User-Agent` with email for more |

The server currently uses unauthenticated access. For production use, pass API keys via environment variables and extend the `_search_*` helpers accordingly.
