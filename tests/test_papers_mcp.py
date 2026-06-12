"""
Unit tests for researchcrew.mcp.papers — no network required.
Uses respx to intercept httpx calls at the transport level.
"""
from __future__ import annotations

import sys

import httpx
import pytest
import respx

from researchcrew.mcp.papers import (
    _reconstruct_abstract,
    _resolve_s2_id,
    fetch_paper,
    find_related,
    get_citations,
    get_references,
    search_papers,
)


# ── shared fixtures ──────────────────────────────────────────────────────────────────

ARXIV_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models use complex RNNs.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1810.04805v2</id>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce a new language representation model called BERT.</summary>
    <published>2018-10-11T00:00:00Z</published>
    <author><name>Jacob Devlin</name></author>
  </entry>
</feed>"""

S2_SEARCH = {
    "data": [{
        "paperId": "abc123",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "year": 2017,
        "authors": [{"name": "Vaswani"}],
        "citationCount": 90000,
        "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
        "url": "https://www.semanticscholar.org/paper/abc123",
        "publicationTypes": ["JournalArticle"],
    }]
}

S2_PAPER = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models...",
    "year": 2017,
    "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
    "citationCount": 90000,
    "externalIds": {"DOI": "10.48550/arXiv.1706.03762", "ArXiv": "1706.03762"},
    "url": "https://www.semanticscholar.org/paper/abc123",
    "publicationTypes": ["JournalArticle"],
}

S2_CITATIONS = {
    "data": [
        {"citingPaper": {
            "paperId": "cite1", "title": "Transformer-XL",
            "abstract": "Attentive language models.", "year": 2019,
            "authors": [{"name": "Dai"}], "citationCount": 5000,
            "externalIds": {"ArXiv": "1901.02860"}, "url": None,
        }},
        {"citingPaper": {
            "paperId": "cite2", "title": "GPT-2",
            "abstract": "Language models are unsupervised multitask learners.", "year": 2019,
            "authors": [{"name": "Radford"}], "citationCount": 20000,
            "externalIds": {}, "url": None,
        }},
    ]
}

S2_REFS = {
    "data": [
        {"citedPaper": {
            "paperId": "ref1", "title": "Sequence to Sequence Learning",
            "abstract": "General end-to-end approach.", "year": 2014,
            "authors": [{"name": "Sutskever"}], "citationCount": 25000,
            "externalIds": {"ArXiv": "1409.3215"}, "url": None,
        }}
    ]
}

S2_RECOM = {
    "recommendedPapers": [{
        "paperId": "rel1", "title": "BERT",
        "abstract": "Bidirectional Encoder Representations.", "year": 2018,
        "authors": [{"name": "Devlin"}], "citationCount": 80000,
        "externalIds": {"ArXiv": "1810.04805"}, "url": None,
        "publicationTypes": [],
    }]
}

OPENALEX = {
    "results": [{
        "id": "https://openalex.org/W123",
        "title": "OpenAlex Paper",
        "abstract_inverted_index": {"OpenAlex": [0], "is": [1], "great": [2]},
        "authorships": [{"author": {"display_name": "Alex"}}],
        "publication_year": 2022,
        "doi": "https://doi.org/10.1234/test",
        "cited_by_count": 42,
        "primary_location": {"landing_page_url": "https://example.com/paper"},
    }]
}


# ── _resolve_s2_id ────────────────────────────────────────────────────────────────────────────

class TestResolveS2Id:
    def test_doi(self):
        assert _resolve_s2_id("10.1145/1234567") == "DOI:10.1145/1234567"

    def test_bare_arxiv_id(self):
        assert _resolve_s2_id("1706.03762") == "ARXIV:1706.03762"

    def test_arxiv_prefix_lowercase(self):
        assert _resolve_s2_id("arxiv:2106.09685") == "ARXIV:2106.09685"

    def test_arxiv_prefix_mixed_case(self):
        assert _resolve_s2_id("ArXiv:2106.09685") == "ARXIV:2106.09685"

    def test_raw_s2_id_passthrough(self):
        raw = "abc123def456" * 4
        assert _resolve_s2_id(raw) == raw


# ── _reconstruct_abstract ───────────────────────────────────────────────────────────────────

class TestReconstructAbstract:
    def test_basic(self):
        inv = {"Hello": [0], "world": [1], "there": [2], "again": [3]}
        assert _reconstruct_abstract(inv) == "Hello world there again"

    def test_out_of_order_keys(self):
        inv = {"world": [1], "Hello": [0]}
        assert _reconstruct_abstract(inv) == "Hello world"

    def test_empty(self):
        assert _reconstruct_abstract({}) == ""

    def test_single_word(self):
        assert _reconstruct_abstract({"only": [0]}) == "only"


# ── search_papers ─────────────────────────────────────────────────────────────────────────────

class TestSearchPapers:
    @respx.mock
    def test_returns_merged_deduped_results(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM)
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=S2_SEARCH)
        )
        results = search_papers("transformer", max_results=10)
        titles = [r["title"] for r in results]
        assert len(results) >= 1
        assert any("Attention" in t for t in titles)
        assert len(results) == len({r["title"] for r in results})

    @respx.mock
    def test_sorted_by_citation_count(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM)
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=S2_SEARCH)
        )
        results = search_papers("transformer", max_results=10)
        counts = [r["citation_count"] for r in results]
        assert counts == sorted(counts, reverse=True)

    @respx.mock
    def test_year_filter_arxiv(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM)
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        results = search_papers("transformer", max_results=10, year_from=2018)
        arxiv_only = [r for r in results if r["source"] == "arxiv"]
        years = [r["year"] for r in arxiv_only if r["year"]]
        assert all(y >= 2018 for y in years)
        assert not any("Vaswani" in str(r.get("authors")) for r in arxiv_only)

    @respx.mock
    def test_arxiv_failure_still_returns_s2(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(500, text="error")
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=S2_SEARCH)
        )
        results = search_papers("transformer", max_results=5)
        assert len(results) >= 1
        assert all(r["source"] == "semantic_scholar" for r in results)

    @respx.mock
    def test_openalex_source(self):
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=OPENALEX)
        )
        results = search_papers("transformer", sources=["openalex"], max_results=5)
        assert len(results) == 1
        r = results[0]
        assert r["source"] == "openalex"
        assert r["doi"] == "10.1234/test"
        assert r["abstract"] == "OpenAlex is great"
        assert r["openalex_id"] == "W123"

    @respx.mock
    def test_max_results_cap(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM)
        )
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        results = search_papers("transformer", max_results=1)
        assert len(results) <= 1


# ── fetch_paper ─────────────────────────────────────────────────────────────────────────────

class TestFetchPaper:
    @respx.mock
    def test_arxiv_id_resolves(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762").mock(
            return_value=httpx.Response(200, json=S2_PAPER)
        )
        paper = fetch_paper("1706.03762")
        assert paper["title"] == "Attention Is All You Need"
        assert paper["year"] == 2017
        assert paper["citation_count"] == 90000
        assert paper["arxiv_id"] == "1706.03762"
        assert len(paper["authors"]) == 2

    @respx.mock
    def test_doi_resolves(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.48550/arXiv.1706.03762"
        ).mock(return_value=httpx.Response(200, json=S2_PAPER))
        paper = fetch_paper("10.48550/arXiv.1706.03762")
        assert paper["title"] == "Attention Is All You Need"

    @respx.mock
    def test_api_error_returns_error_dict(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762").mock(
            return_value=httpx.Response(500, json={"message": "error"})
        )
        result = fetch_paper("1706.03762")
        assert "error" in result
        assert result.get("paper_id") == "1706.03762"


# ── get_citations ────────────────────────────────────────────────────────────────────────────

class TestGetCitations:
    @respx.mock
    def test_returns_citing_papers(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/citations"
        ).mock(return_value=httpx.Response(200, json=S2_CITATIONS))
        cites = get_citations("1706.03762", limit=5)
        assert len(cites) == 2
        titles = [c["title"] for c in cites]
        assert "GPT-2" in titles
        assert "Transformer-XL" in titles

    @respx.mock
    def test_sorted_by_citation_count_desc(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/citations"
        ).mock(return_value=httpx.Response(200, json=S2_CITATIONS))
        cites = get_citations("1706.03762", limit=5)
        counts = [c["citation_count"] for c in cites]
        assert counts == sorted(counts, reverse=True)
        assert cites[0]["title"] == "GPT-2"

    @respx.mock
    def test_api_error_returns_error_list(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/citations"
        ).mock(return_value=httpx.Response(500, json={}))
        result = get_citations("1706.03762", limit=5)
        assert result and "error" in result[0]

    @respx.mock
    def test_doi_paper_id(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/1234/citations"
        ).mock(return_value=httpx.Response(200, json=S2_CITATIONS))
        cites = get_citations("10.1145/1234", limit=5)
        assert len(cites) == 2


# ── get_references ────────────────────────────────────────────────────────────────────────────

class TestGetReferences:
    @respx.mock
    def test_returns_referenced_papers(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/references"
        ).mock(return_value=httpx.Response(200, json=S2_REFS))
        refs = get_references("1706.03762", limit=10)
        assert len(refs) == 1
        assert refs[0]["title"] == "Sequence to Sequence Learning"
        assert refs[0]["arxiv_id"] == "1409.3215"
        assert refs[0]["citation_count"] == 25000

    @respx.mock
    def test_api_error_returns_error_list(self):
        respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/references"
        ).mock(return_value=httpx.Response(500, json={}))
        result = get_references("1706.03762", limit=10)
        assert result and "error" in result[0]


# ── find_related ────────────────────────────────────────────────────────────────────────────

class TestFindRelated:
    @respx.mock
    def test_returns_related_papers(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=S2_SEARCH)
        )
        respx.post("https://api.semanticscholar.org/recommendations/v1/papers").mock(
            return_value=httpx.Response(200, json=S2_RECOM)
        )
        related = find_related("transformer attention", limit=5)
        assert len(related) == 1
        assert related[0]["title"] == "BERT"
        assert related[0]["arxiv_id"] == "1810.04805"

    @respx.mock
    def test_empty_seed_returns_empty(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        related = find_related("xyzzy unknown topic no match", limit=5)
        assert related == []

    @respx.mock
    def test_recommendations_failure_returns_error(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=S2_SEARCH)
        )
        respx.post("https://api.semanticscholar.org/recommendations/v1/papers").mock(
            return_value=httpx.Response(500, json={})
        )
        result = find_related("transformer", limit=5)
        assert result and "error" in result[0]


# ── MCP tool registration ──────────────────────────────────────────────────────────────────

class TestMCPRegistration:
    def test_all_five_tools_registered(self):
        import asyncio
        from researchcrew.mcp.papers import mcp

        async def _check():
            expected = {"search_papers", "fetch_paper", "get_citations", "get_references", "find_related"}
            for name in expected:
                t = await mcp.get_tool(name)
                assert t is not None, f"{name} not registered"
                params = set(t.parameters.get("properties", {}).keys())
                assert params, f"{name} has no parameters"
            return True

        assert asyncio.run(_check())

    def test_search_papers_schema(self):
        import asyncio
        from researchcrew.mcp.papers import mcp

        async def _check():
            t = await mcp.get_tool("search_papers")
            wire = t.to_mcp_tool(tool_name="search_papers")
            schema = wire.inputSchema
            assert schema["required"] == ["query"]
            props = schema["properties"]
            assert "query" in props
            assert "sources" in props
            assert "max_results" in props
            assert "year_from" in props
            assert props["max_results"]["default"] == 10
            return True

        assert asyncio.run(_check())

    def test_fetch_paper_schema(self):
        import asyncio
        from researchcrew.mcp.papers import mcp

        async def _check():
            t = await mcp.get_tool("fetch_paper")
            wire = t.to_mcp_tool(tool_name="fetch_paper")
            schema = wire.inputSchema
            assert "paper_id" in schema["required"]
            return True

        assert asyncio.run(_check())
