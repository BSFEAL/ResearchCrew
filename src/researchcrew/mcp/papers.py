"""
MCP server: academic paper search.

Tools
-----
search_papers   – query arXiv + Semantic Scholar (+ optionally PubMed / OpenAlex)
fetch_paper     – full metadata by arXiv ID, DOI, or Semantic Scholar paper ID
get_citations   – papers that cite a given paper (S2 forward citations)
get_references  – papers referenced by a given paper (S2 backward citations)
find_related    – semantically similar papers via S2 recommendations

Run standalone::

    python -m researchcrew.mcp.papers          # stdio MCP server
    python -m researchcrew.mcp.papers --test   # live smoke-test
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from fastmcp import FastMCP


# ── constants ────────────────────────────────────────────────────────────────────────────

_ARXIV       = "https://export.arxiv.org/api/query"
_S2_SEARCH   = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_PAPER    = "https://api.semanticscholar.org/graph/v1/paper/{id}"
_S2_CITES    = "https://api.semanticscholar.org/graph/v1/paper/{id}/citations"
_S2_REFS     = "https://api.semanticscholar.org/graph/v1/paper/{id}/references"
_S2_RECOMS   = "https://api.semanticscholar.org/recommendations/v1/papers"
_OPENALEX    = "https://api.openalex.org/works"

_S2_FIELDS   = "title,abstract,year,authors,citationCount,externalIds,url,publicationTypes"
_TIMEOUT     = 20.0

# ── helpers ────────────────────────────────────────────────────────────────────────────

def _paper_dict(
    title: str,
    abstract: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    pubmed_id: str | None = None,
    openalex_id: str | None = None,
    url: str | None = None,
    citation_count: int = 0,
    source: str = "unknown",
) -> dict[str, Any]:
    return dict(
        title=title,
        abstract=abstract,
        authors=authors or [],
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_id=s2_id,
        pubmed_id=pubmed_id,
        openalex_id=openalex_id,
        url=url,
        citation_count=citation_count,
        source=source,
    )


def _s2_item_to_dict(item: dict[str, Any], source: str = "semantic_scholar") -> dict[str, Any]:
    ext = item.get("externalIds") or {}
    pid = (
        item.get("paperId")
        or item.get("citedPaper", {}).get("paperId")
        or item.get("citingPaper", {}).get("paperId")
    )
    return _paper_dict(
        title=(
            item.get("title")
            or item.get("citedPaper", {}).get("title")
            or item.get("citingPaper", {}).get("title")
            or ""
        ),
        abstract=item.get("abstract") or "",
        authors=[a.get("name", "") for a in (item.get("authors") or [])],
        year=item.get("year"),
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        s2_id=pid,
        url=item.get("url"),
        citation_count=item.get("citationCount") or 0,
        source=source,
    )


def _s2_nested(item: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract a paper from a citations/references response (paper nested under *key*)."""
    paper = item.get(key, {})
    ext = paper.get("externalIds") or {}
    return _paper_dict(
        title=paper.get("title") or "",
        abstract=paper.get("abstract") or "",
        authors=[a.get("name", "") for a in (paper.get("authors") or [])],
        year=paper.get("year"),
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        s2_id=paper.get("paperId"),
        url=paper.get("url"),
        citation_count=paper.get("citationCount") or 0,
        source="semantic_scholar",
    )


# ── MCP server ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="researchcrew-papers",
    instructions=(
        "Tools for searching and retrieving academic papers from arXiv, "
        "Semantic Scholar, PubMed, and OpenAlex."
    ),
)


@mcp.tool()
def search_papers(
    query: str,
    sources: list[str] | None = None,
    max_results: int = 10,
    year_from: int | None = None,
) -> list[dict[str, Any]]:
    """Search for academic papers across multiple sources.

    Args:
        query:       Free-text search query (title, keywords, author).
        sources:     Subset of ["arxiv", "semantic_scholar", "pubmed", "openalex"].
                     Defaults to ["arxiv", "semantic_scholar"].
        max_results: Maximum papers per source (cap 50).
        year_from:   Exclude papers published before this year.

    Returns:
        List of paper dicts, deduplicated by DOI/arXiv ID, sorted by citation count.
    """
    active = set(sources or ["arxiv", "semantic_scholar"])
    limit  = min(max_results, 50)
    seen: dict[str, dict[str, Any]] = {}  # canonical_id → paper
    seen_arxiv: set[str] = set()           # version-stripped arxiv_ids already added

    def _add(paper: dict[str, Any]) -> None:
        axiv = paper.get("arxiv_id")
        if axiv and axiv in seen_arxiv:
            return
        cid = paper.get("doi") or axiv or paper.get("s2_id") or paper["title"]
        if cid and cid not in seen:
            seen[cid] = paper
            if axiv:
                seen_arxiv.add(axiv)

    with httpx.Client(timeout=_TIMEOUT) as client:
        if "arxiv" in active:
            _search_arxiv(client, query, limit, year_from, _add)
        if "semantic_scholar" in active:
            _search_s2(client, query, limit, _add)
        if "pubmed" in active:
            _search_pubmed(client, query, limit, _add)
        if "openalex" in active:
            _search_openalex(client, query, limit, year_from, _add)

    results = list(seen.values())
    results.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
    return results[:limit]


@mcp.tool()
def fetch_paper(paper_id: str) -> dict[str, Any]:
    """Fetch full metadata for a single paper.

    Args:
        paper_id: arXiv ID (e.g. "2106.09685"), DOI (e.g. "10.1145/1234"),
                  or Semantic Scholar paper ID (40-char hex).

    Returns:
        Paper dict with title, abstract, authors, year, DOI, arXiv ID, citations, URL.
    """
    with httpx.Client(timeout=_TIMEOUT) as client:
        s2_id = _resolve_s2_id(paper_id)
        url   = _S2_PAPER.format(id=s2_id)
        params = {"fields": _S2_FIELDS + ",references,citations"}
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"error": str(exc), "paper_id": paper_id}

        ext = data.get("externalIds") or {}
        return _paper_dict(
            title=data.get("title") or "",
            abstract=data.get("abstract") or "",
            authors=[a.get("name", "") for a in (data.get("authors") or [])],
            year=data.get("year"),
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            s2_id=data.get("paperId"),
            url=data.get("url"),
            citation_count=data.get("citationCount") or 0,
            source="semantic_scholar",
        )


@mcp.tool()
def get_citations(paper_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return papers that cite the given paper (forward citations).

    Args:
        paper_id: arXiv ID, DOI, or Semantic Scholar paper ID.
        limit:    Maximum number of citing papers to return (cap 100).

    Returns:
        List of paper dicts sorted by citation count descending.
    """
    s2_id  = _resolve_s2_id(paper_id)
    url    = _S2_CITES.format(id=s2_id)
    params = {
        "fields": "title,abstract,year,authors,citationCount,externalIds,url",
        "limit": min(limit, 100),
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return [{"error": str(exc)}]

    papers = [_s2_nested(item, "citingPaper") for item in data.get("data", [])]
    papers.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
    return papers


@mcp.tool()
def get_references(paper_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return papers referenced by the given paper (backward citations).

    Args:
        paper_id: arXiv ID, DOI, or Semantic Scholar paper ID.
        limit:    Maximum number of reference papers to return (cap 500).

    Returns:
        List of paper dicts sorted by citation count descending.
    """
    s2_id  = _resolve_s2_id(paper_id)
    url    = _S2_REFS.format(id=s2_id)
    params = {
        "fields": "title,abstract,year,authors,citationCount,externalIds,url",
        "limit": min(limit, 500),
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return [{"error": str(exc)}]

    papers = [_s2_nested(item, "citedPaper") for item in data.get("data", [])]
    papers.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
    return papers


@mcp.tool()
def find_related(title_or_abstract: str, limit: int = 10) -> list[dict[str, Any]]:
    """Find papers semantically related to the given title or abstract snippet.

    Uses a two-step approach: search Semantic Scholar for the best matching
    seed paper, then call the S2 recommendations endpoint for related work.

    Args:
        title_or_abstract: Title or abstract of the paper you want related work for.
        limit:             Maximum results (cap 50).

    Returns:
        List of related paper dicts.
    """
    limit = min(limit, 50)
    with httpx.Client(timeout=_TIMEOUT) as client:
        search_params = {
            "query": title_or_abstract[:200],
            "fields": "paperId,title",
            "limit": 1,
        }
        try:
            sr = client.get(_S2_SEARCH, params=search_params)
            sr.raise_for_status()
            hits = sr.json().get("data", [])
        except Exception as exc:
            return [{"error": f"seed search failed: {exc}"}]

        if not hits:
            return []

        seed_id = hits[0]["paperId"]
        payload = {"positivePaperIds": [seed_id], "negativePaperIds": []}
        rec_params = {"fields": _S2_FIELDS, "limit": limit}
        try:
            rr = client.post(_S2_RECOMS, json=payload, params=rec_params)
            rr.raise_for_status()
            recs = rr.json().get("recommendedPapers", [])
        except Exception as exc:
            return [{"error": f"recommendations failed: {exc}"}]

    return [_s2_item_to_dict(r) for r in recs]


# ── private search helpers ──────────────────────────────────────────────────────────────────

def _resolve_s2_id(paper_id: str) -> str:
    if paper_id.startswith("10."):
        return f"DOI:{paper_id}"
    lower = paper_id.lower()
    if lower.startswith("arxiv:"):
        return "ARXIV:" + paper_id[len("arxiv:"):]
    if "/" not in paper_id and "." in paper_id and len(paper_id) < 20:
        return f"ARXIV:{paper_id}"
    return paper_id


def _search_arxiv(
    client: httpx.Client,
    query: str,
    limit: int,
    year_from: int | None,
    add: Any,
) -> None:
    params: dict[str, Any] = {
        "search_query": f"all:{query}",
        "max_results": limit,
        "sortBy": "relevance",
    }
    try:
        resp = client.get(_ARXIV, params=params)
        resp.raise_for_status()
    except Exception:
        return
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    for entry in root.findall("a:entry", ns):
        try:
            title     = (entry.findtext("a:title", "", ns) or "").strip()
            abstract  = (entry.findtext("a:summary", "", ns) or "").strip()
            published = entry.findtext("a:published", "", ns) or ""
            year      = int(published[:4]) if published else None
            if year_from and year and year < year_from:
                continue
            arxiv_id = ""
            id_el = entry.find("a:id", ns)
            if id_el is not None and id_el.text:
                raw_id = id_el.text.split("/abs/")[-1].strip()
                # strip version suffix: "1706.03762v5" → "1706.03762"
                arxiv_id = raw_id.split("v")[0] if "v" in raw_id else raw_id
            authors = [
                (a.findtext("a:name", "", ns) or "").strip()
                for a in entry.findall("a:author", ns)
            ]
            add(_paper_dict(
                title=title, abstract=abstract, authors=authors, year=year,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                source="arxiv",
            ))
        except Exception:
            continue


def _search_s2(client: httpx.Client, query: str, limit: int, add: Any) -> None:
    params = {"query": query, "fields": _S2_FIELDS, "limit": limit}
    try:
        resp = client.get(_S2_SEARCH, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return
    for item in data.get("data", []):
        try:
            add(_s2_item_to_dict(item))
        except Exception:
            continue


def _search_pubmed(client: httpx.Client, query: str, limit: int, add: Any) -> None:
    try:
        resp = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
        )
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return
    if not pmids:
        return
    try:
        fetch = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        )
        fetch.raise_for_status()
        root = ET.fromstring(fetch.text)
    except Exception:
        return
    for article in root.findall(".//PubmedArticle"):
        try:
            title    = article.findtext(".//ArticleTitle") or ""
            abstract = " ".join(t.text or "" for t in article.findall(".//AbstractText"))
            pmid_el  = article.find(".//PMID")
            pmid     = pmid_el.text if pmid_el is not None else None
            year_el  = article.find(".//PubDate/Year")
            year     = int(year_el.text) if year_el is not None and year_el.text else None
            authors  = []
            for a in article.findall(".//Author"):
                last = a.findtext("LastName") or ""
                fore = a.findtext("ForeName") or ""
                authors.append(f"{last} {fore}".strip())
            add(_paper_dict(
                title=title, abstract=abstract, authors=authors, year=year,
                pubmed_id=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                source="pubmed",
            ))
        except Exception:
            continue


def _search_openalex(
    client: httpx.Client,
    query: str,
    limit: int,
    year_from: int | None,
    add: Any,
) -> None:
    params: dict[str, Any] = {
        "search": query,
        "per_page": min(limit, 25),
        "select": "id,title,abstract_inverted_index,authorships,publication_year,doi,cited_by_count,primary_location",
    }
    if year_from:
        params["filter"] = f"publication_year:>{year_from - 1}"
    try:
        resp = client.get(_OPENALEX, params=params, headers={"User-Agent": "researchcrew/0.1"})
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return
    for item in data.get("results", []):
        try:
            inv      = item.get("abstract_inverted_index") or {}
            abstract = _reconstruct_abstract(inv)
            doi_raw  = item.get("doi") or ""
            doi      = doi_raw.replace("https://doi.org/", "") if doi_raw else None
            authors  = [
                a.get("author", {}).get("display_name", "") for a in (item.get("authorships") or [])
            ]
            oa_id    = (item.get("id") or "").replace("https://openalex.org/", "")
            loc      = item.get("primary_location") or {}
            url      = loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else None)
            add(_paper_dict(
                title=item.get("title") or "",
                abstract=abstract,
                authors=authors,
                year=item.get("publication_year"),
                doi=doi,
                openalex_id=oa_id,
                url=url,
                citation_count=item.get("cited_by_count") or 0,
                source="openalex",
            ))
        except Exception:
            continue


def _reconstruct_abstract(inv: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inv:
        return ""
    max_pos = max(pos for positions in inv.values() for pos in positions)
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            words[pos] = word
    return " ".join(w for w in words if w)


# ── module entrypoint (for python -m researchcrew.mcp.papers) ──────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        QUERY    = "attention is all you need transformer"
        ARXIV_ID = "1706.03762"

        print("=" * 60)
        print(f"LIVE SMOKE TEST  query='{QUERY}'")
        print("=" * 60)

        print("\n[1] search_papers (arxiv + semantic_scholar, max_results=5)")
        results = search_papers(QUERY, max_results=5)
        print(f"  → {len(results)} results")
        for p in results[:3]:
            print(f"    • {p['title'][:70]}  [{p['source']}]  cites={p['citation_count']}")
        assert len(results) > 0

        print(f"\n[2] fetch_paper (arXiv:{ARXIV_ID})")
        paper = fetch_paper(ARXIV_ID)
        print(f"  → {paper.get('title', '')[:70]}")

        print(f"\n[3] get_citations (limit=5)")
        cites = get_citations(ARXIV_ID, limit=5)
        print(f"  → {len(cites)} citing papers")

        print(f"\n[4] get_references (limit=10)")
        refs = get_references(ARXIV_ID, limit=10)
        print(f"  → {len(refs)} references")

        print(f"\n[5] find_related (limit=5)")
        related = find_related("transformer self-attention", limit=5)
        print(f"  → {len(related)} related papers")

        print("\n" + "=" * 60)
        print("SMOKE TEST COMPLETE")
        print("=" * 60)
    else:
        mcp.run()
