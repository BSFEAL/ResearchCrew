from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from researchcrew.literature.corpus import LiteratureCorpus, Paper, SearchQuery


class MultiSourceSearcher:
    """Parallel search across ArXiv, PubMed, and Semantic Scholar.

    Requires ``researchcrew[literature]`` (httpx).
    Each source is queried independently; results are merged and deduplicated
    by DOI/ArXiv ID before being added to the corpus.
    """

    _ARXIV_URL = "https://export.arxiv.org/api/query"
    _S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    _PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def search(
        self,
        query: str,
        corpus: LiteratureCorpus,
        sources: list[str] | None = None,
        max_results: int = 20,
        year_from: int | None = None,
    ) -> int:
        """Run search, add results to corpus. Returns number of new papers added."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "Install researchcrew[literature] for live search."
            ) from exc

        active = sources or ["arxiv", "semantic_scholar"]
        before = corpus.size

        with httpx.Client(timeout=self.timeout) as client:
            if "arxiv" in active:
                self._search_arxiv(client, query, corpus, max_results, year_from)
            if "semantic_scholar" in active:
                self._search_s2(client, query, corpus, max_results)
            if "pubmed" in active:
                self._search_pubmed(client, query, corpus, max_results)

        added = corpus.size - before
        corpus.query_log.append(
            SearchQuery(
                query=query,
                sources=active,
                max_results=max_results,
                result_count=added,
            )
        )
        return added

    # ── ArXiv (Atom feed, no key required) ────────────────────────────

    def _search_arxiv(
        self,
        client: Any,
        query: str,
        corpus: LiteratureCorpus,
        max_results: int,
        year_from: int | None,
    ) -> None:
        params: dict[str, Any] = {
            "search_query": f"all:{query}",
            "max_results": min(max_results, 100),
            "sortBy": "relevance",
        }
        try:
            resp = client.get(self._ARXIV_URL, params=params)
            resp.raise_for_status()
        except Exception:
            return

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ns):
            try:
                title = (entry.findtext("atom:title", "", ns) or "").strip()
                abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
                published = entry.findtext("atom:published", "", ns) or ""
                year = int(published[:4]) if published else None
                if year_from and year and year < year_from:
                    continue
                arxiv_id = ""
                for link in entry.findall("atom:id", ns):
                    raw = link.text or ""
                    arxiv_id = raw.split("/abs/")[-1].strip()
                authors = [
                    (a.findtext("atom:name", "", ns) or "").strip()
                    for a in entry.findall("atom:author", ns)
                ]
                corpus.add(Paper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    source="arxiv",
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                ))
            except Exception:
                continue

    # ── Semantic Scholar (free, 100 req/5min) ────────────────────────

    def _search_s2(
        self,
        client: Any,
        query: str,
        corpus: LiteratureCorpus,
        max_results: int,
    ) -> None:
        params = {
            "query": query,
            "fields": "title,abstract,year,authors,citationCount,externalIds,url",
            "limit": min(max_results, 100),
        }
        try:
            resp = client.get(self._S2_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return

        for item in data.get("data", []):
            try:
                ext = item.get("externalIds") or {}
                corpus.add(Paper(
                    title=item.get("title", ""),
                    abstract=item.get("abstract") or "",
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    year=item.get("year"),
                    source="semantic_scholar",
                    doi=ext.get("DOI"),
                    arxiv_id=ext.get("ArXiv"),
                    citation_count=item.get("citationCount") or 0,
                    url=item.get("url"),
                ))
            except Exception:
                continue

    # ── PubMed (E-utilities, free) ─────────────────────────────────

    def _search_pubmed(
        self,
        client: Any,
        query: str,
        corpus: LiteratureCorpus,
        max_results: int,
    ) -> None:
        try:
            resp = client.get(
                self._PUBMED_SEARCH,
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            )
            resp.raise_for_status()
            pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception:
            return

        if not pmids:
            return

        try:
            fetch_resp = client.get(
                self._PUBMED_FETCH,
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            )
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)
        except Exception:
            return

        for article in root.findall(".//PubmedArticle"):
            try:
                title = article.findtext(".//ArticleTitle") or ""
                abstract_texts = article.findall(".//AbstractText")
                abstract = " ".join(t.text or "" for t in abstract_texts)
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None
                year_el = article.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None
                author_els = article.findall(".//Author")
                authors = []
                for a in author_els:
                    last = a.findtext("LastName") or ""
                    fore = a.findtext("ForeName") or ""
                    authors.append(f"{last} {fore}".strip())
                corpus.add(Paper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    source="pubmed",
                    pubmed_id=pmid,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                ))
            except Exception:
                continue
