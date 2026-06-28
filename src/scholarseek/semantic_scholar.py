from __future__ import annotations

from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request

from .http_retry import ServiceRateLimited, request_json, semantic_scholar_gate
from .models import Paper, QueryPlan


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20, pause: float = 0.35):
        self.api_key = api_key
        self.timeout = timeout
        self.pause = pause
        self.request_gate = semantic_scholar_gate(api_key)

    def search(self, plan: QueryPlan, per_query: int = 10) -> List[Paper]:
        papers: Dict[str, Paper] = {}
        for search_query in plan.search_queries:
            try:
                for paper in self._search_once(search_query, plan, per_query):
                    current = papers.get(paper.id)
                    if current is None or paper.citation_count > current.citation_count:
                        papers[paper.id] = paper
            except ServiceRateLimited as exc:
                print(f"[warn] {exc}; skipping remaining Semantic Scholar queries", flush=True)
                break
            except RuntimeError as exc:
                print(f"[warn] Semantic Scholar query '{search_query}' skipped: {exc}", flush=True)
        return list(papers.values())

    def _search_once(self, query: str, plan: QueryPlan, per_query: int) -> Iterable[Paper]:
        params = {
            "query": query,
            "limit": max(1, min(per_query, 100)),
            "fields": "paperId,title,abstract,year,venue,authors,url,externalIds,citationCount,publicationVenue,openAccessPdf",
        }
        year_filter = _year_filter(plan)
        if year_filter:
            params["year"] = year_filter

        url = f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{urlencode(params)}"
        headers = {"User-Agent": "ScholarSeek-Agent/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = Request(url, headers=headers)
        payload = request_json(
            request,
            service="Semantic Scholar",
            gate=self.request_gate,
            timeout=self.timeout,
        )

        for paper in payload.get("data", []):
            yield _parse_paper(paper)


def _year_filter(plan: QueryPlan) -> Optional[str]:
    if plan.year_from and plan.year_to:
        return f"{plan.year_from}-{plan.year_to}"
    if plan.year_from:
        return f"{plan.year_from}-"
    if plan.year_to:
        return f"-{plan.year_to}"
    return None


def _parse_paper(data: Dict) -> Paper:
    external_ids = data.get("externalIds") or {}
    venue = data.get("venue") or ""
    publication_venue = data.get("publicationVenue") or {}
    if not venue and publication_venue:
        venue = publication_venue.get("name") or ""
    open_access = data.get("openAccessPdf") or {}
    url = data.get("url") or open_access.get("url")
    doi = external_ids.get("DOI")
    if doi and not doi.startswith("http"):
        doi = f"https://doi.org/{doi}"
    return Paper(
        id=data.get("paperId") or doi or data.get("title") or "",
        title=data.get("title") or "",
        year=data.get("year"),
        venue=venue,
        authors=[author.get("name", "") for author in data.get("authors", []) if author.get("name")],
        abstract=data.get("abstract") or "",
        doi=doi,
        url=url,
        citation_count=data.get("citationCount") or 0,
        source="Semantic Scholar",
        raw=data,
    )
