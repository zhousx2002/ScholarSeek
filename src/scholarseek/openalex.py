from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request

from .http_retry import OPENALEX_GATE, ServiceRateLimited, request_json
from .models import Paper, QueryPlan


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_WORK_URL = "https://api.openalex.org/works/{work_id}"


class OpenAlexClient:
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 20,
        pause: float = 0.15,
    ):
        self.email = email
        self.api_key = api_key
        self.timeout = timeout
        self.pause = pause

    def search(self, plan: QueryPlan, per_query: int = 10) -> List[Paper]:
        papers: Dict[str, Paper] = {}
        search_queries = list(plan.search_queries)
        max_workers = max(1, min(len(search_queries), _env_int("SCHOLARSEEK_OPENALEX_WORKERS", 3)))
        if max_workers <= 1 or len(search_queries) <= 1:
            return self._search_sequential(search_queries, plan, per_query)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(lambda q: list(self._search_once(q, plan, per_query)), search_query): search_query
                for search_query in search_queries
            }
            for future in as_completed(futures):
                search_query = futures[future]
                try:
                    results = future.result()
                except ServiceRateLimited as exc:
                    print(f"[warn] {exc}; skipping remaining OpenAlex queries", flush=True)
                    break
                except RuntimeError as exc:
                    print(f"[warn] {exc}")
                    continue
                for paper in results:
                    current = papers.get(paper.id)
                    if current is None or paper.citation_count > current.citation_count:
                        papers[paper.id] = paper
        return list(papers.values())

    def _search_sequential(self, search_queries: List[str], plan: QueryPlan, per_query: int) -> List[Paper]:
        papers: Dict[str, Paper] = {}
        for search_query in search_queries:
            try:
                for paper in self._search_once(search_query, plan, per_query):
                    current = papers.get(paper.id)
                    if current is None or paper.citation_count > current.citation_count:
                        papers[paper.id] = paper
            except ServiceRateLimited as exc:
                print(f"[warn] {exc}; skipping remaining OpenAlex queries", flush=True)
                break
            except RuntimeError as exc:
                print(f"[warn] {exc}")
                continue
        return list(papers.values())

    def _search_once(self, query: str, plan: QueryPlan, per_query: int) -> Iterable[Paper]:
        clean_query = _clean_search_query(query)
        if not clean_query:
            return
        params = {
            "search": clean_query,
            "per_page": max(1, min(per_query, 50)),
            "select": "id,doi,title,display_name,publication_year,primary_location,authorships,cited_by_count,open_access,abstract_inverted_index",
        }
        filters = []
        if plan.year_from:
            filters.append(f"from_publication_date:{plan.year_from}-01-01")
        if plan.year_to:
            filters.append(f"to_publication_date:{plan.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{OPENALEX_WORKS_URL}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "ScholarSeek-Agent/0.1"})
        try:
            payload = request_json(
                req,
                service="OpenAlex",
                gate=OPENALEX_GATE,
                timeout=self.timeout,
            )
        except ServiceRateLimited:
            raise
        except RuntimeError as exc:
            raise RuntimeError(f"OpenAlex request failed for query '{clean_query}': {exc}") from exc

        for work in payload.get("results", []):
            yield _parse_work(work)

    def get_work(self, work_id: str, select: str | None = None) -> Dict:
        clean_id = _openalex_work_id(work_id)
        if not clean_id:
            raise RuntimeError(f"invalid OpenAlex work id: {work_id}")
        params = {}
        if select:
            params["select"] = select
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        query = f"?{urlencode(params)}" if params else ""
        req = Request(
            f"{OPENALEX_WORK_URL.format(work_id=clean_id)}{query}",
            headers={"User-Agent": "ScholarSeek-Agent/0.1"},
        )
        try:
            return request_json(
                req,
                service="OpenAlex",
                gate=OPENALEX_GATE,
                timeout=self.timeout,
            )
        except ServiceRateLimited:
            raise
        except RuntimeError as exc:
            raise RuntimeError(f"OpenAlex work lookup failed for '{clean_id}': {exc}") from exc

    def cited_by(self, cited_by_api_url: str, limit: int = 8) -> List[Paper]:
        if not cited_by_api_url:
            return []
        params = {
            "per_page": max(1, min(limit, 50)),
            "select": "id,doi,title,display_name,publication_year,primary_location,authorships,cited_by_count,open_access,abstract_inverted_index",
        }
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        separator = "&" if "?" in cited_by_api_url else "?"
        req = Request(
            f"{cited_by_api_url}{separator}{urlencode(params)}",
            headers={"User-Agent": "ScholarSeek-Agent/0.1"},
        )
        try:
            payload = request_json(
                req,
                service="OpenAlex cited-by",
                gate=OPENALEX_GATE,
                timeout=self.timeout,
                max_retries=1,
            )
        except ServiceRateLimited:
            raise
        except RuntimeError as exc:
            raise RuntimeError(f"OpenAlex cited-by lookup failed: {exc}") from exc
        return [_parse_work(work) for work in payload.get("results", [])]


def _parse_work(work: Dict) -> Paper:
    title = work.get("display_name") or work.get("title") or ""
    venue = _venue_name(work)
    authors = [
        (authorship.get("author") or {}).get("display_name", "")
        for authorship in work.get("authorships", [])[:8]
    ]
    authors = [author for author in authors if author]
    url = _best_url(work)
    return Paper(
        id=work.get("id") or work.get("doi") or title,
        title=title,
        year=work.get("publication_year"),
        venue=venue,
        authors=authors,
        abstract=_invert_abstract(work.get("abstract_inverted_index") or {}),
        doi=work.get("doi"),
        url=url,
        citation_count=work.get("cited_by_count") or 0,
        source="OpenAlex",
        raw=work,
    )


def _venue_name(work: Dict) -> str:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name") or ""


def _best_url(work: Dict) -> Optional[str]:
    location = work.get("primary_location") or {}
    if location.get("landing_page_url"):
        return location["landing_page_url"]
    open_access = work.get("open_access") or {}
    return open_access.get("oa_url") or work.get("doi")


def _invert_abstract(index: Dict[str, List[int]]) -> str:
    if not index:
        return ""
    positions = []
    for word, offsets in index.items():
        for offset in offsets:
            positions.append((offset, word))
    return " ".join(word for _, word in sorted(positions))


def _clean_search_query(query: str) -> str:
    cleaned = " ".join(str(query).split())
    cleaned = cleaned.replace("?", " ").replace("？", " ")
    cleaned = cleaned.replace(":", " ").replace("：", " ")
    cleaned = cleaned.replace("“", " ").replace("”", " ").replace('"', " ")
    return " ".join(cleaned.split())


def _openalex_work_id(value: str | None) -> str:
    if not value:
        return ""
    value = str(value).strip()
    if value.startswith("https://openalex.org/"):
        return value.rstrip("/").rsplit("/", 1)[-1]
    if value.startswith("https://api.openalex.org/works/"):
        return value.rstrip("/").rsplit("/", 1)[-1]
    return value


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
