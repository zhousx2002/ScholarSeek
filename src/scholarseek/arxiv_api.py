from __future__ import annotations

import socket
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .http_retry import RequestGate, ServiceRateLimited
from .models import Paper, QueryPlan


ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_GATE = RequestGate(3.0)


class ArxivClient:
    def __init__(self, timeout: int = 25, pause: float = 3.0, max_retries: int = 1):
        self.timeout = timeout
        self.pause = pause
        self.max_retries = max(0, max_retries)

    def search(self, plan: QueryPlan, per_query: int = 10) -> List[Paper]:
        papers: Dict[str, Paper] = {}
        for search_query in plan.search_queries:
            for paper in self._search_once(search_query, plan, per_query):
                current = papers.get(paper.id)
                if current is None:
                    papers[paper.id] = paper
            time.sleep(self.pause)
        return list(papers.values())

    def _search_once(self, query: str, plan: QueryPlan, per_query: int) -> Iterable[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(per_query, 50)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_SEARCH_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "ScholarSeek-Agent/0.1"})
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                ARXIV_GATE.wait()
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8")
                break
            except ServiceRateLimited as exc:
                raise RuntimeError(f"arXiv temporarily disabled: {exc}") from exc
            except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code == 429:
                    retry_after = (exc.headers or {}).get("Retry-After")
                    try:
                        block_seconds = max(60.0, float(retry_after))
                    except (TypeError, ValueError):
                        block_seconds = 120.0
                    ARXIV_GATE.block(min(300.0, block_seconds))
                    raise RuntimeError(
                        f"arXiv rate limited; source disabled for {min(300.0, block_seconds):.0f}s"
                    ) from exc
                if attempt >= self.max_retries:
                    ARXIV_GATE.block(10.0)
                    raise RuntimeError(
                        f"arXiv request failed for query '{query}' after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = min(12.0, 3.0 * (2**attempt))
                print(
                    f"[warn] arXiv request failed; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.max_retries}): {exc}",
                    flush=True,
                )
                time.sleep(delay)

        for entry in ET.fromstring(payload).findall("atom:entry", ATOM_NS):
            paper = _parse_entry(entry)
            if _matches_year(paper, plan):
                yield paper


def _parse_entry(entry: ET.Element) -> Paper:
    entry_id = _text(entry, "atom:id")
    arxiv_id = entry_id.rstrip("/").split("/")[-1].split("v")[0]
    title = " ".join(_text(entry, "atom:title").split())
    abstract = " ".join(_text(entry, "atom:summary").split())
    published = _text(entry, "atom:published")
    year = _parse_year(published)
    authors = [
        _text(author, "atom:name")
        for author in entry.findall("atom:author", ATOM_NS)
        if _text(author, "atom:name")
    ]
    doi = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "doi":
            doi = link.attrib.get("href")
    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=title,
        year=year,
        venue="arXiv",
        authors=authors,
        abstract=abstract,
        doi=doi,
        url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_id,
        citation_count=0,
        source="arXiv",
        raw={"arxiv_id": arxiv_id, "published": published},
    )


def _text(entry: ET.Element, path: str) -> str:
    value = entry.findtext(path, default="", namespaces=ATOM_NS)
    return value or ""


def _parse_year(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).year
    except ValueError:
        return None


def _matches_year(paper: Paper, plan: QueryPlan) -> bool:
    if paper.year is None:
        return True
    if plan.year_from and paper.year < plan.year_from:
        return False
    if plan.year_to and paper.year > plan.year_to:
        return False
    return True
