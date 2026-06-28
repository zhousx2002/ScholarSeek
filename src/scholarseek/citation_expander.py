from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote, urlencode
from urllib.request import Request

from .http_retry import request_json, semantic_scholar_gate
from .models import Paper
from .openalex import OpenAlexClient, _parse_work
from .semantic_scholar import _parse_paper


SEMANTIC_GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper"
CITATION_FIELDS = "paperId,title,abstract,year,venue,authors,url,externalIds,citationCount,publicationVenue,openAccessPdf"


class CitationExpander:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 15,
        enabled: bool = True,
        openalex_email: Optional[str] = None,
        openalex_api_key: Optional[str] = None,
        use_openalex: bool = False,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.enabled = enabled
        self.request_gate = semantic_scholar_gate(api_key)
        self.use_openalex = use_openalex
        self.include_reverse_citations = bool(api_key) or bool(use_openalex and openalex_api_key)
        self.openalex = OpenAlexClient(
            email=openalex_email,
            api_key=openalex_api_key,
            timeout=timeout,
        )

    def expand(
        self,
        seeds: Iterable[Paper],
        max_seeds: int = 3,
        per_seed: int = 8,
        include_citations: bool = True,
    ) -> List[Paper]:
        if not self.enabled:
            return []
        if not self.api_key and self.use_openalex:
            return self._expand_openalex(seeds, max_seeds, per_seed, include_citations)
        expanded: Dict[str, Paper] = {}
        tasks = []
        for seed in list(seeds)[:max_seeds]:
            paper_id = semantic_paper_id(seed)
            if not paper_id:
                continue
            relations = ["references"]
            if include_citations:
                relations.append("citations")
            for relation in relations:
                tasks.append((paper_id, relation))
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(tasks)))) as executor:
            futures = {
                executor.submit(self._fetch_relation, paper_id, relation, per_seed): (paper_id, relation)
                for paper_id, relation in tasks
            }
            for future in as_completed(futures):
                paper_id, relation = futures[future]
                try:
                    papers = future.result()
                except RuntimeError as exc:
                    print(f"[warn] {exc}")
                    continue
                for paper in papers:
                    key = paper.doi or _title_key(paper.title)
                    if key:
                        expanded[key] = paper
        return list(expanded.values())

    def _fetch_relation(self, paper_id: str, relation: str, limit: int) -> List[Paper]:
        params = urlencode({"fields": CITATION_FIELDS, "limit": max(1, min(limit, 100))})
        url = f"{SEMANTIC_GRAPH_URL}/{quote(paper_id, safe=':')}/{relation}?{params}"
        headers = {"User-Agent": "ScholarSeek-Agent/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            payload = request_json(
                Request(url, headers=headers),
                service=f"Semantic Scholar {relation}",
                gate=self.request_gate,
                timeout=self.timeout,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Semantic Scholar {relation} expansion failed for '{paper_id}': {exc}") from exc
        nested_key = "citedPaper" if relation == "references" else "citingPaper"
        return [_parse_paper(item[nested_key]) for item in payload.get("data", []) if item.get(nested_key)]

    def _expand_openalex(
        self,
        seeds: Iterable[Paper],
        max_seeds: int,
        per_seed: int,
        include_citations: bool,
    ) -> List[Paper]:
        expanded: Dict[str, Paper] = {}
        for seed in list(seeds)[:max_seeds]:
            work_id = openalex_work_id(seed)
            if not work_id:
                continue
            try:
                work = self.openalex.get_work(
                    work_id,
                    select=(
                        "id,doi,title,display_name,publication_year,primary_location,"
                        "authorships,cited_by_count,open_access,abstract_inverted_index,"
                        "referenced_works,cited_by_api_url"
                    ),
                )
            except RuntimeError as exc:
                print(f"[warn] OpenAlex citation seed lookup failed for '{seed.title}': {exc}", flush=True)
                continue
            for reference_id in (work.get("referenced_works") or [])[: max(1, per_seed)]:
                try:
                    reference = self.openalex.get_work(
                        reference_id,
                        select=(
                            "id,doi,title,display_name,publication_year,primary_location,"
                            "authorships,cited_by_count,open_access,abstract_inverted_index"
                        ),
                    )
                except RuntimeError as exc:
                    print(f"[warn] OpenAlex reference lookup failed for '{reference_id}': {exc}", flush=True)
                    continue
                paper = _parse_work(reference)
                key = paper.doi or _title_key(paper.title)
                if key:
                    expanded[key] = paper
            if include_citations:
                try:
                    for paper in self.openalex.cited_by(work.get("cited_by_api_url") or "", limit=per_seed):
                        key = paper.doi or _title_key(paper.title)
                        if key:
                            expanded[key] = paper
                except RuntimeError as exc:
                    print(f"[warn] OpenAlex cited-by expansion failed for '{seed.title}': {exc}", flush=True)
                    continue
        return list(expanded.values())


def semantic_paper_id(paper: Paper) -> Optional[str]:
    raw_id = (paper.raw or {}).get("paperId")
    if raw_id:
        return str(raw_id)
    if paper.doi:
        doi = paper.doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return f"DOI:{doi}"
    match = re.search(r"(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", paper.url or "")
    if match:
        return f"ARXIV:{match.group(1)}"
    if re.fullmatch(r"[0-9a-f]{40}", paper.id or "", flags=re.I):
        return paper.id
    return None


def openalex_work_id(paper: Paper) -> Optional[str]:
    raw_id = (paper.raw or {}).get("id")
    if raw_id and "openalex" in str(raw_id).lower():
        return str(raw_id)
    if paper.source.lower().startswith("openalex") and paper.id:
        return paper.id
    return None


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())
