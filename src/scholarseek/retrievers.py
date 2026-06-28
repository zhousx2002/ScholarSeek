from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .arxiv_api import ArxivClient
from .local_corpus import LocalCorpusClient
from .models import Paper, QueryPlan
from .openalex import OpenAlexClient
from .semantic_scholar import SemanticScholarClient


SUPPORTED_SOURCES = ("local", "openalex", "semantic-scholar", "arxiv")


class MultiSourceRetriever:
    def __init__(
        self,
        sources: Iterable[str] = SUPPORTED_SOURCES,
        openalex_email: Optional[str] = None,
        openalex_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
    ):
        self.sources = [_normalize_source(source) for source in sources]
        unknown = sorted(set(self.sources) - set(SUPPORTED_SOURCES))
        if unknown:
            raise ValueError(f"Unsupported source(s): {', '.join(unknown)}")
        self.openalex_email = openalex_email
        self.openalex_api_key = openalex_api_key
        self.semantic_scholar_api_key = semantic_scholar_api_key

    def search(self, plan: QueryPlan, per_query: int = 10) -> List[Paper]:
        merged: Dict[str, Paper] = {}
        title_index: Dict[str, str] = {}
        for source in self.sources:
            try:
                papers = self._client_for(source).search(plan, per_query=per_query)
            except RuntimeError as exc:
                print(f"[warn] {exc}")
                continue
            for paper in papers:
                key = _paper_key(paper)
                title_key = _title_key(paper)
                existing_key = key if key in merged else title_index.get(title_key)
                current = merged.get(existing_key) if existing_key else None
                if current is None:
                    merged[key] = paper
                    if title_key:
                        title_index[title_key] = key
                else:
                    merged[existing_key] = _merge_papers(current, paper)
        return list(merged.values())

    def _client_for(self, source: str):
        if source == "local":
            return LocalCorpusClient()
        if source == "openalex":
            return OpenAlexClient(email=self.openalex_email, api_key=self.openalex_api_key)
        if source == "semantic-scholar":
            return SemanticScholarClient(api_key=self.semantic_scholar_api_key)
        if source == "arxiv":
            return ArxivClient()
        raise ValueError(source)


def parse_sources(value: str) -> List[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(SUPPORTED_SOURCES)
    return [_normalize_source(item) for item in value.split(",") if item.strip()]


def _normalize_source(source: str) -> str:
    normalized = source.strip().lower().replace("_", "-")
    aliases = {
        "s2": "semantic-scholar",
        "semantic": "semantic-scholar",
        "semanticscholar": "semantic-scholar",
        "open-alex": "openalex",
    }
    return aliases.get(normalized, normalized)


def _paper_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower().removeprefix('https://doi.org/')}"
    title = _title_key(paper)
    if title:
        return f"title:{title}"
    return f"{paper.source}:{paper.id}"


def _title_key(paper: Paper) -> str:
    return re.sub(r"[^a-z0-9]+", "", paper.title.lower())


def _merge_papers(primary: Paper, secondary: Paper) -> Paper:
    if secondary.citation_count > primary.citation_count:
        primary.citation_count = secondary.citation_count
    if not primary.abstract and secondary.abstract:
        primary.abstract = secondary.abstract
    if not primary.venue and secondary.venue:
        primary.venue = secondary.venue
    if not primary.doi and secondary.doi:
        primary.doi = secondary.doi
    if not primary.url and secondary.url:
        primary.url = secondary.url
    if not primary.authors and secondary.authors:
        primary.authors = secondary.authors
    if secondary.source not in primary.source:
        primary.source = f"{primary.source}, {secondary.source}"
    return primary
