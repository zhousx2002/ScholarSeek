from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class QueryPlan:
    original: str
    search_queries: List[str]
    must_terms: List[str] = field(default_factory=list)
    optional_terms: List[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    planner: str = "heuristic"


@dataclass
class Paper:
    id: str
    title: str
    year: Optional[int]
    venue: str
    authors: List[str]
    abstract: str
    doi: Optional[str]
    url: Optional[str]
    citation_count: int
    source: str
    raw: Dict
    score: float = 0.0
    relevance_reason: str = ""
