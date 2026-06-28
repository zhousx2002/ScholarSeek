from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from .models import Paper
from .qwen import request_qwen_json
from .query_planner import STOPWORDS
from .spar_prompts import QUERY_EVOLUTION_SYSTEM, query_evolution_payload


EVOLUTION_DIRECTIONS = (
    ("methodology", "methods comparison alternatives"),
    ("applications", "applications implementation evaluation"),
    ("limitations", "limitations challenges failure analysis"),
)


def evolve_queries(
    original_query: str,
    relevant_papers: Iterable[Paper],
    searched_queries: Iterable[str],
    max_queries: int = 3,
    qwen_base_url: Optional[str] = None,
    qwen_model: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
) -> List[str]:
    papers = list(relevant_papers)[:5]
    searched = list(searched_queries)
    if qwen_api_key:
        try:
            parsed = request_qwen_json(
                QUERY_EVOLUTION_SYSTEM,
                query_evolution_payload(original_query, searched, papers),
                base_url=qwen_base_url,
                model=qwen_model,
                api_key=qwen_api_key,
                temperature=0.0,
            )
            generated = _queries_from_response(parsed)
            cleaned = _novel_queries(generated, searched)
            if cleaned:
                return cleaned[:max_queries]
        except RuntimeError:
            pass
    return _heuristic_evolution(original_query, papers, searched, max_queries)


def _heuristic_evolution(original_query, papers, searched, max_queries):
    topic_terms = _content_terms(original_query)
    evidence_terms = []
    for paper in papers[:3]:
        evidence_terms.extend(_content_terms(paper.title))
    topic_keys = set(topic_terms)
    technical_terms = [term for term in _dedupe(evidence_terms) if term not in topic_keys and len(term) >= 5]
    topic = " ".join(_dedupe([*topic_terms[:5], *technical_terms[:2]]))
    if not topic:
        topic = " ".join(original_query.split()[:8])
    candidates = [f"{topic} {suffix}" for _, suffix in EVOLUTION_DIRECTIONS]
    return _novel_queries(candidates, searched)[:max_queries]


def _queries_from_response(parsed):
    if not isinstance(parsed, dict):
        return []
    values = parsed.get("evolved_queries") or []
    queries = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("query")
        if value:
            queries.append(str(value))
    return queries


def _novel_queries(queries, searched):
    searched_keys = {_query_key(query) for query in searched}
    output = []
    for query in queries:
        cleaned = " ".join(str(query).replace("?", " ").split())
        key = _query_key(cleaned)
        if len(cleaned.split()) >= 2 and key not in searched_keys:
            searched_keys.add(key)
            output.append(cleaned)
    return output


def _content_terms(text):
    tokens = re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _query_key(query):
    return re.sub(r"[^a-z0-9]+", "", query.lower())


def _dedupe(values):
    return list(dict.fromkeys(value for value in values if value))
