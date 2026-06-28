from __future__ import annotations

import math
import os
import re
from typing import Iterable, List, Tuple

from .models import Paper, QueryPlan


def rank_papers(papers: Iterable[Paper], plan: QueryPlan, limit: int = 20) -> List[Paper]:
    scored = []
    for paper in papers:
        paper.score, paper.relevance_reason = score_paper(paper, plan)
        scored.append(paper)
    scored.sort(key=lambda item: item.score, reverse=True)
    reranker_paths = [
        path
        for path in (
            os.getenv("SCHOLARSEEK_RERANKER_PATH"),
            os.getenv("SCHOLARSEEK_FALLBACK_RERANKER_PATH"),
        )
        if path
    ]
    for reranker_path in dict.fromkeys(reranker_paths):
        try:
            from .trainable_reranker import rerank_papers_with_model

            return rerank_papers_with_model(scored, plan, reranker_path, limit)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[warn] trainable reranker unavailable: {exc}")
    return scored[:limit]


def score_paper(paper: Paper, plan: QueryPlan) -> Tuple[float, str]:
    text = f"{paper.title} {paper.abstract} {paper.venue}".lower()
    title = paper.title.lower()

    must_hits = [term for term in plan.must_terms if _contains(text, term)]
    optional_hits = [term for term in plan.optional_terms if _contains(text, term)]
    title_hits = [term for term in plan.must_terms if _contains(title, term)]
    title_phrase_bonus = _title_phrase_bonus(title, plan)
    query_coverage_bonus = _query_coverage_bonus(title, plan)
    named_entity_bonus = _named_entity_bonus(title, plan)
    canonical_model_bonus = _canonical_model_bonus(title, plan)

    must_ratio = len(must_hits) / max(1, len(plan.must_terms))
    optional_ratio = len(optional_hits) / max(1, len(plan.optional_terms))
    title_ratio = len(title_hits) / max(1, len(plan.must_terms))
    citation_bonus = min(1.0, math.log10(max(1, paper.citation_count)) / 4)
    year_bonus = _year_bonus(paper, plan)

    score = (
        0.48 * must_ratio
        + 0.10 * optional_ratio
        + 0.24 * title_ratio
        + 0.07 * title_phrase_bonus
        + 0.10 * query_coverage_bonus
        + 0.10 * named_entity_bonus
        + 0.16 * canonical_model_bonus
        + 0.03 * citation_bonus
        + 0.02 * year_bonus
    )

    reasons = []
    if title_phrase_bonus:
        reasons.append("matched topic phrase in title")
    if query_coverage_bonus:
        reasons.append("strong query-token coverage")
    if named_entity_bonus:
        reasons.append("matched named model/task tokens")
    if canonical_model_bonus:
        reasons.append("matched canonical foundation-model title pattern")
    if must_hits:
        reasons.append("matched core terms: " + ", ".join(must_hits[:5]))
    if optional_hits:
        reasons.append("matched expanded terms: " + ", ".join(optional_hits[:5]))
    if paper.year:
        reasons.append(f"published in {paper.year}")
    if paper.citation_count:
        reasons.append(f"{paper.citation_count} citations")
    return score, "; ".join(reasons) or "weak lexical match"


def _contains(text: str, term: str) -> bool:
    normalized = term.lower()
    if " " in normalized:
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}\b", text) is not None


def _title_phrase_bonus(title: str, plan: QueryPlan) -> float:
    for query in plan.search_queries[1:]:
        normalized = query.lower()
        if 2 <= len(normalized.split()) <= 5 and normalized in title:
            return 1.0
    return 0.0


def _query_coverage_bonus(title: str, plan: QueryPlan) -> float:
    title_tokens = set(_tokens(title))
    best = 0.0
    for query in plan.search_queries:
        query_tokens = [token for token in _tokens(query) if len(token) >= 3]
        if not query_tokens:
            continue
        overlap = len(title_tokens & set(query_tokens)) / len(set(query_tokens))
        best = max(best, overlap)
    return best


def _named_entity_bonus(title: str, plan: QueryPlan) -> float:
    title_tokens = set(_tokens(title))
    plan_tokens = set(_tokens(" ".join([plan.original, *plan.search_queries, *plan.must_terms])))
    named_tokens = {
        token
        for token in plan_tokens
        if token in {"bert", "gpt", "t5", "palm", "llama", "gat"}
        or len(token) >= 8
        or token in {"bandits", "bandit"}
    }
    if not named_tokens:
        return 0.0
    return min(1.0, len(title_tokens & named_tokens) / max(1, min(3, len(named_tokens))))


def _canonical_model_bonus(title: str, plan: QueryPlan) -> float:
    plan_text = " ".join([plan.original, *plan.search_queries, *plan.must_terms]).lower()
    if not any(token in plan_text for token in ("foundation", "natural language processing", "bert", "gpt", "t5", "palm", "llama")):
        return 0.0
    normalized = title.lower()
    patterns = (
        r"^bert[:\s-]",
        r"language models are few[-\s]?shot learners",
        r"exploring the limits of transfer learning.*text[-\s]?to[-\s]?text transformer",
        r"^palm[:\s-].*scaling language modeling",
        r"^llama[:\s-].*foundation language models",
        r"pre[-\s]?training.*bidirectional transformers.*language understanding",
    )
    return 1.0 if any(re.search(pattern, normalized) for pattern in patterns) else 0.0


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _year_bonus(paper: Paper, plan: QueryPlan) -> float:
    if paper.year is None:
        return 0.0
    if plan.year_from and paper.year < plan.year_from:
        return 0.0
    if plan.year_to and paper.year > plan.year_to:
        return 0.0
    if plan.year_from or plan.year_to:
        return 1.0
    return max(0.0, min(1.0, (paper.year - 2015) / 10))
