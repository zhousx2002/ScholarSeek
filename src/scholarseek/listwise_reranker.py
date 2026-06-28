from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .models import Paper
from .qwen import request_qwen_json
from .spar_prompts import AUTHORITY_RERANK_SYSTEM


def rerank_with_qwen(
    query: str,
    papers: Iterable[Paper],
    limit: int,
    *,
    base_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    max_candidates: Optional[int] = None,
) -> List[Paper]:
    max_candidates = max_candidates or _env_int("QWEN_LISTWISE_MAX_CANDIDATES", 24)
    candidates = list(papers)[:max_candidates]
    if not candidates or not api_key:
        return candidates[:limit]
    payload = {
        "query": query,
        "instructions": (
            "Rank the candidates as a set. Prioritize direct topical relevance, then coverage of the "
            "request, venue/citation authority, and a useful mix of surveys and concrete methods. "
            "Return 1-based candidate indices in best-first order."
        ),
        "candidates": [
            {
                "index": index,
                "title": paper.title,
                "abstract": (paper.abstract or "")[:220],
                "year": paper.year,
                "venue": paper.venue,
                "citation_count": paper.citation_count,
            }
            for index, paper in enumerate(candidates, start=1)
        ],
        "required_schema": {
            "papers": [{"index": "1-based integer", "score": "0 to 1", "reason": "short string"}]
        },
    }
    try:
        parsed = request_qwen_json(
            AUTHORITY_RERANK_SYSTEM,
            payload,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.0,
            timeout=_env_int("QWEN_LISTWISE_TIMEOUT", 25),
            max_tokens=500,
        )
    except RuntimeError as exc:
        print(f"[warn] Qwen listwise reranker unavailable: {exc}", flush=True)
        return candidates[:limit]

    selected = []
    selected_indices = set()
    for item in parsed.get("papers") or []:
        try:
            index = int(item.get("index")) - 1
        except (AttributeError, TypeError, ValueError):
            continue
        if index < 0 or index >= len(candidates) or index in selected_indices:
            continue
        paper = candidates[index]
        paper.relevance_reason = f"Qwen listwise: {item.get('reason') or 'set-level relevance'}"
        selected.append(paper)
        selected_indices.add(index)
        if len(selected) >= limit:
            break
    for index, paper in enumerate(candidates):
        if len(selected) >= limit:
            break
        if index not in selected_indices:
            selected.append(paper)
    return selected


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default
