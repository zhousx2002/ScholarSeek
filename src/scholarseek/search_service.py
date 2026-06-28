from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional

from .config import get_api_config
from .formatters import format_json
from .query_planner import build_query_plan
from .qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, build_qwen_query_plan, synthesize_qwen_answer
from .ranker import rank_papers
from .retrievers import MultiSourceRetriever, parse_sources

SEARCH_CACHE: dict[str, tuple[float, Dict[str, Any]]] = {}
SEARCH_CACHE_TTL_SECONDS = 600


def search_papers(
    query: str,
    planner: str = "heuristic",
    answer: str = "none",
    sources: str = "openalex,arxiv",
    max_queries: int = 3,
    per_query: int = 5,
    limit: int = 10,
    openalex_email: Optional[str] = None,
    openalex_api_key: Optional[str] = None,
    semantic_scholar_api_key: Optional[str] = None,
    qwen_base_url: Optional[str] = None,
    qwen_model: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    strict_qwen: bool = False,
    strategy: str = "standard",
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    cache_key = json.dumps(
        {
            "query": query,
            "planner": planner,
            "answer": answer,
            "sources": sources,
            "max_queries": max_queries,
            "per_query": per_query,
            "limit": limit,
            "reranker_path": os.getenv("SCHOLARSEEK_RERANKER_PATH") or "",
            "strategy": strategy,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    cached = _get_cached(cache_key)
    if cached is not None:
        cached.setdefault("timing", {})["cache_hit"] = True
        return cached

    plan_started_at = time.perf_counter()
    config = get_api_config()
    qwen_base_url = qwen_base_url or config.qwen_base_url or DEFAULT_QWEN_BASE_URL
    qwen_model = qwen_model or config.qwen_model or DEFAULT_QWEN_MODEL
    qwen_api_key = qwen_api_key or config.qwen_api_key
    semantic_scholar_api_key = semantic_scholar_api_key or config.semantic_scholar_api_key
    openalex_email = openalex_email or config.openalex_email
    openalex_api_key = openalex_api_key or config.openalex_api_key

    if planner == "qwen":
        plan = build_qwen_query_plan(
            query,
            max_queries=max_queries,
            base_url=qwen_base_url,
            model=qwen_model,
            api_key=qwen_api_key,
            fallback_on_error=not strict_qwen,
        )
    else:
        plan = build_query_plan(query, max_queries=max_queries)
    plan_seconds = time.perf_counter() - plan_started_at
    _progress("query_plan_ready", queries=len(plan.search_queries), planner=plan.planner)

    retrieval_started_at = time.perf_counter()
    selected_sources = parse_sources(sources)
    retriever = MultiSourceRetriever(
        sources=selected_sources,
        openalex_email=openalex_email,
        openalex_api_key=openalex_api_key,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )
    papers = retriever.search(plan, per_query=per_query)
    retrieval_seconds = time.perf_counter() - retrieval_started_at
    _progress("initial_retrieval_ready", candidates=len(papers))
    pipeline_trace = {
        "strategy": "standard",
        "initial_candidates": len(papers),
        "final_papers": 0,
        "retrieved_candidate_titles": [paper.title for paper in papers],
        "retrieved_candidate_ids": [_paper_identifier(paper) for paper in papers],
    }
    if strategy in {"spar", "spar-qwen"}:
        from .citation_expander import CitationExpander
        from .judgement import JudgementAgent
        from .refchain_planner import RefChainPlanner

        judgement = JudgementAgent(
            reranker_path=config.reranker_path,
            use_qwen=strategy == "spar-qwen",
            qwen_base_url=qwen_base_url,
            qwen_model=qwen_model,
            qwen_api_key=qwen_api_key,
        )
        enable_openalex_refchain = _env_bool("SCHOLARSEEK_ENABLE_OPENALEX_REFCHAIN", False)
        enable_citation_expansion = ("semantic-scholar" in selected_sources) or (
            enable_openalex_refchain and "openalex" in selected_sources
        )
        spar_result = RefChainPlanner(
            retriever=retriever,
            judgement_agent=judgement,
            citation_expander=CitationExpander(
                api_key=semantic_scholar_api_key,
                enabled=enable_citation_expansion,
                openalex_email=openalex_email,
                openalex_api_key=openalex_api_key,
                use_openalex=enable_openalex_refchain and "openalex" in selected_sources,
            ),
            per_query=per_query,
        ).run(
            plan,
            papers,
            limit,
            qwen_config={
                "base_url": qwen_base_url,
                "model": qwen_model,
                "api_key": qwen_api_key if strategy == "spar-qwen" else None,
            },
        )
        ranked = spar_result.papers
        pipeline_trace = spar_result.trace
        _progress("spar_pipeline_ready", papers=len(ranked))
    else:
        ranking_started_at = time.perf_counter()
        ranked = rank_papers(papers, plan, limit=limit)
        pipeline_trace["ranking_seconds"] = round(time.perf_counter() - ranking_started_at, 3)
        pipeline_trace["final_papers"] = len(ranked)
    pipeline_trace["query_plan"] = {
        "planner": plan.planner,
        "search_queries": plan.search_queries,
        "must_terms": plan.must_terms,
        "optional_terms": plan.optional_terms,
        "year_from": plan.year_from,
        "year_to": plan.year_to,
    }

    synthesized_answer = None
    answer_seconds = 0.0
    if answer == "qwen":
        answer_started_at = time.perf_counter()
        synthesized_answer = synthesize_qwen_answer(
            query,
            ranked,
            base_url=qwen_base_url,
            model=qwen_model,
            api_key=qwen_api_key,
            fallback_on_error=not strict_qwen,
        )
        answer_seconds = time.perf_counter() - answer_started_at

    result = json.loads(format_json(plan, ranked, answer=synthesized_answer))
    result["reranker"] = _reranker_metadata()
    result["pipeline_trace"] = pipeline_trace
    result["timing"] = {
        "cache_hit": False,
        "planning_seconds": round(plan_seconds, 3),
        "retrieval_seconds": round(retrieval_seconds, 3),
        "search_seconds": round(time.perf_counter() - started_at, 3),
        "llm_answer_seconds": round(answer_seconds, 3),
    }
    _set_cached(cache_key, result)
    return result


def synthesize_answer_for_papers(
    query: str,
    papers: list[dict],
    qwen_base_url: Optional[str] = None,
    qwen_model: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    strict_qwen: bool = False,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    from .models import Paper

    config = get_api_config()
    qwen_base_url = qwen_base_url or config.qwen_base_url or DEFAULT_QWEN_BASE_URL
    qwen_model = qwen_model or config.qwen_model or DEFAULT_QWEN_MODEL
    qwen_api_key = qwen_api_key or config.qwen_api_key
    paper_objects = [
        Paper(
            id=str(item.get("id") or item.get("url") or item.get("title") or index),
            title=item.get("title") or "",
            year=item.get("year"),
            venue=item.get("venue") or "",
            authors=item.get("authors") or [],
            abstract=item.get("abstract") or "",
            doi=item.get("doi"),
            url=item.get("url"),
            citation_count=item.get("citation_count") or 0,
            source=item.get("source") or "",
            raw={},
            score=item.get("score") or 0.0,
            relevance_reason=item.get("reason") or "",
        )
        for index, item in enumerate(papers, start=1)
    ]
    answer = synthesize_qwen_answer(
        query,
        paper_objects,
        base_url=qwen_base_url,
        model=qwen_model,
        api_key=qwen_api_key,
        fallback_on_error=not strict_qwen,
    )
    return {
        "answer": answer,
        "timing": {
            "llm_answer_seconds": round(time.perf_counter() - started_at, 3),
        },
    }


def _get_cached(key: str) -> Optional[Dict[str, Any]]:
    item = SEARCH_CACHE.get(key)
    if item is None:
        return None
    created_at, value = item
    if time.time() - created_at > SEARCH_CACHE_TTL_SECONDS:
        SEARCH_CACHE.pop(key, None)
        return None
    return json.loads(json.dumps(value, ensure_ascii=False))


def _set_cached(key: str, value: Dict[str, Any]) -> None:
    SEARCH_CACHE[key] = (time.time(), json.loads(json.dumps(value, ensure_ascii=False)))


def _reranker_metadata() -> Dict[str, Any]:
    reranker_path = os.getenv("SCHOLARSEEK_RERANKER_PATH") or ""
    if not reranker_path:
        return {"enabled": False, "type": "lexical"}
    compact_path = os.path.join(reranker_path, "compact_reranker.json")
    if reranker_path.endswith("compact_reranker.json") or os.path.exists(compact_path):
        return {"enabled": True, "type": "PaSa compact trainable reranker", "path": reranker_path}
    return {"enabled": True, "type": "cross-encoder trainable reranker", "path": reranker_path}


def _paper_identifier(paper) -> str:
    raw = paper.raw or {}
    return raw.get("arxiv_id") or raw.get("paperId") or paper.doi or paper.id or paper.url or ""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _progress(stage: str, **details) -> None:
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)
