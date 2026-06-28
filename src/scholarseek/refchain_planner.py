from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .citation_expander import CitationExpander
from .judgement import JudgementAgent
from .listwise_reranker import rerank_with_qwen
from .models import Paper, QueryPlan
from .query_evolver import evolve_queries
from .query_planner import build_query_plan
from .ranker import rank_papers, score_paper
from .trainable_reranker import get_reranker


@dataclass
class RefChainResult:
    papers: List[Paper]
    trace: Dict[str, Any]


class RefChainPlanner:
    """One-layer SPAR-style retrieval expansion with bounded latency and cost."""

    def __init__(
        self,
        retriever,
        judgement_agent: JudgementAgent,
        citation_expander: CitationExpander,
        per_query: int = 8,
        max_evolved_queries: int = 3,
        max_seed_papers: int = 3,
    ):
        self.retriever = retriever
        self.judgement_agent = judgement_agent
        self.citation_expander = citation_expander
        self.per_query = per_query
        self.max_evolved_queries = max_evolved_queries
        self.max_seed_papers = max_seed_papers

    def run(self, plan: QueryPlan, initial_papers: List[Paper], limit: int, qwen_config=None) -> RefChainResult:
        judgement_pool_size = max(100, limit * 10)
        initial_ranked = rank_papers(initial_papers, plan, limit=judgement_pool_size)
        initial_judgements = self.judgement_agent.judge(plan.original, initial_ranked)
        judged_initial = _accepted_papers(
            initial_judgements,
            keep_uncertain=True,
            min_keep=judgement_pool_size,
        )
        _progress("initial_judgement_ready", candidates=len(initial_ranked), kept=len(judged_initial))
        seeds = [result.paper for result in initial_judgements if result.decision == "related"][: self.max_seed_papers]
        if len(seeds) < self.max_seed_papers:
            seed_ids = {id(paper) for paper in seeds}
            for result in sorted(initial_judgements, key=lambda item: item.score, reverse=True):
                if id(result.paper) not in seed_ids:
                    seeds.append(result.paper)
                    seed_ids.add(id(result.paper))
                if len(seeds) >= self.max_seed_papers:
                    break

        citation_candidates = self.citation_expander.expand(
            seeds,
            max_seeds=self.max_seed_papers,
            per_seed=max(3, min(self.per_query, 8)),
            include_citations=self.citation_expander.include_reverse_citations,
        )
        judged_citations = self.judgement_agent.filter(
            plan.original,
            citation_candidates,
            keep_uncertain=False,
            min_keep=0,
        )
        _progress("citation_expansion_ready", candidates=len(citation_candidates), kept=len(judged_citations))

        qwen_config = qwen_config or {}
        evolved_queries = evolve_queries(
            plan.original,
            seeds,
            plan.search_queries,
            max_queries=self.max_evolved_queries,
            qwen_base_url=qwen_config.get("base_url"),
            qwen_model=qwen_config.get("model"),
            qwen_api_key=qwen_config.get("api_key"),
        )
        _progress("query_evolution_ready", queries=len(evolved_queries))
        evolved_candidates = []
        for evolved_query in evolved_queries:
            try:
                evolved_candidates.extend(self._search_evolved(evolved_query))
            except RuntimeError as exc:
                print(f"[warn] evolved query failed: {exc}")
        judged_evolved = self.judgement_agent.filter(
            plan.original,
            evolved_candidates,
            keep_uncertain=True,
            min_keep=min(judgement_pool_size, len(evolved_candidates)),
        )
        _progress("evolved_retrieval_ready", candidates=len(evolved_candidates), kept=len(judged_evolved))

        retrieved_pool = _merge_papers([*initial_papers, *citation_candidates, *evolved_candidates])
        review_candidates = _review_candidates(retrieved_pool, plan, limit=max(10, limit))
        rescue_candidates = _lexical_rescue_candidates(retrieved_pool, plan, limit=max(30, limit * 4))
        merged = _merge_papers(
            [*judged_initial, *judged_citations, *judged_evolved, *review_candidates, *rescue_candidates]
        )
        ranked_pool = rank_papers(merged, plan, limit=max(judgement_pool_size, limit))
        listwise_pool = _merge_papers([*ranked_pool[:32], *review_candidates[:6], *rescue_candidates[:8]])
        listwise_ranked = rerank_with_qwen(
            plan.original,
            listwise_pool,
            limit=max(limit, 20),
            base_url=qwen_config.get("base_url"),
            model=qwen_config.get("model"),
            api_key=qwen_config.get("api_key"),
            max_candidates=24,
        )
        final_papers = _ensure_lexical_rescue(
            listwise_ranked,
            rescue_candidates,
            limit,
            minimum_rescues=0,
        )
        final_papers = _ensure_review_diversity(
            final_papers,
            review_candidates,
            limit,
            minimum_reviews=0,
        )
        _progress("final_ranking_ready", candidates=len(merged), selected=len(final_papers))
        trace = {
            "strategy": "spar-one-layer",
            "initial_candidates": len(initial_papers),
            "judged_initial": len(judged_initial),
            "citation_candidates": len(citation_candidates),
            "accepted_citations": len(judged_citations),
            "evolved_queries": evolved_queries,
            "evolved_candidates": len(evolved_candidates),
            "accepted_evolved": len(judged_evolved),
            "final_papers": len(final_papers),
            "listwise_candidates": len(listwise_pool),
            "listwise_reranker": "qwen" if qwen_config.get("api_key") else "disabled",
            "lexical_rescue_candidates": len(rescue_candidates),
            "retrieved_candidate_titles": [paper.title for paper in retrieved_pool],
            "retrieved_candidate_ids": [_paper_identifier(paper) for paper in retrieved_pool],
            "max_refchain_depth": 1,
        }
        return RefChainResult(final_papers, trace)

    def _search_evolved(self, evolved_query):
        evolved_plan = build_query_plan(evolved_query, max_queries=1)
        return self.retriever.search(evolved_plan, per_query=self.per_query)


def _merge_papers(papers):
    merged = {}
    title_index = {}
    for paper in papers:
        title_key = re.sub(r"[^a-z0-9]+", "", paper.title.lower())
        if not title_key:
            continue
        key = _paper_key(paper)
        existing_key = key if key in merged else title_index.get(title_key)
        current = merged.get(existing_key) if existing_key else None
        if current is None or paper.score > current.score:
            target_key = existing_key or key
            merged[target_key] = paper
            title_index[title_key] = target_key
    return list(merged.values())


def _accepted_papers(judgements, keep_uncertain, min_keep=0):
    accepted = []
    for result in judgements:
        if result.decision == "related" or (keep_uncertain and result.decision == "uncertain"):
            result.paper.score = result.score
            result.paper.relevance_reason = f"Judgement Agent: {result.decision}; {result.reason}"
            accepted.append(result.paper)
    accepted_ids = {id(paper) for paper in accepted}
    for result in sorted(judgements, key=lambda item: item.score, reverse=True):
        if len(accepted) >= min(min_keep, len(judgements)):
            break
        if id(result.paper) in accepted_ids:
            continue
        result.paper.score = result.score
        result.paper.relevance_reason = (
            f"Judgement Agent: relative-rank fallback; original={result.decision}; {result.reason}"
        )
        accepted.append(result.paper)
        accepted_ids.add(id(result.paper))
    return accepted


def _paper_key(paper):
    if paper.doi:
        return "doi:" + paper.doi.lower().removeprefix("https://doi.org/")
    return "title:" + re.sub(r"[^a-z0-9]+", "", paper.title.lower())


def _paper_identifier(paper):
    raw = paper.raw or {}
    return (
        raw.get("arxiv_id")
        or raw.get("paperId")
        or paper.doi
        or paper.id
        or paper.url
        or ""
    )


def _review_candidates(papers, plan, limit=10):
    candidates = [
        paper
        for paper in papers
        if re.search(r"\b(survey|review|overview)\b", paper.title, flags=re.I)
    ]
    lexical_scores = [score_paper(paper, plan)[0] for paper in candidates]
    fallback_path = os.getenv("SCHOLARSEEK_FALLBACK_RERANKER_PATH")
    if fallback_path and candidates:
        try:
            compact_scores = get_reranker(fallback_path).score_pairs(
                plan.original,
                [paper.title for paper in candidates],
            )
            scored = list(zip(compact_scores, lexical_scores, candidates))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return [paper for _, _, paper in scored[:limit]]
        except (RuntimeError, OSError, ValueError):
            pass
    candidates.sort(key=lambda paper: score_paper(paper, plan)[0], reverse=True)
    return candidates[:limit]


def _lexical_rescue_candidates(papers, plan, limit=30):
    scored = []
    for paper in papers:
        score, reason = score_paper(paper, plan)
        paper.score = score
        paper.relevance_reason = f"lexical rescue; {reason}"
        scored.append(paper)
    scored.sort(key=lambda paper: paper.score, reverse=True)
    return scored[:limit]


def _ensure_review_diversity(ranked_papers, review_candidates, limit, minimum_reviews=1):
    selected = list(ranked_papers[:limit])
    selected_keys = {_paper_key(paper) for paper in selected}
    review_count = sum(
        bool(re.search(r"\b(survey|review|overview)\b", paper.title, flags=re.I))
        for paper in selected
    )
    insertion_points = (2, 4, 6, 8)
    for review in review_candidates:
        if review_count >= min(minimum_reviews, limit):
            break
        key = _paper_key(review)
        if key in selected_keys:
            continue
        review.relevance_reason = f"result-type diversity (survey/review); {review.relevance_reason}"
        position = min(insertion_points[min(review_count, len(insertion_points) - 1)], len(selected))
        selected.insert(position, review)
        selected_keys.add(key)
        review_count += 1
        if len(selected) > limit:
            removed = selected.pop()
            selected_keys.discard(_paper_key(removed))
    return selected[:limit]


def _ensure_lexical_rescue(ranked_papers, rescue_candidates, limit, minimum_rescues=4):
    selected = list(ranked_papers[:limit])
    selected_keys = {_paper_key(paper) for paper in selected}
    rescue_count = sum(_paper_key(paper) in {_paper_key(candidate) for candidate in rescue_candidates[:minimum_rescues]} for paper in selected)
    insertion_points = (0, 2, 4, 6)
    for rescue in rescue_candidates:
        if rescue_count >= min(minimum_rescues, limit):
            break
        key = _paper_key(rescue)
        if key in selected_keys:
            continue
        rescue.relevance_reason = f"lexical rescue selected; {rescue.relevance_reason}"
        position = min(insertion_points[min(rescue_count, len(insertion_points) - 1)], len(selected))
        selected.insert(position, rescue)
        selected_keys.add(key)
        rescue_count += 1
        if len(selected) > limit:
            removed = selected.pop()
            selected_keys.discard(_paper_key(removed))
    return selected[:limit]


def _progress(stage, **details):
    print({"stage": stage, **details}, flush=True)
