from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .models import Paper
from .qwen import request_qwen_json
from .spar_prompts import JUDGEMENT_SYSTEM, judgement_payload
from .trainable_reranker import get_reranker, pair_features


@dataclass(frozen=True)
class Judgement:
    paper: Paper
    decision: str
    score: float
    reason: str
    judge: str


class JudgementAgent:
    def __init__(
        self,
        reranker_path: Optional[str] = None,
        related_threshold: float = 0.62,
        unrelated_threshold: float = 0.32,
        use_qwen: bool = False,
        qwen_base_url: Optional[str] = None,
        qwen_model: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        max_qwen_reviews: int = 4,
    ):
        self.reranker_path = reranker_path
        self.related_threshold = related_threshold
        self.unrelated_threshold = unrelated_threshold
        self.use_qwen = use_qwen
        self.qwen_base_url = qwen_base_url
        self.qwen_model = qwen_model
        self.qwen_api_key = qwen_api_key
        self.max_qwen_reviews = max(0, max_qwen_reviews)

    def judge(self, query: str, papers: Iterable[Paper]) -> List[Judgement]:
        candidates = list(papers)
        if not candidates:
            return []
        scores = self._scores(query, candidates)
        midpoint = (self.related_threshold + self.unrelated_threshold) / 2.0
        uncertain_indices = [
            index
            for index, score in enumerate(scores)
            if self.unrelated_threshold <= score < self.related_threshold
        ]
        uncertain_indices.sort(key=lambda index: abs(scores[index] - midpoint))
        qwen_indices = set(uncertain_indices[: self.max_qwen_reviews])
        return [
            self._decision(query, paper, score, allow_qwen=index in qwen_indices)
            for index, (paper, score) in enumerate(zip(candidates, scores))
        ]

    def filter(
        self,
        query: str,
        papers: Iterable[Paper],
        keep_uncertain: bool = True,
        min_keep: int = 0,
    ) -> List[Paper]:
        accepted = []
        judgements = self.judge(query, papers)
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

    def _scores(self, query, papers):
        reranker_paths = [self.reranker_path, os.getenv("SCHOLARSEEK_FALLBACK_RERANKER_PATH")]
        for reranker_path in dict.fromkeys(path for path in reranker_paths if path):
            try:
                reranker = get_reranker(reranker_path)
                return reranker.score_pairs(query, [paper.title for paper in papers])
            except (RuntimeError, OSError, ValueError):
                pass
        scores = []
        for paper in papers:
            features = pair_features(query, f"{paper.title} {paper.abstract[:600]}")
            topical = 0.50 * features[0] + 0.25 * features[1] + 0.25 * features[2]
            scores.append(max(float(paper.score), min(1.0, topical)))
        return scores

    def _decision(self, query, paper, score, allow_qwen=True):
        if score >= self.related_threshold:
            return Judgement(paper, "related", score, f"relevance score={score:.4f}", "compact-or-rule")
        if score < self.unrelated_threshold:
            return Judgement(paper, "unrelated", score, f"relevance score={score:.4f}", "compact-or-rule")
        if allow_qwen and self.use_qwen and self.qwen_api_key:
            try:
                parsed = request_qwen_json(
                    JUDGEMENT_SYSTEM,
                    judgement_payload(query, paper),
                    base_url=self.qwen_base_url,
                    model=self.qwen_model,
                    api_key=self.qwen_api_key,
                    temperature=0.0,
                )
                decision = str(parsed.get("decision") or "uncertain").lower()
                if decision not in {"related", "unrelated", "uncertain"}:
                    decision = "uncertain"
                qwen_score = _bounded_score(parsed.get("score"), score)
                return Judgement(paper, decision, qwen_score, str(parsed.get("reason") or "Qwen review"), "qwen")
            except RuntimeError:
                pass
        return Judgement(paper, "uncertain", score, f"borderline relevance score={score:.4f}", "compact-or-rule")


def _bounded_score(value, fallback):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback
