from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .models import Paper, QueryPlan


class CompactFeatureReranker:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if path.is_dir():
            path = path / "compact_reranker.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.weights = [float(value) for value in payload["weights"]]
        self.bias = float(payload.get("bias", 0.0))
        self.feature_names = payload.get("feature_names") or FEATURE_NAMES

    def score_pairs(self, query: str, titles: List[str]) -> List[float]:
        scores = []
        for title in titles:
            logit = self.bias + sum(weight * value for weight, value in zip(self.weights, pair_features(query, title)))
            scores.append(_sigmoid(logit))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_path: str, batch_size: int = 4, max_length: Optional[int] = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Trainable reranker requires torch and transformers.") from exc

        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as exc:
            raise RuntimeError(f"Trainable reranker model could not be loaded from {model_path}.") from exc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        metadata_path = Path(model_path) / "scholarseek_reranker.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.max_length = max_length or int(metadata.get("max_length") or 256)
        self.use_amp = self.device.type == "cuda"

    def score_pairs(self, query: str, titles: List[str]) -> List[float]:
        scores: List[float] = []
        with self.torch.no_grad():
            for start in range(0, len(titles), self.batch_size):
                batch_titles = titles[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch_titles),
                    batch_titles,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with self.torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(**encoded).logits.squeeze(-1)
                scores.extend(self.torch.sigmoid(logits).detach().cpu().tolist())
        return scores


FEATURE_NAMES = [
    "query_coverage",
    "title_coverage",
    "jaccard",
    "ordered_phrase",
    "bigram_overlap",
    "length_balance",
    "rare_token_overlap",
    "numeric_overlap",
]

FEATURE_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "based",
    "by",
    "can",
    "could",
    "field",
    "fields",
    "for",
    "find",
    "from",
    "in",
    "looking",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "related",
    "research",
    "show",
    "some",
    "study",
    "that",
    "the",
    "tell",
    "to",
    "using",
    "what",
    "with",
    "work",
    "works",
    "you",
    "your",
    "please",
    "provide",
}


def pair_features(query: str, title: str) -> List[float]:
    query_tokens = _tokens(query)
    title_tokens = _tokens(title)
    query_set = set(query_tokens)
    title_set = set(title_tokens)
    overlap = query_set & title_set
    union = query_set | title_set
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    title_bigrams = set(zip(title_tokens, title_tokens[1:]))
    bigram_union = query_bigrams | title_bigrams
    query_numbers = {token for token in query_tokens if token.isdigit()}
    title_numbers = {token for token in title_tokens if token.isdigit()}
    rare_query = {token for token in query_set if len(token) >= 7}
    rare_title = {token for token in title_set if len(token) >= 7}
    length_balance = min(len(query_tokens), len(title_tokens)) / max(1, max(len(query_tokens), len(title_tokens)))
    return [
        len(overlap) / max(1, len(query_set)),
        len(overlap) / max(1, len(title_set)),
        len(overlap) / max(1, len(union)),
        1.0 if " ".join(query_tokens[:6]) and " ".join(query_tokens[:6]) in " ".join(title_tokens) else 0.0,
        len(query_bigrams & title_bigrams) / max(1, len(bigram_union)),
        length_balance,
        len(rare_query & rare_title) / max(1, len(rare_query)),
        len(query_numbers & title_numbers) / max(1, len(query_numbers)) if query_numbers else 0.0,
    ]


def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in FEATURE_STOPWORDS]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@lru_cache(maxsize=2)
def get_reranker(model_path: str):
    path = Path(model_path)
    compact_path = path / "compact_reranker.json" if path.is_dir() else path
    if compact_path.name == "compact_reranker.json" and compact_path.exists():
        return CompactFeatureReranker(str(compact_path))
    return CrossEncoderReranker(model_path)


def rerank_papers_with_model(
    papers: List[Paper],
    plan: QueryPlan,
    model_path: str,
    limit: int,
) -> List[Paper]:
    if not papers:
        return []
    reranker = get_reranker(model_path)
    titles = [paper.title for paper in papers]
    semantic_scores = reranker.score_pairs(plan.original, titles)
    semantic_order = sorted(range(len(papers)), key=lambda index: semantic_scores[index], reverse=True)
    semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    fusion_k = 20.0
    is_compact = "compact_reranker" in str(model_path)
    semantic_weight = 0.25 if is_compact else 0.65
    lexical_weight = 1.0 - semantic_weight
    lexical_scores = [paper.score for paper in papers]
    max_lexical = max(lexical_scores) if lexical_scores else 1.0
    ranked_items = []
    for lexical_rank, (paper, semantic_score) in enumerate(zip(papers, semantic_scores), start=1):
        semantic_rank = semantic_ranks[lexical_rank - 1]
        rrf_score = (fusion_k + 1.0) * (
            semantic_weight / (fusion_k + semantic_rank)
            + lexical_weight / (fusion_k + lexical_rank)
        )
        lexical_confidence = paper.score / max_lexical if max_lexical > 0 else 0.0
        display_score = min(0.99, max(0.0, 0.55 * semantic_score + 0.45 * lexical_confidence))
        paper.score = float(display_score)
        paper.relevance_reason = (
            f"RRF semantic_rank={semantic_rank}, lexical_rank={lexical_rank}, "
            f"semantic_score={semantic_score:.4f}, lexical_confidence={lexical_confidence:.4f}, "
            f"rrf_score={rrf_score:.4f}"
        )
        ranked_items.append((rrf_score, paper))
    ranked_items.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in ranked_items[:limit]]
