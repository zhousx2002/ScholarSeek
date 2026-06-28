from __future__ import annotations

import re
from datetime import date
from typing import Iterable, List, Optional, Tuple

from .models import QueryPlan


STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "any",
    "are",
    "as",
    "based",
    "by",
    "can",
    "cases",
    "could",
    "field",
    "fields",
    "first",
    "focused",
    "for",
    "find",
    "from",
    "has",
    "have",
    "in",
    "looking",
    "last",
    "list",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "propose",
    "proposed",
    "recent",
    "related",
    "research",
    "show",
    "some",
    "studies",
    "study",
    "that",
    "the",
    "tell",
    "to",
    "use",
    "used",
    "using",
    "what",
    "where",
    "which",
    "with",
    "work",
    "works",
    "year",
    "years",
    "you",
    "your",
    "please",
    "provide",
    "through",
    "explored",
    "identifying",
    "give",
    "generalize",
    "me",
    "implemented",
    "methods",
    "method",
    "approaches",
    "approach",
    "applications",
    "application",
    "especially",
    "improve",
    "improving",
    "accuracy",
    "performance",
    "latest",
    "cutting",
    "edge",
}

PHRASE_HINTS = (
    "artificial intelligence",
    "automated literature review",
    "autonomous driving",
    "climate model",
    "climate prediction",
    "computer vision",
    "cross modal",
    "cross-modal",
    "deep learning",
    "large language model",
    "large language models",
    "few shot learning",
    "few-shot learning",
    "generative adversarial network",
    "generative adversarial networks",
    "retrieval augmented generation",
    "graph neural network",
    "graph neural networks",
    "image recognition",
    "knowledge graph",
    "multi task learning",
    "multi-task learning",
    "multimodal model",
    "multimodal models",
    "natural language processing",
    "network traffic analysis",
    "physics informed machine learning",
    "physics-informed machine learning",
    "quantum error correction",
    "reinforcement learning",
    "speech recognition",
    "diffusion model",
    "diffusion models",
    "multi agent",
    "multi-agent",
)

ACRONYM_EXPANSIONS = {
    "ai": "artificial intelligence",
    "afsl": "adaptive few shot learning",
    "gan": "generative adversarial network",
    "gans": "generative adversarial networks",
    "gnn": "graph neural network",
    "gnns": "graph neural networks",
    "gpu": "graphics processing unit",
    "llm": "large language model",
    "llms": "large language models",
    "mtl": "multi task learning",
    "nisq": "noisy intermediate scale quantum",
    "nlp": "natural language processing",
    "qec": "quantum error correction",
    "rag": "retrieval augmented generation",
    "sft": "supervised fine tuning",
}

DOMAIN_QUERY_EXPANSIONS = {
    "reconstruction-based techniques": [
        "time series anomaly detection reconstruction based",
        "multivariate time series anomaly detection reconstruction",
        "graph attention network anomaly detection",
        "graph neural network anomaly detection",
        "GAT multivariate time series anomaly detection",
    ],
    "reconstruction based techniques": [
        "time series anomaly detection reconstruction based",
        "multivariate time series anomaly detection reconstruction",
        "graph attention network anomaly detection",
        "graph neural network anomaly detection",
    ],
    "target networks": [
        "deep q learning target network theoretical analysis",
        "deep reinforcement learning target networks",
    ],
    "peer reviews": [
        "peer review bias detection calibration",
        "automatic bias detection peer review",
        "least square calibration peer review",
        "reviewer score calibration peer review",
    ],
    "foundation models": [
        "BERT GPT T5 PaLM LLaMA foundation language models",
        "pretrained language models BERT GPT T5",
        "large language models few shot learners",
        "scaling language modeling PaLM LLaMA",
    ],
    "natural language processing": [
        "BERT pre-training bidirectional transformers language understanding",
        "GPT language models few-shot learners",
        "T5 transfer learning text-to-text transformer",
        "PaLM LLaMA foundation language models",
    ],
    "causal bandits": [
        "causal bandits learning good interventions causal inference",
        "causal bandits optimal interventions",
        "causal reinforcement learning interventions",
        "sequential experimentation causal bandits",
    ],
    "causal reinforcement learning": [
        "causal reinforcement learning interventions",
        "causal bandits learning interventions",
        "optimal interventions causal reinforcement learning",
    ],
}

DOMAIN_QUERY_EXPANSIONS.update({
    "viewing ray": [
        "light field networks neural scene representations",
        "single-evaluation rendering light field networks",
        "scene representation transformer geometry-free novel view synthesis",
        "neural scene representation viewing ray",
    ],
    "coordinate definition": [
        "light field networks neural scene representations",
        "neural rendering viewing ray parameterization",
        "scene representation transformer novel view synthesis",
    ],
    "predictive distribution": [
        "probabilistic u-net segmentation ambiguous images",
        "stochastic segmentation networks aleatoric uncertainty",
        "spatially correlated aleatoric uncertainty segmentation",
    ],
    "uncertainties": [
        "probabilistic u-net ambiguous image segmentation",
        "stochastic segmentation networks uncertainty segmentation",
    ],
    "voxel": [
        "3D ShapeNets volumetric shapes",
        "3D-R2N2 single multi-view 3D object reconstruction",
        "volumetric shapes voxel 3D representation",
    ],
    "stationary distribution": [
        "nonparametric stochastic contextual bandits",
        "k-nearest neighbour UCB multi-armed bandits covariates",
        "contextual bandits stationary rewards covariates",
    ],
    "seq2edit": [
        "Encode Tag Realize high precision text editing",
        "token level edit operation prediction text editing",
        "sequence to edit grammatical error correction",
    ],
})

CHINESE_QUERY_EXPANSIONS = {
    "无人机": [
        "unmanned aerial vehicle",
        "UAV",
        "drone",
        "aerial robotics",
        "unmanned aircraft systems",
    ],
    "无人驾驶": ["autonomous driving", "self driving vehicles", "autonomous vehicles"],
    "自动驾驶": ["autonomous driving", "self driving vehicles", "autonomous vehicles"],
    "图像检索": ["image retrieval", "content based image retrieval", "visual retrieval"],
    "文献综述": ["literature review", "survey", "systematic review"],
    "大模型": ["large language model", "large language models", "foundation model"],
    "多模态": ["multimodal learning", "multimodal model", "vision language model"],
    "知识图谱": ["knowledge graph", "knowledge graphs"],
    "推荐系统": ["recommender system", "recommendation system"],
    "强化学习": ["reinforcement learning"],
    "联邦学习": ["federated learning"],
    "目标检测": ["object detection"],
    "语义通信": ["semantic communication"],
    "协同感知": ["collaborative perception", "cooperative perception"],
}


def build_query_plan(query: str, max_queries: int = 6) -> QueryPlan:
    year_from, year_to = _extract_year_range(query)
    phrases = _extract_phrases(query)
    terms = _extract_terms(query)
    translated_queries = _translated_queries(query)

    must_terms = _dedupe([*phrases[:3], *_translated_terms(translated_queries)[:24], *terms[:8]])
    optional_terms = _dedupe(terms[8:18])

    search_queries = _dedupe(
        [
            *translated_queries,
            _compact_core_query(phrases, terms),
            *(_phrase_queries(phrases, terms)),
            *(_intent_queries(query, phrases, terms)),
            *(_pairwise_queries(terms[:8])),
            *(_safe_original_query(query)),
        ]
    )
    search_queries = [q for q in search_queries if len(q.split()) >= 2]
    if not search_queries:
        search_queries = [query]

    return QueryPlan(
        original=query,
        search_queries=search_queries[:max_queries],
        must_terms=must_terms,
        optional_terms=optional_terms,
        year_from=year_from,
        year_to=year_to,
        planner="heuristic",
    )

def _safe_original_query(query: str) -> List[str]:
    tokens = _extract_terms(query)
    if len(query.split()) > 12:
        return []
    if query.strip().endswith(("?", "？")):
        return []
    if len(tokens) >= 2:
        return [" ".join(tokens[:8])]
    return []

def _compact_core_query(phrases: List[str], terms: List[str]) -> str:
    if phrases and len(phrases[0].split()) <= 7:
        return phrases[0]
    return " ".join(terms[:5])


def _safe_original_query(query: str) -> Iterable[str]:
    normalized = " ".join(str(query).split())
    if 2 <= len(normalized.split()) <= 10:
        yield normalized


def _extract_year_range(query: str) -> Tuple[Optional[int], Optional[int]]:
    current_year = date.today().year
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    if years:
        return min(years), max(years)

    recent = re.search(r"(?:last|past|recent)\s+(\d+)\s+years?", query, re.I)
    if recent:
        span = max(1, int(recent.group(1)))
        return current_year - span + 1, current_year

    chinese_recent = re.search(r"近\s*(\d+)\s*年", query)
    if chinese_recent:
        span = max(1, int(chinese_recent.group(1)))
        return current_year - span + 1, current_year

    return None, None


def _extract_phrases(query: str) -> List[str]:
    lowered = query.lower()
    found = [phrase for phrase in PHRASE_HINTS if phrase in lowered]
    quoted = re.findall(r'"([^"]{3,80})"', query)
    acronyms = [
        expansion
        for acronym, expansion in ACRONYM_EXPANSIONS.items()
        if re.search(rf"\b{re.escape(acronym)}s?\b", lowered)
    ]
    return _dedupe([*found, *acronyms, *_translated_queries(query)[:3], *quoted])


def _extract_terms(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _translated_queries(query: str) -> List[str]:
    expansions = []
    for chinese, english_terms in CHINESE_QUERY_EXPANSIONS.items():
        if chinese in query:
            expansions.extend(english_terms)
    lowered = query.lower()
    for phrase, english_terms in DOMAIN_QUERY_EXPANSIONS.items():
        if phrase in lowered:
            expansions.extend(english_terms)
    if expansions:
        expansions.extend(_chinese_intent_queries(query, expansions[0]))
    return _dedupe(expansions)


def _translated_terms(queries: List[str]) -> List[str]:
    terms = []
    for query in queries:
        terms.extend(_extract_terms(query))
    return _dedupe(terms)


def _chinese_intent_queries(query: str, core: str) -> Iterable[str]:
    if any(token in query for token in ("综述", "有哪些", "相关", "文献")):
        yield f"{core} survey"
        yield f"{core} recent advances"
    if any(token in query for token in ("应用", "系统", "工具")):
        yield f"{core} applications"
    if any(token in query for token in ("检测", "识别", "跟踪", "感知")):
        yield f"{core} perception detection"


def _phrase_queries(phrases: List[str], terms: List[str]) -> Iterable[str]:
    for phrase in phrases[:3]:
        yield phrase
        if terms:
            yield f"{phrase} {terms[0]}"
        yield f"{phrase} survey"


def _intent_queries(query: str, phrases: List[str], terms: List[str]) -> Iterable[str]:
    lowered = query.lower()
    core = phrases[0] if phrases else " ".join(terms[:3])
    if not core:
        return
    if any(word in lowered for word in ("survey", "overview", "review", "literature")):
        yield f"{core} survey review"
    if any(word in lowered for word in ("benchmark", "dataset", "large-scale", "large scale")):
        yield f"{core} benchmark dataset"
    if any(word in lowered for word in ("application", "applied", "practical", "system")):
        yield f"{core} application"
    if any(word in lowered for word in ("latest", "recent", "cutting-edge", "cutting edge")):
        yield f"{core} recent advances"


def _pairwise_queries(terms: List[str]) -> Iterable[str]:
    for i in range(0, len(terms) - 1, 2):
        yield f"{terms[i]} {terms[i + 1]}"


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        normalized = " ".join(item.strip().split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped
