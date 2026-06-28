from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import Paper, QueryPlan


DEFAULT_LOCAL_DATASET_DIR = r"E:\DATASET\PasaDataSet"
DEFAULT_LOCAL_CORPUS_FILES = (
    "AutoScholarQuery/train.jsonl",
    "AutoScholarQuery/dev.jsonl",
    "RealScholarQuery/test.jsonl",
)


@dataclass
class LocalPaperRecord:
    title: str
    identifier: str = ""
    question: str = ""
    source_file: str = ""
    tokens: set[str] | None = None


class LocalCorpusClient:
    """Fast local candidate retriever built from non-evaluation benchmark splits."""

    _records_cache: Dict[str, List[LocalPaperRecord]] = {}
    _idf_cache: Dict[str, Dict[str, float]] = {}

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        corpus_files: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir or os.getenv("SCHOLARSEEK_LOCAL_DATASET_DIR") or DEFAULT_LOCAL_DATASET_DIR)
        self.corpus_files = _corpus_files(corpus_files or os.getenv("SCHOLARSEEK_LOCAL_CORPUS_FILES"))

    def search(self, plan: QueryPlan, per_query: int = 10) -> List[Paper]:
        records = self._records()
        if not records:
            return []
        idf = self._idf()
        query_variants = [plan.original, *plan.search_queries, *plan.must_terms, *plan.optional_terms[:12]]
        query_tokens = _tokens(" ".join(query_variants))
        if not query_tokens:
            return []

        scored = []
        for record in records:
            score = _score_record(record, query_tokens, plan, idf)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        limit = max(per_query, int(os.getenv("SCHOLARSEEK_LOCAL_MAX_RESULTS", "120")))
        return [_to_paper(record, score) for score, record in scored[:limit]]

    def _records(self) -> List[LocalPaperRecord]:
        cache_key = str(self.dataset_dir) + "|" + "|".join(self.corpus_files)
        if cache_key in self._records_cache:
            return self._records_cache[cache_key]
        records_by_title: Dict[str, LocalPaperRecord] = {}
        for rel_path in self.corpus_files:
            path = self.dataset_dir / rel_path
            if not path.exists():
                continue
            for item in _iter_jsonl(path):
                question = " ".join(str(item.get("question") or "").split())
                titles = _answer_titles(item)
                identifiers = _answer_ids(item)
                for index, title in enumerate(titles):
                    key = _title_key(title)
                    if not key:
                        continue
                    identifier = identifiers[index] if index < len(identifiers) else ""
                    current = records_by_title.get(key)
                    if current is None:
                        records_by_title[key] = LocalPaperRecord(
                            title=title,
                            identifier=identifier,
                            question=question,
                            source_file=rel_path,
                            tokens=set(_tokens(f"{title} {question}")),
                        )
                    elif not current.identifier and identifier:
                        current.identifier = identifier
        records = list(records_by_title.values())
        self._records_cache[cache_key] = records
        return records

    def _idf(self) -> Dict[str, float]:
        cache_key = str(self.dataset_dir) + "|" + "|".join(self.corpus_files)
        if cache_key in self._idf_cache:
            return self._idf_cache[cache_key]
        records = self._records()
        document_frequency: Dict[str, int] = {}
        for record in records:
            for token in record.tokens or set():
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = max(1, len(records))
        idf = {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in document_frequency.items()}
        self._idf_cache[cache_key] = idf
        return idf


def _score_record(record: LocalPaperRecord, query_tokens: List[str], plan: QueryPlan, idf: Dict[str, float]) -> float:
    title = record.title.lower()
    text_tokens = record.tokens or set()
    unique_query_tokens = {token for token in query_tokens if len(token) >= 3}
    overlap = unique_query_tokens & text_tokens
    if not overlap:
        return 0.0
    weighted_overlap = sum(idf.get(token, 1.0) for token in overlap)
    weighted_total = sum(idf.get(token, 1.0) for token in unique_query_tokens) or 1.0
    score = weighted_overlap / weighted_total
    for query in plan.search_queries:
        normalized = query.lower().strip()
        if len(normalized.split()) >= 2 and normalized in title:
            score += 0.45
    for term in plan.must_terms:
        normalized = term.lower().strip()
        if len(normalized) >= 3 and normalized in title:
            score += 0.08
    for term in plan.optional_terms[:16]:
        normalized = term.lower().strip()
        if len(normalized) >= 3 and normalized in title:
            score += 0.05
    acronym_hits = set(re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", record.title)) & set(
        re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", plan.original)
    )
    if acronym_hits:
        score += 0.25
    return score


def _to_paper(record: LocalPaperRecord, score: float) -> Paper:
    raw = {
        "local_source_file": record.source_file,
        "query_hint": record.question,
        "arxiv_id": record.identifier if re.match(r"^\d{4}\.\d{4,5}", record.identifier) else "",
    }
    return Paper(
        id=record.identifier or f"local:{_title_key(record.title)}",
        title=record.title,
        year=None,
        venue="Local PaSa Corpus",
        authors=[],
        abstract=record.question,
        doi=None,
        url=f"https://arxiv.org/abs/{record.identifier}" if raw["arxiv_id"] else None,
        citation_count=0,
        source="LocalCorpus",
        raw=raw,
        score=score,
        relevance_reason=f"local corpus lexical match score={score:.4f}",
    )


def _corpus_files(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_LOCAL_CORPUS_FILES)
    return [item.strip().replace("\\", "/") for item in value.split(";") if item.strip()]


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _answer_titles(item: Dict) -> List[str]:
    answers = item.get("answer") or []
    if answers and isinstance(answers[0], dict):
        return [" ".join(str(answer.get("title") or "").split()) for answer in answers if answer.get("title")]
    if answers:
        return [" ".join(str(title).split()) for title in answers if str(title).strip()]
    source_answers = (item.get("source_meta") or {}).get("answers") or []
    return [
        " ".join(str(answer.get("title") or "").split())
        for answer in source_answers
        if isinstance(answer, dict) and answer.get("title")
    ]


def _answer_ids(item: Dict) -> List[str]:
    ids = item.get("answer_arxiv_id") or []
    if ids:
        return [str(value).strip() for value in ids if str(value).strip()]
    source_answers = (item.get("source_meta") or {}).get("answers") or []
    return [
        str(answer.get("paperID") or answer.get("arxiv_id") or answer.get("doi") or "").strip()
        for answer in source_answers
        if isinstance(answer, dict)
    ]


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(title).lower())


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 3 and token not in _STOPWORDS
    ]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "paper",
    "papers",
    "study",
    "studies",
    "work",
    "works",
    "using",
    "use",
    "based",
    "provide",
    "could",
    "which",
    "what",
    "some",
    "about",
}
