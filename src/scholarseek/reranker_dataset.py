from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .query_planner import STOPWORDS


@dataclass(frozen=True)
class PasaQueryRecord:
    qid: str
    question: str
    answers: List[str]
    answer_arxiv_ids: List[str]
    published_time: Optional[str] = None


@dataclass(frozen=True)
class RerankerPair:
    query: str
    title: str
    label: int
    qid: str
    arxiv_id: Optional[str] = None


def read_pasa_jsonl(path: str | Path) -> Iterator[PasaQueryRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            answers = [_clean_title(title) for title in item.get("answer", []) if _clean_title(title)]
            arxiv_ids = [str(value) for value in item.get("answer_arxiv_id", [])]
            meta = item.get("source_meta") or {}
            yield PasaQueryRecord(
                qid=str(item.get("qid") or ""),
                question=" ".join(str(item.get("question") or "").split()),
                answers=answers,
                answer_arxiv_ids=arxiv_ids,
                published_time=meta.get("published_time"),
            )


def read_many(paths: Iterable[str | Path], limit: Optional[int] = None) -> List[PasaQueryRecord]:
    records: List[PasaQueryRecord] = []
    for path in paths:
        for record in read_pasa_jsonl(path):
            if record.question and record.answers:
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return records
    return records


def build_title_pool(records: Iterable[PasaQueryRecord]) -> List[str]:
    seen = set()
    titles = []
    for record in records:
        for title in record.answers:
            key = _title_key(title)
            if key and key not in seen:
                seen.add(key)
                titles.append(title)
    return titles


def generate_pairs(
    records: Iterable[PasaQueryRecord],
    negative_pool: List[str],
    negatives_per_positive: int = 4,
    hard_negatives_per_positive: int = 0,
    seed: int = 13,
) -> Iterator[RerankerPair]:
    rng = random.Random(seed)
    negative_pool_keys = [_title_key(title) for title in negative_pool]
    hard_negative_miner = HardNegativeMiner(negative_pool) if hard_negatives_per_positive > 0 else None
    for record in records:
        positive_keys = {_title_key(title) for title in record.answers}
        hard_negatives = (
            hard_negative_miner.mine(record.question, positive_keys, hard_negatives_per_positive)
            if hard_negative_miner
            else []
        )
        for index, title in enumerate(record.answers):
            arxiv_id = record.answer_arxiv_ids[index] if index < len(record.answer_arxiv_ids) else None
            yield RerankerPair(record.question, title, 1, record.qid, arxiv_id)
            used_negative_keys = set()
            for negative in hard_negatives:
                key = _title_key(negative)
                if key not in used_negative_keys:
                    used_negative_keys.add(key)
                    yield RerankerPair(record.question, negative, 0, record.qid, None)
            for negative in _sample_negatives(
                rng,
                negative_pool,
                negative_pool_keys,
                positive_keys | used_negative_keys,
                negatives_per_positive,
            ):
                yield RerankerPair(record.question, negative, 0, record.qid, None)


class HardNegativeMiner:
    def __init__(self, titles: List[str]):
        self.titles = titles
        self.title_tokens = [set(_search_tokens(title)) for title in titles]
        self.postings = defaultdict(list)
        for index, tokens in enumerate(self.title_tokens):
            for token in tokens:
                self.postings[token].append(index)
        total = max(1, len(titles))
        self.idf = {
            token: math.log((total + 1) / (len(indices) + 1)) + 1.0
            for token, indices in self.postings.items()
        }
        self.max_postings = max(200, total // 8)

    def mine(self, query: str, excluded_keys: set[str], count: int) -> List[str]:
        query_tokens = set(_search_tokens(query))
        scores = defaultdict(float)
        for token in query_tokens:
            indices = self.postings.get(token, [])
            if len(indices) > self.max_postings:
                continue
            weight = self.idf.get(token, 1.0)
            for index in indices:
                scores[index] += weight
        ranked = sorted(
            scores,
            key=lambda index: (
                scores[index],
                len(query_tokens & self.title_tokens[index]) / max(1, len(query_tokens | self.title_tokens[index])),
            ),
            reverse=True,
        )
        output = []
        for index in ranked:
            title = self.titles[index]
            if _title_key(title) in excluded_keys:
                continue
            output.append(title)
            if len(output) >= count:
                break
        return output


def write_pairs_jsonl(pairs: Iterable[RerankerPair], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "query": pair.query,
                        "title": pair.title,
                        "label": pair.label,
                        "qid": pair.qid,
                        "arxiv_id": pair.arxiv_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def read_pairs_jsonl(path: str | Path, limit: Optional[int] = None) -> List[RerankerPair]:
    pairs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            pairs.append(
                RerankerPair(
                    query=str(item["query"]),
                    title=str(item["title"]),
                    label=int(item["label"]),
                    qid=str(item.get("qid") or ""),
                    arxiv_id=item.get("arxiv_id"),
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def _clean_title(title: str) -> str:
    return " ".join(str(title).replace("\n", " ").split())


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _search_tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
        if token not in STOPWORDS
    ]


def _sample_negatives(
    rng: random.Random,
    pool: List[str],
    pool_keys: List[str],
    positive_keys: set[str],
    count: int,
) -> List[str]:
    if count <= 0 or not pool:
        return []

    selected: List[str] = []
    selected_keys = set()
    max_attempts = max(100, count * 30)
    for _ in range(max_attempts):
        if len(selected) >= count:
            return selected
        index = rng.randrange(len(pool))
        key = pool_keys[index]
        if key in positive_keys or key in selected_keys:
            continue
        selected.append(pool[index])
        selected_keys.add(key)

    # Extremely small or highly overlapping pools can defeat rejection sampling.
    for title, key in zip(pool, pool_keys):
        if len(selected) >= count:
            break
        if key in positive_keys or key in selected_keys:
            continue
        selected.append(title)
        selected_keys.add(key)
    return selected
