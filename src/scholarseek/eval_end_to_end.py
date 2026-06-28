from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence

from .search_service import search_papers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end ScholarSeek evaluation on PaSa or SPAR benchmarks.")
    parser.add_argument("--dataset-file", required=True, help="JSONL file or SPAR zip archive")
    parser.add_argument("--zip-member", default="SPAR-master/benchmark/AutoScholarQuery_test.jsonl")
    parser.add_argument("--strategy", choices=("standard", "spar", "spar-qwen"), default="spar")
    parser.add_argument("--sources", default="openalex,semantic-scholar,arxiv")
    parser.add_argument("--planner", choices=("heuristic", "qwen"), default="heuristic")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum search-query variants generated for each benchmark record.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum benchmark records to evaluate; omit for the full dataset.",
    )
    parser.add_argument("--output", default="outputs/eval/end_to_end_results.jsonl")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows in the output and continue with missing record indices.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_rows(output) if args.resume else []
    completed_indices = {int(row["index"]) for row in rows if row.get("index") is not None}
    mode = "a" if args.resume else "w"
    if completed_indices:
        print(
            json.dumps(
                {"status": "resuming", "completed_indices": sorted(completed_indices)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    with output.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(iter_benchmark(args.dataset_file, args.zip_member), start=1):
            if args.max_records is not None and index > args.max_records:
                break
            if index in completed_indices:
                continue
            print(
                json.dumps(
                    {
                        "index": index,
                        "status": "started",
                        "message": "planning and retrieving candidates",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            started = time.perf_counter()
            result = search_papers(
                query=record["question"],
                planner=args.planner,
                answer="none",
                sources=args.sources,
                max_queries=args.max_queries,
                per_query=args.per_query,
                limit=args.top_k,
                strategy=args.strategy,
            )
            predicted_papers = result.get("papers", [])
            predicted = [paper.get("title") or "" for paper in predicted_papers]
            predicted_ids = [_paper_identifier(paper) for paper in predicted_papers]
            metrics = evaluate_predictions(predicted, record["answers"], predicted_ids, record.get("answer_ids"), args.top_k)
            candidate_titles = (result.get("pipeline_trace") or {}).get("retrieved_candidate_titles") or predicted
            candidate_ids = (result.get("pipeline_trace") or {}).get("retrieved_candidate_ids") or predicted_ids
            retrieval_metrics = evaluate_retrieval(candidate_titles, record["answers"], candidate_ids, record.get("answer_ids"))
            row = {
                "index": index,
                "qid": record.get("qid"),
                "question": record["question"],
                "gold_titles": record["answers"],
                "gold_ids": record.get("answer_ids") or [],
                "predicted_titles": predicted,
                "predicted_ids": predicted_ids,
                "metrics": metrics,
                "retrieval_metrics": retrieval_metrics,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "pipeline_trace": result.get("pipeline_trace"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "index": index,
                        **metrics,
                        **retrieval_metrics,
                        "latency_seconds": row["latency_seconds"],
                    }
                ),
                flush=True,
            )

    summary = aggregate_metrics(rows)
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "saved": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def _read_existing_rows(path: Path) -> list[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] ignoring malformed resume row {line_number} in {path}", flush=True)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def iter_benchmark(dataset_file: str, zip_member: str = "") -> Iterator[Dict]:
    path = Path(dataset_file)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(zip_member) as binary:
                for raw_line in binary:
                    record = _benchmark_record(json.loads(raw_line.decode("utf-8")))
                    if record:
                        yield record
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = _benchmark_record(json.loads(line))
                if record:
                    yield record


def evaluate_titles(predicted: Sequence[str], gold: Sequence[str], top_k: int = 10) -> Dict[str, float]:
    return evaluate_predictions(predicted, gold, None, None, top_k)


def evaluate_predictions(
    predicted: Sequence[str],
    gold: Sequence[str],
    predicted_ids: Sequence[str] | None = None,
    gold_ids: Sequence[str] | None = None,
    top_k: int = 10,
) -> Dict[str, float]:
    predicted_keys = [_title_key(title) for title in predicted[:top_k] if _title_key(title)]
    predicted_id_keys = [_id_key(value) for value in (predicted_ids or [])[:top_k] if _id_key(value)]
    gold_items = _gold_items(gold, gold_ids)
    matched_positions: list[int] = []
    matched_gold: set[int] = set()
    for index, key in enumerate(predicted_keys):
        id_key = predicted_id_keys[index] if index < len(predicted_id_keys) else ""
        for gold_index, gold_item in enumerate(gold_items):
            if key and key == gold_item["title"]:
                matched_positions.append(index)
                matched_gold.add(gold_index)
                break
            if id_key and id_key == gold_item["id"]:
                matched_positions.append(index)
                matched_gold.add(gold_index)
                break
    precision = len(matched_positions) / max(1, len(predicted_keys))
    recall = len(matched_gold) / max(1, len(gold_items))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    reciprocal_rank = 0.0
    for rank, index in enumerate(range(len(predicted_keys)), start=1):
        key = predicted_keys[index]
        id_key = predicted_id_keys[index] if index < len(predicted_id_keys) else ""
        if any(
            (key and key == gold_item["title"]) or (id_key and id_key == gold_item["id"])
            for gold_item in gold_items
        ):
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        f"hit@{top_k}": 1.0 if matched_positions else 0.0,
        "mrr": reciprocal_rank,
    }


def evaluate_retrieval(
    candidates: Sequence[str],
    gold: Sequence[str],
    candidate_ids: Sequence[str] | None = None,
    gold_ids: Sequence[str] | None = None,
) -> Dict[str, float]:
    candidate_keys = {_title_key(title) for title in candidates if _title_key(title)}
    candidate_id_keys = {_id_key(value) for value in (candidate_ids or []) if _id_key(value)}
    gold_items = _gold_items(gold, gold_ids)
    matched = {
        index
        for index, item in enumerate(gold_items)
        if item["title"] in candidate_keys or (item["id"] and item["id"] in candidate_id_keys)
    }
    return {
        "retrieval_recall": len(matched) / max(1, len(gold_items)),
        "retrieval_hit": 1.0 if matched else 0.0,
        "retrieved_candidates": len(candidate_keys),
    }


def aggregate_metrics(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"queries": 0}
    metric_names = list(rows[0]["metrics"])
    summary = {"queries": len(rows)}
    for name in metric_names:
        summary[name] = sum(row["metrics"][name] for row in rows) / len(rows)
    total_predicted = 0
    total_gold = 0
    total_matched = 0
    for row in rows:
        predicted = [_title_key(title) for title in row["predicted_titles"] if _title_key(title)]
        predicted_ids = [_id_key(value) for value in row.get("predicted_ids", []) if _id_key(value)]
        gold_items = _gold_items(row["gold_titles"], row.get("gold_ids"))
        total_predicted += len(predicted)
        total_gold += len(gold_items)
        total_matched += len(_matched_gold_indices(predicted, predicted_ids, gold_items))
    micro_precision = total_matched / max(1, total_predicted)
    micro_recall = total_matched / max(1, total_gold)
    summary["micro_precision"] = micro_precision
    summary["micro_recall"] = micro_recall
    summary["micro_f1"] = 2 * micro_precision * micro_recall / max(1e-12, micro_precision + micro_recall)
    retrieval_rows = [row.get("retrieval_metrics") for row in rows if row.get("retrieval_metrics")]
    if retrieval_rows:
        summary["retrieval_recall"] = sum(item["retrieval_recall"] for item in retrieval_rows) / len(retrieval_rows)
        summary["retrieval_hit"] = sum(item["retrieval_hit"] for item in retrieval_rows) / len(retrieval_rows)
        summary["average_retrieved_candidates"] = sum(
            item["retrieved_candidates"] for item in retrieval_rows
        ) / len(retrieval_rows)
    summary["average_latency_seconds"] = sum(row["latency_seconds"] for row in rows) / len(rows)
    return summary


def _benchmark_record(item):
    question = " ".join(str(item.get("question") or "").split())
    answers = item.get("answer") or []
    if answers and isinstance(answers[0], dict):
        answers = [answer.get("title") or "" for answer in answers]
    if not answers:
        source_answers = (item.get("source_meta") or {}).get("answers") or []
        answers = [answer.get("title") or "" for answer in source_answers]
    answers = [" ".join(str(title).split()) for title in answers if str(title).strip()]
    answer_ids = item.get("answer_arxiv_id") or []
    if not answer_ids:
        source_answers = (item.get("source_meta") or {}).get("answers") or []
        answer_ids = [
            answer.get("paperID") or answer.get("arxiv_id") or answer.get("doi") or ""
            for answer in source_answers
            if isinstance(answer, dict)
        ]
    answer_ids = [str(value).strip() for value in answer_ids if str(value).strip()]
    if not question or not answers:
        return None
    return {"qid": item.get("qid"), "question": question, "answers": answers, "answer_ids": answer_ids}


def _paper_identifier(paper: Dict) -> str:
    raw = paper.get("raw") or {}
    return (
        raw.get("arxiv_id")
        or raw.get("paperId")
        or paper.get("doi")
        or paper.get("id")
        or paper.get("url")
        or ""
    )


def _gold_items(titles: Sequence[str], identifiers: Sequence[str] | None = None) -> list[Dict[str, str]]:
    identifiers = list(identifiers or [])
    count = max(len(titles), len(identifiers))
    items = []
    for index in range(count):
        title = titles[index] if index < len(titles) else ""
        identifier = identifiers[index] if index < len(identifiers) else ""
        title_key = _title_key(title)
        id_key = _id_key(identifier)
        if title_key or id_key:
            items.append({"title": title_key, "id": id_key})
    return items


def _matched_gold_indices(
    predicted_titles: Sequence[str],
    predicted_ids: Sequence[str],
    gold_items: Sequence[Dict[str, str]],
) -> set[int]:
    matched: set[int] = set()
    for index, key in enumerate(predicted_titles):
        id_key = predicted_ids[index] if index < len(predicted_ids) else ""
        for gold_index, gold_item in enumerate(gold_items):
            if key and key == gold_item["title"]:
                matched.add(gold_index)
            if id_key and id_key == gold_item["id"]:
                matched.add(gold_index)
    return matched


def _title_key(title):
    return re.sub(r"[^a-z0-9]+", "", str(title).lower())


def _id_key(value):
    text = str(value or "").strip().lower()
    text = text.removeprefix("arxiv:")
    text = text.removeprefix("https://arxiv.org/abs/")
    text = text.removeprefix("http://arxiv.org/abs/")
    text = text.split("v")[0] if re.match(r"^\d{4}\.\d{4,5}v\d+$", text) else text
    text = text.removeprefix("https://doi.org/")
    return re.sub(r"[^a-z0-9.]+", "", text)


if __name__ == "__main__":
    raise SystemExit(main())
