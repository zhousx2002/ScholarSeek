from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List

from .reranker_dataset import (
    PasaQueryRecord,
    build_title_pool,
    generate_pairs,
    read_many,
    read_pairs_jsonl,
    write_pairs_jsonl,
)
from .trainable_reranker import FEATURE_NAMES, CompactFeatureReranker, pair_features


DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate a ScholarSeek trainable reranker.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-dir", default="E:/DATASET/PasaDataSet")
    prepare.add_argument("--output-dir", default="outputs/reranker_data")
    prepare.add_argument("--negatives-per-positive", type=int, default=4)
    prepare.add_argument("--hard-negatives-per-positive", type=int, default=4)
    prepare.add_argument("--max-train-records", type=int, default=None)
    prepare.add_argument("--seed", type=int, default=13)

    train = subparsers.add_parser("train")
    train.add_argument("--train-file", default="outputs/reranker_data/train_pairs.jsonl")
    train.add_argument("--dev-file", default="outputs/reranker_data/dev_pairs.jsonl")
    train.add_argument("--output-dir", default="outputs/reranker_model")
    train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--max-length", type=int, default=384)
    train.add_argument("--max-train-pairs", type=int, default=None)
    train.add_argument("--max-dev-pairs", type=int, default=4000)
    train.add_argument("--loss", choices=("pairwise", "pointwise"), default="pairwise")
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--log-every", type=int, default=1000)
    train.add_argument("--seed", type=int, default=13)

    compact = subparsers.add_parser("train-compact")
    compact.add_argument("--train-file", default="outputs/reranker_data/train_pairs.jsonl")
    compact.add_argument("--dev-file", default="outputs/reranker_data/dev_pairs.jsonl")
    compact.add_argument("--output-dir", default="outputs/reranker_compact")
    compact.add_argument("--epochs", type=int, default=6)
    compact.add_argument("--learning-rate", type=float, default=0.08)
    compact.add_argument("--l2", type=float, default=0.001)
    compact.add_argument("--max-train-pairs", type=int, default=None)
    compact.add_argument("--max-dev-pairs", type=int, default=12000)
    compact.add_argument("--seed", type=int, default=13)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--dataset-dir", default="E:/DATASET/PasaDataSet")
    evaluate.add_argument("--model-dir", default="outputs/reranker_model")
    evaluate.add_argument("--split", choices=("dev", "test", "real"), default="dev")
    evaluate.add_argument("--candidates-per-query", type=int, default=64)
    evaluate.add_argument("--limit-records", type=int, default=None)
    evaluate.add_argument(
        "--output",
        default=None,
        help="Result JSON path (default: outputs/eval/reranker_<split>_results.json).",
    )
    evaluate.add_argument("--log-every", type=int, default=25, help="Print progress every N queries; 0 disables it.")
    evaluate.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_pairs(args)
    if args.command == "train":
        return train_model(args)
    if args.command == "train-compact":
        return train_compact_model(args)
    if args.command == "eval":
        return evaluate_model(args)
    return 1


def prepare_pairs(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    train_records = read_many([dataset_dir / "AutoScholarQuery" / "train.jsonl"], limit=args.max_train_records)
    dev_records = read_many([dataset_dir / "AutoScholarQuery" / "dev.jsonl"])
    title_pool = build_title_pool([*train_records, *dev_records])
    output_dir = Path(args.output_dir)
    train_count = write_pairs_jsonl(
        generate_pairs(
            train_records,
            title_pool,
            negatives_per_positive=args.negatives_per_positive,
            hard_negatives_per_positive=args.hard_negatives_per_positive,
            seed=args.seed,
        ),
        output_dir / "train_pairs.jsonl",
    )
    dev_count = write_pairs_jsonl(
        generate_pairs(
            dev_records,
            title_pool,
            negatives_per_positive=args.negatives_per_positive,
            hard_negatives_per_positive=args.hard_negatives_per_positive,
            seed=args.seed + 1,
        ),
        output_dir / "dev_pairs.jsonl",
    )
    metadata = {
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "title_pool": len(title_pool),
        "train_pairs": train_count,
        "dev_pairs": dev_count,
        "negatives_per_positive": args.negatives_per_positive,
        "hard_negatives_per_positive": args.hard_negatives_per_positive,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


def train_compact_model(args) -> int:
    random.seed(args.seed)
    train_pairs = read_pairs_jsonl(args.train_file, limit=args.max_train_pairs)
    dev_pairs = read_pairs_jsonl(args.dev_file, limit=args.max_dev_pairs)
    weights = [0.0 for _ in FEATURE_NAMES]
    bias = 0.0
    best = {"loss": math.inf, "weights": list(weights), "bias": bias, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_pairs)
        train_loss = 0.0
        for pair in train_pairs:
            features = pair_features(pair.query, pair.title)
            logit = bias + sum(weight * value for weight, value in zip(weights, features))
            prob = _sigmoid(logit)
            error = prob - pair.label
            train_loss += _binary_loss(prob, pair.label)
            for index, value in enumerate(features):
                weights[index] -= args.learning_rate * (error * value + args.l2 * weights[index])
            bias -= args.learning_rate * error

        dev_loss, dev_auc = _compact_dev_metrics(dev_pairs, weights, bias)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / max(1, len(train_pairs)),
                    "dev_loss": dev_loss,
                    "dev_auc": dev_auc,
                    "model": "compact-feature-logistic-reranker",
                },
                ensure_ascii=False,
            )
        )
        if dev_loss < best["loss"]:
            best = {"loss": dev_loss, "weights": list(weights), "bias": bias, "epoch": epoch, "dev_auc": dev_auc}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "compact-feature-logistic-reranker",
        "feature_names": FEATURE_NAMES,
        "weights": best["weights"],
        "bias": best["bias"],
        "best_epoch": best["epoch"],
        "best_dev_loss": best["loss"],
        "best_dev_auc": best.get("dev_auc"),
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "source_dataset": "PaSa AutoScholarQuery pairs",
    }
    (output_dir / "compact_reranker.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "compact_reranker.json"), "size_bytes": (output_dir / "compact_reranker.json").stat().st_size}, indent=2))
    return 0


def train_model(args) -> int:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Training requires optional dependencies: torch and transformers. "
            "Install them in your intended environment before training."
        ) from exc
    _require_torch_version(torch, transformers.__version__)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_pairs = read_pairs_jsonl(args.train_file, limit=args.max_train_pairs)
    dev_pairs = read_pairs_jsonl(args.dev_file, limit=args.max_dev_pairs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss == "pairwise":
        train_items = build_pairwise_triplets(train_pairs)
        dev_items = build_pairwise_triplets(dev_pairs)
        if not train_items:
            raise SystemExit("Pairwise training found no positive-negative triplets in the training file.")
        train_loader = DataLoader(
            PairwiseDataset(train_items),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_pairwise_collate(tokenizer, args.max_length),
        )
        dev_loader = DataLoader(
            PairwiseDataset(dev_items),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_pairwise_collate(tokenizer, args.max_length),
        )
    else:
        train_loader = DataLoader(
            PairDataset(train_pairs, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )
        dev_loader = DataLoader(
            PairDataset(dev_pairs, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    accumulation_steps = max(1, args.gradient_accumulation_steps)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dev_loss = math.inf
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.loss == "pairwise":
                    positive = {key: value.to(device) for key, value in batch["positive"].items()}
                    negative = {key: value.to(device) for key, value in batch["negative"].items()}
                    positive_logits = model(**positive).logits.squeeze(-1)
                    negative_logits = model(**negative).logits.squeeze(-1)
                    loss = -torch.nn.functional.logsigmoid(positive_logits - negative_logits).mean()
                else:
                    labels = batch.pop("labels").to(device)
                    encoded = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**encoded).logits.squeeze(-1)
                    loss = loss_fn(logits, labels)
            scaler.scale(loss / accumulation_steps).backward()
            should_step = step % accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item())
            if step == 1 or step == len(train_loader) or step % max(1, args.log_every) == 0:
                elapsed = max(1e-6, time.perf_counter() - epoch_started)
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "total_steps": len(train_loader),
                            "progress_percent": round(100.0 * step / max(1, len(train_loader)), 2),
                            "running_loss": train_loss / step,
                            "pairs_per_second": round(step * args.batch_size / elapsed, 2),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    flush=True,
                )
        if args.loss == "pairwise":
            dev_loss, dev_pair_accuracy = evaluate_pairwise(model, dev_loader, device, use_amp)
        else:
            dev_loss = evaluate_loss(model, dev_loader, loss_fn, device)
            dev_pair_accuracy = None
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / max(1, len(train_loader)),
                    "dev_loss": dev_loss,
                    "dev_pair_accuracy": dev_pair_accuracy,
                    "device": str(device),
                    "loss": args.loss,
                    "fp16": use_amp,
                }
            )
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "scholarseek_reranker.json").write_text(
                json.dumps(
                    {
                        "base_model": args.base_model,
                        "max_length": args.max_length,
                        "best_dev_loss": best_dev_loss,
                        "loss": args.loss,
                        "dev_pair_accuracy": dev_pair_accuracy,
                        "gradient_accumulation_steps": accumulation_steps,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return 0


def _require_torch_version(torch_module, transformers_version: str) -> None:
    version = str(torch_module.__version__).split("+", 1)[0]
    torch_major, torch_minor = _major_minor(version)
    transformers_major, transformers_minor = _major_minor(transformers_version)
    if (torch_major, torch_minor) < (2, 1) and (transformers_major, transformers_minor) >= (4, 57):
        raise SystemExit(
            "Trainable reranker requires PyTorch >= 2.1 because the installed "
            f"Transformers backend disables older PyTorch versions. Found torch {torch_module.__version__} "
            f"and transformers {transformers_version}. Either upgrade torch or run with "
            "PYTHONNOUSERSITE=1 and transformers==4.40.2 inside the active conda environment."
        )


def _major_minor(version: str) -> tuple[int, int]:
    parts = []
    for item in version.split(".")[:2]:
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    return tuple((parts + [0, 0])[:2])


def evaluate_model(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    records = _records_for_split(dataset_dir, args.split, args.limit_records)
    pool_records = read_many(
        [dataset_dir / "AutoScholarQuery" / "train.jsonl", dataset_dir / "AutoScholarQuery" / "dev.jsonl"]
    )
    title_pool = build_title_pool(pool_records)
    rng = random.Random(args.seed)
    reranker = _load_any_reranker(args.model_dir)
    started_at = time.time()
    metrics = evaluate_records(
        reranker,
        records,
        title_pool,
        args.candidates_per_query,
        rng,
        progress_every=max(0, args.log_every),
    )
    duration_seconds = time.time() - started_at
    output = Path(args.output or f"outputs/eval/reranker_{args.split}_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset_dir": str(dataset_dir.resolve()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "split": args.split,
        "candidates_per_query": args.candidates_per_query,
        "limit_records": args.limit_records,
        "seed": args.seed,
        "duration_seconds": round(duration_seconds, 3),
        "queries_per_second": round(metrics["queries"] / max(duration_seconds, 1e-9), 3),
        "metrics": metrics,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "saved_to": str(output), **result}, ensure_ascii=False, indent=2))
    return 0


def _load_any_reranker(model_dir: str):
    path = Path(model_dir)
    if path.is_dir() and (path / "compact_reranker.json").exists():
        return CompactFeatureReranker(str(path))
    if path.name == "compact_reranker.json":
        return CompactFeatureReranker(str(path))
    from .trainable_reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_dir)


class PairDataset:
    def __init__(self, pairs, tokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        encoded = self.tokenizer(
            pair.query,
            pair.title,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = float(pair.label)
        return encoded


class PairwiseDataset:
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]


def build_pairwise_triplets(pairs):
    triplets = []
    current_positive = None
    for pair in pairs:
        if pair.label == 1:
            current_positive = pair
            continue
        if current_positive is None:
            continue
        if pair.query != current_positive.query or pair.qid != current_positive.qid:
            continue
        triplets.append((pair.query, current_positive.title, pair.title))
    return triplets


def make_pairwise_collate(tokenizer, max_length):
    def collate(items):
        queries = [item[0] for item in items]
        positives = [item[1] for item in items]
        negatives = [item[2] for item in items]
        options = {
            "truncation": True,
            "padding": True,
            "max_length": max_length,
            "return_tensors": "pt",
        }
        return {
            "positive": tokenizer(queries, positives, **options),
            "negative": tokenizer(queries, negatives, **options),
        }

    return collate


def collate_batch(items):
    import torch

    labels = torch.tensor([item.pop("labels") for item in items], dtype=torch.float)
    keys = items[0].keys()
    max_len = max(len(item["input_ids"]) for item in items)
    batch = {}
    for key in keys:
        pad_value = 0
        batch[key] = torch.tensor(
            [item[key] + [pad_value] * (max_len - len(item[key])) for item in items],
            dtype=torch.long,
        )
    batch["labels"] = labels
    return batch


def evaluate_loss(model, loader, loss_fn, device) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            total += float(loss_fn(logits, labels).item())
    return total / max(1, len(loader))


def evaluate_pairwise(model, loader, device, use_amp=False) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for batch in loader:
            positive = {key: value.to(device) for key, value in batch["positive"].items()}
            negative = {key: value.to(device) for key, value in batch["negative"].items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                positive_logits = model(**positive).logits.squeeze(-1)
                negative_logits = model(**negative).logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(positive_logits - negative_logits).mean()
            total_loss += float(loss.item())
            correct += int((positive_logits > negative_logits).sum().item())
            examples += int(positive_logits.numel())
    return total_loss / max(1, len(loader)), correct / max(1, examples)


def _records_for_split(dataset_dir: Path, split: str, limit: int | None) -> List[PasaQueryRecord]:
    if split == "real":
        normal_path = dataset_dir / "RealScholarQuery" / "test.jsonl"
        legacy_path = dataset_dir / "RealScholarQuery" / "test .jsonl"
        return read_many([normal_path if normal_path.exists() else legacy_path], limit=limit)
    return read_many([dataset_dir / "AutoScholarQuery" / f"{split}.jsonl"], limit=limit)


def evaluate_records(reranker, records, title_pool, candidates_per_query, rng, progress_every=0):
    total = 0
    recall_at_1 = 0
    recall_at_5 = 0
    mrr = 0.0
    started_at = time.time()
    record_count = len(records)
    for record in records:
        positives = record.answers
        positive_keys = {title.lower() for title in positives}
        negatives = [title for title in title_pool if title.lower() not in positive_keys]
        sampled = rng.sample(negatives, k=min(max(0, candidates_per_query - len(positives)), len(negatives)))
        candidates = positives + sampled
        scores = reranker.score_pairs(record.question, candidates)
        ranked = [title for _, title in sorted(zip(scores, candidates), reverse=True)]
        total += 1
        if ranked and ranked[0] in positives:
            recall_at_1 += 1
        if any(title in positives for title in ranked[:5]):
            recall_at_5 += 1
        for rank, title in enumerate(ranked, start=1):
            if title in positives:
                mrr += 1.0 / rank
                break
        if progress_every and (total == 1 or total % progress_every == 0 or total == record_count):
            elapsed = time.time() - started_at
            speed = total / max(elapsed, 1e-9)
            remaining = max(0, record_count - total)
            print(
                json.dumps(
                    {
                        "phase": "eval",
                        "processed": total,
                        "total": record_count,
                        "progress_percent": round(total / max(1, record_count) * 100, 2),
                        "elapsed_seconds": round(elapsed, 1),
                        "queries_per_second": round(speed, 3),
                        "eta_seconds": round(remaining / max(speed, 1e-9), 1),
                        "running_recall@1": round(recall_at_1 / total, 4),
                        "running_recall@5": round(recall_at_5 / total, 4),
                        "running_mrr": round(mrr / total, 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {
        "queries": total,
        "recall@1": recall_at_1 / max(1, total),
        "recall@5": recall_at_5 / max(1, total),
        "mrr": mrr / max(1, total),
    }


def _compact_dev_metrics(pairs, weights, bias) -> tuple[float, float]:
    scored = []
    total_loss = 0.0
    for pair in pairs:
        features = pair_features(pair.query, pair.title)
        score = _sigmoid(bias + sum(weight * value for weight, value in zip(weights, features)))
        scored.append((score, pair.label))
        total_loss += _binary_loss(score, pair.label)
    return total_loss / max(1, len(pairs)), _auc(scored)


def _binary_loss(prob: float, label: int) -> float:
    clipped = min(1.0 - 1e-8, max(1e-8, prob))
    return -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _auc(scored_labels) -> float:
    positives = sum(1 for _, label in scored_labels if label == 1)
    negatives = len(scored_labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(scored_labels, key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


if __name__ == "__main__":
    raise SystemExit(main())
