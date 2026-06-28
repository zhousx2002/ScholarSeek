from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "eval"
FIG_DIR = ROOT / "outputs" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    autoscholar = {
        10: _load_summary("autoscholar_local_openalex_20_summary.json"),
        20: _load_summary("autoscholar_local_openalex_top5_20_summary.json"),
    }
    sparbench = {
        5: _load_summary("sparbench_local_openalex_top5_20_summary.json"),
        10: _load_summary("sparbench_local_openalex_top10_20_summary.json"),
        20: _load_summary("sparbench_local_openalex_top20_20_summary.json"),
    }
    reranker = _load_json("reranker_test_results.json")["metrics"]

    _plot_topk_curves("AutoScholar", autoscholar, FIG_DIR / "autoscholar_topk_metrics.png")
    _plot_topk_curves("SPARBench", sparbench, FIG_DIR / "sparbench_topk_metrics.png")
    _plot_candidate_gap(autoscholar, sparbench, FIG_DIR / "candidate_recall_gap.png")
    _plot_latency(autoscholar, sparbench, FIG_DIR / "latency_comparison.png")
    _plot_reranker(reranker, FIG_DIR / "reranker_module_metrics.png")
    _write_table_data(autoscholar, sparbench, reranker)

    print(f"Saved figures to {FIG_DIR}")


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 240


def _load_summary(name: str) -> dict:
    return _load_json(name)


def _load_json(name: str) -> dict:
    path = EVAL_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_topk_curves(dataset: str, data: dict[int, dict], output: Path) -> None:
    ks = sorted(data)
    precision = [data[k]["precision"] for k in ks]
    recall = [data[k]["recall"] for k in ks]
    f1 = [data[k]["f1"] for k in ks]
    hit = [data[k].get(f"hit@{k}", 0.0) for k in ks]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, precision, marker="o", linewidth=2.2, label="Precision@K")
    ax.plot(ks, recall, marker="s", linewidth=2.2, label="Recall@K")
    ax.plot(ks, f1, marker="^", linewidth=2.2, label="F1@K")
    ax.plot(ks, hit, marker="D", linewidth=2.2, label="Hit@K")
    ax.set_title(f"{dataset} Top-K Evaluation")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Score")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(0.7, max(hit + recall + precision + f1) * 1.15))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_gap(autoscholar: dict[int, dict], sparbench: dict[int, dict], output: Path) -> None:
    rows = [
        ("AutoScholar@10", autoscholar[10], 10),
        ("AutoScholar@20", autoscholar[20], 20),
        ("SPARBench@10", sparbench[10], 10),
        ("SPARBench@20", sparbench[20], 20),
    ]
    labels = [row[0] for row in rows]
    retrieval_recall = [row[1]["retrieval_recall"] for row in rows]
    final_recall = [row[1]["recall"] for row in rows]
    retrieval_hit = [row[1]["retrieval_hit"] for row in rows]
    final_hit = [row[1].get(f"hit@{row[2]}", 0.0) for row in rows]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 1.5 * width, retrieval_recall, width, label="Candidate Recall")
    ax.bar(x - 0.5 * width, final_recall, width, label="Final Recall")
    ax.bar(x + 0.5 * width, retrieval_hit, width, label="Candidate Hit")
    ax.bar(x + 1.5 * width, final_hit, width, label="Final Hit")
    ax.set_title("Candidate Pool vs Final Top-K Results")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_latency(autoscholar: dict[int, dict], sparbench: dict[int, dict], output: Path) -> None:
    rows = [
        ("AutoScholar@10", autoscholar[10]),
        ("AutoScholar@20", autoscholar[20]),
        ("SPARBench@5", sparbench[5]),
        ("SPARBench@10", sparbench[10]),
        ("SPARBench@20", sparbench[20]),
    ]
    labels = [row[0] for row in rows]
    values = [row[1]["average_latency_seconds"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#4C78A8", "#59A14F", "#59A14F", "#59A14F"])
    ax.set_title("Average Query Latency")
    ax.set_ylabel("Seconds / query")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_reranker(metrics: dict, output: Path) -> None:
    labels = ["Recall@1", "Recall@5", "MRR"]
    values = [metrics["recall@1"], metrics["recall@5"], metrics["mrr"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F28E2B", "#59A14F"])
    ax.set_title("Trainable Reranker Module Evaluation")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _write_table_data(autoscholar: dict[int, dict], sparbench: dict[int, dict], reranker: dict) -> None:
    payload = {
        "autoscholar": autoscholar,
        "sparbench": sparbench,
        "reranker": reranker,
        "figures": [
            "autoscholar_topk_metrics.png",
            "sparbench_topk_metrics.png",
            "candidate_recall_gap.png",
            "latency_comparison.png",
            "reranker_module_metrics.png",
        ],
    }
    (FIG_DIR / "figure_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
