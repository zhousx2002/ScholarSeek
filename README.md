# ScholarCP / ScholarSeek Agent

ScholarCP is an academic paper search and recommendation agent built for the Huawei Track 3 task,
"intelligent paper search and recommendation for complex academic queries in research scenarios".
The system accepts natural-language research needs, retrieves candidate papers from local and online
sources, reranks them with PaSa-trained relevance models, and generates evidence-grounded answers
with Qwen.

The project currently includes:

- Web UI with home, result list, paper detail expansion, and related-paper graph page.
- Fast and SPAR search modes.
- LocalCorpus retrieval from PaSa/SPAR data, plus OpenAlex, arXiv, and Semantic Scholar API adapters.
- Qwen-based query planning, query evolution, answer generation, and optional listwise reranking.
- PaSa compact trainable reranker and optional BAAI/bge-reranker-base cross-encoder training pipeline.
- SPAR-style Judgement Agent, one-layer citation expansion, and query evolution.
- End-to-end evaluation on AutoScholarQuery and SPARBench with Precision, Recall, F1, Hit@K, MRR, and latency.
- Experimental figure generation for project reports.

## Project Layout

```text
frontend/                 Web pages, styles, and browser logic
src/scholarseek/           Backend service, retrievers, planners, rankers, evaluators
scripts/                   Utility scripts, including experimental figure generation
outputs/eval/              Evaluation result JSON/JSONL files
outputs/figures/           Generated experiment figures and figure_metrics.json
outputs/reranker_compact/  Compact fallback reranker
outputs/reranker_model_base/ Optional Transformer reranker output
submission_docs/           Competition document drafts
start_web.ps1              One-command web startup script
```

## Environment

The project is normally run in the `DP_learn` conda environment:

```powershell
conda activate DP_learn
cd E:\VsCodeProjects\ScholarSeek_Agent
python -m pip install -e .
```

Install reranker dependencies only when training or loading the Transformer cross-encoder:

```powershell
python -m pip install -e ".[reranker]"
```

If PowerShell cannot import the local package, set:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:PYTHONPATH="src"
```

The provided startup script already sets these two variables.

## Configuration

Copy the example configuration and fill in keys in `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Important fields:

```text
DASHSCOPE_API_KEY=replace_with_your_dashscope_api_key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

OPENALEX_EMAIL=your_email@example.com
OPENALEX_API_KEY=replace_with_your_openalex_api_key
SEMANTIC_SCHOLAR_API_KEY=replace_with_your_semantic_scholar_api_key

SCHOLARSEEK_SOURCES=local,openalex,semantic-scholar,arxiv
SCHOLARSEEK_LOCAL_DATASET_DIR=E:\DATASET\PasaDataSet
SCHOLARSEEK_LOCAL_CORPUS_FILES=AutoScholarQuery/train.jsonl;AutoScholarQuery/dev.jsonl;RealScholarQuery/test.jsonl
SCHOLARSEEK_LOCAL_MAX_RESULTS=120

SCHOLARSEEK_RERANKER_PATH=outputs/reranker_model_base
SCHOLARSEEK_FALLBACK_RERANKER_PATH=outputs/reranker_compact
```

Use `.env` for real API keys. `.env.example` is only a template and should not contain secrets.

## Start The Web System

Recommended:

```powershell
cd E:\VsCodeProjects\ScholarSeek_Agent
.\start_web.ps1
```

If script execution is blocked:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_web.ps1
```

Open:

```text
http://127.0.0.1:5174
```

The server serves both the frontend and backend APIs:

- `/api/search`: search endpoint used by the frontend.
- `/api/config`: masked runtime configuration.

Manual startup:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:PYTHONPATH="src"
E:\miniconda3\envs\DP_learn\python.exe -m scholarseek.web_server --host 127.0.0.1 --port 5174
```

## Frontend Usage

The UI provides:

- Search Mode:
  - `Fast`: direct query planning, retrieval, deduplication, and reranking. Recommended for demos.
  - `SPAR`: enhanced pipeline with judgement, citation expansion, and query evolution.
- Sources:
  - `LocalCorpus`: local PaSa/SPAR-derived candidate corpus; fast and useful for benchmark-style queries.
  - `OpenAlex`: online academic search API.
  - `arXiv`: online preprint search API, useful for AI/CS topics but may be slower.
  - `Semantic Scholar`: enabled after configuring an API key.
- Retrieval controls:
  - `Top-K`: number of final papers shown.
  - `Per query`: candidates fetched per query variant.
  - `Queries`: number of generated query variants.
- Answer:
  - Uses Qwen to synthesize an evidence-based answer from retrieved papers.

Clicking a paper expands its abstract and relevance information. The related-paper action opens a
graph view that shows the selected paper and associated papers through topic, author, source, or
year-proximity links.

## Backend CLI

Basic retrieval:

```powershell
python -m scholarseek.cli "papers on LLM agents for academic paper search" --sources local,openalex --limit 10
```

Qwen planner:

```powershell
python -m scholarseek.cli "有哪些无人机协同感知与任务导向通信相关的论文？" --planner qwen --sources local,openalex --limit 10
```

Qwen answer generation:

```powershell
python -m scholarseek.cli "Find papers about query reformulation for retrieval augmented generation" `
  --planner qwen `
  --answer qwen `
  --sources local,openalex `
  --limit 10
```

Show masked API configuration:

```powershell
python -m scholarseek.cli "test" --show-api-config
```

## Retrieval Sources

ScholarCP supports both local and online retrieval.

- `local`: local corpus built from configured PaSa/SPAR data files.
- `openalex`: OpenAlex Works search.
- `arxiv`: arXiv API search.
- `semantic-scholar`: Semantic Scholar Graph API search and citation expansion.

Recommended demo setting:

```text
local,openalex
```

This keeps latency low while preserving online search coverage. Use `arxiv` for AI/CS-heavy queries
when additional coverage is needed. Use `semantic-scholar` only after obtaining an API key.

## Reranking

The backend first performs lexical and metadata scoring, then optionally reranks candidates using a
trainable reranker.

Current reranker options:

- Compact fallback reranker: small PaSa-trained feature model stored in `outputs/reranker_compact`.
- Transformer reranker: optional cross-encoder stored in `outputs/reranker_model_base`.

In the web server, if the installed PyTorch/Transformers environment cannot safely load the
Transformer reranker, the system automatically falls back to the compact reranker.

The displayed relevance value in the frontend is a calibrated user-facing score. It is not the same
as official evaluation Precision, Recall, or F1.

## Train The Reranker

Prepare PaSa-style pairwise data:

```powershell
python -m scholarseek.train_reranker prepare `
  --dataset-dir E:\DATASET\PasaDataSet `
  --output-dir outputs\reranker_data_hard `
  --negatives-per-positive 2 `
  --hard-negatives-per-positive 4
```

Train the BAAI/bge-reranker-base cross-encoder with pairwise RankNet loss:

```powershell
python -m scholarseek.train_reranker train `
  --train-file outputs\reranker_data_hard\train_pairs.jsonl `
  --dev-file outputs\reranker_data_hard\dev_pairs.jsonl `
  --output-dir outputs\reranker_model_base `
  --base-model BAAI/bge-reranker-base `
  --epochs 5 `
  --batch-size 1 `
  --gradient-accumulation-steps 16 `
  --loss pairwise `
  --fp16 `
  --gradient-checkpointing `
  --learning-rate 2e-5 `
  --max-length 256
```

Evaluate the reranker:

```powershell
python -m scholarseek.train_reranker eval `
  --dataset-dir E:\DATASET\PasaDataSet `
  --model-dir outputs\reranker_model_base `
  --split test `
  --candidates-per-query 64 `
  --log-every 25 `
  --output outputs\eval\reranker_test_results.json
```

Train the compact submission-friendly reranker:

```powershell
python -m scholarseek.train_reranker train-compact `
  --train-file outputs\reranker_data\train_pairs.jsonl `
  --dev-file outputs\reranker_data\dev_pairs.jsonl `
  --output-dir outputs\reranker_compact `
  --epochs 8
```

## SPAR-Style Pipeline

The enhanced pipeline follows a SPAR-inspired process:

1. Query planning.
2. Initial multi-source retrieval.
3. Judgement Agent for related, unrelated, or uncertain candidate filtering.
4. One-layer citation/reference expansion when supported.
5. Query evolution based on relevant papers.
6. Final reranking and Top-K selection.
7. Optional Qwen evidence-grounded answer generation.

The search API accepts:

```text
standard
spar
spar-qwen
```

`spar-qwen` uses Qwen for query planning and selected relevance/reranking steps. It can improve
semantic reasoning but is slower and may time out under network or API pressure. For stable demos,
prefer `Fast` mode with `local,openalex`.

## End-To-End Evaluation

AutoScholarQuery benchmark:

```powershell
python -m scholarseek.eval_end_to_end `
  --dataset-file E:\DATASET\PasaDataSet\benchmark\AutoScholarQuery_test.jsonl `
  --strategy standard `
  --planner heuristic `
  --sources local,openalex `
  --top-k 10 `
  --per-query 80 `
  --max-queries 3 `
  --max-records 20 `
  --output outputs\eval\autoscholar_local_openalex_20.jsonl
```

SPARBench Top-5:

```powershell
python -m scholarseek.eval_end_to_end `
  --dataset-file E:\DATASET\PasaDataSet\benchmark\spar_bench.jsonl `
  --strategy standard `
  --planner heuristic `
  --sources local,openalex `
  --top-k 5 `
  --per-query 80 `
  --max-queries 3 `
  --max-records 20 `
  --output outputs\eval\sparbench_local_openalex_top5_20.jsonl
```

SPARBench Top-10:

```powershell
python -m scholarseek.eval_end_to_end `
  --dataset-file E:\DATASET\PasaDataSet\benchmark\spar_bench.jsonl `
  --strategy standard `
  --planner heuristic `
  --sources local,openalex `
  --top-k 10 `
  --per-query 80 `
  --max-queries 3 `
  --max-records 20 `
  --output outputs\eval\sparbench_local_openalex_top10_20.jsonl
```

SPARBench Top-20:

```powershell
python -m scholarseek.eval_end_to_end `
  --dataset-file E:\DATASET\PasaDataSet\benchmark\spar_bench.jsonl `
  --strategy standard `
  --planner heuristic `
  --sources local,openalex `
  --top-k 20 `
  --per-query 80 `
  --max-queries 3 `
  --max-records 20 `
  --output outputs\eval\sparbench_local_openalex_top20_20.jsonl
```

Use `--resume` to continue an interrupted run:

```powershell
python -m scholarseek.eval_end_to_end `
  --dataset-file E:\DATASET\PasaDataSet\benchmark\spar_bench.jsonl `
  --strategy standard `
  --planner heuristic `
  --sources local,openalex `
  --top-k 10 `
  --per-query 80 `
  --max-queries 3 `
  --output outputs\eval\sparbench_full.jsonl `
  --resume
```

The evaluator prints progress for each record and saves:

- Per-query JSONL results.
- Summary JSON with Precision, Recall, F1, Hit@K, MRR, retrieval recall, retrieval hit, and latency.

## Generate Experiment Figures

After evaluation summaries are available:

```powershell
E:\miniconda3\envs\DP_learn\python.exe scripts\generate_eval_figures.py
```

Generated files:

```text
outputs/figures/autoscholar_topk_metrics.png
outputs/figures/sparbench_topk_metrics.png
outputs/figures/candidate_recall_gap.png
outputs/figures/latency_comparison.png
outputs/figures/reranker_module_metrics.png
outputs/figures/figure_metrics.json
```

Do not set `PYTHONNOUSERSITE=1` when running the plotting script if the active environment relies
on user-site matplotlib dependencies.

## Reported Metrics

The evaluator reports:

- Precision
- Recall
- F1
- Micro Precision / Micro Recall / Micro F1
- Hit@K
- MRR
- Retrieval Recall
- Retrieval Hit
- Average retrieved candidates
- Average latency

These metrics are intended for project documentation and comparison tables. Retrieval Recall and
Retrieval Hit diagnose whether errors come from candidate recall or final reranking.

## Tests

```powershell
python -m unittest discover -s tests
```

## Notes

- `.env` is ignored by git and should contain real API keys.
- `Semantic Scholar` may be unavailable without an API key or under public rate limits.
- Online sources are affected by network latency and API throttling. For stable demonstrations,
  enable `LocalCorpus` and `OpenAlex`.
- If frontend changes do not appear, restart the server and hard-refresh the browser with `Ctrl+F5`.
