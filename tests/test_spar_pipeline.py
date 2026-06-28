import json
import tempfile
import unittest
from unittest.mock import patch

from scholarseek.citation_expander import CitationExpander
from pathlib import Path

from scholarseek.eval_end_to_end import (
    _read_existing_rows,
    aggregate_metrics,
    evaluate_retrieval,
    evaluate_titles,
    iter_benchmark,
)
from scholarseek.judgement import JudgementAgent
from scholarseek.models import Paper
from scholarseek.query_evolver import evolve_queries
from scholarseek.refchain_planner import _ensure_review_diversity
from scholarseek.search_service import _env_bool


class SparPipelineTests(unittest.TestCase):
    def test_reads_existing_rows_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps({"index": 1}) + "\n", encoding="utf-8")

            rows = _read_existing_rows(path)

        self.assertEqual([1], [row["index"] for row in rows])

    def test_disabled_citation_expander_makes_no_api_requests(self):
        self.assertEqual([], CitationExpander(enabled=False).expand([]))

    def test_citation_expander_uses_openalex_without_semantic_key(self):
        seed = _paper("Seed Paper", "")
        seed.id = "https://openalex.org/W1"
        seed.source = "OpenAlex"
        expander = CitationExpander(enabled=True, use_openalex=True)

        class FakeOpenAlex:
            def get_work(self, work_id, select=None):
                if str(work_id).endswith("W1"):
                    return {"referenced_works": ["https://openalex.org/W2"], "cited_by_api_url": ""}
                return {
                    "id": "https://openalex.org/W2",
                    "display_name": "Referenced Paper",
                    "publication_year": 2024,
                    "primary_location": {"source": {"display_name": "Test Venue"}},
                    "authorships": [],
                    "cited_by_count": 3,
                    "open_access": {},
                    "abstract_inverted_index": {},
                }

            def cited_by(self, cited_by_api_url, limit=8):
                return []

        expander.openalex = FakeOpenAlex()

        expanded = expander.expand([seed], max_seeds=1, per_seed=1)

        self.assertEqual(["Referenced Paper"], [paper.title for paper in expanded])

    def test_openalex_refchain_env_flag(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_env_bool("SCHOLARSEEK_ENABLE_OPENALEX_REFCHAIN", False))
        with patch.dict("os.environ", {"SCHOLARSEEK_ENABLE_OPENALEX_REFCHAIN": "1"}):
            self.assertTrue(_env_bool("SCHOLARSEEK_ENABLE_OPENALEX_REFCHAIN", False))

    def test_judgement_retains_relative_top_candidates(self):
        papers = [_paper("Paper A", ""), _paper("Paper B", ""), _paper("Paper C", "")]
        agent = JudgementAgent(related_threshold=0.9, unrelated_threshold=0.8)
        agent._scores = lambda _query, _papers: [0.1, 0.3, 0.2]

        accepted = agent.filter("query", papers, keep_uncertain=False, min_keep=2)

        self.assertEqual(["Paper B", "Paper C"], [paper.title for paper in accepted])

    def test_judgement_limits_qwen_reviews(self):
        papers = [_paper(f"Paper {index}", "") for index in range(10)]
        agent = JudgementAgent(use_qwen=True, qwen_api_key="test", max_qwen_reviews=4)
        agent._scores = lambda _query, _papers: [0.5] * len(papers)
        with patch("scholarseek.judgement.request_qwen_json", return_value={"decision": "uncertain"}) as mocked:
            agent.judge("query", papers)

        self.assertEqual(4, mocked.call_count)

    def test_final_results_keep_review_diversity(self):
        methods = [_paper(f"Method {index}", "") for index in range(10)]
        reviews = [_paper("Topic Survey", ""), _paper("Systematic Review", "")]

        selected = _ensure_review_diversity(methods, reviews, limit=10)

        self.assertIn("Topic Survey", [paper.title for paper in selected])
        self.assertIn("Systematic Review", [paper.title for paper in selected])
        self.assertEqual(10, len(selected))

    def test_query_evolution_generates_three_novel_directions(self):
        paper = _paper("Dense retrieval methods", "We compare dense encoders and discuss limitations.")
        queries = evolve_queries("dense retrieval", [paper], ["dense retrieval"], max_queries=3)
        self.assertEqual(3, len(queries))
        self.assertTrue(any("limitations" in query for query in queries))

    def test_end_to_end_title_metrics(self):
        metrics = evaluate_titles(["Paper A", "Other"], ["Paper A", "Paper B"], top_k=2)
        self.assertEqual(0.5, metrics["precision"])
        self.assertEqual(0.5, metrics["recall"])
        self.assertEqual(1.0, metrics["hit@2"])
        self.assertEqual(1.0, metrics["mrr"])

    def test_retrieval_metrics_measure_candidate_pool_before_top_k(self):
        metrics = evaluate_retrieval(["Paper A", "Other"], ["Paper A", "Paper B"])

        self.assertEqual(0.5, metrics["retrieval_recall"])
        self.assertEqual(1.0, metrics["retrieval_hit"])
        self.assertEqual(2, metrics["retrieved_candidates"])

    def test_reads_pasa_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(
                json.dumps({"qid": "1", "question": "Find papers", "answer": ["Paper A"]}) + "\n",
                encoding="utf-8",
            )
            rows = list(iter_benchmark(str(path)))
        self.assertEqual("Paper A", rows[0]["answers"][0])

    def test_aggregate_includes_micro_metrics(self):
        rows = [
            {
                "predicted_titles": ["Paper A", "Other"],
                "gold_titles": ["Paper A"],
                "metrics": evaluate_titles(["Paper A", "Other"], ["Paper A"], 2),
                "latency_seconds": 1.0,
            }
        ]
        summary = aggregate_metrics(rows)
        self.assertEqual(0.5, summary["micro_precision"])
        self.assertEqual(1.0, summary["micro_recall"])


def _paper(title, abstract):
    return Paper("id", title, 2024, "venue", [], abstract, None, None, 0, "test", {})


if __name__ == "__main__":
    unittest.main()
