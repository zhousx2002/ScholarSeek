import unittest
from unittest.mock import patch

from scholarseek.listwise_reranker import rerank_with_qwen
from scholarseek.models import Paper


class ListwiseRerankerTest(unittest.TestCase):
    def test_uses_qwen_best_first_indices_and_fills_missing_results(self):
        papers = [_paper("A"), _paper("B"), _paper("C")]
        response = {
            "papers": [
                {"index": 3, "score": 0.9, "reason": "best"},
                {"index": 1, "score": 0.8, "reason": "second"},
            ]
        }
        with patch("scholarseek.listwise_reranker.request_qwen_json", return_value=response):
            ranked = rerank_with_qwen(
                "query",
                papers,
                3,
                base_url="https://example.com",
                model="qwen-test",
                api_key="key",
            )

        self.assertEqual(["C", "A", "B"], [paper.title for paper in ranked])


def _paper(title):
    return Paper(title, title, 2025, "venue", [], "abstract", None, None, 0, "test", {})


if __name__ == "__main__":
    unittest.main()
