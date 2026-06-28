import unittest
import socket
from unittest.mock import patch

from scholarseek.models import Paper
from scholarseek.qwen import QwenAnswerSynthesizer, QwenPlanner, build_qwen_query_plan, synthesize_qwen_answer


class FakeQwenPlanner(QwenPlanner):
    def _post_chat(self, payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "search_queries": [
                            "graph neural networks molecular property prediction MoleculeNet",
                            "MoleculeNet benchmark graph neural networks"
                          ],
                          "must_terms": ["graph neural networks", "molecular property prediction", "MoleculeNet"],
                          "optional_terms": ["benchmark", "recent"],
                          "year_from": 2024,
                          "year_to": 2026
                        }
                        """
                    }
                }
            ]
        }


class FakeQwenAnswerSynthesizer(QwenAnswerSynthesizer):
    def _post_chat(self, payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Direct answer: use [1] as the strongest candidate and verify [2]."
                    }
                }
            ]
        }


class QwenPlannerTest(unittest.TestCase):
    def test_qwen_planner_falls_back_on_socket_timeout(self):
        with patch("scholarseek.qwen.urlopen", side_effect=socket.timeout("timed out")):
            plan = build_qwen_query_plan(
                "papers on LLM agents",
                model="qwen-test",
                fallback_on_error=True,
            )

        self.assertTrue(plan.planner.startswith("heuristic-fallback:qwen-test"))

    def test_qwen_planner_prioritizes_concise_generated_queries(self):
        planner = FakeQwenPlanner(model="qwen-test")
        plan = planner.build_plan(
            "Find recent graph neural network papers for molecular property prediction on MoleculeNet.",
            max_queries=3,
        )

        self.assertEqual(plan.planner, "qwen:qwen-test")
        self.assertEqual(plan.year_from, 2024)
        self.assertIn("MoleculeNet", plan.must_terms)
        self.assertLessEqual(len(plan.search_queries), 3)
        self.assertEqual(
            plan.search_queries[0],
            "graph neural networks molecular property prediction MoleculeNet",
        )

    def test_qwen_planner_falls_back_when_endpoint_unavailable(self):
        plan = build_qwen_query_plan(
            "papers on LLM agents for academic paper search",
            base_url="http://127.0.0.1:1/v1",
            model="qwen-test",
            fallback_on_error=True,
        )

        self.assertTrue(plan.planner.startswith("heuristic-fallback:qwen-test"))

    def test_qwen_answer_synthesizer_returns_text(self):
        synthesizer = FakeQwenAnswerSynthesizer(model="qwen-test")
        answer = synthesizer.synthesize("test query", [_paper("1", "Relevant Paper")])

        self.assertIn("Direct answer", answer)
        self.assertIn("[1]", answer)

    def test_qwen_answer_fallback_returns_ranked_summary(self):
        answer = synthesize_qwen_answer(
            "test query",
            [_paper("1", "Relevant Paper")],
            base_url="http://127.0.0.1:1/v1",
            model="qwen-test",
            fallback_on_error=True,
        )

        self.assertIn("fallback summarizes", answer)
        self.assertIn("Relevant Paper", answer)


def _paper(identifier, title):
    return Paper(
        id=identifier,
        title=title,
        year=2025,
        venue="Test Venue",
        authors=["A. Researcher"],
        abstract="This paper is relevant evidence.",
        doi=None,
        url=None,
        citation_count=3,
        source="test",
        raw={},
        score=0.9,
        relevance_reason="matched core terms",
    )


if __name__ == "__main__":
    unittest.main()
