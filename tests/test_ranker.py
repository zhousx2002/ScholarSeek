import unittest
from unittest.mock import patch

from scholarseek.models import Paper
from scholarseek.query_planner import build_query_plan
from scholarseek.ranker import rank_papers
from scholarseek.trainable_reranker import rerank_papers_with_model


class RankerTest(unittest.TestCase):
    def test_trainable_reranker_preserves_strong_lexical_evidence(self):
        plan = build_query_plan("graph neural networks molecular prediction")
        lexical_first = _paper("1", "Graph Neural Networks for Molecular Prediction")
        semantic_first = _paper("2", "Molecular Representation Learning")

        class FakeReranker:
            def score_pairs(self, _query, _titles):
                return [0.1, 0.9]

        with patch("scholarseek.trainable_reranker.get_reranker", return_value=FakeReranker()):
            ranked = rerank_papers_with_model([lexical_first, semantic_first], plan, "unused", 2)

        self.assertEqual("1", ranked[0].id)
        self.assertIn("semantic_rank", ranked[0].relevance_reason)

    def test_ranker_prefers_constraint_matching_paper(self):
        plan = build_query_plan(
            "graph neural networks for molecular property prediction on MoleculeNet in the last 3 years"
        )
        relevant = Paper(
            id="1",
            title="Graph Neural Networks for Molecular Property Prediction on MoleculeNet",
            year=2025,
            venue="Example",
            authors=[],
            abstract="We study molecular property prediction with graph neural networks on MoleculeNet.",
            doi=None,
            url=None,
            citation_count=3,
            source="test",
            raw={},
        )
        generic = Paper(
            id="2",
            title="A Review of Graph Neural Networks",
            year=2025,
            venue="Example",
            authors=[],
            abstract="This survey studies graph neural networks for many applications.",
            doi=None,
            url=None,
            citation_count=5000,
            source="test",
            raw={},
        )

        ranked = rank_papers([generic, relevant], plan)

        self.assertEqual(ranked[0].id, "1")
        self.assertGreater(ranked[0].score, ranked[1].score)


def _paper(identifier, title):
    return Paper(
        id=identifier,
        title=title,
        year=2025,
        venue="Example",
        authors=[],
        abstract=title,
        doi=None,
        url=None,
        citation_count=0,
        source="test",
        raw={},
    )


if __name__ == "__main__":
    unittest.main()
