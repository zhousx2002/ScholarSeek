import unittest

from scholarseek.query_planner import build_query_plan


class QueryPlannerTest(unittest.TestCase):
    def test_build_query_plan_extracts_recent_year_range(self):
        plan = build_query_plan(
            "graph neural networks for molecular property prediction on MoleculeNet in the last 3 years",
            max_queries=4,
        )

        self.assertIsNotNone(plan.year_from)
        self.assertIsNotNone(plan.year_to)
        self.assertEqual(plan.year_to - plan.year_from, 2)
        self.assertIn("graph neural network", plan.must_terms)
        self.assertIn("moleculenet", plan.must_terms)
        self.assertLessEqual(len(plan.search_queries), 4)

    def test_build_query_plan_extracts_explicit_year_range(self):
        plan = build_query_plan("retrieval augmented generation for LLM hallucination detection 2022 2024")

        self.assertEqual(plan.year_from, 2022)
        self.assertEqual(plan.year_to, 2024)
        self.assertIn("retrieval augmented generation", plan.must_terms)

    def test_build_query_plan_translates_chinese_domain_terms(self):
        plan = build_query_plan("有哪些无人机相关的文献？", max_queries=5)

        joined = " | ".join(plan.search_queries).lower()
        self.assertIn("unmanned aerial vehicle", joined)
        self.assertIn("drone", joined)


if __name__ == "__main__":
    unittest.main()
