import json
import tempfile
import unittest
from pathlib import Path

from scholarseek.reranker_dataset import HardNegativeMiner, build_title_pool, generate_pairs, read_many, write_pairs_jsonl
from scholarseek.train_reranker import build_pairwise_triplets


class RerankerDatasetTest(unittest.TestCase):
    def test_generate_pairs_uses_positive_and_negative_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "train.jsonl"
            data_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question": "find graph neural network papers",
                                "answer": ["Graph Neural Networks for Molecules"],
                                "answer_arxiv_id": ["1234.5678"],
                                "qid": "q1",
                            }
                        ),
                        json.dumps(
                            {
                                "question": "find retrieval papers",
                                "answer": ["Neural Information Retrieval"],
                                "answer_arxiv_id": ["2222.3333"],
                                "qid": "q2",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            records = read_many([data_file])
            pool = build_title_pool(records)
            pairs = list(generate_pairs(records[:1], pool, negatives_per_positive=1, seed=1))

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].label, 1)
        self.assertEqual(pairs[0].title, "Graph Neural Networks for Molecules")
        self.assertEqual(pairs[1].label, 0)
        self.assertNotEqual(pairs[1].title, pairs[0].title)

    def test_write_pairs_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "train.jsonl"
            out_file = Path(directory) / "pairs.jsonl"
            data_file.write_text(
                json.dumps(
                    {
                        "question": "q",
                        "answer": ["A", "B"],
                        "answer_arxiv_id": ["1", "2"],
                        "qid": "qid",
                    }
                ),
                encoding="utf-8",
            )
            records = read_many([data_file])
            count = write_pairs_jsonl(generate_pairs(records, build_title_pool(records), 1), out_file)

            self.assertEqual(count, 2)
            self.assertTrue(out_file.exists())

    def test_hard_negative_miner_prefers_lexically_related_title(self):
        miner = HardNegativeMiner(
            [
                "Dense Retrieval with Transformer Encoders",
                "Graph Neural Networks for Molecules",
                "Image Segmentation with U-Net",
            ]
        )
        negatives = miner.mine("transformer dense retrieval", set(), count=1)
        self.assertEqual(["Dense Retrieval with Transformer Encoders"], negatives)

    def test_pairwise_triplets_follow_each_positive(self):
        from scholarseek.reranker_dataset import RerankerPair

        pairs = [
            RerankerPair("query", "positive one", 1, "q1"),
            RerankerPair("query", "negative one", 0, "q1"),
            RerankerPair("query", "positive two", 1, "q1"),
            RerankerPair("query", "negative two", 0, "q1"),
        ]
        self.assertEqual(
            [("query", "positive one", "negative one"), ("query", "positive two", "negative two")],
            build_pairwise_triplets(pairs),
        )


if __name__ == "__main__":
    unittest.main()
