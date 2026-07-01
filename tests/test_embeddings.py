import unittest
from unittest.mock import patch

from app.core.embeddings import SIMILARITY_THRESHOLD, TOP_K, cosine_similarity, find_similar_pairs, get_embeddings


class EmbeddingsTests(unittest.TestCase):
    def test_cosine_similarity_returns_expected_score(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 2], [2, 4]), 1.0)

    @patch("app.core.embeddings._get_model")
    def test_get_embeddings_uses_model_output(self, mock_get_model):
        mock_get_model.return_value.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

        embeddings = get_embeddings(["first", "second"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])

    @patch("app.core.embeddings.get_embeddings")
    def test_find_similar_pairs_uses_threshold_and_top_k(self, mock_get_embeddings):
        mock_get_embeddings.return_value = [
            [1.0, 0.0],
            [0.95, 0.31],
            [0.2, 0.1],
            [0.95, 0.30],
        ]

        results = find_similar_pairs(["a", "b", "c", "d"], threshold=0.9, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["left_index"] < results[0]["right_index"])
        self.assertGreaterEqual(results[0]["similarity"], 0.9)
        self.assertLessEqual(len(results), TOP_K)


if __name__ == "__main__":
    unittest.main()
