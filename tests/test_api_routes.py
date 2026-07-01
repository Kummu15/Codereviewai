import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class ReviewRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_review_returns_grouped_findings(self):
        payload = {
            "code": "def sample():\n    return 1\n",
            "filename": "example.py",
        }

        response = self.client.post("/review", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("summary", body)
        self.assertIn("scopes", body)
        self.assertGreaterEqual(body["summary"]["total_issues"], 1)
        self.assertTrue(any(scope["name"] == "sample" for scope in body["scopes"]))

    def test_review_snippet_returns_snippet_result(self):
        payload = {
            "code": "def sample():\n    return 1\n",
            "filename": "example.py",
        }

        response = self.client.post("/review/snippet", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("summary", body)
        self.assertIn("issues", body)
        self.assertEqual(body["source"], "snippet")

    def test_review_attaches_similarity_matches_to_scopes(self):
        payload = {
            "code": "def first():\n    return 1\n\ndef second():\n    return 1\n",
            "filename": "example.py",
        }

        with patch("app.api.routes.find_similar_pairs", return_value=[{"left_index": 0, "right_index": 1, "similarity": 0.93}]):
            response = self.client.post("/review", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        scopes = {scope["name"]: scope for scope in body["scopes"]}
        self.assertEqual(scopes["first"]["similar_scopes"][0]["similarity"], 0.93)


if __name__ == "__main__":
    unittest.main()
