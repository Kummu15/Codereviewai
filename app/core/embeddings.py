from __future__ import annotations

import math
from typing import List, Sequence, Tuple

SIMILARITY_THRESHOLD = 0.85
TOP_K = 5

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers is required for embeddings. Install it via requirements.txt."
            ) from exc
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_embeddings(code_strings: Sequence[str]) -> List[List[float]]:
    if not code_strings:
        return []
    model = _get_model()
    embeddings = model.encode(list(code_strings), convert_to_numpy=False)
    return [list(map(float, embedding)) for embedding in embeddings]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot_product = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(value) * float(value) for value in vec_a))
    norm_b = math.sqrt(sum(float(value) * float(value) for value in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_similar_pairs(code_strings: Sequence[str], threshold: float = SIMILARITY_THRESHOLD, top_k: int = TOP_K) -> List[dict]:
    if len(code_strings) < 2:
        return []

    embeddings = get_embeddings(code_strings)
    results: List[dict] = []
    for left_index in range(len(code_strings)):
        for right_index in range(left_index + 1, len(code_strings)):
            similarity = cosine_similarity(embeddings[left_index], embeddings[right_index])
            if similarity >= threshold:
                results.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "similarity": round(similarity, 4),
                    }
                )

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[:top_k]
