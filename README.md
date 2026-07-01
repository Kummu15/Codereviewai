# CodeReview AI

CodeReview AI is a lightweight code review service that combines AST-based static checks with semantic similarity analysis for grouped function review.

## Similarity threshold

The semantic similarity module uses a cosine-similarity threshold of 0.85 for pairing related functions. This value is deliberately high enough to catch near-duplicate logic and repeated implementation patterns, while avoiding noisy matches caused by common constructs such as loops, error handling, or boilerplate. It is a practical default for early iterations and can be tuned later with labeled review data.
