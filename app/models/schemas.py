from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    issue_type: str
    line_number: int
    severity: str
    message: str


class ReviewSummary(BaseModel):
    total_issues: int
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    scope_count: int = 0
    risk_score: Optional[float] = None
    similar_examples: List[Dict[str, Any]] = Field(default_factory=list)


class SimilarityMatch(BaseModel):
    left_scope: str
    right_scope: str
    similarity: float


class ReviewScope(BaseModel):
    name: str
    kind: str
    line_number: int
    issues: List[ReviewIssue] = Field(default_factory=list)
    similar_scopes: List[SimilarityMatch] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    code: str
    filename: str = "snippet.py"


class ReviewFileResponse(BaseModel):
    source: str = "file"
    filename: str
    summary: ReviewSummary
    scopes: List[ReviewScope] = Field(default_factory=list)


class ReviewSnippetResponse(BaseModel):
    source: str = "snippet"
    filename: str
    summary: ReviewSummary
    issues: List[ReviewIssue] = Field(default_factory=list)
