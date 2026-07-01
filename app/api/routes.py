from typing import Any, Dict, List

from fastapi import APIRouter

from app.core.ast_analyzer import analyze_code
from app.core.embeddings import find_similar_pairs
from app.models.schemas import (
    ReviewFileResponse,
    ReviewIssue,
    ReviewRequest,
    ReviewScope,
    ReviewSnippetResponse,
    ReviewSummary,
    SimilarityMatch,
)

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@router.post("/review", tags=["Review"], response_model=ReviewFileResponse)
def review_code(payload: ReviewRequest) -> ReviewFileResponse:
    issues = analyze_code(payload.code)
    scopes = _group_issues_by_scope(payload.code)
    summary = _build_summary(issues, scopes)
    return ReviewFileResponse(
        filename=payload.filename,
        summary=summary,
        scopes=scopes,
    )


@router.post("/review/snippet", tags=["Review"], response_model=ReviewSnippetResponse)
def review_snippet(payload: ReviewRequest) -> ReviewSnippetResponse:
    issues = analyze_code(payload.code)
    summary = _build_summary(issues, [])
    return ReviewSnippetResponse(
        filename=payload.filename,
        summary=summary,
        issues=[ReviewIssue(**issue) for issue in issues],
    )


def _group_issues_by_scope(code: str) -> List[ReviewScope]:
    import ast

    tree = ast.parse(code)
    function_nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    scopes: List[ReviewScope] = []
    function_snippets: List[str] = []
    function_names: List[str] = []

    for node in function_nodes:
        snippet = ast.get_source_segment(code, node) or ""
        function_issues = [ReviewIssue(**issue) for issue in analyze_code(snippet)]
        if function_issues:
            scopes.append(
                ReviewScope(
                    name=node.name,
                    kind="function",
                    line_number=node.lineno,
                    issues=function_issues,
                )
            )
            function_snippets.append(snippet)
            function_names.append(node.name)

    if len(function_snippets) >= 2:
        similar_pairs = find_similar_pairs(function_snippets)
        for pair in similar_pairs:
            left_name = function_names[pair["left_index"]]
            right_name = function_names[pair["right_index"]]
            scopes[pair["left_index"]].similar_scopes.append(
                SimilarityMatch(
                    left_scope=left_name,
                    right_scope=right_name,
                    similarity=pair["similarity"],
                )
            )
            scopes[pair["right_index"]].similar_scopes.append(
                SimilarityMatch(
                    left_scope=right_name,
                    right_scope=left_name,
                    similarity=pair["similarity"],
                )
            )

    return scopes


def _build_summary(issues: List[Dict[str, Any]], scopes: List[ReviewScope]) -> ReviewSummary:
    severity_counts: Dict[str, int] = {}
    for issue in issues:
        severity = issue["severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return ReviewSummary(
        total_issues=len(issues),
        severity_counts=severity_counts,
        scope_count=len(scopes),
        risk_score=None,
        similar_examples=[],
    )
