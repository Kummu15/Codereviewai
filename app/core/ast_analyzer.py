import ast
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


LONG_FUNCTION_LINES = 20
MAX_NESTING_DEPTH = 3


class AnalyzerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[Dict[str, Any]] = []
        self._imported_names: Set[str] = set()
        self._used_names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._imported_names.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name.split(".")[0]
            self._imported_names.add(name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._used_names.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add_issue(
                issue_type="bare_except",
                line_number=node.lineno,
                severity="medium",
                message="Catch all exceptions with a bare except clause.",
            )
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.generic_visit(node)

    def _check_function(self, node: ast.AST) -> None:
        self._check_long_function(node)
        self._check_missing_docstring(node)

    def _check_long_function(self, node: ast.AST) -> None:
        if not hasattr(node, "body"):
            return
        body_lines = self._get_body_line_count(node.body)
        if body_lines > LONG_FUNCTION_LINES:
            self._add_issue(
                issue_type="long_function",
                line_number=node.lineno,
                severity="medium",
                message="Function body is longer than the recommended size.",
            )

    def _check_missing_docstring(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if not node.body or not self._is_docstring(node.body[0]):
            self._add_issue(
                issue_type="missing_docstring",
                line_number=node.lineno,
                severity="low",
                message="Function is missing a docstring.",
            )

    def _get_body_line_count(self, body: Sequence[ast.stmt]) -> int:
        if not body:
            return 0
        start_line = body[0].lineno
        end_line = body[-1].end_lineno or body[-1].lineno
        return max(0, end_line - start_line + 1)

    def _is_docstring(self, node: Optional[ast.AST]) -> bool:
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)

    def _add_issue(self, issue_type: str, line_number: int, severity: str, message: str) -> None:
        self.issues.append(
            {
                "issue_type": issue_type,
                "line_number": line_number,
                "severity": severity,
                "message": message,
            }
        )


def analyze_code(code: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = AnalyzerVisitor()
    visitor.visit(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self_check = _check_nesting(node)
            visitor.issues.extend(self_check)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults:
                if _is_mutable_default(default):
                    visitor.issues.append(
                        {
                            "issue_type": "mutable_default_argument",
                            "line_number": default.lineno,
                            "severity": "medium",
                            "message": "Default argument uses a mutable value.",
                        }
                    )

    for imported_name in sorted(visitor._imported_names):
        if imported_name not in visitor._used_names:
            visitor.issues.append(
                {
                    "issue_type": "unused_import",
                    "line_number": 1,
                    "severity": "low",
                    "message": f"Imported name '{imported_name}' is never used.",
                }
            )

    issues = sorted(visitor.issues, key=lambda item: (item["line_number"], item["issue_type"]))
    return issues


def _check_nesting(node: ast.AST, depth: int = 0) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.Try, ast.ExceptHandler)):
        if depth >= MAX_NESTING_DEPTH:
            issues.append(
                {
                    "issue_type": "deep_nesting",
                    "line_number": node.lineno,
                    "severity": "medium",
                    "message": "Control structure nesting exceeds the recommended depth.",
                }
            )
        depth += 1

    for child in ast.iter_child_nodes(node):
        issues.extend(_check_nesting(child, depth))
    return issues


def _is_mutable_default(node: ast.AST) -> bool:
    return isinstance(node, (ast.List, ast.Dict, ast.Set))
