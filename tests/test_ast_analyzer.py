import unittest

from app.core.ast_analyzer import analyze_code


class AstAnalyzerTests(unittest.TestCase):
    def test_detects_long_function_issue(self):
        bad_code = "def sample():\n"
        for i in range(25):
            bad_code += f"    value_{i} = {i}\n"
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "long_function" for issue in bad_issues))

        good_code = "def sample():\n    value = 1\n    return value\n"
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "long_function" for issue in good_issues))

    def test_detects_deep_nesting_issue(self):
        bad_code = """
def sample(value):
    if value > 0:
        if value > 1:
            if value > 2:
                if value > 3:
                    return True
    return False
"""
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "deep_nesting" for issue in bad_issues))

        good_code = """
def sample(value):
    if value > 0:
        return True
    return False
"""
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "deep_nesting" for issue in good_issues))

    def test_detects_bare_except_issue(self):
        bad_code = """
try:
    run()
except:
    pass
"""
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "bare_except" for issue in bad_issues))

        good_code = """
try:
    run()
except ValueError:
    pass
"""
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "bare_except" for issue in good_issues))

    def test_detects_mutable_default_argument_issue(self):
        bad_code = "def sample(items=[]):\n    return items\n"
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "mutable_default_argument" for issue in bad_issues))

        good_code = "def sample(items=None):\n    if items is None:\n        items = []\n    return items\n"
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "mutable_default_argument" for issue in good_issues))

    def test_detects_missing_docstring_issue(self):
        bad_code = "def sample():\n    return 1\n"
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "missing_docstring" for issue in bad_issues))

        good_code = """def sample():
    \"\"\"Return a value.\"\"\"
    return 1
"""
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "missing_docstring" for issue in good_issues))

    def test_detects_unused_import_issue(self):
        bad_code = "import os\n\ndef sample():\n    return 1\n"
        bad_issues = analyze_code(bad_code)
        self.assertTrue(any(issue["issue_type"] == "unused_import" for issue in bad_issues))

        good_code = "import os\n\ndef sample():\n    return os.getcwd()\n"
        good_issues = analyze_code(good_code)
        self.assertFalse(any(issue["issue_type"] == "unused_import" for issue in good_issues))


if __name__ == "__main__":
    unittest.main()
