import os
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utilities.git_utils import get_git_commit_sha


class TestGetGitCommitSha(unittest.TestCase):
    """Unit tests for get_git_commit_sha()."""

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_returns_string(self):
        result = get_git_commit_sha()
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # Happy-path tests (mocked subprocess)
    # ------------------------------------------------------------------

    def test_returns_short_sha_in_git_repo(self):
        """When git returns a valid SHA, the function returns it stripped."""
        mock_sha = b"a3f9c12\n"
        with patch(
            "utilities.git_utils.subprocess.check_output", return_value=mock_sha
        ):
            result = get_git_commit_sha()
        self.assertEqual(result, "a3f9c12")

    def test_sha_has_no_leading_trailing_whitespace(self):
        """Output must be stripped of newlines and spaces."""
        mock_sha = b"  abc1234  \n"
        with patch(
            "utilities.git_utils.subprocess.check_output", return_value=mock_sha
        ):
            result = get_git_commit_sha()
        self.assertEqual(result, result.strip())

    # ------------------------------------------------------------------
    # Error-path tests
    # ------------------------------------------------------------------

    def test_returns_empty_string_when_not_git_repo(self):
        """CalledProcessError (not a git repo) must return empty string."""
        with patch(
            "utilities.git_utils.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            result = get_git_commit_sha()
        self.assertEqual(result, "")

    def test_returns_empty_string_on_file_not_found(self):
        """FileNotFoundError (git not installed) must return empty string."""
        with patch(
            "utilities.git_utils.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            result = get_git_commit_sha()
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
