import subprocess


def get_git_commit_sha() -> str:
    """Returns the short git commit SHA of the current HEAD.

    Returns an empty string if the working directory is not a git repository
    or if git is not installed, so callers never need to handle exceptions.

    Returns
    -------
    str
        The short (7-character) git commit SHA, or an empty string.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""
