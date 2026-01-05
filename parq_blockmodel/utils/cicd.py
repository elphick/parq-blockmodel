import os

def is_github_runner() -> bool:
    """Check if the code is running inside a GitHub Actions runner.

    Returns:
        bool: True if running in GitHub Actions, False otherwise.
    """
    return os.getenv("GITHUB_ACTIONS") == "true"
