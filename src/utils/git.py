"""Utils to interact with git."""
import os


def get_current_git_commit() -> str:
    """Get current commit."""
    return os.popen("git rev-parse HEAD").read().strip()


def get_current_git_branch() -> str:
    """Get current git branch user is in."""
    return os.popen("git branch --show-current").read().strip()
