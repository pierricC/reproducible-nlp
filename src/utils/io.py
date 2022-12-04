"""Class and functions dealing with I/O logic."""
import os


def ensure_directory(path: str):
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
