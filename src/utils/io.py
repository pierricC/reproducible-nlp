"""Class and functions dealing with I/O logic."""
import json
import os
from typing import Any, Dict


def ensure_file_directory(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_to_json(filepath: str, object_to_save: Dict[str, Any]) -> None:
    """Dump an object to a json path."""
    ensure_file_directory(filepath)
    with open(filepath, "w") as f:
        json.dump(object_to_save, f)
