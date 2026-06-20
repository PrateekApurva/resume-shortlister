"""
Loads candidates from either:
  - NDJSON (candidates.json, full 100K): one JSON object per line
  - JSON array (sample_candidates.json): standard [...] array

Detects the format automatically by peeking at the first character.
"""

import json
from pathlib import Path
from typing import Generator


def load_candidates(path: str | Path, limit: int | None = None) -> list[dict]:
    """
    Load all candidates from a file into a list.
    Use limit to load only the first N candidates (useful for quick tests).
    """
    return list(_iter_candidates(path, limit=limit))


def iter_candidates(path: str | Path, limit: int | None = None) -> Generator[dict, None, None]:
    """
    Iterate candidates one at a time without loading all into memory.
    Use this for the full 100K to avoid RAM overflow.
    """
    yield from _iter_candidates(path, limit=limit)


def _iter_candidates(path: str | Path, limit: int | None = None) -> Generator[dict, None, None]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON array format (sample_candidates.json)
            candidates = json.load(f)
            for i, c in enumerate(candidates):
                if limit is not None and i >= limit:
                    break
                yield c
        else:
            # NDJSON format (candidates.json) — one object per line
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)
