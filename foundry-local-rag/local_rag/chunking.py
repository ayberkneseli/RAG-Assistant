from __future__ import annotations

import re
from collections.abc import Sequence

from .types import LoadedSection, PreparedChunk

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def chunk_sections(
    sections: Sequence[LoadedSection],
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[PreparedChunk]:
    """Split sections into deterministic word windows with a small overlap."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    step = max_words - overlap_words
    chunks: list[PreparedChunk] = []

    for section in sections:
        cleaned = normalize_text(section.text)
        if not cleaned:
            continue
        words = cleaned.split(" ")
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if not window:
                break
            chunks.append(
                PreparedChunk(content=" ".join(window), locator=section.locator)
            )
            if start + max_words >= len(words):
                break

    return chunks
