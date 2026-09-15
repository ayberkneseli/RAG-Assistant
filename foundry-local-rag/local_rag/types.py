from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LoadedSection:
    text: str
    locator: str | None = None


@dataclass(frozen=True)
class PreparedChunk:
    content: str
    locator: str | None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source_name: str
    source_path: str
    locator: str | None
    content: str
    score: float

    @property
    def citation_label(self) -> str:
        return (
            f"{self.source_name} ({self.locator})" if self.locator else self.source_name
        )


@dataclass(frozen=True)
class IngestionReport:
    files_seen: int
    files_indexed: int
    files_skipped: int
    files_removed: int
    chunks_written: int


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: Sequence[RetrievedChunk] = field(default_factory=tuple)
    grounded: bool = True


class EmbeddingProvider(Protocol):
    @property
    def embedding_model_name(self) -> str: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class CompletionProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class LocalAIProvider(EmbeddingProvider, CompletionProvider, Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class DocumentFile:
    path: Path
    relative_path: str
    checksum: str
    sections: Sequence[LoadedSection]
