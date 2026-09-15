from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application settings with beginner-friendly, laptop-sized defaults."""

    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    database_path: Path = PROJECT_ROOT / "data" / "rag.db"
    chat_model: str = "qwen2.5-0.5b"
    embedding_model: str = "qwen3-embedding-0.6b"
    chunk_words: int = 180
    chunk_overlap_words: int = 30
    embedding_batch_size: int = 8
    top_k: int = 1
    min_score: float = 0.35
    max_output_tokens: int = 96

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            documents_dir=Path(
                os.getenv("RAG_DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "documents"))
            ).expanduser(),
            database_path=Path(
                os.getenv("RAG_DATABASE_PATH", str(PROJECT_ROOT / "data" / "rag.db"))
            ).expanduser(),
            chat_model=os.getenv("RAG_CHAT_MODEL", "qwen2.5-0.5b"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
            chunk_words=int(os.getenv("RAG_CHUNK_WORDS", "180")),
            chunk_overlap_words=int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "30")),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "8")),
            top_k=int(os.getenv("RAG_TOP_K", "1")),
            min_score=float(os.getenv("RAG_MIN_SCORE", "0.35")),
            max_output_tokens=int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "96")),
        )

    def validate(self) -> None:
        if self.chunk_words < 20:
            raise ValueError("chunk_words must be at least 20")
        if not 0 <= self.chunk_overlap_words < self.chunk_words:
            raise ValueError(
                "chunk_overlap_words must be between 0 and chunk_words - 1"
            )
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
