from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .types import RetrievedChunk

SCHEMA_VERSION = "1"


def encode_embedding(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def decode_embedding(blob: bytes, dimensions: int) -> list[float]:
    expected_size = dimensions * 4
    if len(blob) != expected_size:
        raise ValueError(
            f"Corrupt embedding: expected {expected_size} bytes, received {len(blob)}"
        )
    return list(struct.unpack(f"<{dimensions}f", blob))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorStore:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    locator TEXT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def get_metadata(self, key: str) -> str | None:
        self.initialize()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.initialize()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                (key, value),
            )

    def clear_content(self) -> None:
        self.initialize()
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM metadata WHERE key != 'schema_version'")

    def document_checksum(self, source_path: str) -> str | None:
        self.initialize()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT checksum FROM documents WHERE source_path = ?", (source_path,)
            ).fetchone()
        return str(row["checksum"]) if row else None

    def replace_document(
        self,
        source_path: str,
        source_name: str,
        checksum: str,
        chunks: Sequence[tuple[str, str | None, Sequence[float]]],
    ) -> None:
        if not chunks:
            raise ValueError(f"No text chunks were produced for {source_path}")

        indexed_at = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO documents(source_path, source_name, checksum, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_name = excluded.source_name,
                    checksum = excluded.checksum,
                    indexed_at = excluded.indexed_at
                """,
                (source_path, source_name, checksum, indexed_at),
            )
            document_id = int(
                connection.execute(
                    "SELECT id FROM documents WHERE source_path = ?", (source_path,)
                ).fetchone()["id"]
            )
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?", (document_id,)
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, locator, content, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        index,
                        locator,
                        content,
                        encode_embedding(embedding),
                        len(embedding),
                    )
                    for index, (content, locator, embedding) in enumerate(chunks)
                ],
            )

    def remove_documents_not_in(self, source_paths: Iterable[str]) -> int:
        keep = set(source_paths)
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id, source_path FROM documents"
            ).fetchall()
            remove_ids = [
                int(row["id"]) for row in rows if row["source_path"] not in keep
            ]
            connection.executemany(
                "DELETE FROM documents WHERE id = ?",
                [(item_id,) for item_id in remove_ids],
            )
        return len(remove_ids)

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[RetrievedChunk]:
        self.initialize()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.locator,
                    c.content,
                    c.embedding,
                    c.embedding_dim,
                    d.source_name,
                    d.source_path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                """
            ).fetchall()

        scored = [
            RetrievedChunk(
                chunk_id=int(row["chunk_id"]),
                source_name=str(row["source_name"]),
                source_path=str(row["source_path"]),
                locator=str(row["locator"]) if row["locator"] else None,
                content=str(row["content"]),
                score=cosine_similarity(
                    query_embedding,
                    decode_embedding(row["embedding"], int(row["embedding_dim"])),
                ),
            )
            for row in rows
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def stats(self) -> dict[str, int | str | None]:
        self.initialize()
        with closing(self._connect()) as connection:
            documents = int(
                connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            chunks = int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
            latest = connection.execute(
                "SELECT MAX(indexed_at) FROM documents"
            ).fetchone()[0]
        return {"documents": documents, "chunks": chunks, "last_indexed_at": latest}
