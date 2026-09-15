from dataclasses import replace

from local_rag.config import Settings
from local_rag.database import VectorStore
from local_rag.ingestion import ingest_documents


class FakeEmbedder:
    embedding_model_name = "fake-embedding-v1"

    def embed_texts(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_ingestion_skips_unchanged_documents(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    document = docs / "notes.md"
    document.write_text("A short local knowledge-base document.", encoding="utf-8")
    settings = replace(
        Settings(),
        documents_dir=docs,
        database_path=tmp_path / "rag.db",
        chunk_words=20,
        chunk_overlap_words=2,
    )
    store = VectorStore(settings.database_path)
    embedder = FakeEmbedder()

    first = ingest_documents(store, embedder, settings)
    second = ingest_documents(store, embedder, settings)

    assert first.files_indexed == 1
    assert first.chunks_written == 1
    assert second.files_skipped == 1
    assert second.files_indexed == 0
    assert store.stats()["chunks"] == 1


def test_ingestion_reindexes_changed_document(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    document = docs / "notes.txt"
    document.write_text("First version", encoding="utf-8")
    settings = replace(
        Settings(),
        documents_dir=docs,
        database_path=tmp_path / "rag.db",
        chunk_words=20,
        chunk_overlap_words=2,
    )
    store = VectorStore(settings.database_path)
    embedder = FakeEmbedder()
    ingest_documents(store, embedder, settings)

    document.write_text("Second version with changed content", encoding="utf-8")
    report = ingest_documents(store, embedder, settings)

    assert report.files_indexed == 1
    assert report.files_skipped == 0
    assert store.stats()["documents"] == 1
