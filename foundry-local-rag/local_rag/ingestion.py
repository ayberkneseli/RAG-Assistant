from __future__ import annotations

from collections.abc import Callable

from .chunking import chunk_sections
from .config import Settings
from .database import VectorStore
from .loaders import discover_documents, load_document
from .types import EmbeddingProvider, IngestionReport

StatusCallback = Callable[[str], None]


def _batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest_documents(
    store: VectorStore,
    embedder: EmbeddingProvider,
    settings: Settings,
    rebuild: bool = False,
    status_callback: StatusCallback | None = None,
) -> IngestionReport:
    settings.validate()
    store.initialize()

    if rebuild:
        store.clear_content()
    else:
        current_model = store.get_metadata("embedding_model")
        if current_model and current_model != embedder.embedding_model_name:
            raise RuntimeError(
                "Veritabanı farklı bir embedding modeliyle oluşturulmuş. "
                "İndeksleme komutunu --rebuild ile yeniden çalıştırın."
            )

    paths = discover_documents(settings.documents_dir)
    if not paths:
        raise RuntimeError(
            f"{settings.documents_dir} içinde .md, .txt veya .pdf belgesi bulunamadı"
        )

    indexed = 0
    skipped = 0
    chunks_written = 0
    relative_paths: list[str] = []

    for path in paths:
        document = load_document(path, settings.documents_dir)
        relative_paths.append(document.relative_path)
        if (
            not rebuild
            and store.document_checksum(document.relative_path) == document.checksum
        ):
            skipped += 1
            if status_callback:
                status_callback(f"Değişmeyen dosya atlandı: {document.relative_path}")
            continue

        chunks = chunk_sections(
            document.sections,
            max_words=settings.chunk_words,
            overlap_words=settings.chunk_overlap_words,
        )
        if not chunks:
            raise RuntimeError(
                f"{document.relative_path} içinde okunabilir metin bulunamadı"
            )

        if status_callback:
            status_callback(
                f"{len(chunks)} metin parçası işleniyor: {document.relative_path}"
            )

        vectors: list[list[float]] = []
        contents = [chunk.content for chunk in chunks]
        for batch in _batched(contents, settings.embedding_batch_size):
            vectors.extend(embedder.embed_texts(batch))

        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding sayısı metin parçası sayısıyla eşleşmiyor")

        store.replace_document(
            source_path=document.relative_path,
            source_name=path.name,
            checksum=document.checksum,
            chunks=[
                (chunk.content, chunk.locator, vector)
                for chunk, vector in zip(chunks, vectors)
            ],
        )
        indexed += 1
        chunks_written += len(chunks)

    removed = store.remove_documents_not_in(relative_paths)
    store.set_metadata("embedding_model", embedder.embedding_model_name)

    return IngestionReport(
        files_seen=len(paths),
        files_indexed=indexed,
        files_skipped=skipped,
        files_removed=removed,
        chunks_written=chunks_written,
    )
