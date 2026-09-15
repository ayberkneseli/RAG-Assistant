from local_rag.database import VectorStore, cosine_similarity


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_vector_store_returns_best_chunk(tmp_path):
    store = VectorStore(tmp_path / "rag.db")
    store.initialize()
    store.replace_document(
        "alpha.md",
        "alpha.md",
        "checksum-a",
        [
            ("Python is a programming language.", None, [1.0, 0.0]),
            ("SQLite is a local database.", None, [0.0, 1.0]),
        ],
    )

    results = store.search([0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0].content.startswith("Python")
    assert results[0].source_path == "alpha.md"
