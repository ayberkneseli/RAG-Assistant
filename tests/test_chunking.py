from local_rag.chunking import chunk_sections
from local_rag.types import LoadedSection


def test_chunking_creates_overlapping_windows():
    section = LoadedSection("one two three four five six seven eight nine", "page 2")

    chunks = chunk_sections([section], max_words=5, overlap_words=2)

    assert [chunk.content for chunk in chunks] == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine",
    ]
    assert all(chunk.locator == "page 2" for chunk in chunks)


def test_chunking_ignores_empty_sections():
    assert chunk_sections([LoadedSection("  \n\t  ")], 20, 2) == []
