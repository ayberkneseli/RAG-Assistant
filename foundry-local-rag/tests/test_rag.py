from local_rag.database import VectorStore
from local_rag.rag import NO_INFORMATION_MESSAGE, RagService, _compact_generated_answer


class FakeAI:
    embedding_model_name = "fake"

    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed_texts(self, texts):
        return [list(self.vector) for _ in texts]

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "The answer is grounded. [S1]"


def make_store(tmp_path):
    store = VectorStore(tmp_path / "rag.db")
    store.initialize()
    store.replace_document(
        "policy.md",
        "policy.md",
        "checksum",
        [("The deadline is Friday at 18:00.", None, [1.0, 0.0])],
    )
    return store


def test_answer_adds_retrieved_context_and_citation_label(tmp_path):
    ai = FakeAI([1.0, 0.0])
    service = RagService(make_store(tmp_path), ai, ai, min_score=0.5)

    result = service.answer("When is the deadline?")

    assert result.grounded is True
    assert result.answer.endswith("[S1]")
    assert "[S1] Kaynak: policy.md" in ai.calls[0][1]
    assert "Friday at 18:00" in ai.calls[0][1]


def test_low_similarity_returns_fallback_without_calling_model(tmp_path):
    ai = FakeAI([0.0, 1.0])
    service = RagService(make_store(tmp_path), ai, ai, min_score=0.5)

    result = service.answer("What is the cafeteria menu?")

    assert result.grounded is False
    assert result.answer == NO_INFORMATION_MESSAGE
    assert ai.calls == []


def test_compact_answer_stops_repeated_word_sequences():
    phrase = "öğrenciler dizüstü bilgisayar şarj cihazı ve kulaklık getirmelidir"
    answer = _compact_generated_answer(f"{phrase} {phrase} {phrase}")

    assert answer == phrase


def test_compact_answer_keeps_at_most_two_sentences():
    answer = _compact_generated_answer("Birinci. İkinci. Üçüncü.")

    assert answer == "Birinci. İkinci."
