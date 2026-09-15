from __future__ import annotations

import re

from .database import VectorStore
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .types import AnswerResult, CompletionProvider, EmbeddingProvider, RetrievedChunk

NO_INFORMATION_MESSAGE = (
    "The knowledge base does not contain enough information to answer this reliably."
)
NO_INFORMATION_MESSAGE_TR = "Bilgi tabanında bu soruyu güvenilir biçimde yanıtlamak için yeterli bilgi bulunamadı."


def _fallback_message(question: str) -> str:
    lowered = f" {question.lower()} "
    turkish_markers = ("ı", "ğ", "ü", "ş", "ö", "ç", " nedir ", " nasıl ", " kaç ")
    return (
        NO_INFORMATION_MESSAGE_TR
        if any(marker in lowered for marker in turkish_markers)
        else NO_INFORMATION_MESSAGE
    )


def _compact_generated_answer(answer: str) -> str:
    """Keep small local models from flooding the UI with repeated output."""
    cleaned = " ".join(answer.split())
    if not cleaned:
        return cleaned

    # Stop when an eight-word sequence starts repeating. Tiny greedy models
    # occasionally enter a loop even with a short output-token limit.
    words = cleaned.split()
    seen: dict[tuple[str, ...], int] = {}
    cut_at = len(words)
    for index in range(max(0, len(words) - 7)):
        window = tuple(word.casefold() for word in words[index : index + 8])
        if window in seen:
            cut_at = index
            break
        seen[window] = index
    cleaned = " ".join(words[:cut_at]).strip()

    # The UI is designed for short answers. Preserve at most two completed
    # sentences and apply a final character guard for malformed generations.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2]).strip()
    if len(cleaned) > 500:
        cleaned = cleaned[:500].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return cleaned


class RagService:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        generator: CompletionProvider,
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Soru boş bırakılamaz")
        stats = self.store.stats()
        if stats["chunks"] == 0:
            raise RuntimeError("Bilgi tabanı boş. Önce indeksleme komutunu çalıştırın.")
        query_vector = self.embedder.embed_texts([cleaned])[0]
        return self.store.search(query_vector, top_k or self.top_k)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AnswerResult:
        matches = self.retrieve(question, top_k=top_k)
        threshold = self.min_score if min_score is None else min_score
        if not matches or matches[0].score < threshold:
            return AnswerResult(
                answer=_fallback_message(question),
                sources=matches,
                grounded=False,
            )

        contexts = [
            f"[S{index}] Kaynak: {chunk.citation_label}\n{chunk.content}"
            for index, chunk in enumerate(matches, start=1)
        ]
        prompt = build_user_prompt(question.strip(), contexts)
        answer = _compact_generated_answer(
            self.generator.complete(SYSTEM_PROMPT, prompt)
        )
        return AnswerResult(answer=answer, sources=matches, grounded=True)
