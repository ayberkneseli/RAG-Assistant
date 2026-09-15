"""Local, source-grounded RAG application powered by Foundry Local."""

from .config import Settings
from .database import VectorStore
from .foundry_runtime import FoundryLocalAI
from .ingestion import ingest_documents
from .rag import RagService

__all__ = [
    "FoundryLocalAI",
    "RagService",
    "Settings",
    "VectorStore",
    "ingest_documents",
]
