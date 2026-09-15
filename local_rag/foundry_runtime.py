from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

ProgressCallback = Callable[[str, float], None]


class FoundryLocalAI:
    """Thin adapter around Foundry Local 2.x's typed, in-process Session API."""

    def __init__(
        self,
        chat_model: str,
        embedding_model: str,
        max_output_tokens: int = 450,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.chat_model_name = chat_model
        self._embedding_model_name = embedding_model
        self.max_output_tokens = max_output_tokens
        self.progress_callback = progress_callback
        self._manager: Any | None = None
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    def _notify(self, stage: str, percent: float) -> None:
        if self.progress_callback:
            self.progress_callback(stage, percent)

    def _get_manager(self) -> Any:
        if self._manager is not None:
            return self._manager

        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager
        except ImportError as exc:
            raise RuntimeError(
                "Foundry Local SDK kurulu değil. Şu komutu çalıştırın: "
                "pip install -r requirements.txt"
            ) from exc

        config = Configuration(app_name="local-rag-assistant")
        FoundryLocalManager.initialize(config)
        self._manager = FoundryLocalManager.instance
        return self._manager

    def _get_loaded_model(self, alias: str) -> Any:
        existing = self._models.get(alias)
        if existing is not None and existing.is_loaded:
            return existing

        manager = self._get_manager()
        model = existing or manager.catalog.get_model(alias)
        if not model.is_cached:
            self._notify(f"{alias} indiriliyor", 0.0)
            model.download(lambda pct: self._notify(f"{alias} indiriliyor", float(pct)))
        if not model.is_loaded:
            self._notify(f"{alias} yükleniyor", 100.0)
            model.load()
        self._models[alias] = model
        return model

    @staticmethod
    def _tensor_to_vector(tensor: Any) -> list[float]:
        """Convert Foundry's typed TensorItem bytes to a Python float list."""
        data = tensor.data
        if isinstance(data, list):
            return [float(value) for value in data]

        try:
            import numpy as np
            from foundry_local_sdk import TensorDataType
        except ImportError as exc:
            raise RuntimeError("Tensor conversion requires numpy") from exc

        dtype_map = {
            TensorDataType.FLOAT: np.dtype("<f4"),
            TensorDataType.DOUBLE: np.dtype("<f8"),
            TensorDataType.FLOAT16: np.dtype("<f2"),
        }
        dtype = dtype_map.get(tensor.data_type)
        if dtype is None:
            raise RuntimeError(f"Unsupported embedding tensor type: {tensor.data_type}")

        vector = np.frombuffer(data, dtype=dtype)
        shape = getattr(tensor, "shape", getattr(tensor, "dimensions", None))
        if shape:
            expected = math.prod(int(value) for value in shape)
            if vector.size != expected:
                raise RuntimeError(
                    f"Embedding tensor size mismatch: expected {expected}, got {vector.size}"
                )
        return vector.astype(float).tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty text")

        with self._lock:
            from foundry_local_sdk import (
                EmbeddingsSession,
                Request,
                TensorItem,
                TextItem,
            )

            model = self._get_loaded_model(self.embedding_model_name)
            with EmbeddingsSession(model) as session, Request() as request:
                for text in cleaned:
                    request.add_item(TextItem(text))
                with session.process_request(request) as response:
                    vectors = [
                        self._tensor_to_vector(item)
                        for item in response
                        if isinstance(item, TensorItem)
                    ]

        if len(vectors) != len(cleaned):
            raise RuntimeError(
                f"Foundry returned {len(vectors)} embeddings for {len(cleaned)} inputs"
            )
        return vectors

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not user_prompt.strip():
            raise ValueError("Prompt cannot be empty")

        with self._lock:
            from foundry_local_sdk import (
                ChatSession,
                MessageItem,
                Request,
                RequestOptions,
                SearchOptions,
                TextItem,
                TextItemType,
            )

            model = self._get_loaded_model(self.chat_model_name)
            with ChatSession(model) as session:
                session.set_options(
                    RequestOptions(
                        search=SearchOptions(
                            temperature=0.2,
                            top_p=0.9,
                            top_k=30,
                            max_output_tokens=self.max_output_tokens,
                            frequency_penalty=0.7,
                            presence_penalty=0.2,
                            seed=42,
                            early_stopping=True,
                            do_sample=True,
                        )
                    )
                )
                # Keep message wrappers alive until the request finishes because
                # MessageItem borrows the handles of its child TextItems.
                system_message = MessageItem.system(system_prompt)
                user_message = MessageItem.user(user_prompt)
                with Request() as request:
                    request.add_item(system_message)
                    request.add_item(user_message)
                    with session.process_request(request) as response:
                        regular_text: list[str] = []
                        for item in response:
                            # Depending on the Foundry Local runtime/model, the
                            # assistant output can be returned either as a
                            # top-level TextItem or inside a MessageItem.
                            if (
                                isinstance(item, TextItem)
                                and item.type == TextItemType.DEFAULT
                            ):
                                regular_text.append(item.text)
                            elif isinstance(item, MessageItem):
                                regular_text.extend(
                                    part.text
                                    for part in item.parts
                                    if isinstance(part, TextItem)
                                    and part.type == TextItemType.DEFAULT
                                )

        answer = "".join(regular_text).strip()
        if not answer:
            raise RuntimeError(
                "Foundry Local boş bir yanıt döndürdü. Modeli yeniden deneyin "
                "veya uygulamayı yeniden başlatın."
            )
        return answer

    def close(self) -> None:
        with self._lock:
            for model in self._models.values():
                try:
                    if model.is_loaded:
                        model.unload()
                except Exception:
                    pass
            self._models.clear()
