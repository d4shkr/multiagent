"""Utilities for optional RAG retrieval inside specialized agents."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RAGEvidence:
    """Minimal retrieval payload used by agents."""

    query: str
    hits: list[dict[str, Any]]
    available: bool
    backend: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OptionalRAGClient:
    """Best-effort wrapper around the project retriever.

    The project should remain runnable even when FAISS/Ollama/RAG storage are absent.
    """

    def __init__(self, storage_path: str | Path | None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.logger = logging.getLogger("OptionalRAGClient")
        self._retriever = None
        self._error: str | None = None
        self._available = False
        self._backend = "disabled"
        self._initialize()

    def _initialize(self) -> None:
        if self.storage_path is None:
            self._error = "RAG storage path is not configured"
            return
        if not self.storage_path.exists():
            self._error = f"RAG storage not found: {self.storage_path}"
            return
        try:
            from langflow_components.rag.retriever_backend import HybridRetriever

            self._retriever = HybridRetriever.from_storage(str(self.storage_path))
            self._available = True
            self._backend = "hybrid"
        except Exception as exc:
            self._error = str(exc)
            db_path = self.storage_path / "chunks.db"
            if db_path.exists():
                self._retriever = "sqlite_lexical"
                self._available = True
                self._backend = "sqlite_lexical"
            else:
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def backend(self) -> str:
        return self._backend

    def _fallback_sqlite_search(self, query: str, k: int, cell_type_filter: str | None) -> list[dict[str, Any]]:
        assert self.storage_path is not None
        db_path = self.storage_path / "chunks.db"
        tokens = [t.lower() for t in query.split() if t.strip()]
        if not tokens:
            return []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM chunks").fetchall()]
        conn.close()
        scored = []
        for row in rows:
            if cell_type_filter and row.get("cell_type") != cell_type_filter:
                continue
            text = (row.get("code") or "").lower()
            score = sum(text.count(token) for token in tokens)
            if score > 0:
                scored.append({
                    "chunk_id": row.get("id"),
                    "source": row.get("source"),
                    "cell_index": row.get("cell_index"),
                    "cell_type": row.get("cell_type"),
                    "heading": row.get("heading"),
                    "score": float(score),
                    "preview": (row.get("code") or "")[:300],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def retrieve(self, query: str, k: int = 3, cell_type_filter: str | None = None) -> RAGEvidence:
        if not self._available or self._retriever is None:
            return RAGEvidence(query=query, hits=[], available=False, backend="disabled", error=self._error)
        if self._backend == "sqlite_lexical":
            try:
                return RAGEvidence(
                    query=query,
                    hits=self._fallback_sqlite_search(query=query, k=k, cell_type_filter=cell_type_filter),
                    available=True,
                    backend=self._backend,
                    error=self._error,
                )
            except Exception as exc:
                return RAGEvidence(query=query, hits=[], available=False, backend=self._backend, error=str(exc))
        try:
            results = self._retriever.retrieve(query=query, k=k, cell_type_filter=cell_type_filter)
            hits = []
            for item in results:
                payload = item.to_dict()
                payload["preview"] = payload.get("code", "")[:300]
                payload.pop("code", None)
                hits.append(payload)
            return RAGEvidence(query=query, hits=hits, available=True, backend=self._backend)
        except Exception as exc:
            return RAGEvidence(query=query, hits=[], available=False, backend=self._backend, error=str(exc))

    def close(self) -> None:
        if hasattr(self._retriever, "close"):
            try:
                self._retriever.close()
            except Exception:
                self.logger.exception("Failed to close retriever cleanly")
