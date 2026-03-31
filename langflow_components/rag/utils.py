"""Shared RAG utilities for Langflow components."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _get_ollama_url() -> str:
    """Get Ollama URL from environment or default."""
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Embeddings
    embedding_model: str = "bge-m3"
    ollama_url: str = None  # Will be set from env
    embedding_dim: int = 1024  # BGE-M3 dimension
    batch_size: int = 32

    # Search
    top_k: int = 5
    alpha: float = 0.5
    threshold: float = 0.3

    # Storage
    storage_path: str = "./rag_storage"

    def __post_init__(self):
        """Set ollama_url from environment if not provided."""
        if self.ollama_url is None:
            self.ollama_url = _get_ollama_url()


@dataclass
class RetrievedChunk:
    """Retrieved code chunk with scores."""

    chunk_id: str
    code: str
    source: str
    cell_index: int
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "code": self.code,
            "source": self.source,
            "cell_index": self.cell_index,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "heading": self.heading,
        }


def tokenize_code(text: str) -> list[str]:
    """
    Tokenize code for BM25 search.

    Splits on:
    - Underscores: snake_case -> snake, case
    - CamelCase: CamelCase -> Camel, Case
    - Operators and punctuation
    - Numbers
    """
    tokens = re.split(r'[^a-zA-Z0-9]+', text)

    result = []
    for token in tokens:
        if not token:
            continue

        camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)

        if camel_split:
            result.extend(camel_split)
        else:
            result.append(token)

    return [t.lower() for t in result if t]


def rrf_rerank(
    bm25_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion for combining multiple ranking lists.

    RRF score = sum(1 / (k + rank)) for each ranking list

    Args:
        bm25_results: List of (chunk_id, score) from BM25
        semantic_results: List of (chunk_id, score) from semantic search
        k: RRF parameter (default 60)

    Returns:
        Combined and sorted list of (chunk_id, rrf_score)
    """
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] += 1 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(semantic_results):
        scores[chunk_id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
