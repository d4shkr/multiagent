"""Backend implementation for Hybrid Retriever."""

import json
import pickle
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from langflow_components.rag.utils import (
    RAGConfig,
    RetrievedChunk,
    rrf_rerank,
)


def tokenize_code(text: str) -> list[str]:
    """Tokenize code for BM25 search."""
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


class OllamaEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = config.embedding_model
        self.url = config.ollama_url
        self.dimension = config.embedding_dim
        self._client = None

    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.url)
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
        return self._client

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            embedding = response.get("embedding", [])
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")


class BM25Index:
    """BM25 lexical search index for code."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus: list[str] = []

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Search for relevant chunks."""
        if self._bm25 is None:
            return []

        query_tokens = tokenize_code(query)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))

        return results

    def load(self, path: Path) -> None:
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._chunk_ids = data["chunk_ids"]
        self._corpus = data["corpus"]

        if self._corpus:
            tokenized = [tokenize_code(code) for code in self._corpus]
            self._bm25 = BM25Okapi(tokenized)


class FAISSVectorStore:
    """FAISS-based vector store for fast ANN search."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.dimension = config.embedding_dim
        self._index = None
        self._id_map: list[str] = []

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search for k nearest neighbors."""
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        k = min(k, len(self._id_map))
        if k == 0:
            return []

        distances, indices = self._index.search(query, k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self._id_map):
                results.append((self._id_map[i], float(dist)))

        return results

    def load(self, path: Path) -> None:
        """Load index and ID map from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        import faiss
        self._index = faiss.read_index(str(path))

        id_map_path = path.with_suffix(".ids.json")
        if id_map_path.exists():
            with open(id_map_path, "r") as f:
                self._id_map = json.load(f)


class ChunkStore:
    """SQLite-based chunk storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self):
        """Lazy connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get(self, chunk_id: str) -> dict | None:
        """Get chunk by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search."""

    def __init__(
        self,
        config: RAGConfig,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
        chunk_store: ChunkStore,
        embedder: OllamaEmbedder,
    ):
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store
        self.embedder = embedder

    @classmethod
    def from_storage(cls, storage_path: str, config: RAGConfig | None = None) -> "HybridRetriever":
        """Load retriever from storage."""
        config = config or RAGConfig()
        storage_path = Path(storage_path)

        # Load vector store
        vector_store = FAISSVectorStore(config)
        faiss_path = storage_path / "faiss_index.bin"
        if faiss_path.exists():
            vector_store.load(faiss_path)

        # Load BM25 index
        bm25_index = BM25Index()
        bm25_path = storage_path / "bm25_index.pkl"
        if bm25_path.exists():
            bm25_index.load(bm25_path)

        # Load chunk store
        chunk_store = ChunkStore(storage_path / "chunks.db")

        # Create embedder
        embedder = OllamaEmbedder(config)

        return cls(
            config=config,
            vector_store=vector_store,
            bm25_index=bm25_index,
            chunk_store=chunk_store,
            embedder=embedder,
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        cell_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid search with RRF reranking."""
        start_time = time.perf_counter()
        k = k or self.config.top_k

        # Get more results for filtering
        search_k = k * 3 if (source_filter or cell_type_filter) else k * 2

        # BM25 search
        bm25_results = self.bm25_index.search(query, k=search_k)

        # Semantic search
        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(query_embedding, k=search_k)

        # RRF rerank
        combined = rrf_rerank(bm25_results, semantic_results)

        # Get top-k with filtering
        results: list[RetrievedChunk] = []

        for chunk_id, rrf_score in combined:
            if rrf_score <= 0:
                continue

            chunk = self.chunk_store.get(chunk_id)
            if chunk is None:
                continue

            # Apply filters
            if source_filter and source_filter not in chunk.get("source", ""):
                continue
            if cell_type_filter and chunk.get("cell_type") != cell_type_filter:
                continue

            # Get individual scores
            bm25_score = next((s for cid, s in bm25_results if cid == chunk_id), 0.0)
            semantic_score = next((s for cid, s in semantic_results if cid == chunk_id), 0.0)

            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                code=chunk.get("code", ""),
                source=chunk.get("source", ""),
                cell_index=chunk.get("cell_index", 0),
                score=rrf_score,
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                heading=chunk.get("heading"),
            ))

            if len(results) >= k:
                break

        return results

    def semantic_search(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        cell_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Pure semantic search."""
        k = k or self.config.top_k
        search_k = k * 2 if (source_filter or cell_type_filter) else k

        query_embedding = self.embedder.embed(query)
        semantic_results = self.vector_store.search(query_embedding, k=search_k)

        results: list[RetrievedChunk] = []
        for chunk_id, score in semantic_results:
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            if source_filter and source_filter not in chunk.get("source", ""):
                continue
            if cell_type_filter and chunk.get("cell_type") != cell_type_filter:
                continue

            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                code=chunk.get("code", ""),
                source=chunk.get("source", ""),
                cell_index=chunk.get("cell_index", 0),
                score=score,
                bm25_score=0.0,
                semantic_score=score,
                heading=chunk.get("heading"),
            ))

            if len(results) >= k:
                break

        return results

    def bm25_search(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        cell_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Pure BM25 search."""
        k = k or self.config.top_k
        search_k = k * 2 if (source_filter or cell_type_filter) else k

        bm25_results = self.bm25_index.search(query, k=search_k)

        results: list[RetrievedChunk] = []
        for chunk_id, score in bm25_results:
            chunk = self.chunk_store.get(chunk_id)
            if not chunk:
                continue

            if source_filter and source_filter not in chunk.get("source", ""):
                continue
            if cell_type_filter and chunk.get("cell_type") != cell_type_filter:
                continue

            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                code=chunk.get("code", ""),
                source=chunk.get("source", ""),
                cell_index=chunk.get("cell_index", 0),
                score=score,
                bm25_score=score,
                semantic_score=0.0,
                heading=chunk.get("heading"),
            ))

            if len(results) >= k:
                break

        return results

    def format_for_prompt(
        self,
        results: list[RetrievedChunk],
        max_chunks: int = 5,
        max_code_len: int = 500,
    ) -> str:
        """Format retrieved chunks for insertion into prompt."""
        if not results:
            return "No relevant code examples found."

        lines = ["=== RELEVANT CODE EXAMPLES ===\n"]

        for i, r in enumerate(results[:max_chunks], 1):
            code = r.code
            if len(code) > max_code_len:
                code = code[:max_code_len] + "\n... (truncated)"

            header = f"[{r.source}"
            if r.heading:
                header += f" - {r.heading}"
            header += f"] (score: {r.score:.3f})"

            lines.append(f"### Example {i}: {header}")
            lines.append("```python")
            lines.append(code)
            lines.append("```\n")

        return "\n".join(lines)

    def close(self) -> None:
        """Close connections."""
        self.chunk_store.close()
