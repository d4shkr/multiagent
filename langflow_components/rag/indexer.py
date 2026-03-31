"""RAG Indexer for Jupyter Notebooks."""

import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from langflow_components.rag.utils import RAGConfig, tokenize_code

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Index Jupyter notebooks for RAG search."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []

    def _get_embedder(self):
        """Get Ollama embedder."""
        import ollama
        return ollama.Client(host=self.config.ollama_url)

    def parse_notebook(self, notebook_path: Path) -> list[dict[str, Any]]:
        """Parse Jupyter notebook into chunks."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        chunks = []
        current_heading = None

        for cell_idx, cell in enumerate(nb.get("cells", [])):
            cell_type = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))

            if not source.strip():
                continue

            # Extract headings from markdown
            if cell_type == "markdown":
                for line in source.split("\n"):
                    if line.startswith("# "):
                        current_heading = line[2:].strip()
                    elif line.startswith("## "):
                        current_heading = line[3:].strip()

            # Create chunk
            chunk_id = f"{notebook_path.stem}_{cell_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "code": source,
                "source": str(notebook_path.name),
                "cell_index": cell_idx,
                "cell_type": cell_type,
                "heading": current_heading,
            })

        return chunks

    def generate_embeddings(self, chunks: list[dict], allow_fallback: bool = True) -> list[np.ndarray]:
        """Generate embeddings for chunks using Ollama."""
        try:
            embedder = self._get_embedder()
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.warning("Embeddings backend unavailable, using zero vectors fallback: %s", exc)
            return [np.zeros(self.config.embedding_dim, dtype=np.float32) for _ in chunks]
        embeddings = []

        logger.info("Generating embeddings for %s chunks...", len(chunks))

        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                logger.info("Progress: %s/%s", i + 1, len(chunks))

            try:
                response = embedder.embeddings(
                    model=self.config.embedding_model,
                    prompt=chunk["code"],
                )
                embedding = np.array(response.get("embedding", []), dtype=np.float32)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning("Error embedding chunk %s: %s", chunk["chunk_id"], e)
                # Use zero vector as fallback
                embeddings.append(np.zeros(self.config.embedding_dim, dtype=np.float32))

        return embeddings

    def build_bm25_index(self, chunks: list[dict]) -> BM25Okapi:
        """Build BM25 index from chunks."""
        corpus = [tokenize_code(chunk["code"]) for chunk in chunks]
        return BM25Okapi(corpus)

    def build_faiss_index(self, embeddings: list[np.ndarray]):
        """Build FAISS index from embeddings."""
        import faiss

        # Stack embeddings into matrix
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)

        # Create index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)

        return index

    def save(self, storage_path: str) -> None:
        """Save all indices to disk."""
        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.embeddings:
            try:
                import faiss
                faiss_index = self.build_faiss_index(self.embeddings)
                faiss.write_index(faiss_index, str(storage / "faiss_index.bin"))

                # Save ID map
                id_map = [chunk["chunk_id"] for chunk in self.chunks]
                with open(storage / "faiss_index.ids.json", "w") as f:
                    json.dump(id_map, f)
            except Exception as exc:
                logger.warning("FAISS unavailable, saving lexical store only: %s", exc)

        # Save BM25 index
        if self.chunks:
            corpus = [chunk["code"] for chunk in self.chunks]
            bm25_data = {
                "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
                "corpus": corpus,
            }
            with open(storage / "bm25_index.pkl", "wb") as f:
                pickle.dump(bm25_data, f)

        # Save chunks to SQLite
        db_path = storage / "chunks.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                code TEXT,
                source TEXT,
                cell_index INTEGER,
                cell_type TEXT,
                heading TEXT
            )
        """)

        for chunk in self.chunks:
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks (id, code, source, cell_index, cell_type, heading)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk["code"],
                    chunk["source"],
                    chunk["cell_index"],
                    chunk["cell_type"],
                    chunk["heading"],
                ),
            )

        conn.commit()
        conn.close()

        logger.info("Saved %s chunks to %s", len(self.chunks), storage)

    def index_notebooks(self, notebook_paths: list[str | Path]) -> int:
        """Index multiple notebooks."""
        all_chunks = []

        for path in notebook_paths:
            path = Path(path)
            if not path.exists():
                logger.warning("Notebook not found: %s", path)
                continue

            logger.info("Processing: %s", path.name)
            chunks = self.parse_notebook(path)
            all_chunks.extend(chunks)
            logger.info("Found %s cells", len(chunks))

        if not all_chunks:
            logger.info("No chunks to index")
            return 0

        # Generate embeddings
        embeddings = self.generate_embeddings(all_chunks, allow_fallback=True)

        # Store
        self.chunks = all_chunks
        self.embeddings = embeddings

        return len(all_chunks)


def main():
    """Index notebooks from test_notebooks directory."""
    import argparse
    import os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Index Jupyter notebooks for RAG")
    parser.add_argument(
        "--notebooks-dir",
        default="./old_code_project/test_notebooks",
        help="Directory containing notebooks",
    )
    parser.add_argument(
        "--storage-path",
        default="./rag_storage",
        help="Path to store RAG indices",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama server URL",
    )

    args = parser.parse_args()

    # Find notebooks
    notebooks_dir = Path(args.notebooks_dir)
    notebooks = list(notebooks_dir.glob("*.ipynb"))

    if not notebooks:
        logger.info("No notebooks found in %s", notebooks_dir)
        return

    logger.info("Found %s notebooks:", len(notebooks))
    for nb in notebooks:
        logger.info(" - %s", nb)

    # Create config
    config = RAGConfig(
        storage_path=args.storage_path,
        ollama_url=args.ollama_url,
    )

    # Index
    indexer = RAGIndexer(config)
    total_chunks = indexer.index_notebooks(notebooks)

    if total_chunks > 0:
        indexer.save(args.storage_path)
        logger.info("Indexing complete: %s chunks indexed", total_chunks)
    else:
        logger.info("No chunks indexed")


if __name__ == "__main__":
    main()
