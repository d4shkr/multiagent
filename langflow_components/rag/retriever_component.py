"""Hybrid RAG Retriever Component for Langflow."""

import time
from pathlib import Path
from typing import Any

from langflow.custom import Component
from langflow.io import StrInput, IntInput, DropdownInput, MessageTextInput, DataInput, Output
from langflow.schema import Data

from langflow_components.rag.utils import (
    RAGConfig,
    RetrievedChunk,
    rrf_rerank,
)


class HybridRetrieverComponent(Component):
    """Hybrid RAG Retriever combining BM25 + Semantic search with RRF reranking."""

    display_name = "Hybrid RAG Retriever"
    description = "Retrieves code examples using hybrid BM25 + semantic search"
    icon = "search"

    inputs = [
        DataInput(
            name="query_input",
            display_name="Query Input",
            info="Connect from another component (e.g., Text Input, Chat Input)",
            input_types=["Data"],
        ),
        StrInput(
            name="query",
            display_name="Search Query (manual)",
            info="The search query (or connect Query Input)",
            tool_mode=True,
            value="",
        ),
        IntInput(
            name="k",
            display_name="Top K Results",
            info="Number of results to return",
            value=5,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Type of search to perform",
            options=["hybrid", "semantic", "bm25"],
            value="hybrid",
        ),
        StrInput(
            name="storage_path",
            display_name="Storage Path",
            info="Path to RAG storage directory",
            value="./rag_storage",
        ),
        StrInput(
            name="source_filter",
            display_name="Source Filter",
            info="Filter by source notebook path (partial match)",
            value="",
            advanced=True,
        ),
        StrInput(
            name="cell_type_filter",
            display_name="Cell Type Filter",
            info="Filter by cell type: 'code' or 'markdown'",
            value="",
            advanced=True,
        ),
        IntInput(
            name="max_code_len",
            display_name="Max Code Length",
            info="Maximum code length per result",
            value=500,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Results",
            name="results",
            method="retrieve",
        ),
        Output(
            display_name="Formatted Context",
            name="context",
            method="retrieve_formatted",
        ),
        Output(
            display_name="Text Output",
            name="text_output",
            method="get_text_output",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._retriever = None

    def _get_retriever(self):
        """Lazy-load the retriever."""
        if self._retriever is None:
            from langflow_components.rag.retriever_backend import HybridRetriever

            config = RAGConfig(storage_path=self.storage_path)
            self._retriever = HybridRetriever.from_storage(
                storage_path=self.storage_path,
                config=config,
            )
        return self._retriever

    def _get_query(self) -> str:
        """Get query from input connection or manual field."""
        # Try to get from query_input (DataInput connection)
        if self.query_input is not None and self.query_input != "":
            try:
                if hasattr(self.query_input, 'data'):
                    input_data = self.query_input.data
                else:
                    input_data = self.query_input

                if isinstance(input_data, dict):
                    # Try common field names
                    query = (
                        input_data.get("query")
                        or input_data.get("text")
                        or input_data.get("message")
                        or input_data.get("input")
                        or input_data.get("task")
                    )
                    if query:
                        return str(query)
                elif isinstance(input_data, str):
                    return input_data
            except (AttributeError, TypeError):
                pass

        # Fallback to manual query field
        return self.query if self.query else ""

    def retrieve(self) -> list[Data]:
        """
        Retrieve code examples matching the query.

        Returns:
            List of Data objects containing retrieved chunks
        """
        query = self._get_query()
        if not query:
            return []

        retriever = self._get_retriever()

        start_time = time.perf_counter()

        # Build filters
        source_filter = self.source_filter if self.source_filter else None
        cell_type_filter = self.cell_type_filter if self.cell_type_filter else None

        # Perform search based on type
        if self.search_type == "hybrid":
            results = retriever.retrieve(
                query=query,
                k=self.k,
                source_filter=source_filter,
                cell_type_filter=cell_type_filter,
            )
        elif self.search_type == "semantic":
            results = retriever.semantic_search(
                query=query,
                k=self.k,
                source_filter=source_filter,
                cell_type_filter=cell_type_filter,
            )
        else:  # bm25
            results = retriever.bm25_search(
                query=query,
                k=self.k,
                source_filter=source_filter,
                cell_type_filter=cell_type_filter,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.status = f"Found {len(results)} results in {latency_ms:.1f}ms"

        # Convert to Data objects
        data_results = []
        for r in results:
            data_results.append(
                Data(
                    data={
                        "chunk_id": r.chunk_id,
                        "code": r.code[:self.max_code_len] if len(r.code) > self.max_code_len else r.code,
                        "source": r.source,
                        "cell_index": r.cell_index,
                        "score": r.score,
                        "bm25_score": r.bm25_score,
                        "semantic_score": r.semantic_score,
                        "heading": r.heading,
                    }
                )
            )

        return data_results

    def retrieve_formatted(self) -> Data:
        """
        Retrieve and format code examples for prompt context.

        Returns:
            Data object with formatted context string
        """
        query = self._get_query()
        if not query:
            return Data(data={"context": "No query provided.", "results_count": 0})

        retriever = self._get_retriever()

        source_filter = self.source_filter if self.source_filter else None
        cell_type_filter = self.cell_type_filter if self.cell_type_filter else None

        results = retriever.retrieve(
            query=query,
            k=self.k,
            source_filter=source_filter,
            cell_type_filter=cell_type_filter,
        )

        # Format for prompt
        formatted = retriever.format_for_prompt(
            results=results,
            max_chunks=self.k,
            max_code_len=self.max_code_len,
        )

        self.status = f"Formatted {len(results)} results"

        return Data(data={"context": formatted, "results_count": len(results)})

    def get_text_output(self) -> Data:
        """
        Get results as simple text for chaining to next component.

        Returns:
            Data object with text representation of results
        """
        query = self._get_query()
        if not query:
            return Data(data={
                "text": "No query provided.",
                "query": "",
                "results_count": 0,
            })

        retriever = self._get_retriever()

        source_filter = self.source_filter if self.source_filter else None
        cell_type_filter = self.cell_type_filter if self.cell_type_filter else None

        results = retriever.retrieve(
            query=query,
            k=self.k,
            source_filter=source_filter,
            cell_type_filter=cell_type_filter,
        )

        # Build text output
        lines = [f"=== RAG Search Results (query: {query}) ==="]
        lines.append(f"Found: {len(results)} results\n")

        for i, r in enumerate(results, 1):
            code_preview = r.code[:200] + "..." if len(r.code) > 200 else r.code
            lines.append(f"[{i}] Source: {r.source} | Score: {r.score:.3f}")
            lines.append(f"    Code: {code_preview}\n")

        text_output = "\n".join(lines)
        self.status = f"Generated text output with {len(results)} results"

        return Data(data={
            "text": text_output,
            "query": query,
            "results_count": len(results),
        })
