"""
ChainMind Knowledge Base — High-level manager for the hybrid RAG pipeline.

Orchestrates document ingestion, chunking, indexing, and querying
across the full retrieval stack.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from chainmind.config.settings import RetrievalMode, Settings
from chainmind.core.types import RetrievalResult
from chainmind.retrieval.bm25_retriever import BM25Retriever
from chainmind.retrieval.dense_retriever import DenseRetriever
from chainmind.retrieval.hybrid_retriever import CrossEncoderReranker, HybridRetriever

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base manager for the hybrid RAG pipeline.

    Supports three retrieval modes:
    - 'bm25': Vectorless keyword matching only
    - 'vector': Dense semantic search only
    - 'hybrid': BM25 + Dense with RRF fusion + reranking
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._documents: dict[str, dict[str, Any]] = {}  # In-memory doc store

        # Initialize retrievers
        self._bm25 = BM25Retriever(
            persist_dir=str(settings.chromadb_path / "bm25_cache")
        )
        self._dense = DenseRetriever(
            collection_name="supply_chain_kb",
            persist_dir=str(settings.chromadb_path),
            embedding_model=settings.embedding_model,
        )

        # Initialize reranker (lazy)
        self._reranker = CrossEncoderReranker()

        # Initialize hybrid retriever
        self._hybrid = HybridRetriever(
            retrievers=[self._bm25, self._dense],
            reranker=self._reranker,
            rerank_top_k=settings.reranker_top_k,
        )

        logger.info(f"Knowledge base initialized (mode={settings.retrieval_mode.value})")

    async def search(
        self, query: str, top_k: int = 5, mode: str | None = None
    ) -> list[RetrievalResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            mode: Override retrieval mode ('bm25', 'vector', 'hybrid')
        """
        effective_mode = mode or self._settings.retrieval_mode.value

        if effective_mode == "bm25":
            return await self._bm25.retrieve(query, top_k)
        elif effective_mode == "vector":
            return await self._dense.retrieve(query, top_k)
        else:  # hybrid
            return await self._hybrid.retrieve(query, top_k)

    async def ingest(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Ingest a document into the knowledge base.

        Chunks the content and indexes into all retriever backends.
        """
        doc_id = str(uuid.uuid4())[:12]
        metadata = metadata or {}

        # Chunk the document
        chunks = self._chunk_document(content, doc_id, metadata)

        # Index in all retrievers
        await self._hybrid.index(chunks)

        # Store in memory for direct retrieval
        self._documents[doc_id] = {
            "content": content,
            "metadata": metadata,
            "chunk_count": len(chunks),
        }

        logger.info(f"KB: Ingested doc {doc_id} ({len(chunks)} chunks)")
        return doc_id

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a specific document by ID."""
        return self._documents.get(doc_id)

    def _chunk_document(
        self,
        content: str,
        doc_id: str,
        metadata: dict[str, Any],
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Semantic-aware chunking with overlap.

        Splits on paragraph boundaries first, then falls back
        to fixed-size chunks with overlap.
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunk_id = f"{doc_id}-{len(chunks)}"
                    chunks.append({
                        "id": chunk_id,
                        "content": current_chunk,
                        "metadata": {
                            **metadata,
                            "parent_doc_id": doc_id,
                            "chunk_index": len(chunks),
                        },
                    })
                current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunk_id = f"{doc_id}-{len(chunks)}"
            chunks.append({
                "id": chunk_id,
                "content": current_chunk,
                "metadata": {
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": len(chunks),
                },
            })

        # If no paragraph-based chunks, do fixed-size chunking
        if not chunks:
            words = content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i : i + chunk_size])
                chunk_id = f"{doc_id}-{len(chunks)}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "parent_doc_id": doc_id,
                        "chunk_index": len(chunks),
                    },
                })

        return chunks

    @property
    def document_count(self) -> int:
        return len(self._documents)
