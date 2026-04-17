"""
ChainMind Dense Retriever — Semantic vector search using ChromaDB.

Captures semantic meaning, synonyms, and conceptual intent.
Uses sentence-transformers for embedding generation.
ChromaDB for local vector storage (no external service required).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from chainmind.core.interfaces import IRetriever
from chainmind.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class DenseRetriever(IRetriever):
    """
    Dense vector retriever using ChromaDB + sentence-transformers.

    Embeds documents and queries into dense vector space for
    semantic similarity search.
    """

    def __init__(
        self,
        collection_name: str = "supply_chain_kb",
        persist_dir: str = "./data/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

    @property
    def retriever_name(self) -> str:
        return "dense"

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB and embedding model."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model_name
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(
                f"Dense retriever initialized: model={self._embedding_model_name}, "
                f"collection={self._collection_name}, "
                f"docs={self._collection.count()}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize dense retriever: {e}")
            raise

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Semantic similarity search."""
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            retrieval_results = []
            if results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
                ids = results["ids"][0] if results["ids"] else [f"dense-{i}" for i in range(len(docs))]

                for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
                    # Convert cosine distance to similarity score
                    score = 1.0 - dist
                    retrieval_results.append(
                        RetrievalResult(
                            doc_id=doc_id,
                            content=doc,
                            score=float(score),
                            metadata=meta or {},
                            retriever=self.retriever_name,
                        )
                    )

            return retrieval_results

        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            return []

    async def index(self, documents: list[dict[str, Any]]) -> int:
        """Index documents into ChromaDB."""
        self._ensure_initialized()

        ids = []
        docs = []
        metadatas = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            doc_id = doc.get("id", hashlib.md5(content.encode()).hexdigest()[:12])
            ids.append(doc_id)
            docs.append(content)
            metadatas.append(doc.get("metadata", {}))

        if docs:
            # Upsert to handle duplicates
            self._collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metadatas,
            )

        logger.info(f"Dense: Indexed {len(docs)} documents (total: {self._collection.count()})")
        return len(docs)

    @property
    def document_count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()
