"""
ChainMind BM25 Retriever — Vectorless keyword-based retrieval.

Pure lexical matching with no embeddings or vector storage.
Excels at exact keyword matches (SKU codes, error IDs, technical terms).
Zero cost, zero latency from embedding models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from chainmind.core.interfaces import IRetriever
from chainmind.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever(IRetriever):
    """
    BM25-based vectorless retriever.

    Tokenizes documents at index time and queries using
    Okapi BM25 scoring (Robertson et al., 1994).
    """

    def __init__(self, persist_dir: str | None = None):
        self._documents: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._persist_dir = persist_dir
        self._is_indexed = False

        # Load persisted index if available
        if persist_dir:
            self._load_index()

    @property
    def retriever_name(self) -> str:
        return "bm25"

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve documents by BM25 keyword matching."""
        if not self._is_indexed or self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue  # Skip zero-score results

            doc = self._documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.get("id", f"bm25-{idx}"),
                    content=doc["content"],
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    retriever=self.retriever_name,
                )
            )

        return results

    async def index(self, documents: list[dict[str, Any]]) -> int:
        """Index documents for BM25 retrieval."""
        for doc in documents:
            if "content" not in doc:
                continue

            # Generate ID if not present
            if "id" not in doc:
                doc["id"] = hashlib.md5(doc["content"].encode()).hexdigest()[:12]

            self._documents.append(doc)
            self._tokenized_corpus.append(self._tokenize(doc["content"]))

        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self._is_indexed = True

        # Persist if configured
        if self._persist_dir:
            self._save_index()

        logger.info(f"BM25: Indexed {len(documents)} new docs (total: {len(self._documents)})")
        return len(documents)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        # Basic tokenization — handles most supply chain terminology
        return text.lower().split()

    def _save_index(self) -> None:
        """Persist document store to disk."""
        if not self._persist_dir:
            return
        path = Path(self._persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        index_file = path / "bm25_index.json"
        with open(index_file, "w") as f:
            json.dump(self._documents, f, default=str)

    def _load_index(self) -> None:
        """Load persisted document store."""
        if not self._persist_dir:
            return
        index_file = Path(self._persist_dir) / "bm25_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    self._documents = json.load(f)
                self._tokenized_corpus = [
                    self._tokenize(doc["content"]) for doc in self._documents
                ]
                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
                    self._is_indexed = True
                logger.info(f"BM25: Loaded {len(self._documents)} docs from cache")
            except Exception as e:
                logger.warning(f"BM25: Failed to load index: {e}")

    @property
    def document_count(self) -> int:
        return len(self._documents)
