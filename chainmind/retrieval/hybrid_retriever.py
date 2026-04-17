"""
ChainMind Hybrid Retriever — Reciprocal Rank Fusion across multiple retrievers.

Combines BM25 (vectorless) + Dense (vector) + optional SPLADE (learned sparse)
using Reciprocal Rank Fusion (RRF), which is score-agnostic and robust.

Architecture:
  Query → [BM25] → Rank List 1 ─┐
                                  ├→ RRF Fusion → Top-K → Reranker → Results
  Query → [Dense] → Rank List 2 ─┘

Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods" (Cormack et al., 2009)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from chainmind.config.constants import RRF_K_CONSTANT
from chainmind.core.interfaces import IRetriever
from chainmind.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever(IRetriever):
    """
    Hybrid retriever using Reciprocal Rank Fusion.

    Runs multiple retrievers in parallel, fuses their ranked results
    using RRF, and optionally applies cross-encoder reranking.
    """

    def __init__(
        self,
        retrievers: list[IRetriever],
        reranker: Any | None = None,  # CrossEncoderReranker
        rrf_k: int = RRF_K_CONSTANT,
        rerank_top_k: int = 5,
    ):
        self._retrievers = retrievers
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._rerank_top_k = rerank_top_k

    @property
    def retriever_name(self) -> str:
        return "hybrid"

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        Hybrid retrieval with RRF fusion.

        1. Query all retrievers in parallel
        2. Apply Reciprocal Rank Fusion
        3. Optionally rerank top results
        """
        # Stage 1: Parallel retrieval
        retrieval_tasks = [
            retriever.retrieve(query, top_k=top_k * 2)  # Over-retrieve for better fusion
            for retriever in self._retrievers
        ]

        all_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Collect valid results
        ranked_lists: list[list[RetrievalResult]] = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Retriever '{self._retrievers[i].retriever_name}' failed: {result}"
                )
                continue
            ranked_lists.append(result)

        if not ranked_lists:
            logger.warning("All retrievers failed — returning empty results")
            return []

        # Stage 2: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(ranked_lists, top_k)

        # Stage 3: Reranking (optional)
        if self._reranker and fused:
            try:
                fused = await self._reranker.rerank(
                    query, fused, top_k=self._rerank_top_k
                )
            except Exception as e:
                logger.warning(f"Reranker failed, using RRF scores: {e}")

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = Σ 1 / (k + rank_i)

        where k is a constant (default 60) that dampens the impact
        of high-ranking documents.
        """
        # Accumulate RRF scores per document
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, RetrievalResult] = {}
        doc_sources: dict[str, list[str]] = defaultdict(list)

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                rrf_score = 1.0 / (self._rrf_k + rank + 1)
                rrf_scores[result.doc_id] += rrf_score

                # Keep the best metadata and content
                if result.doc_id not in doc_map or result.score > doc_map[result.doc_id].score:
                    doc_map[result.doc_id] = result

                doc_sources[result.doc_id].append(result.retriever)

        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results with RRF scores
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            original = doc_map[doc_id]
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=original.content,
                    score=rrf_scores[doc_id],
                    metadata={
                        **original.metadata,
                        "rrf_score": rrf_scores[doc_id],
                        "sources": doc_sources[doc_id],
                    },
                    retriever="hybrid_rrf",
                )
            )

        return results

    async def index(self, documents: list[dict[str, Any]]) -> int:
        """Index documents into all sub-retrievers."""
        total = 0
        for retriever in self._retrievers:
            try:
                count = await retriever.index(documents)
                total += count
            except Exception as e:
                logger.error(
                    f"Indexing failed for {retriever.retriever_name}: {e}"
                )
        return total


class CrossEncoderReranker:
    """
    Cross-encoder reranker for precision refinement.

    Takes the top-K candidates from hybrid retrieval and
    re-scores them with a cross-encoder model for higher precision.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._initialized:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            self._initialized = True
            logger.info(f"Reranker initialized: {self._model_name}")
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
            raise

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 5
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder scoring."""
        self._ensure_initialized()

        if not results or self._model is None:
            return results

        # Create query-document pairs
        pairs = [(query, r.content) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Assign reranked scores
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    score=float(score),
                    metadata={
                        **result.metadata,
                        "reranker_score": float(score),
                        "original_score": result.score,
                    },
                    retriever="reranked",
                )
            )

        # Sort by reranker score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
