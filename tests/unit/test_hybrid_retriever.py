"""
Unit tests for the Hybrid Retriever — BM25 + Dense + RRF fusion.
"""

import pytest
from chainmind.retrieval.bm25_retriever import BM25Retriever
from chainmind.retrieval.hybrid_retriever import HybridRetriever
from chainmind.core.types import RetrievalResult


@pytest.fixture
def sample_documents():
    return [
        {"id": "doc-1", "content": "Safety stock calculation requires demand variability and lead time data", "metadata": {"category": "inventory"}},
        {"id": "doc-2", "content": "SKU-001 Widget A current stock level is 1250 units at warehouse WH-EAST", "metadata": {"category": "stock"}},
        {"id": "doc-3", "content": "Supplier Acme Components has a reliability score of 95% and average lead time of 14 days", "metadata": {"category": "supplier"}},
        {"id": "doc-4", "content": "Just-in-time procurement reduces holding costs but increases supply risk", "metadata": {"category": "procurement"}},
        {"id": "doc-5", "content": "Route optimization using vehicle routing algorithms can reduce transport costs by 15-20%", "metadata": {"category": "logistics"}},
    ]


@pytest.mark.unit
class TestBM25Retriever:

    @pytest.mark.asyncio
    async def test_index_documents(self, sample_documents):
        retriever = BM25Retriever()
        count = await retriever.index(sample_documents)
        assert count == 5
        assert retriever.document_count == 5

    @pytest.mark.asyncio
    async def test_keyword_retrieval(self, sample_documents):
        retriever = BM25Retriever()
        await retriever.index(sample_documents)

        results = await retriever.retrieve("safety stock calculation", top_k=3)
        assert len(results) > 0
        assert results[0].retriever == "bm25"
        # The safety stock doc should rank highly
        assert any("safety stock" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_exact_sku_retrieval(self, sample_documents):
        """BM25 excels at exact keyword matching — critical for supply chain."""
        retriever = BM25Retriever()
        await retriever.index(sample_documents)

        results = await retriever.retrieve("SKU-001", top_k=3)
        assert len(results) > 0
        assert "SKU-001" in results[0].content

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self):
        retriever = BM25Retriever()
        results = await retriever.retrieve("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_persistence(self, sample_documents, tmp_path):
        # Index and persist
        retriever1 = BM25Retriever(persist_dir=str(tmp_path / "bm25"))
        await retriever1.index(sample_documents)
        assert retriever1.document_count == 5

        # Load from persistence
        retriever2 = BM25Retriever(persist_dir=str(tmp_path / "bm25"))
        assert retriever2.document_count == 5

        results = await retriever2.retrieve("supplier", top_k=3)
        assert len(results) > 0


@pytest.mark.unit
class TestHybridRetriever:

    @pytest.mark.asyncio
    async def test_rrf_fusion(self, sample_documents):
        """Test Reciprocal Rank Fusion across multiple retrievers."""
        bm25_1 = BM25Retriever()
        bm25_2 = BM25Retriever()  # Simulate two different retrievers

        await bm25_1.index(sample_documents)
        await bm25_2.index(sample_documents)

        hybrid = HybridRetriever(retrievers=[bm25_1, bm25_2])
        results = await hybrid.retrieve("safety stock supplier lead time", top_k=3)

        assert len(results) > 0
        # RRF scores should be present in metadata
        assert all(r.retriever == "hybrid_rrf" for r in results)
        # Results should be sorted by RRF score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_hybrid_index_propagates(self, sample_documents):
        bm25 = BM25Retriever()
        hybrid = HybridRetriever(retrievers=[bm25])

        await hybrid.index(sample_documents)
        assert bm25.document_count == 5

    @pytest.mark.asyncio
    async def test_resilient_when_retriever_fails(self, sample_documents):
        """Hybrid should still work if one retriever fails."""
        bm25 = BM25Retriever()
        await bm25.index(sample_documents)

        # Create a deliberately broken retriever
        class BrokenRetriever:
            @property
            def retriever_name(self):
                return "broken"
            async def retrieve(self, query, top_k=10):
                raise Exception("I'm broken")
            async def index(self, docs):
                return 0

        hybrid = HybridRetriever(retrievers=[bm25, BrokenRetriever()])
        results = await hybrid.retrieve("safety stock", top_k=3)

        # Should still return results from the working retriever
        assert len(results) > 0
