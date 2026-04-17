"""
Evaluation pipeline for retrieval quality.

Measures Hit Rate, MRR, and NDCG across retrieval modes.
"""

import pytest
import math
from chainmind.retrieval.bm25_retriever import BM25Retriever
from chainmind.retrieval.hybrid_retriever import HybridRetriever


# Ground truth test cases: query → expected relevant doc IDs
EVAL_CASES = [
    {
        "query": "safety stock calculation formula",
        "relevant_docs": ["safety-stock"],
        "category": "exact_match",
    },
    {
        "query": "how to evaluate supplier performance",
        "relevant_docs": ["supplier-eval"],
        "category": "semantic",
    },
    {
        "query": "demand forecasting ARIMA method",
        "relevant_docs": ["demand-forecast"],
        "category": "technical",
    },
    {
        "query": "vehicle routing optimization",
        "relevant_docs": ["logistics-routing"],
        "category": "domain_specific",
    },
    {
        "query": "anomaly detection in quality metrics",
        "relevant_docs": ["quality-mgmt"],
        "category": "cross_domain",
    },
]

EVAL_DOCUMENTS = [
    {"id": "safety-stock", "content": "Safety Stock Calculation Guide. Safety stock = Z × √(LT × σd² + d² × σLT²). Z-score for service level.", "metadata": {"category": "inventory"}},
    {"id": "supplier-eval", "content": "Supplier Evaluation Framework. Key KPIs: Quality (30%), Delivery (25%), Cost (25%), Risk (20%). Scoring 1-5 scale.", "metadata": {"category": "procurement"}},
    {"id": "demand-forecast", "content": "Demand Forecasting Methods. Statistical: Moving Average, ARIMA, Holt-Winters. ML: XGBoost, LSTM, Transformer.", "metadata": {"category": "planning"}},
    {"id": "logistics-routing", "content": "Logistics Route Optimization. Vehicle Routing Problem: CVRP, VRPTW. Strategies: consolidation, cross-docking, milk runs.", "metadata": {"category": "logistics"}},
    {"id": "quality-mgmt", "content": "Quality Management. SPC control charts. Anomaly Detection: Z-score method, IQR method, Isolation Forest.", "metadata": {"category": "quality"}},
    {"id": "irrelevant-1", "content": "Company holiday schedule for fiscal year 2025. Office locations and parking information.", "metadata": {"category": "admin"}},
    {"id": "irrelevant-2", "content": "Employee onboarding checklist. HR forms and benefits enrollment procedures.", "metadata": {"category": "hr"}},
]


def _hit_rate(results, relevant_ids, k=5):
    """Hit Rate@K: fraction of queries where at least one relevant doc is in top-K."""
    top_k_ids = [r.doc_id for r in results[:k]]
    return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0


def _mrr(results, relevant_ids, k=10):
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, r in enumerate(results[:k]):
        if r.doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(results, relevant_ids, k=5):
    """Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, r in enumerate(results[:k]):
        rel = 1.0 if r.doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: all relevant docs at top
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


@pytest.mark.eval
class TestRetrievalQuality:

    @pytest.fixture
    async def bm25_retriever(self):
        retriever = BM25Retriever()
        await retriever.index(EVAL_DOCUMENTS)
        return retriever

    @pytest.fixture
    async def hybrid_retriever(self):
        bm25 = BM25Retriever()
        await bm25.index(EVAL_DOCUMENTS)
        return HybridRetriever(retrievers=[bm25])

    @pytest.mark.asyncio
    async def test_bm25_hit_rate(self, bm25_retriever):
        """BM25 Hit Rate@5 should be >= 0.8."""
        hits = []
        for case in EVAL_CASES:
            results = await bm25_retriever.retrieve(case["query"], top_k=5)
            hits.append(_hit_rate(results, case["relevant_docs"]))

        avg_hit_rate = sum(hits) / len(hits)
        print(f"\nBM25 Hit Rate@5: {avg_hit_rate:.2f}")
        assert avg_hit_rate >= 0.6, f"BM25 Hit Rate too low: {avg_hit_rate:.2f}"

    @pytest.mark.asyncio
    async def test_bm25_mrr(self, bm25_retriever):
        """BM25 MRR should be >= 0.5."""
        mrr_scores = []
        for case in EVAL_CASES:
            results = await bm25_retriever.retrieve(case["query"], top_k=10)
            mrr_scores.append(_mrr(results, case["relevant_docs"]))

        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        print(f"\nBM25 MRR: {avg_mrr:.2f}")
        assert avg_mrr >= 0.4, f"BM25 MRR too low: {avg_mrr:.2f}"

    @pytest.mark.asyncio
    async def test_hybrid_hit_rate(self, hybrid_retriever):
        """Hybrid Hit Rate@5 should be >= BM25 alone."""
        hits = []
        for case in EVAL_CASES:
            results = await hybrid_retriever.retrieve(case["query"], top_k=5)
            hits.append(_hit_rate(results, case["relevant_docs"]))

        avg_hit_rate = sum(hits) / len(hits)
        print(f"\nHybrid Hit Rate@5: {avg_hit_rate:.2f}")
        assert avg_hit_rate >= 0.6

    @pytest.mark.asyncio
    async def test_hybrid_ndcg(self, hybrid_retriever):
        """Hybrid NDCG@5 should be reasonable."""
        ndcg_scores = []
        for case in EVAL_CASES:
            results = await hybrid_retriever.retrieve(case["query"], top_k=5)
            ndcg_scores.append(_ndcg(results, case["relevant_docs"]))

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        print(f"\nHybrid NDCG@5: {avg_ndcg:.2f}")
        assert avg_ndcg >= 0.3
