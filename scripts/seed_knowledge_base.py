"""
Seed the ChainMind knowledge base with sample supply chain documents.

Run: python scripts/seed_knowledge_base.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chainmind.config.settings import get_settings
from chainmind.retrieval.knowledge_base import KnowledgeBase


SUPPLY_CHAIN_DOCUMENTS = [
    {
        "title": "Safety Stock Calculation Guide",
        "category": "inventory",
        "content": """Safety Stock Calculation Guide

Safety stock is buffer inventory held to protect against variability in demand and lead time.

Formula: Safety Stock = Z × √(LT × σd² + d² × σLT²)

Where:
- Z = Z-score for desired service level (1.65 for 95%, 2.33 for 99%)
- LT = Average lead time in days
- σd = Standard deviation of daily demand
- d = Average daily demand
- σLT = Standard deviation of lead time

Best Practices:
1. Review safety stock levels quarterly
2. Use ABC classification to set different service levels
3. Consider seasonal demand patterns
4. Factor in supplier reliability scores
5. Monitor stockout frequency vs. holding costs

Common Service Levels:
- A items (high value): 99% → Z = 2.33
- B items (medium value): 95% → Z = 1.65
- C items (low value): 90% → Z = 1.28""",
    },
    {
        "title": "Supplier Evaluation Framework",
        "category": "procurement",
        "content": """Supplier Evaluation Framework

Key Performance Indicators for Supplier Evaluation:

1. Quality (Weight: 30%)
   - Defect rate (PPM)
   - Quality certification (ISO 9001, etc.)
   - Return/rejection rate
   - Corrective action responsiveness

2. Delivery (Weight: 25%)
   - On-time delivery percentage
   - Lead time consistency
   - Flexibility in order changes
   - Emergency order capability

3. Cost (Weight: 25%)
   - Unit price competitiveness
   - Total Cost of Ownership (TCO)
   - Payment terms
   - Volume discount structures

4. Risk (Weight: 20%)
   - Financial stability
   - Geographic risk
   - Single-source dependency
   - Business continuity planning

Scoring: Each KPI on 1-5 scale. Weighted average determines overall score.
Actions:
- Score > 4.0: Preferred supplier
- Score 3.0-4.0: Approved supplier
- Score < 3.0: Probation — improvement plan required""",
    },
    {
        "title": "Demand Forecasting Methods",
        "category": "demand_planning",
        "content": """Demand Forecasting Methods in Supply Chain

Statistical Methods:
1. Moving Average — Simple, weighted, or exponential
2. ARIMA — Best for stationary time series with trend/seasonality
3. Exponential Smoothing (Holt-Winters) — Handles trend and seasonal components
4. Prophet — Facebook's model for business time series with holidays

Machine Learning Methods:
1. Random Forest / XGBoost — Feature-rich datasets with external variables
2. LSTM Neural Networks — Long-term dependencies in sequential data
3. Transformer models — State-of-the-art for complex patterns

Forecast Accuracy Metrics:
- MAPE (Mean Absolute Percentage Error): Target < 15%
- WMAPE (Weighted MAPE): Better for SKU mix evaluation
- Bias: Should be near zero (over/under-forecasting detection)
- Forecast Value Added (FVA): Compare against naive forecast

Best Practices:
- Use multiple models and ensemble results
- Segment products by demand pattern (smooth, intermittent, lumpy)
- Incorporate external signals (promotions, weather, economic indicators)
- Review forecast accuracy weekly, model fit monthly""",
    },
    {
        "title": "Logistics Route Optimization",
        "category": "logistics",
        "content": """Logistics Route Optimization Guide

Vehicle Routing Problem (VRP) Solutions:
1. Capacitated VRP (CVRP) — Vehicle capacity constraints
2. VRP with Time Windows (VRPTW) — Delivery time requirements
3. Multi-Depot VRP — Multiple warehouse origins

Optimization Strategies:
- Consolidation: Combine LTL shipments to reduce cost per unit
- Cross-docking: Reduce warehouse handling by direct transfer
- Milk runs: Scheduled pickup routes for multiple suppliers
- Hub-and-spoke: Central distribution hub model

Carrier Selection Criteria:
- Cost per mile/kg
- Transit time reliability
- Coverage area
- Damage/claims history
- Technology integration (EDI, API tracking)

KPIs:
- Cost per unit shipped
- On-time delivery rate (target: >95%)
- Average transit time
- Order accuracy rate
- Carbon footprint per shipment""",
    },
    {
        "title": "Quality Management in Supply Chain",
        "category": "quality",
        "content": """Quality Management in Supply Chain

Statistical Process Control (SPC):
- Control charts (X-bar, R-chart, p-chart)
- Process capability indices (Cp, Cpk)
- Upper/Lower control limits (UCL/LCL)

Root Cause Analysis Methods:
1. 5 Whys — Simple iterative questioning
2. Fishbone (Ishikawa) — Categorize causes (Man, Machine, Method, Material, Environment)
3. Pareto Analysis — Focus on vital few (80/20 rule)
4. Fault Tree Analysis — Systematic failure analysis

Anomaly Detection:
- Z-score method: Flag data points > 2σ from mean
- IQR method: Flag points outside Q1 - 1.5×IQR to Q3 + 1.5×IQR
- Machine learning: Isolation Forest, Autoencoder-based detection

Corrective Action Process:
1. Contain — Isolate affected inventory
2. Identify — Root cause analysis
3. Correct — Implement fix
4. Verify — Confirm effectiveness
5. Prevent — Systemic improvement""",
    },
]


async def main():
    print("🔧 Initializing ChainMind Knowledge Base...")
    settings = get_settings()
    kb = KnowledgeBase(settings)

    print(f"📚 Seeding {len(SUPPLY_CHAIN_DOCUMENTS)} documents...")
    for doc in SUPPLY_CHAIN_DOCUMENTS:
        doc_id = await kb.ingest(
            content=doc["content"],
            metadata={"title": doc["title"], "category": doc["category"]},
        )
        print(f"  ✓ {doc['title']} → {doc_id}")

    print(f"\n✅ Knowledge base seeded with {kb.document_count} documents")

    # Test search
    print("\n🔍 Testing hybrid search...")
    results = await kb.search("safety stock calculation formula", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] {r.content[:80]}...")

    print("\n🔍 Testing BM25 search...")
    results = await kb.search("SKU service level 95%", top_k=3, mode="bm25")
    for r in results:
        print(f"  [{r.score:.4f}] {r.content[:80]}...")

    print("\n✅ Seed complete!")


if __name__ == "__main__":
    asyncio.run(main())
