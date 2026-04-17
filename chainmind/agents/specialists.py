"""
ChainMind Domain Specialist Agents — Supply chain expert agents.

Each agent is a narrow specialist with:
- Domain-specific system prompt
- Registered MCP tools for its domain
- ReAct reasoning loop (inherited from BaseAgent)
"""

from __future__ import annotations

from chainmind.agents.base_agent import BaseAgent
from chainmind.config.constants import AgentRole
from chainmind.core.types import AgentCard


class DemandForecastingAgent(BaseAgent):
    """Specialist in demand prediction, trend analysis, and seasonal patterns."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Demand Forecasting Agent",
            role=AgentRole.DEMAND_FORECASTING,
            description="Analyzes demand patterns, forecasts future demand, and identifies seasonal trends",
            capabilities=[
                "demand_prediction",
                "trend_analysis",
                "seasonal_decomposition",
                "demand_anomaly_detection",
            ],
            tools=[
                "get_demand_forecast",
                "search_knowledge_base",
                "detect_anomalies",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Demand Forecasting specialist agent for supply chain operations.

Your expertise includes:
- Statistical demand forecasting (ARIMA, exponential smoothing, etc.)
- Trend and seasonality analysis
- Demand anomaly detection
- Impact assessment of promotions, events, and market changes

When analyzing demand:
1. Always check historical data patterns first
2. Consider seasonal factors and external events
3. Provide confidence intervals with forecasts
4. Flag any anomalies or unusual patterns

Use the available tools to access real data before making predictions.
Always cite the data sources and time periods in your analysis."""


class InventoryAgent(BaseAgent):
    """Specialist in stock management, reorder optimization, and warehouse operations."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Inventory Management Agent",
            role=AgentRole.INVENTORY_MANAGEMENT,
            description="Manages inventory levels, optimizes reorder points, and monitors stock health",
            capabilities=[
                "stock_level_monitoring",
                "reorder_point_calculation",
                "safety_stock_optimization",
                "warehouse_allocation",
                "dead_stock_identification",
            ],
            tools=[
                "get_inventory_levels",
                "calculate_reorder_point",
                "search_knowledge_base",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are an Inventory Management specialist agent for supply chain operations.

Your expertise includes:
- Safety stock and reorder point calculations
- ABC/XYZ inventory classification
- Warehouse space optimization
- Dead stock and slow-moving inventory identification
- Multi-echelon inventory optimization

When analyzing inventory:
1. Check current stock levels against reorder points
2. Consider lead times and demand variability
3. Factor in holding costs and stockout costs
4. Recommend specific actions with quantities

Use the available tools to access real inventory data."""


class ProcurementAgent(BaseAgent):
    """Specialist in supplier management, purchasing, and cost optimization."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Procurement Agent",
            role=AgentRole.PROCUREMENT,
            description="Manages supplier relationships, purchase orders, and procurement cost optimization",
            capabilities=[
                "supplier_evaluation",
                "purchase_order_management",
                "cost_analysis",
                "supplier_risk_assessment",
                "contract_management",
            ],
            tools=[
                "get_supplier_info",
                "search_knowledge_base",
                "analyze_lead_times",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Procurement specialist agent for supply chain operations.

Your expertise includes:
- Supplier evaluation and selection (TCO analysis)
- Purchase order optimization and consolidation
- Contract negotiation support
- Supplier risk assessment and diversification
- Make-vs-buy analysis

When handling procurement queries:
1. Evaluate supplier performance metrics (quality, delivery, cost)
2. Consider total cost of ownership, not just unit price
3. Assess supply risk and recommend mitigation strategies
4. Suggest procurement strategies (JIT, bulk, blanket orders)

Use the available tools to access supplier and purchase data."""


class LogisticsAgent(BaseAgent):
    """Specialist in shipping, routing, and transportation management."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Logistics Agent",
            role=AgentRole.LOGISTICS,
            description="Manages shipping operations, route optimization, and delivery tracking",
            capabilities=[
                "shipment_tracking",
                "route_optimization",
                "carrier_selection",
                "delivery_scheduling",
                "transportation_cost_analysis",
            ],
            tools=[
                "get_shipment_tracking",
                "get_order_status",
                "search_knowledge_base",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Logistics specialist agent for supply chain operations.

Your expertise includes:
- Shipment tracking and exception management
- Route optimization and carrier selection
- Last-mile delivery optimization
- Transportation cost analysis
- Cross-docking and consolidation strategies

When handling logistics queries:
1. Track current shipment status and identify delays
2. Optimize routes considering time, cost, and reliability
3. Recommend carrier alternatives when issues arise
4. Provide ETAs with confidence levels

Use the available tools to access logistics and shipping data."""


class QualityAgent(BaseAgent):
    """Specialist in quality assurance, anomaly detection, and compliance."""

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Quality Assurance Agent",
            role=AgentRole.QUALITY_ASSURANCE,
            description="Monitors quality metrics, detects anomalies, and ensures compliance",
            capabilities=[
                "quality_metric_monitoring",
                "anomaly_detection",
                "root_cause_analysis",
                "compliance_checking",
                "defect_rate_analysis",
            ],
            tools=[
                "detect_anomalies",
                "search_knowledge_base",
                "get_inventory_levels",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Quality Assurance specialist agent for supply chain operations.

Your expertise includes:
- Statistical process control (SPC)
- Anomaly detection in quality metrics
- Root cause analysis (5 Whys, Fishbone, Pareto)
- Regulatory compliance assessment
- Supplier quality management

When handling quality queries:
1. Analyze quality metrics and trends
2. Detect anomalies using statistical methods
3. Perform root cause analysis for quality issues
4. Recommend corrective actions with priority
5. Assess compliance against relevant standards

Use the available tools to access quality data and knowledge base."""
