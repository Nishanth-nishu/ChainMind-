"""EXP006 — Structured Output Enforcement runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp006(BaseExperiment):
    experiment_id = "exp006_structured_output"
    paper_ref = "Willard & Louf 2023 — Outlines + JSON mode"
    hypothesis = (
        "Mandating JSON-format tool calls and adding a repair layer eliminates "
        "parsing errors, recovering TSR lost to regex fragility."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from experiments.exp006_structured_output.agent import StructuredOutputAgent

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()
        reg = AgentRegistry()
        reg.register(StructuredOutputAgent(llm_router=router, mcp_servers=[mol]))
        reg.register(WebResearchAgent(llm_router=router, mcp_servers=[res]))
        reg.register(KnowledgeGraphAgent(llm_router=router, mcp_servers=[res]))
        return OrchestratorAgent(llm_router=router, agent_registry=reg)


if __name__ == "__main__":
    run_experiment(Exp006)
