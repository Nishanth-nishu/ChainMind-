"""EXP003 — Chain-of-Verification runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp003(BaseExperiment):
    experiment_id = "exp003_cove"
    paper_ref = "Dhuliawala et al., Meta AI 2023 — Chain-of-Verification"
    hypothesis = (
        "Post-hoc verification questions answered independently against raw "
        "tool outputs catch molecular hallucinations before the final answer."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from experiments.exp003_cove.agent import CoVeAgent

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()
        reg = AgentRegistry()
        reg.register(CoVeAgent(llm_router=router, mcp_servers=[mol]))
        reg.register(WebResearchAgent(llm_router=router, mcp_servers=[res]))
        reg.register(KnowledgeGraphAgent(llm_router=router, mcp_servers=[res]))
        return OrchestratorAgent(llm_router=router, agent_registry=reg)


if __name__ == "__main__":
    run_experiment(Exp003)
