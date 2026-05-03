"""EXP008 — Multi-Agent Debate runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp008(BaseExperiment):
    experiment_id = "exp008_debate"
    paper_ref = "Du et al., ICML 2023 — Multiagent Debate"
    hypothesis = (
        "Two parallel agents independently answer each question; a judge "
        "resolves disagreements against raw tool outputs, reducing hallucination."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from chainmind.agents.specialists import ComputationalChemistAgent
        from experiments.exp008_debate.agent import DebateOrchestrator

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()

        # Two independent chemical agents for debate
        agent_a = ComputationalChemistAgent(llm_router=router, mcp_servers=[mol])
        agent_b = ComputationalChemistAgent(llm_router=router, mcp_servers=[mol])
        debate = DebateOrchestrator(agent_a=agent_a, agent_b=agent_b, llm_router=router)

        # Wrap in a minimal registry+orchestrator so base_experiment.run_task works
        # The debate orchestrator exposes .process() directly
        return debate


if __name__ == "__main__":
    run_experiment(Exp008)
