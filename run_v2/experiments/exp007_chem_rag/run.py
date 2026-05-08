"""EXP007 — Chemistry-RAG Pre-Context runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp007(BaseExperiment):
    experiment_id = "exp007_chem_rag"
    paper_ref = "Lewis et al., NeurIPS 2020 — RAG + ChemCrow domain adaptation"
    hypothesis = (
        "BM25 retrieval of chemistry knowledge chunks (Lipinski rules, SMILES "
        "conventions, tool return formats) before each THINK step grounds the "
        "7B model and reduces tool output misinterpretation."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from experiments.exp007_chem_rag.agent import ChemRAGAgent

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()
        reg = AgentRegistry()
        reg.register(ChemRAGAgent(llm_router=router, mcp_servers=[mol]))
        reg.register(WebResearchAgent(llm_router=router, mcp_servers=[res]))
        reg.register(KnowledgeGraphAgent(llm_router=router, mcp_servers=[res]))
        return OrchestratorAgent(llm_router=router, agent_registry=reg)


if __name__ == "__main__":
    run_experiment(Exp007)
