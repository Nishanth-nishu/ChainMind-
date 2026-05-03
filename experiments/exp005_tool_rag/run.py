"""EXP005 — Dynamic Tool Selection (Tool-RAG) runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp005(BaseExperiment):
    experiment_id = "exp005_tool_rag"
    paper_ref = "Qin et al., ICLR 2024 — ToolLLM"
    hypothesis = (
        "BM25 retriever surfaces top-2 relevant tools per query instead of "
        "listing all tools, reducing selection confusion for the 7B model."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from experiments.exp005_tool_rag.agent import ToolRAGAgent

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()
        reg = AgentRegistry()
        reg.register(ToolRAGAgent(llm_router=router, mcp_servers=[mol]))
        reg.register(WebResearchAgent(llm_router=router, mcp_servers=[res]))
        reg.register(KnowledgeGraphAgent(llm_router=router, mcp_servers=[res]))
        return OrchestratorAgent(llm_router=router, agent_registry=reg)


if __name__ == "__main__":
    run_experiment(Exp005)
