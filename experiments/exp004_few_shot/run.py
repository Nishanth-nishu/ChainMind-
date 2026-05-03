"""EXP004 — Chemistry Few-Shot Prompting runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.base_experiment import BaseExperiment
from experiments.shared.run_template import run_experiment


class Exp004(BaseExperiment):
    experiment_id = "exp004_few_shot"
    paper_ref = "Bran et al., NeurIPS AI4Sci 2023 — ChemCrow"
    hypothesis = (
        "4 curated chemistry examples (Lipinski, Tanimoto, PubChem, conformer) "
        "in the system prompt teach the 7B model correct tool format, reducing "
        "malformed tool calls. Expected highest TSR gain (+17-20%)."
    )

    def build_orchestrator(self, settings):
        from chainmind.llm.router import LLMRouter
        from chainmind.agents.orchestrator import OrchestratorAgent
        from chainmind.agents.specialists import WebResearchAgent, KnowledgeGraphAgent
        from chainmind.a2a.protocol import AgentRegistry
        from chainmind.mcp.molecular_server import MolecularMCPServer
        from chainmind.mcp.research_server import ResearchMCPServer
        from experiments.exp004_few_shot.agent import FewShotChemistAgent

        router = LLMRouter(settings)
        mol, res = MolecularMCPServer(), ResearchMCPServer()
        reg = AgentRegistry()
        reg.register(FewShotChemistAgent(llm_router=router, mcp_servers=[mol]))
        reg.register(WebResearchAgent(llm_router=router, mcp_servers=[res]))
        reg.register(KnowledgeGraphAgent(llm_router=router, mcp_servers=[res]))
        return OrchestratorAgent(llm_router=router, agent_registry=reg)


if __name__ == "__main__":
    run_experiment(Exp004)
