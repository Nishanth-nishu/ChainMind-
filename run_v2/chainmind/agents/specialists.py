"""
ChainMind D4 Specialist Agents — Drug Discovery Decision Decomposition (D4).

Three narrow specialists, each with:
- Domain-specific system prompt enforcing ReAct step-by-step reasoning
- Isolated MCP tool set (agent-level tool isolation — no cross-contamination)
- Force-convergence loop (inherited from BaseAgent) to prevent infinite loops

Agents
------
ComputationalChemistAgent
    Molecular analysis via RDKit + PubChem. Handles Lipinski, SMILES validation,
    Tanimoto similarity, PDB/MOL2 parsing, and 3D conformer generation.

WebResearchAgent
    Literature search via DuckDuckGo + ArXiv API + TDC leaderboard retrieval.

KnowledgeGraphAgent
    Visual knowledge graph generation (Mermaid.js flowcharts) for explaining
    drug discovery concepts to non-coding scientists.

Architecture
-----------
Each specialist is registered with the A2A AgentRegistry and receives tasks
exclusively via the A2A protocol from the OrchestratorAgent. This separation
ensures that tool usage, reasoning traces, and errors are cleanly attributable
to individual agents — a design requirement for the ablation study.

Note on Tool Isolation
----------------------
ComputationalChemistAgent  →  MolecularMCPServer  (RDKit + BioPython + PubChem)
WebResearchAgent           →  ResearchMCPServer   (DuckDuckGo + ArXiv + TDC)
KnowledgeGraphAgent        →  ResearchMCPServer   (generate_knowledge_graph + search_literature)

No agent has access to tools outside its domain. This is enforced at registration
time in cli.py / api_server.py by passing only the relevant MCP server to each
BaseAgent constructor. Cross-agent tool contamination is a controlled variable
in the ablation study (see chainmind/eval/ablation.py).
"""

from __future__ import annotations

from chainmind.agents.base_agent import BaseAgent
from chainmind.config.constants import AgentRole
from chainmind.core.types import AgentCard


# =============================================================================
# D4 (Drug Discovery Decision Decomposition) Specialist Agents
# =============================================================================


class ComputationalChemistAgent(BaseAgent):
    """
    D4 Specialist: Computational Chemistry & Structural Biology.

    Connects exclusively to the MolecularMCPServer, which provides RDKit-
    and BioPython-backed tools for:
    - Lipinski's Rule of 5 assessment
    - Tanimoto molecular similarity (Morgan fingerprints, radius=2, 2048 bits)
    - PubChem compound lookup (name / SMILES namespace)
    - PDB and MOL2 file parsing (atom count, chain count, formula)
    - 3D conformer generation (ETKDG + MMFF94 force-field optimisation)
    - 3D coordinate validation (energy-based steric clash detection)

    Agent isolation: this agent CANNOT call web search or KG generation tools.
    All computations are deterministic (RDKit is seeded at 42) — results are
    reproducible across runs, which is a prerequisite for benchmark grading.
    """

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Computational Chemist Agent",
            role=AgentRole.COMPUTATIONAL_CHEMISTRY,
            description=(
                "Analyzes molecular structures using RDKit and PubChem. "
                "Handles drug-likeness assessment, structural similarity, "
                "PDB/MOL2 parsing, and 3D conformer generation."
            ),
            capabilities=[
                "lipinski_analysis",
                "molecular_similarity",
                "pubchem_search",
                "structure_parsing",
                "conformer_generation",
                "molecule_validation",
            ],
            tools=[
                "parse_molecule_file",
                "assess_lipinski_rules",
                "calculate_similarity",
                "pubchem_search",
                "generate_3d_conformer",
                "validate_molecule_3d",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Computational Chemistry specialist agent for drug discovery operations.

Your expertise includes:
- Molecular structure analysis using RDKit (SMILES parsing, descriptors)
- Lipinski's Rule of 5 for drug-likeness assessment
- Tanimoto molecular similarity using Morgan fingerprints
- PubChem database queries for compound metadata
- PDB and MOL2 file parsing for structural biology
- 3D conformer generation via ETKDG + MMFF94 force-field optimization

When analyzing molecules, think step by step:
1. ALWAYS validate SMILES strings before computing properties
2. Report all four Lipinski descriptors (MW, LogP, HBD, HBA) with pass/fail
3. For similarity calculations, use Morgan radius=2, nBits=2048
4. When querying PubChem, provide CID, IUPAC name, formula, and weight
5. For 3D coordinates, report whether MMFF94 optimization converged

## Example
Query: "Does Aspirin pass Lipinski's Rule of 5?"
Thinking: First, I need to validate the SMILES. Then call assess_lipinski_rules tool.
Tool call: assess_lipinski_rules(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
Result: MW=180.16 (✅ <500), LogP=1.24 (✅ <5), HBD=1 (✅ ≤5), HBA=4 (✅ ≤10)
Answer: Aspirin PASSES all Lipinski rules. It is drug-like.

IMPORTANT: You must use the available tools for all chemistry computations.
NEVER hallucinate molecular properties — always use the deterministic tools.
If a SMILES string is invalid, report the error clearly."""


class WebResearchAgent(BaseAgent):
    """
    D4 Specialist: Literature & Web Research.

    Connects exclusively to the ResearchMCPServer for:
    - DuckDuckGo web search (recent papers, reviews, drug discovery news)
    - ArXiv direct API search (academic preprints, sorted by submission date)
    - TDC (Therapeutics Data Commons) benchmark lookup (ADMET tasks, SOTA scores)

    Agent isolation: this agent CANNOT call molecular tools (RDKit, PubChem).
    All queries are open-ended and exploratory — the agent is designed to surface
    recent literature and synthesize key findings into actionable summaries.
    """

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Web Research Agent",
            role=AgentRole.WEB_RESEARCH,
            description=(
                "Searches the web and ArXiv for recent scientific papers, "
                "ML drug discovery approaches, and TDC benchmark data."
            ),
            capabilities=[
                "literature_search",
                "arxiv_search",
                "tdc_benchmark_retrieval",
                "research_summarization",
            ],
            tools=[
                "search_literature",
                "search_arxiv",
                "fetch_tdc_benchmark",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Web Research specialist agent for drug discovery literature.

Your expertise includes:
- Searching for recent ML/AI papers on drug discovery
- Finding ADMET prediction methodologies and benchmarks
- Retrieving TDC (Therapeutics Data Commons) task information
- Summarizing research trends in computational chemistry

When handling research queries, think step by step:
1. Search multiple sources (DuckDuckGo + ArXiv) for comprehensive coverage
2. Prioritize recent papers (2023-2025) for state-of-the-art methods
3. Include paper titles, key findings, and URLs in your responses
4. For TDC benchmarks, report the SOTA model, metric, and score
5. Synthesize findings into clear, actionable summaries

## Example
Query: "Latest GNN approaches for molecular property prediction"
Thinking: I need to search ArXiv for recent GNN papers, then DuckDuckGo for broader coverage.
Step 1: search_arxiv(query="GNN molecular property prediction 2024")
Step 2: search_literature(query="graph neural network ADMET prediction")
Step 3: Synthesize top 5 papers with titles, methods, and key results.

Use the available tools for all searches.
Cite your sources with titles and URLs."""


class KnowledgeGraphAgent(BaseAgent):
    """
    D4 Specialist: Knowledge Graph Generation.

    Connects to the ResearchMCPServer's generate_knowledge_graph and
    search_literature tools to:
    - Convert scientific concepts into Subject → Predicate → Object triplets
    - Generate valid Mermaid.js flowcharts (graph TD format)
    - Optionally search literature for context before generating the graph

    Agent isolation: this agent CANNOT call molecular computation tools.
    Knowledge graphs are validated in the benchmark by: (a) checking that
    the Mermaid block is syntactically parseable, and (b) counting edges
    (minimum 5 required for a meaningful graph).

    Output format: the agent MUST produce a ```mermaid ... ``` code block
    followed by a plain-language explanation. Both are required for grading.
    """

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Knowledge Graph Agent",
            role=AgentRole.KNOWLEDGE_GRAPH,
            description=(
                "Generates visual Knowledge Graphs (Mermaid.js flowcharts) "
                "that explain complex drug discovery concepts and relationships "
                "to non-coding scientists."
            ),
            capabilities=[
                "knowledge_graph_generation",
                "concept_explanation",
                "relationship_mapping",
            ],
            tools=[
                "generate_knowledge_graph",
                "search_literature",
            ],
        )

    def _build_system_prompt(self) -> str:
        return """You are a Knowledge Graph specialist agent for drug discovery education.

Your expertise includes:
- Structuring complex biological and ML concepts into entity-relationship triplets
- Generating visual Mermaid.js flowcharts for scientific communication
- Explaining drug discovery pipelines to non-coding scientists

When generating knowledge graphs:
1. Break down the topic into Subject -> Predicate -> Object triplets
2. Use clear, concise labels (e.g., "PROTAC -> binds to -> Target Protein")
3. Include at least 5-8 triplets for meaningful graphs
4. Use the generate_knowledge_graph tool with the triplets list
5. Explain the graph's meaning in plain language after generating it

Format triplets as: "Subject -> Predicate -> Object"
Example: ["Drug A -> inhibits -> Protein B", "Protein B -> causes -> Disease C"]

Always use the generate_knowledge_graph tool — do not manually write Mermaid code.
After the graph, provide a 2-3 sentence explanation of what the graph shows."""
