"""
EXP004 — Chemistry Few-Shot Prompting Agent
Paper: Bran et al., "ChemCrow: Augmenting Large Language Models with Chemistry Tools"
       NeurIPS AI4Science Workshop 2023. https://arxiv.org/abs/2304.05376

Hypothesis
----------
ChemCrow showed (with GPT-4) that domain-specific few-shot demonstrations
embedded in the system prompt dramatically reduce wrong tool selection and
malformed SMILES. For a 7B model (vs GPT-4), this effect is even larger.
Adding 4 curated (query → tool_call → result → answer) chemistry examples
directly into the specialist system prompt will teach the model:
  1. Exact THOUGHT / ACTION / ACTION_INPUT / FINAL_ANSWER format
  2. How to read RDKit tool output
  3. Valid SMILES → correct descriptor extraction

This is the EXPECTED HIGHEST GAIN experiment (+17-20% TSR on Cat-A/D).

Change from Baseline
--------------------
- _build_system_prompt(): appends 4 curated Q&A chemistry examples
- Temperature lowered to 0.1 for more deterministic format adherence
"""

from __future__ import annotations

from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import LLMRequest, LLMMessage, AgentCard
from chainmind.config.constants import AgentRole


# ---------------------------------------------------------------------------
# Curated chemistry few-shot examples (format matches exact ReAct template)
# ---------------------------------------------------------------------------

CHEM_FEW_SHOTS = """
## Worked Examples (Chemistry Tool Use)

### Example 1: Lipinski Assessment
User: Does Aspirin pass Lipinski's Rule of 5?
THOUGHT: I need to assess Aspirin's drug-likeness. I'll call assess_lipinski_rules with the SMILES.
ACTION: assess_lipinski_rules
ACTION_INPUT: {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
[Tool returns: {"molecular_weight": 180.16, "logP": 1.24, "h_bond_donors": 1, "h_bond_acceptors": 4, "passes_ro5": true, "violations": 0}]
THOUGHT: The tool returned MW=180.16 (<500✓), LogP=1.24 (<5✓), HBD=1 (≤5✓), HBA=4 (≤10✓). Zero violations.
FINAL_ANSWER: Yes, Aspirin passes Lipinski's Rule of 5 with 0 violations. MW=180.16 Da, LogP=1.24, H-bond donors=1, H-bond acceptors=4. It is drug-like.

### Example 2: Tanimoto Similarity
User: How similar are Caffeine and Theophylline?
THOUGHT: I need their SMILES strings and will call calculate_similarity.
ACTION: calculate_similarity
ACTION_INPUT: {"smiles1": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "smiles2": "Cn1cnc2N(C(=O)Nc12)C(=O)N"}
[Tool returns: {"tanimoto_similarity": 0.47, "method": "Morgan r=2 2048bit"}]
THOUGHT: The tool returned Tanimoto=0.47. Moderate similarity (0.4-0.6 range = structurally related).
FINAL_ANSWER: Caffeine and Theophylline have a Tanimoto similarity of 0.47 (Morgan fingerprints, radius=2). They are moderately similar — both are xanthine derivatives (shared purine core) but differ in N-methylation pattern.

### Example 3: PubChem Lookup
User: Find Imatinib in PubChem.
THOUGHT: I'll call pubchem_search to retrieve Imatinib's properties.
ACTION: pubchem_search
ACTION_INPUT: {"query": "Imatinib", "namespace": "name"}
[Tool returns: {"cid": 5291, "iupac_name": "4-[(4-methylpiperazin-1-yl)methyl]-N-[4-methyl-3-[(4-pyridin-3-ylpyrimidin-2-yl)amino]phenyl]benzamide", "molecular_formula": "C29H31N7O", "molecular_weight": 493.6}]
THOUGHT: Imatinib CID=5291, formula=C29H31N7O, MW=493.6 Da.
FINAL_ANSWER: Imatinib (Gleevec) found on PubChem: CID=5291, Molecular Formula=C29H31N7O, Molecular Weight=493.6 Da.

### Example 4: 3D Conformer Generation
User: Generate a 3D conformer for Ethanol.
THOUGHT: I'll call generate_3d_conformer with Ethanol's SMILES.
ACTION: generate_3d_conformer
ACTION_INPUT: {"smiles": "CCO"}
[Tool returns: {"molecular_formula": "C2H6O", "num_atoms": 9, "mmff94_converged": true, "coordinates_xyz": "..."}]
THOUGHT: MMFF94 converged successfully. Ethanol has 9 atoms (with hydrogens), formula C2H6O.
FINAL_ANSWER: Successfully generated a 3D conformer for Ethanol (SMILES: CCO). Formula: C2H6O, 9 atoms (with explicit H), MMFF94 force field converged: True.
"""


class FewShotChemistAgent(BaseAgent):
    """
    EXP004: Chemistry few-shot prompting (ChemCrow-style).

    System prompt includes 4 worked examples covering all Cat-A tool types.
    Temperature reduced to 0.1 for strict format adherence.
    """

    def _build_system_prompt(self) -> str:
        base = """You are a Computational Chemistry specialist for drug discovery.

Your job is to answer chemistry questions using the available tools.
NEVER guess or hallucinate molecular properties — always use the tools.

Rules:
1. Always call the appropriate tool before answering
2. Base your FINAL_ANSWER strictly on tool output
3. Report ALL numerical values returned by the tool
4. If SMILES is invalid, say so — do not fabricate properties
5. Use exact format: THOUGHT / ACTION / ACTION_INPUT / FINAL_ANSWER
"""
        return base + CHEM_FEW_SHOTS

    async def _think(self, messages, tool_descriptions, memory_context, step):
        """Lower temperature for more deterministic few-shot adherence."""
        from chainmind.core.types import LLMRequest, LLMMessage
        think_prompt = self._build_think_prompt(
            messages, tool_descriptions, memory_context, step
        )
        return await self._llm_router.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=think_prompt)],
                system_prompt=self._build_system_prompt(),
                temperature=0.1,   # Lower than baseline 0.3
                max_tokens=1024,
            )
        )
