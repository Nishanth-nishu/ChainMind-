"""
experiments/shared/mock_provider.py
Deterministic mock LLM for dry-run testing of all 8 experiments.

The mock uses keyword matching to produce well-formed ReAct responses.
Key insight: since MCP tools (RDKit, PubChem, ArXiv) are REAL, the mock
TSR reflects true tool accuracy — identical to what a well-prompted LLM
would achieve. This validates the pipeline without needing vLLM.

Mock strategy per category:
  Cat-A (Molecular): call the correct molecular tool → FINAL_ANSWER from result
  Cat-B (Literature): call search_arxiv or fetch_tdc_benchmark → FINAL_ANSWER
  Cat-C (Knowledge Graph): call generate_knowledge_graph → FINAL_ANSWER
  Cat-D (Multi-Step): sequence of tool calls based on task type
"""
from __future__ import annotations

import re
from chainmind.core.interfaces import ILLMProvider
from chainmind.core.types import (
    LLMRequest, LLMResponse, TokenUsage,
)


# ---------------------------------------------------------------------------
# Routing rules: (pattern, tool_name, arg_extractor)
# ---------------------------------------------------------------------------

def _extract_smiles(text: str) -> str:
    """Pull a SMILES string from task query (quoted or after 'SMILES:')."""
    drugs = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
        "celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)OC(C(C(=O)c5ccccc5)(C(CC2(C(C1OC(=O)C)O)O)OC(=O)C)O)C)OC6CC(CC(C6OC(=O)C)(C)O)OC(=O)C)OC(=O)C)O)O4)C",
        "cyclosporine": "CC[C@@H]1OC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)1",
        "cyclosporin": "CC[C@@H]1OC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)NC(=O)1",
        "imatinib": "CC1=CN=CC(=C1)NC2=NC=CC(=N2)NC3=CC=C(C=C3)NC(=O)C4=CC=C(C=C4)CN5CCN(CC5)C",
        "morphine": "OC1=CC=C2CC3N(CCC34CC(O)C=C4)C2=C1",
        "penicillin": "CC1(C)SC2C(NC1=O)C(=O)O2",
        "metformin": "CN(C)C(=N)NC(=N)N",
        "lisinopril": "OCC(NC(=O)C(CCN)CCC(=O)O)C(=O)N1CCCC1C(=O)O",
        "atorvastatin": "CC(C)C1=C(C(=CC=C1)F)C2=CC=CC=C2.CC(=O)OCC(CC(CC(=O)O)O)NC(=O)C3=CC(=CC=C3F)F",
        "warfarin": "OC(=O)CCC(=O)C1C(=O)c2ccccc2OC1=O",
        "sildenafil": "CCCC1=NN(C2=CC=CC=C12)C3=NC(=O)C4=C(N3)N(CCO)N=NC4=O",
        "tamoxifen": "CCC(=C(C1=CC=CC=C1)C2=CC=C(OCCN(C)C)C=C2)C3=CC=CC=C3",
        "methotrexate": "CN(CC1=CN=C2N=CN=C(N)C2=N1)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
        "doxorubicin": "COC1=CC=CC2=C1CC(O)(CC(=O)CO)C3=C(O)C4=CC(=O)C(=C4C(=C3O)CC(O)CO)N",
        "taxol": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)OC(C(C(=O)C5=CC=CC=C5)NC(=O)OC(C)(C)C)C)OC(=O)C)OC(C)(C(CC2(C(C1OC(=O)C)O)O)OC(=O)C)O)O4)C)C",
        "theophylline": "Cn1cnc2N(C(=O)Nc12)C(=O)N",
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "sucrose": "OC[C@H]1OC(O[C@@H]2[C@@H](CO)OC(O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
    }
    # Check drug names FIRST (fast path for benchmark queries)
    text_lower = text.lower()
    for name, smi in drugs.items():
        if name in text_lower:
            return smi

    # Regex fallback: strip trailing unbalanced parentheses introduced by sentence structure
    m = re.search(r"SMILES[:\s]+([A-Za-z0-9@+\-\[\]()=#$.,/\\%{}]+)", text)
    if m:
        candidate = m.group(1).strip()
        # Balance parens: remove trailing ) if unbalanced
        while candidate.endswith(")") and candidate.count("(") < candidate.count(")"):
            candidate = candidate[:-1]
        if candidate:
            return candidate

    return "CC(=O)OC1=CC=CC=C1C(=O)O"  # Default: aspirin




def _extract_drug_name(text: str) -> str:
    """Extract a drug/compound name from query."""
    drugs = ["aspirin", "ibuprofen", "caffeine", "paracetamol", "acetaminophen",
             "imatinib", "morphine", "lisinopril", "atorvastatin", "warfarin",
             "sildenafil", "tamoxifen", "methotrexate", "doxorubicin", "metformin",
             "penicillin", "taxol", "theophylline"]
    text_lower = text.lower()
    for d in drugs:
        if d in text_lower:
            return d
    return "aspirin"


def _build_tool_call(tool: str, args: dict) -> str:
    import json
    args_str = json.dumps(args)
    return (
        f"THOUGHT: I need to call {tool} to answer this question accurately.\n"
        f"ACTION: {tool}\n"
        f"ACTION_INPUT: {args_str}"
    )


def _build_final_answer(observation: str, query: str) -> str:
    return (
        f"THOUGHT: Based on the tool results, I can now answer the question.\n"
        f"FINAL_ANSWER: {observation}"
    )


class MockLLMProvider(ILLMProvider):
    """
    Deterministic mock LLM for pipeline validation.
    Routes queries to the correct MCP tool and formats FINAL_ANSWER from observation.
    Network calls are real (RDKit, PubChem REST, ArXiv API).
    """

    def __init__(self):
        self._step = 0
        self._last_observation = ""

    def _route(self, query: str) -> str:
        """Pick the correct tool and arguments for this query.
        
        The 'query' may be the full think_prompt (not just the user question).
        Extract the actual user message from the conversation history section.
        """
        # Extract actual user query from conversation history block
        # The think_prompt embeds it as "[user]: <actual query>"
        user_match = re.search(r"\[user\]: ([^\[]+)", query)
        if user_match:
            q = user_match.group(1).strip().lower()
            q_full = user_match.group(1).strip()  # keep case for SMILES
        else:
            q = query.lower()
            q_full = query

        # Molecular property routing
        if any(k in q for k in ["lipinski", "rule of 5", "drug-like", "bioavailable"]):
            smiles = _extract_smiles(q_full)
            return _build_tool_call("assess_lipinski_rules", {"smiles": smiles})

        if any(k in q for k in ["molecular weight", "mol weight", "mw of", "weight of", "mass of"]):
            smiles = _extract_smiles(q_full)
            return _build_tool_call("assess_lipinski_rules", {"smiles": smiles})

        if any(k in q for k in ["tanimoto", "similarity", "similar"]):
            # Extract two drug names from the user query
            drugs = re.findall(r"\b(aspirin|ibuprofen|caffeine|theophylline|morphine|benzene|glucose)\b", q)
            s1 = _extract_smiles(drugs[0] if drugs else "aspirin")
            s2 = _extract_smiles(drugs[1] if len(drugs) > 1 else "ibuprofen")
            return _build_tool_call("calculate_similarity", {"smiles1": s1, "smiles2": s2})

        if any(k in q for k in ["pubchem", "cid", "compound id", "find in pubchem", "look up"]):
            name = _extract_drug_name(q_full)
            return _build_tool_call("pubchem_search", {"query": name, "namespace": "name"})

        if any(k in q for k in ["3d conformer", "3d structure", "generate conformer", "3d coordinate"]):
            smiles = _extract_smiles(q_full)
            return _build_tool_call("generate_3d_conformer", {"smiles": smiles})

        if any(k in q for k in ["logp", "log p", "partition coefficient", "lipophilicity"]):
            smiles = _extract_smiles(q_full)
            return _build_tool_call("assess_lipinski_rules", {"smiles": smiles})

        if any(k in q for k in ["h-bond donor", "hydrogen bond donor", "hbd", "hba", "h-bond acceptor"]):
            smiles = _extract_smiles(q_full)
            return _build_tool_call("assess_lipinski_rules", {"smiles": smiles})

        # Literature routing
        if any(k in q for k in ["tdc", "therapeutics data commons", "admet benchmark", "admet dataset"]):
            task_map = {
                "caco2": "Caco2_Wang", "bbb": "BBB_Martins", "hia": "HIA_Hou",
                "solubility": "Solubility_AqSolDB", "lipophilicity": "Lipophilicity_AstraZeneca",
                "ppbr": "PPBR_AZ", "cyp": "CYP3A4_Veith",
            }
            task = "Caco2_Wang"
            for k, v in task_map.items():
                if k in q:
                    task = v
                    break
            return _build_tool_call("fetch_tdc_benchmark", {"task_name": task})

        if any(k in q for k in ["arxiv", "recent paper", "research paper", "literature", "study", "published"]):
            # Extract search terms
            terms = re.findall(r"\b(admet|docking|molecular|diffusion|transformer|graph neural|binding|drug discovery|protein|scaffold)\b", q)
            search_q = " ".join(terms[:3]) if terms else "drug discovery machine learning 2024"
            return _build_tool_call("search_arxiv", {"query": search_q, "max_results": 5})

        if any(k in q for k in ["search", "web search", "find information", "duckduckgo"]):
            return _build_tool_call("search_literature", {"query": q_full[:100]})

        # Knowledge graph routing
        if any(k in q for k in ["knowledge graph", "graph", "triplet", "relationship", "mechanism", "pathway"]):
            # Generic pharmacology KG
            topic = re.search(r"(mechanism|pathway|absorption|distribution|metabolism|inhibitor|target|receptor)", q)
            t = topic.group(1) if topic else "drug discovery"
            triplets = [
                f"Drug -> inhibits -> {t}",
                f"{t} -> affects -> Biological target",
                f"Biological target -> expressed in -> Target tissue",
                f"Drug -> undergoes -> ADMET processes",
                f"ADMET -> includes -> Absorption",
                f"ADMET -> includes -> Distribution",
                f"ADMET -> includes -> Metabolism",
            ]
            return _build_tool_call("generate_knowledge_graph", {"triplets": triplets})

        # Default: search literature
        return _build_tool_call("search_literature", {"query": q_full[:80]})

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a scripted ReAct response based on the prompt context."""
        # Figure out the full prompt text
        full_content = " ".join(m.content for m in request.messages)

        # Detect if a tool observation is already in the conversation
        # The base_agent appends observations in the format:
        # Detect if a tool observation is already in the conversation.
        # Covers: base_agent format  →  "Tool 'X' returned: ..."
        #         EXP002 synthesis   →  "Tool Results:\n{...}"
        has_observation = (
            ("Tool '" in full_content and "returned:" in full_content)
            or "Tool Results:" in full_content
            or "based on these tool results" in full_content.lower()
        )

        if has_observation:
            # Try base_agent observation format first
            obs_match = re.search(
                r"Tool '[^']+' returned: (.+?)(?:THOUGHT:|Step \d|$)",
                full_content, re.DOTALL
            )
            if obs_match:
                observation = obs_match.group(1).strip()[:500]
            else:
                # EXP002 synthesis format: "Tool Results:\n{...}"
                obs_match2 = re.search(
                    r"Tool Results:\s*(.+?)(?:Provide|$)",
                    full_content, re.DOTALL
                )
                observation = (
                    obs_match2.group(1).strip()[:500]
                    if obs_match2
                    else full_content[-300:].strip()
                )
            self._last_observation = observation
            answer = _build_final_answer(observation, full_content[:200])
        else:
            # First step: route to correct tool based on user query
            # Extract user query — base_agent puts it in the last 'user' message
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break
            if not user_query:
                # Fallback: last 400 chars of the assembled content
                user_query = full_content[-400:]

            answer = self._route(user_query)

        return LLMResponse(
            content=answer,
            model="mock-deterministic-v1",
            provider="mock",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_ms=5.0,
        )


    async def health_check(self) -> bool:
        return True
