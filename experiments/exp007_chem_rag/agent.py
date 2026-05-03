"""
EXP007 — Chemistry-RAG Pre-Context
Paper: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
       NeurIPS 2020. https://arxiv.org/abs/2005.11401
       + Applied to chemistry domain (cf. ChemCrow, Bran et al. 2023)

Hypothesis: Before the specialist's ReAct loop, retrieve 2-3 relevant chemistry
knowledge chunks (Lipinski definitions, SMILES conventions, RDKit output formats)
from a local BM25 index. Prepending this grounding context reduces misinterpretation
of tool outputs and malformed SMILES inputs.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import LLMMessage


# ---------------------------------------------------------------------------
# Static chemistry knowledge base (BM25-indexed at runtime)
# ---------------------------------------------------------------------------

CHEM_KNOWLEDGE = [
    {
        "id": "lipinski",
        "text": (
            "Lipinski's Rule of Five: A drug is orally bioavailable if MW ≤ 500 Da, "
            "LogP ≤ 5, H-bond donors (HBD) ≤ 5, H-bond acceptors (HBA) ≤ 10. "
            "Molecules violating 2+ rules are generally not orally bioavailable. "
            "assess_lipinski_rules(smiles) returns: molecular_weight, logP, h_bond_donors, "
            "h_bond_acceptors, passes_ro5, violations."
        ),
    },
    {
        "id": "smiles",
        "text": (
            "SMILES (Simplified Molecular Input Line Entry System) encodes molecules as strings. "
            "Aromatic atoms: lowercase c, n, o. Branches: parentheses. Rings: numbers. "
            "Double bonds: =. Triple bonds: #. Stereochemistry: @ and /. "
            "Always validate SMILES before computing properties — invalid SMILES causes RDKit errors."
        ),
    },
    {
        "id": "tanimoto",
        "text": (
            "Tanimoto similarity measures structural overlap between two molecules via fingerprints. "
            "Morgan fingerprints (radius=2, 2048 bits) are standard for drug-likeness comparison. "
            "Score range: 0 (no overlap) to 1 (identical). "
            ">0.85 = very similar (likely same scaffold). 0.4-0.85 = related. <0.4 = dissimilar. "
            "calculate_similarity(smiles1, smiles2) returns: tanimoto_similarity (float 0-1)."
        ),
    },
    {
        "id": "pubchem",
        "text": (
            "PubChem is NCBI's free chemical database. pubchem_search(query, namespace='name') "
            "returns: cid (PubChem compound ID), iupac_name, molecular_formula, molecular_weight. "
            "Use namespace='name' for drug names and namespace='smiles' for SMILES queries. "
            "Common drugs: Aspirin CID=2244, Ibuprofen CID=3672, Caffeine CID=2519."
        ),
    },
    {
        "id": "conformer",
        "text": (
            "3D conformer generation: ETKDG algorithm places atoms in 3D space, then MMFF94 "
            "force field optimizes geometry. Returns: molecular_formula, num_atoms (with H), "
            "mmff94_converged (bool), coordinates_xyz. "
            "Convergence=True means the geometry is physically valid. "
            "generate_3d_conformer(smiles) returns these fields."
        ),
    },
    {
        "id": "mermaid",
        "text": (
            "Mermaid.js flowchart syntax: graph TD for top-down. Nodes: A[Label]. "
            "Edges: A --> B or A -- label --> B. "
            "generate_knowledge_graph(triplets) accepts a list of strings: "
            "'Subject -> Predicate -> Object'. Returns a ```mermaid code block."
        ),
    },
    {
        "id": "arxiv_search",
        "text": (
            "search_arxiv(query, max_results=5) searches ArXiv preprint server. "
            "Returns: list of {title, authors, abstract, arxiv_id, url, published}. "
            "Best queries: specific technical terms + year filter like '2024'. "
            "Drug discovery topics: ADMET, molecular property prediction, de novo design, "
            "protein-ligand docking, graph neural network, diffusion model, SMILES transformer."
        ),
    },
    {
        "id": "tdc_benchmarks",
        "text": (
            "TDC (Therapeutics Data Commons) provides ADMET benchmark datasets. "
            "fetch_tdc_benchmark(task_name) returns SOTA model, metric, score, task description. "
            "Key tasks: Caco2_Wang (absorption, MAE), BBB_Martins (brain penetration, AUROC), "
            "HIA_Hou (intestinal absorption, AUROC), Solubility_AqSolDB (MAE), "
            "Lipophilicity_AstraZeneca (LogD, MAE), PPBR_AZ (plasma protein binding, MAE)."
        ),
    },
]


def _bm25_retrieve(query: str, docs: list[dict], top_k: int = 2) -> list[str]:
    """Lightweight BM25 retrieval over the chemistry knowledge base."""
    q_tokens = re.findall(r"\b\w+\b", query.lower())
    k1, b = 1.5, 0.75
    avg_dl = sum(len(re.findall(r"\b\w+\b", d["text"])) for d in docs) / max(len(docs), 1)

    scores = []
    for doc in docs:
        doc_tokens = re.findall(r"\b\w+\b", doc["text"].lower())
        dl = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0
        for t in q_tokens:
            tf = tf_map.get(t, 0)
            idf = math.log(1 + (len(docs) - sum(1 for d in docs if t in d["text"].lower()) + 0.5) /
                           (sum(1 for d in docs if t in d["text"].lower()) + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append(score)

    top_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [docs[i]["text"] for i in top_idx if scores[i] > 0]


class ChemRAGAgent(BaseAgent):
    """EXP007: Chemistry-RAG — retrieve domain knowledge before THINK steps."""

    def _build_think_prompt(
        self,
        messages: list[LLMMessage],
        tool_descriptions: str,
        memory_context: str,
        step: int,
    ) -> str:
        # Retrieve relevant chemistry context for the user query
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        retrieved = _bm25_retrieve(user_msg, CHEM_KNOWLEDGE, top_k=2)

        if retrieved:
            rag_block = (
                "\n## Chemistry Knowledge (retrieved context — use this to interpret tool output)\n"
                + "\n".join(f"• {chunk}" for chunk in retrieved)
                + "\n"
            )
        else:
            rag_block = ""

        base = super()._build_think_prompt(
            messages, tool_descriptions, memory_context, step
        )
        return base.replace(
            "## Conversation History",
            f"{rag_block}\n## Conversation History",
        )
