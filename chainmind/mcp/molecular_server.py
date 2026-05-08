"""
chainmind/mcp/molecular_server.py
MCP Server providing computational chemistry tools via RDKit and PubChem.

Tools:
  - assess_lipinski_rules     : Lipinski Rule of 5 (MW, LogP, HBD, HBA)
  - get_canonical_smiles      : PubChem name → canonical SMILES
  - calculate_tanimoto        : Morgan fingerprint Tanimoto similarity between two molecules
  - get_molecular_descriptors : Full descriptor set (MW, LogP, TPSA, RotBonds, rings)
  - validate_smiles           : Check if SMILES is valid and return basic info
"""
from __future__ import annotations

import json
from typing import Any

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)
from rdkit.DataStructs import TanimotoSimilarity

from chainmind.core.interfaces import IMCPServer
from chainmind.core.types import MCPToolDefinition, MCPToolResult


class MolecularMCPServer(IMCPServer):
    """MCP Server providing computational chemistry tools via RDKit and PubChem."""

    def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="assess_lipinski_rules",
                description=(
                    "Takes a SMILES string and returns Lipinski Rule of 5 (Ro5) "
                    "properties: MW, LogP, H-bond donors (HBD), H-bond acceptors (HBA), "
                    "and whether the molecule passes all four rules."
                ),
                parameters={"smiles": "The SMILES string of the molecule"},
            ),
            MCPToolDefinition(
                name="get_canonical_smiles",
                description=(
                    "Queries PubChem by compound name to retrieve the canonical SMILES string."
                ),
                parameters={"query": "The common name or IUPAC name of the drug or molecule"},
            ),
            MCPToolDefinition(
                name="calculate_tanimoto",
                description=(
                    "Calculates the Tanimoto (Jaccard) similarity between two molecules "
                    "using Morgan fingerprints (radius=2, nBits=2048). "
                    "Returns a value between 0.0 (completely different) and 1.0 (identical). "
                    "Both inputs must be valid SMILES strings."
                ),
                parameters={
                    "smiles1": "SMILES string of the first molecule",
                    "smiles2": "SMILES string of the second molecule",
                },
            ),
            MCPToolDefinition(
                name="get_molecular_descriptors",
                description=(
                    "Computes a comprehensive set of molecular descriptors from a SMILES string: "
                    "molecular weight, LogP, TPSA, rotatable bonds, ring count, "
                    "H-bond donors/acceptors, and heavy atom count. "
                    "Useful for ADMET property estimation and drug-likeness profiling."
                ),
                parameters={"smiles": "The SMILES string of the molecule"},
            ),
            MCPToolDefinition(
                name="validate_smiles",
                description=(
                    "Checks whether a SMILES string is chemically valid. "
                    "Returns the canonical SMILES, molecular formula, and atom count if valid."
                ),
                parameters={"smiles": "The SMILES string to validate"},
            ),
        ]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> MCPToolResult:
        try:
            if tool_name == "assess_lipinski_rules":
                return self._assess_lipinski(args)

            elif tool_name == "get_canonical_smiles":
                return await self._get_canonical_smiles(args)

            elif tool_name == "calculate_tanimoto":
                return self._calculate_tanimoto(args)

            elif tool_name == "get_molecular_descriptors":
                return self._get_molecular_descriptors(args)

            elif tool_name == "validate_smiles":
                return self._validate_smiles(args)

            else:
                return MCPToolResult(
                    result="",
                    success=False,
                    error=f"Unknown tool: '{tool_name}'. Available: assess_lipinski_rules, "
                          "get_canonical_smiles, calculate_tanimoto, get_molecular_descriptors, validate_smiles",
                )

        except Exception as e:
            return MCPToolResult(result="", success=False, error=str(e))

    # ── Tool implementations ──────────────────────────────────────────────────

    def _assess_lipinski(self, args: dict) -> MCPToolResult:
        smiles = args.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MCPToolResult(result="", success=False, error=f"Invalid SMILES: {smiles!r}")

        mw  = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        passes = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

        violations = []
        if mw  > 500: violations.append(f"MW={mw:.1f} > 500")
        if logp > 5:  violations.append(f"LogP={logp:.2f} > 5")
        if hbd  > 5:  violations.append(f"HBD={hbd} > 5")
        if hba  > 10: violations.append(f"HBA={hba} > 10")

        return MCPToolResult(result=json.dumps({
            "smiles": Chem.MolToSmiles(mol),
            "molecular_weight": round(mw, 2),
            "logP": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "passes_ro5": passes,
            "violations": violations,
            "verdict": "PASSES Lipinski Ro5" if passes else f"FAILS Lipinski Ro5: {'; '.join(violations)}",
        }), success=True)

    async def _get_canonical_smiles(self, args: dict) -> MCPToolResult:
        # Accept any of: query, name, compound (model sends different keys)
        query = (args.get("query") or args.get("name") or args.get("compound") or "").strip()
        if not query:
            return MCPToolResult(result="", success=False,
                                 error='No compound name provided. Pass {"query": "compound name"}')
        compounds = pcp.get_compounds(query, "name")
        if not compounds:
            return MCPToolResult(result="", success=False,
                                 error=f"Compound not found in PubChem: {query!r}")
        c = compounds[0]
        return MCPToolResult(result=json.dumps({
            "name": query,
            "smiles": c.canonical_smiles,
            "iupac_name": c.iupac_name,
            "cid": c.cid,
            "molecular_formula": c.molecular_formula,
        }), success=True)

    def _calculate_tanimoto(self, args: dict) -> MCPToolResult:
        smi1 = args.get("smiles1") or args.get("smiles_1") or args.get("smiles_a") or ""
        smi2 = args.get("smiles2") or args.get("smiles_2") or args.get("smiles_b") or ""

        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        if mol1 is None:
            return MCPToolResult(result="", success=False, error=f"Invalid SMILES (smiles1): {smi1!r}")
        if mol2 is None:
            return MCPToolResult(result="", success=False, error=f"Invalid SMILES (smiles2): {smi2!r}")

        # Morgan fingerprints: radius=2, nBits=2048 (standard for drug discovery)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        tanimoto = TanimotoSimilarity(fp1, fp2)

        # Qualitative interpretation
        if tanimoto >= 0.85:
            interpretation = "Very similar (likely same scaffold)"
        elif tanimoto >= 0.65:
            interpretation = "Moderately similar (related scaffold)"
        elif tanimoto >= 0.40:
            interpretation = "Low similarity (different scaffold)"
        else:
            interpretation = "Very dissimilar (unrelated compounds)"

        return MCPToolResult(result=json.dumps({
            "smiles1": Chem.MolToSmiles(mol1),
            "smiles2": Chem.MolToSmiles(mol2),
            "tanimoto_similarity": round(tanimoto, 4),
            "fingerprint": "Morgan (radius=2, nBits=2048)",
            "interpretation": interpretation,
        }), success=True)

    def _get_molecular_descriptors(self, args: dict) -> MCPToolResult:
        smiles = args.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MCPToolResult(result="", success=False, error=f"Invalid SMILES: {smiles!r}")

        formula = rdMolDescriptors.CalcMolFormula(mol)
        tpsa    = rdMolDescriptors.CalcTPSA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings   = rdMolDescriptors.CalcNumRings(mol)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        return MCPToolResult(result=json.dumps({
            "smiles": Chem.MolToSmiles(mol),
            "molecular_formula": formula,
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logP": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(tpsa, 2),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": rot_bonds,
            "ring_count": rings,
            "aromatic_rings": arom_rings,
            "heavy_atom_count": heavy_atoms,
        }), success=True)

    def _validate_smiles(self, args: dict) -> MCPToolResult:
        smiles = args.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MCPToolResult(result=json.dumps({
                "valid": False,
                "input_smiles": smiles,
                "error": "RDKit cannot parse this SMILES string",
            }), success=True)

        formula = rdMolDescriptors.CalcMolFormula(mol)
        return MCPToolResult(result=json.dumps({
            "valid": True,
            "input_smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_formula": formula,
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
        }), success=True)
