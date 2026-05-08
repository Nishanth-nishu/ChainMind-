import json
from typing import Any
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from chainmind.core.interfaces import IMCPServer
from chainmind.core.types import MCPToolDefinition, MCPToolResult

class MolecularMCPServer(IMCPServer):
    """MCP Server providing computational chemistry tools via RDKit and PubChem."""
    
    def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="assess_lipinski_rules",
                description="Takes a SMILES string and returns Lipinski rule of 5 (Ro5) properties (MW, LogP, HBD, HBA) and whether it passes.",
                parameters={"smiles": "The SMILES string of the molecule"}
            ),
            MCPToolDefinition(
                name="get_canonical_smiles",
                description="Queries PubChem by name to get the canonical SMILES string.",
                parameters={"query": "The common name of the drug or molecule"}
            )
        ]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> MCPToolResult:
        try:
            if tool_name == "assess_lipinski_rules":
                smiles = args.get("smiles", "")
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return MCPToolResult(result="", success=False, error=f"Invalid SMILES string: {smiles}")
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                
                passes = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
                
                res = {
                    "smiles": smiles,
                    "molecular_weight": round(mw, 2),
                    "logP": round(logp, 2),
                    "hbd": hbd,
                    "hba": hba,
                    "passes_ro5": passes
                }
                return MCPToolResult(result=json.dumps(res), success=True)
                
            elif tool_name == "get_canonical_smiles":
                query = args.get("query", "")
                compounds = pcp.get_compounds(query, 'name')
                if not compounds:
                    return MCPToolResult(result="", success=False, error=f"Compound not found for query: {query}")
                return MCPToolResult(result=json.dumps({"name": query, "smiles": compounds[0].canonical_smiles}), success=True)
                
            else:
                return MCPToolResult(result="", success=False, error=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            return MCPToolResult(result="", success=False, error=str(e))
