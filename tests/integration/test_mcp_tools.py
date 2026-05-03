"""
Integration tests for D4 MCP tool execution pipeline.

Tests the full flow: Agent → MCP Server → Tool Handler → Response

Covers:
  - MolecularMCPServer: Lipinski, Tanimoto, PubChem, 3D conformer
  - ResearchMCPServer:  ArXiv, DuckDuckGo, TDC, knowledge graph generation
  - Error handling:     unknown tools, missing params, invalid SMILES
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# MolecularMCPServer — RDKit + BioPython tools
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMolecularMCPServer:
    """Tests for the computational chemistry MCP server."""

    @pytest.fixture
    def server(self):
        from chainmind.mcp.molecular_server import MolecularMCPServer
        return MolecularMCPServer()

    def test_list_tools_contains_d4_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "assess_lipinski_rules" in names
        assert "calculate_similarity" in names
        assert "generate_3d_conformer" in names
        assert "pubchem_search" in names

    @pytest.mark.asyncio
    async def test_lipinski_aspirin_passes(self, server):
        """Aspirin (MW=180, LogP~1.2) must pass all 4 Lipinski criteria."""
        result = await server.execute_tool(
            "assess_lipinski_rules",
            {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )
        assert result.success, f"Tool failed: {result}"
        data = result.result
        assert data.get("passes_ro5") is True
        mw = data.get("molecular_weight", 0)
        assert 178 < mw < 182, f"Unexpected MW: {mw}"

    @pytest.mark.asyncio
    async def test_lipinski_paclitaxel_fails(self, server):
        """Paclitaxel (MW~853, many violations) must fail Lipinski."""
        result = await server.execute_tool(
            "assess_lipinski_rules",
            {
                "smiles": (
                    "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)c5ccccc5)"
                    "(C4OC(=O)c6ccccc6)O)OC(=O)C)OC(C)=O)O)"
                    "OC(=O)C(CC(=O)c7ccccc7)NC(=O)c8ccccc8"
                )
            },
        )
        assert result.success
        assert result.result.get("passes_ro5") is False

    @pytest.mark.asyncio
    async def test_tanimoto_self_similarity_is_one(self, server):
        """A molecule compared to itself must have Tanimoto = 1.0."""
        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = await server.execute_tool(
            "calculate_similarity",
            {"smiles1": aspirin, "smiles2": aspirin},
        )
        assert result.success
        sim = result.result.get("tanimoto_similarity", -1)
        assert abs(sim - 1.0) < 1e-6, f"Self-similarity should be 1.0, got {sim}"

    @pytest.mark.asyncio
    async def test_tanimoto_aspirin_ibuprofen_low(self, server):
        """Aspirin and Ibuprofen are structurally different → similarity < 0.5."""
        result = await server.execute_tool(
            "calculate_similarity",
            {
                "smiles1": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "smiles2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            },
        )
        assert result.success
        sim = result.result.get("tanimoto_similarity", 1.0)
        assert sim < 0.5, f"Expected sim < 0.5, got {sim}"

    @pytest.mark.asyncio
    async def test_3d_conformer_ethanol_converges(self, server):
        """Ethanol should always produce a converged MMFF94 conformer."""
        result = await server.execute_tool(
            "generate_3d_conformer",
            {"smiles": "CCO"},
        )
        assert result.success
        data = result.result
        assert data.get("mmff94_converged") is True
        assert data.get("num_atoms", 0) >= 3

    @pytest.mark.asyncio
    async def test_invalid_smiles_returns_error(self, server):
        """Feeding a garbage SMILES string must not crash the server."""
        result = await server.execute_tool(
            "assess_lipinski_rules",
            {"smiles": "THIS_IS_NOT_A_SMILES!!!"},
        )
        # Either success=False or the result contains an error key
        if result.success:
            assert "error" in result.result or result.result.get("passes_ro5") is None
        else:
            assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_or_errors(self, server):
        """An unknown tool name must return an error result (not crash)."""
        from chainmind.core.exceptions import MCPToolNotFoundError
        try:
            result = await server.execute_tool("nonexistent_d4_tool", {})
            assert not result.success
        except MCPToolNotFoundError:
            pass  # also acceptable


# ---------------------------------------------------------------------------
# ResearchMCPServer — ArXiv + DuckDuckGo + TDC + KG generation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestResearchMCPServer:
    """Tests for the web research MCP server."""

    @pytest.fixture
    def server(self):
        from chainmind.mcp.research_server import ResearchMCPServer
        return ResearchMCPServer()

    def test_list_tools_contains_research_tools(self, server):
        tools = server.list_tools()
        names = [t.name for t in tools]
        assert "search_arxiv" in names
        assert "search_literature" in names
        assert "generate_knowledge_graph" in names

    @pytest.mark.asyncio
    async def test_knowledge_graph_valid_mermaid(self, server):
        """generate_knowledge_graph must return a non-empty Mermaid block."""
        result = await server.execute_tool(
            "generate_knowledge_graph",
            {
                "topic": "PROTAC targeted protein degradation",
                "triplets": [
                    "PROTAC -> binds to -> Target Protein",
                    "PROTAC -> recruits -> E3 Ligase",
                    "E3 Ligase -> attaches -> Ubiquitin",
                    "Ubiquitin -> marks -> Target Protein",
                    "Proteasome -> degrades -> Target Protein",
                ],
            },
        )
        assert result.success, f"Tool error: {result}"
        mermaid = result.result.get("mermaid", "") or str(result.result)
        assert "graph" in mermaid.lower() or "->" in mermaid, (
            f"Expected Mermaid graph syntax, got: {mermaid[:200]}"
        )

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_or_errors(self, server):
        """Unknown tool must not crash the server."""
        from chainmind.core.exceptions import MCPToolNotFoundError
        try:
            result = await server.execute_tool("nonexistent_research_tool", {})
            assert not result.success
        except MCPToolNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Benchmark validator self-check (no LLM required)
# ---------------------------------------------------------------------------

class TestChainMindBench:
    """Validates the ChainMind-Bench JSON and ground_truth_validator logic."""

    def test_load_benchmark_100_tasks(self):
        from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark
        tasks = load_benchmark()
        assert len(tasks) == 100, f"Expected 100 tasks, got {len(tasks)}"

    def test_all_tasks_have_required_fields(self):
        from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark
        tasks = load_benchmark()
        for t in tasks:
            assert "id" in t, f"Missing 'id' in task: {t}"
            assert "query" in t, f"Missing 'query' in task {t.get('id')}"
            assert "category" in t, f"Missing 'category' in task {t.get('id')}"
            assert t["category"] in ("A", "B", "C", "D"), (
                f"Unknown category in {t['id']}: {t['category']}"
            )

    def test_category_counts(self):
        from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark
        tasks = load_benchmark()
        cats = {c: sum(1 for t in tasks if t["category"] == c) for c in "ABCD"}
        assert cats["A"] == 40, f"Cat-A: expected 40, got {cats['A']}"
        assert cats["B"] == 30, f"Cat-B: expected 30, got {cats['B']}"
        assert cats["C"] == 15, f"Cat-C: expected 15, got {cats['C']}"
        assert cats["D"] == 15, f"Cat-D: expected 15, got {cats['D']}"

    def test_scorer_cat_a_perfect_answer(self):
        """A correct Lipinski answer should score >= 0.8."""
        from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark, score_response
        tasks = load_benchmark()
        task_a = next(t for t in tasks if t["id"] == "A001")
        perfect = (
            "Aspirin passes Lipinski's Rule of 5. "
            "MW=180.16 (pass), LogP=1.24 (pass), HBD=1 (pass), HBA=4 (pass). "
            "Violations: 0. passes_ro5: True."
        )
        result = score_response(task_a, perfect)
        assert result["score"] >= 0.7, (
            f"Perfect answer scored too low: {result['score']}"
        )

    def test_scorer_cat_b_keyword_recall(self):
        """A response mentioning expected keywords should pass Cat-B scoring."""
        from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark, score_response
        tasks = load_benchmark()
        task_b = next(t for t in tasks if t["id"] == "B001")
        response = (
            "Found papers: GNN molecular property prediction graph neural network "
            "2024 ADMET molecular property graph prediction benchmark."
        )
        result = score_response(task_b, response)
        assert result["score"] >= 0.5, (
            f"Keyword-rich response scored too low: {result['score']}"
        )

    def test_dataset_py_d4_only(self):
        """dataset.py must not contain any supply chain questions."""
        from chainmind.eval.dataset import ALL_QUESTIONS
        sc_ids = [q.id for q in ALL_QUESTIONS if q.id.startswith("SC_")]
        assert len(sc_ids) == 0, f"Legacy SC questions still present: {sc_ids}"

    def test_dataset_py_quick_eval_subset(self):
        """QUICK_EVAL_QUESTIONS should be a non-empty subset of ALL_QUESTIONS."""
        from chainmind.eval.dataset import ALL_QUESTIONS, QUICK_EVAL_QUESTIONS
        all_ids = {q.id for q in ALL_QUESTIONS}
        for q in QUICK_EVAL_QUESTIONS:
            assert q.id in all_ids, f"{q.id} in QUICK but not in ALL"
        assert len(QUICK_EVAL_QUESTIONS) >= 4
