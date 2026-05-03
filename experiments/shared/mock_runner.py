"""
experiments/shared/mock_runner.py
Run any experiment with the MockLLMProvider (no vLLM required).

Usage:
    python experiments/shared/mock_runner.py --exp 4 --n 20
    python experiments/shared/mock_runner.py --all --n 20
    python experiments/shared/mock_runner.py --all --mode full
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPERIMENTS = {
    "1": ("exp001_reflexion",          "Exp001", "ReflexionAgent"),
    "2": ("exp002_self_consistency",   "Exp002", "SelfConsistencyAgent"),
    "3": ("exp003_cove",               "Exp003", "CoVeAgent"),
    "4": ("exp004_few_shot",           "Exp004", "FewShotChemistAgent"),
    "5": ("exp005_tool_rag",           "Exp005", "ToolRAGAgent"),
    "6": ("exp006_structured_output",  "Exp006", "StructuredOutputAgent"),
    "7": ("exp007_chem_rag",           "Exp007", "ChemRAGAgent"),
    "8": ("exp008_debate",             "Exp008", "DebateOrchestrator"),
}


def build_mock_orchestrator(exp_num: str, exp_module_name: str):
    """Wire specialist directly (bypasses OrchestratorAgent for clean benchmarking)."""
    from experiments.shared.mock_provider import MockLLMProvider
    from chainmind.mcp.molecular_server import MolecularMCPServer
    from chainmind.mcp.research_server import ResearchMCPServer

    mock_provider = MockLLMProvider()

    class MockRouter:
        async def generate(self, request):
            # Fresh provider per call to reset step counter
            p = MockLLMProvider()
            return await p.generate(request)
        async def generate_structured(self, request, schema):
            p = MockLLMProvider()
            return await p.generate(request)
        async def stream(self, request):
            p = MockLLMProvider()
            r = await p.generate(request)
            yield r.content

    router = MockRouter()
    mol = MolecularMCPServer()
    res = ResearchMCPServer()

    agent_mod = importlib.import_module(f"experiments.{exp_module_name}.agent")

    agent_classes = {
        "1": "ReflexionAgent",
        "2": "SelfConsistencyAgent",
        "3": "CoVeAgent",
        "4": "FewShotChemistAgent",
        "5": "ToolRAGAgent",
        "6": "StructuredOutputAgent",
        "7": "ChemRAGAgent",
    }

    if exp_num == "8":
        from chainmind.agents.specialists import ComputationalChemistAgent
        from experiments.exp008_debate.agent import DebateOrchestrator
        a = ComputationalChemistAgent(llm_router=router, mcp_servers=[mol])
        b = ComputationalChemistAgent(llm_router=router, mcp_servers=[mol])
        return DebateOrchestrator(agent_a=a, agent_b=b, llm_router=router)

    AgentClass = getattr(agent_mod, agent_classes[exp_num])
    # Return specialist directly — benchmarks test specialist, not orchestration
    return AgentClass(llm_router=router, mcp_servers=[mol])


async def run_one(exp_num: str, tasks: list, mode: str):
    exp_dir, _, _ = EXPERIMENTS[exp_num]
    run_mod = importlib.import_module(f"experiments.{exp_dir}.run")
    from experiments.shared.base_experiment import BaseExperiment as _Base
    ExpClass = next(
        v for k, v in vars(run_mod).items()
        if isinstance(v, type) and issubclass(v, _Base) and v is not _Base
    )
    exp = ExpClass()

    # Override build_orchestrator to use mock
    exp.build_orchestrator = lambda settings: build_mock_orchestrator(exp_num, exp_dir)

    result = await exp.run(tasks, settings=None, mode=mode)
    result.save()
    return result


async def run_all(tasks: list, mode: str, exps: list[str]):
    results = []
    for exp_num in exps:
        try:
            r = await run_one(exp_num, tasks, mode)
            results.append(r)
        except Exception as e:
            print(f"  ERROR in EXP{exp_num}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Mock-mode experiment runner (no vLLM)")
    parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()), help="Run single exp")
    parser.add_argument("--all", action="store_true", help="Run all 8 experiments")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--category", choices=["A", "B", "C", "D", "all"], default="all")
    args = parser.parse_args()

    from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark
    tasks = load_benchmark()
    if args.category != "all":
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.mode == "sample":
        tasks = tasks[:args.n]

    exps = list(EXPERIMENTS.keys()) if args.all else ([args.exp] if args.exp else ["4"])

    print(f"\nRunning {len(exps)} experiment(s) | mode={args.mode} | n={len(tasks)} tasks")
    asyncio.run(run_all(tasks, args.mode, exps))

    # Print comparison
    print("\n" + "="*60)
    print("  Comparison table:")
    import subprocess
    subprocess.run([sys.executable, "scripts/compare_experiments.py"], check=False)


if __name__ == "__main__":
    main()
