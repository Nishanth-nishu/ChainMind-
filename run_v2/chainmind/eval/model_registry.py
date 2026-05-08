"""
ChainMind Multi-Model Registry — Open-Source LLM Configurations

Research-validated model selection for NeurIPS AI4Science drug discovery benchmarks.
All models are free, open-source, and fit within 24GB VRAM (RTX 3090/A10/A100).

Model rationale (based on 2025–2026 research landscape):
  1. Qwen2.5-7B-Instruct    — Strong general instruction following, our current baseline
  2. Llama-3.1-8B-Instruct  — Meta's best open 8B, widely used research baseline
  3. DeepSeek-R1-Distill-Qwen-7B  — Reasoning-distilled from R1, excellent on multi-step
  4. BioMistral-7B          — Biomedical fine-tuned Mistral, domain-specific upper bound
  5. Phi-3.5-mini-instruct  — Microsoft 3.8B, extreme efficiency for cost-accuracy Pareto
     
These 5 × (direct | chainmind) = 10 system configs → publishable multi-model Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Full configuration for a single vLLM-servable model."""
    # Registry key (used in CLI --model flags)
    key: str
    # HuggingFace model ID
    hf_id: str
    # Short display name for tables
    display_name: str
    # vLLM served-model-name (used in API calls)
    served_name: str
    # Port to serve on (each model gets a unique port to allow hot-swap testing)
    port: int
    # vLLM launch kwargs
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    gpu_mem_util: float = 0.90
    max_num_seqs: int = 32
    trust_remote_code: bool = True
    # Research metadata
    params_b: float = 7.0          # Parameter count in billions
    domain: str = "general"        # general | biomedical | reasoning | efficient
    citation: str = ""
    hf_license: str = "Apache-2.0"
    # Benchmark-specific notes
    notes: str = ""


# =============================================================================
# Registered models (research-validated selection)
# =============================================================================

MODEL_REGISTRY: dict[str, ModelConfig] = {

    # ── 1. Qwen2.5-7B-Instruct — Current ChainMind baseline ─────────────────
    "qwen2.5-7b": ModelConfig(
        key="qwen2.5-7b",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5-7B",
        served_name="chainmind-qwen",
        port=8100,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_mem_util=0.92,
        params_b=7.0,
        domain="general",
        hf_license="Apache-2.0",
        citation="Qwen2.5 Technical Report, team Qwen, 2024",
        notes="Strong instruction following; 128K token context; default ChainMind backbone.",
    ),

    # ── 2. Llama-3.1-8B-Instruct — Meta standard research baseline ───────────
    "llama3.1-8b": ModelConfig(
        key="llama3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B",
        served_name="chainmind-llama",
        port=8101,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_mem_util=0.90,
        params_b=8.0,
        domain="general",
        hf_license="Meta Llama 3.1 Community License",
        citation="Dubey et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024",
        notes="De-facto open-source research baseline. Required for credible comparison.",
    ),

    # ── 3. DeepSeek-R1-Distill-Qwen-7B — Reasoning specialist ───────────────
    "deepseek-r1-7b": ModelConfig(
        key="deepseek-r1-7b",
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        display_name="DeepSeek-R1-7B",
        served_name="chainmind-deepseek",
        port=8102,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_mem_util=0.90,
        params_b=7.0,
        domain="reasoning",
        hf_license="MIT",
        citation="DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning. arXiv:2501.12948, 2025",
        notes=(
            "Chain-of-thought reasoning distilled from DeepSeek-R1-671B. "
            "Expected to excel on Cat-D multi-step chains. Key hypothesis: "
            "reasoning distillation > raw scale for drug discovery workflows."
        ),
    ),

    # ── 4. BioMistral-7B — Biomedical domain upper bound ────────────────────
    "biomistral-7b": ModelConfig(
        key="biomistral-7b",
        hf_id="BioMistral/BioMistral-7B",
        display_name="BioMistral-7B",
        served_name="chainmind-biomistral",
        port=8103,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_mem_util=0.88,
        params_b=7.0,
        domain="biomedical",
        hf_license="Apache-2.0",
        citation=(
            "Labrak et al. BioMistral: A Collection of Open-Source Pretrained LLMs "
            "for Medical Domains. arXiv:2402.10373, 2024"
        ),
        notes=(
            "Fine-tuned on PubMed Central. Provides the domain-specialization "
            "upper bound: does PEFT on biomedical text beat general instruction "
            "tuning + MCP tool augmentation (ChainMind)?"
        ),
    ),

    # ── 5. Phi-3.5-mini — Efficiency / cost-accuracy Pareto ────────────────
    "phi3.5-mini": ModelConfig(
        key="phi3.5-mini",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        display_name="Phi-3.5-mini (3.8B)",
        served_name="chainmind-phi",
        port=8104,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_mem_util=0.70,   # 3.8B fits with much lower utilization
        max_num_seqs=64,     # Higher concurrency given headroom
        params_b=3.8,
        domain="efficient",
        hf_license="MIT",
        citation=(
            "Abdin et al. Phi-3 Technical Report. arXiv:2404.14219, 2024"
        ),
        notes=(
            "3.8B parameter model with strong reasoning-per-FLOP. "
            "Anchors the cost-accuracy Pareto frontier: if ChainMind + Phi-3.5 "
            "matches BioMistral-7B accuracy at 54% the parameter count, "
            "that's a publishable efficiency result."
        ),
    ),
}


# =============================================================================
# Bench system configs (model × tool-use = 10 systems)
# =============================================================================

def get_bench_systems() -> dict[str, dict]:
    """
    Returns a dict of system_key → SystemConfig kwargs for bench_runner.

    Naming: {model_key}_direct | {model_key}_chainmind
    """
    systems = {}
    for mkey, mcfg in MODEL_REGISTRY.items():
        # Direct (no tools — parametric knowledge only)
        systems[f"{mkey}_direct"] = {
            "name": f"{mcfg.display_name} (direct)",
            "use_orchestrator": False,
            "use_tools": False,
            "llm_backend": "local",
            "vllm_port": mcfg.port,
            "served_name": mcfg.served_name,
        }
        # ChainMind (full MCP + A2A)
        systems[f"{mkey}_chainmind"] = {
            "name": f"ChainMind ({mcfg.display_name})",
            "use_orchestrator": True,
            "use_tools": True,
            "llm_backend": "local",
            "vllm_port": mcfg.port,
            "served_name": mcfg.served_name,
        }
    return systems


def get_model(key: str) -> ModelConfig:
    """Retrieve a ModelConfig by key, raise KeyError if not found."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{key}' not in registry. Available: {available}")
    return MODEL_REGISTRY[key]


def list_models() -> None:
    """Print a formatted model table."""
    print(f"\n{'='*80}")
    print(f"  ChainMind Multi-Model Registry ({len(MODEL_REGISTRY)} models)")
    print(f"{'='*80}")
    print(f"  {'Key':<20} {'Display':<22} {'Params':<8} {'Domain':<12} {'Port'}")
    print(f"  {'-'*20} {'-'*22} {'-'*8} {'-'*12} {'-'*6}")
    for key, cfg in MODEL_REGISTRY.items():
        print(f"  {key:<20} {cfg.display_name:<22} {cfg.params_b:<8.1f} {cfg.domain:<12} {cfg.port}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    list_models()
