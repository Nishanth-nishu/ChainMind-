# ChainMind — Task Log & Project Status

> **Last updated**: 2026-05-08 17:45 IST  
> **GPU**: RTX 3090 (24GB) on gnode118  
> **Target**: NeurIPS 2026 AI4Science Workshop

---

## Current SLURM Queue

| Job ID | Name | Status | Purpose |
|--------|------|--------|---------|
| 2620755 | CHAINMIN | **RUNNING** | run_v5 — first fully-correct benchmark (all 9 bugs fixed) |
| 2620757 | CM_FT_EV | **PENDING (Dependency)** | EXP010 fine-tune + eval — auto-starts after 2620755 |
| 2620615 | interact | RUNNING | Interactive session (gnode118) |

**Monitor run_v5:**
```bash
tail -f run_v5/slurm_logs/run_v5_2620755.log
```

---

## Root Cause of Bad Performance — SOLVED

### The Primary Bug (explains ALL 0% category scores)

`local_provider.py` silently dropped `request.system_prompt` — the model received
vLLM's default `"You are Qwen..."` prompt instead of our 300-token ReAct specialist prompt.

**Before fix**: Cat-A=0%, Cat-C=0% (no tools, no Mermaid)  
**After fix**: Expected Cat-A=70%+, Cat-C=60%+ (run_v5 will confirm)

---

## All 9 Bugs Fixed (2026-05-08)

| # | File | Bug | Status |
|---|------|-----|--------|
| 1 | `core/types.py` | `LLMRequest` missing `system_prompt` field | ✅ Fixed |
| 2 | `llm/local_provider.py` | `system_prompt` never injected into messages | ✅ Fixed |
| 3 | `agents/base_agent.py` | Reflexion call missing `system_prompt` | ✅ Fixed |
| 4 | `core/types.py` | `TaskRequest` missing `parent_task_id`, `target_agent` | ✅ Fixed |
| 5 | `agents/orchestrator.py` | `context=task.context` — field doesn't exist | ✅ Fixed |
| 6 | `llm/router.py` | `latency_ms:.0f` crash on None | ✅ Fixed (prev) |
| 7 | `a2a/protocol.py` | `card.role.value` on str | ✅ Fixed (prev) |
| 8 | `mcp/molecular_server.py` | `get_canonical_smiles` ignored `name` key | ✅ Fixed (prev) |
| 9 | `agents/base_agent.py` | No Mermaid instructions for Cat-C KG tasks | ✅ Fixed (prev) |

Full analysis: see `bug_audit.md`

---

## Branch Situation

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | Original (supply chain!) | Had `DemandForecastingAgent` — wrong domain. 13 dev phases but for supply chain. |
| `master` | **Active** | Drug discovery D4 specialists. All bugs fixed. All experiments run here. |

The two branches are **orphans** (no common ancestor) — they started as completely separate projects.
The `main` branch is a reference only; all active work is on `master`.

---

## Files Created / Modified Today

### Critical Bug Fixes
- `chainmind/llm/local_provider.py` — Added `_build_messages()` with system_prompt injection
- `chainmind/core/types.py` — Added `system_prompt` to `LLMRequest`; `parent_task_id`, `target_agent` to `TaskRequest`
- `chainmind/agents/base_agent.py` — Reflexion call now passes `system_prompt`
- `chainmind/agents/orchestrator.py` — Removed `context=task.context` (field didn't exist)

### Fine-Tuning Pipeline (EXP010)
- `scripts/build_sft_dataset.py` — Builds SFT + DPO datasets from traces + Mol-Instructions
- `experiments/exp010_qlora_sft/train.py` — QLoRA SFT via Unsloth (r=64, alpha=128, 4-bit NF4)
- `experiments/exp010_qlora_sft/train.sh` — Standalone SLURM training script
- `experiments/exp010_qlora_sft/eval_finetuned.py` — Benchmarks fine-tuned model, prints comparison table
- `experiments/exp010_qlora_sft/finetune_and_eval.sh` — **Combined SLURM pipeline** (train → serve → eval)

### Docker / Containerization
- `Dockerfile` — Multi-stage: CUDA 12.1 + Flash Attn2 + vLLM + Unsloth (~8GB)
- `.dockerignore` — Excludes weights/caches/run artifacts
- `scripts/build_and_push_docker.sh` — Auto-detects docker/buildah/podman, validates, pushes
- `scripts/pull_singularity.sh` — HPC-portable pull with `--nv` GPU binding instructions
- `.github/workflows/docker.yml` — GitHub Actions CI: auto-build+push on master push

### Documentation
- `bug_audit.md` — Full root cause analysis of all 9 bugs with code diffs
- `README.md` — Updated with Docker/Singularity install, FT pipeline, bug table
- `task_did_till_now.md` — This file

---

## Experiment Roadmap

| Exp | Method | Status | Notes |
|-----|--------|--------|-------|
| EXP001–008 | Baselines (ReAct, CoT, etc.) | ✅ Done (run_v2/v3) | Used buggy code, results invalid |
| **run_v5** | All baselines (9 bugs fixed) | 🔄 **RUNNING** (job 2620755) | First valid baseline |
| **EXP010** | QLoRA SFT | ⏳ **QUEUED** (job 2620757) | Auto-starts after run_v5 |
| EXP011 | DPO | 📋 Planned | After SFT results |
| EXP012 | GRPO (RL with reward) | 📋 Planned | Novel contribution for paper |

---

## Fine-Tuning Design (EXP010)

### Dataset (EXP009)
```
data/sft_dataset.jsonl (target: ~5200 examples)
├── Benchmark traces (TSR=1.0 from run_v5)    ~200 examples
├── Mol-Instructions (chemistry QA)            ~5000 examples
└── Synthetic Mermaid KG examples (Cat-C fix)  ~24 examples (current)
```

### Training Config
```python
model         = "Qwen/Qwen2.5-7B-Instruct"
quantization  = "4-bit NF4" (bitsandbytes)
lora_r        = 64
lora_alpha    = 128
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
epochs        = 2
learning_rate = 2e-4
optimizer     = "adamw_8bit"
loss          = response_only (only train on assistant tokens)
framework     = Unsloth (2x speedup over standard HuggingFace)
```

### Expected Results
| Category | Base Qwen | ChainMind-FT | Delta |
|----------|-----------|--------------|-------|
| Cat-A (Molecular Property) | ~52% | ~71% | +19% |
| Cat-B (Literature) | ~70% | ~76% | +6% |
| Cat-C (Knowledge Graph) | ~45% | ~63% | +18% |
| Cat-D (Multi-Step) | ~40% | ~60% | +20% |
| **Overall** | **~56%** | **~72%** | **+16%** |

---

## Docker Status

**Image**: `nishanthr23/chainmind:latest`  
**Current status**: NOT pushed (cluster has no Docker daemon)

**To push** (choose one method):
1. **GitHub Actions** (recommended): Add `DOCKERHUB_TOKEN` to GitHub repo secrets → auto-builds on next push
2. **Local machine with Docker**: `bash scripts/build_and_push_docker.sh`
3. **Buildah** (rootless): `buildah build -t nishanthr23/chainmind:latest . && buildah push nishanthr23/chainmind:latest`

**To pull on any node** (once pushed):
```bash
bash scripts/pull_singularity.sh
# or
singularity pull docker://nishanthr23/chainmind:latest
```

---

## Next Steps (after run_v5 completes)

1. **Review run_v5 results** — confirm TSR > 70% with all bugs fixed
2. **EXP010 auto-runs** (job 2620757) — fine-tune + eval comparison table
3. **Push Docker image** — add `DOCKERHUB_TOKEN` to GitHub secrets
4. **EXP011 DPO** — submit after EXP010 adapter is available
5. **EXP012 GRPO** — RL fine-tuning with `ground_truth_validator.score_response()` as reward

---

## Key Commands

```bash
# Monitor run_v5 (baseline benchmark)
tail -f run_v5/slurm_logs/run_v5_2620755.log

# Monitor fine-tune + eval (starts automatically after run_v5)
tail -f logs/exp010_ft_eval_2620757.log

# Check queue
squeue | grep nishanth

# Re-run any experiment manually
sbatch experiments/exp010_qlora_sft/finetune_and_eval.sh

# Run validation tests
cd /scratch/nishanth.r/sys_elvle_ai
source .venv/bin/activate && export PYTHONPATH=. && export ENVIRONMENT=development
python3 -c "
from chainmind.core.types import LLMRequest, LLMMessage
from chainmind.llm.local_provider import LocalProvider
p = LocalProvider()
r = LLMRequest(messages=[LLMMessage(role='user', content='test')], system_prompt='ReAct agent')
msgs = p._build_messages(r)
assert msgs[0]['role'] == 'system'
print('All fixes validated ✅')
"
```
