# ChainMind Bug Audit & Root Cause Analysis
**Date**: 2026-05-08  
**Analyst**: Automated deep diff of main vs master + live log inspection

---

## Performance Gap Summary

| Run | TSR Cat-A | TSR Cat-B | TSR Cat-C | TSR Cat-D | Overall |
|-----|-----------|-----------|-----------|-----------|---------|
| Interactive (main branch) | ~75% | ~80% | ~60% | ~55% | ~79% best |
| SLURM run_v2 (master, 10 bugs) | 0% | 65% | 0% | 30% | ~60% best |
| SLURM run_v3 (9 bugs fixed) | ~? | ~70% | ~? | ~? | Pending |
| **SLURM run_v4 (ALL fixed)** | **Expected 70%+** | **Expected 75%+** | **Expected 60%+** | **Expected 60%+** | **Expected 78-85%** |

---

## Root Cause #1 — CRITICAL: `system_prompt` silently dropped (ALL categories broken)

**Severity**: 🔴 Critical — affects 100% of benchmark tasks  
**Files**: `chainmind/core/types.py`, `chainmind/llm/local_provider.py`  
**Discovered**: 2026-05-08, confirmed via git diff main→master

### What happened
```python
# base_agent.py (correct — builds ReAct specialist prompt)
system_prompt = self._build_system_prompt()   # ← 300 tokens of ReAct instructions
LLMRequest(messages=..., system_prompt=system_prompt)  # ← passed in kwarg

# core/types.py (BUG — field didn't exist!)
class LLMRequest(BaseModel):
    messages: List[LLMMessage]
    temperature: float = 0.0
    max_tokens: int = 2048
    # ← NO system_prompt field! Pydantic silently dropped the kwarg.

# local_provider.py (BUG — never injected system_prompt even if it existed)
messages = [{"role": m.role, "content": m.content} for m in request.messages]
# ← request.system_prompt was never prepended as {"role": "system", ...}
```

### Effect
- Model received vLLM's default: `"You are Qwen, created by Alibaba Cloud"`
- Without ReAct specialist prompt: no tool calls, no Mermaid blocks, no step-by-step format
- Cat-A: hallucinated numerical values → fails exact numeric validation → **0%**
- Cat-C: never generated Mermaid blocks → **0%**  
- Cat-D: no multi-step ReAct format → **~30%**
- Cat-B: survived because keyword recall works even with plain text → **65%**

### Fix
```python
# core/types.py — added field
class LLMRequest(BaseModel):
    system_prompt: Optional[str] = None  # ← ADDED

# local_provider.py — inject as first message
def _build_messages(self, request: LLMRequest) -> list[dict]:
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for m in request.messages:
        if m.role == "system" and request.system_prompt:
            continue  # prevent duplication
        messages.append({"role": m.role, "content": m.content})
    return messages
```

---

## Root Cause #2 — `router.py` latency format crash

**Severity**: 🟠 High — killed all successful LLM responses in run_v2  
**File**: `chainmind/llm/router.py`

```python
# BUG
logger.info(f"Request completed in {latency_ms:.0f}ms")  # crashes if latency_ms is None

# FIX
logger.info(f"Request completed in {(latency_ms or 0):.0f}ms")
```

---

## Root Cause #3 — `a2a/protocol.py` `.value` on str crash

**Severity**: 🟠 High — all agent registration failed silently  
**File**: `chainmind/a2a/protocol.py`

```python
# BUG: card.role is a str from simplified types.py, not an Enum
role_key = card.role.value  # AttributeError: 'str' object has no attribute 'value'

# FIX
role_key = card.role.value if hasattr(card.role, 'value') else str(card.role)
```

---

## Root Cause #4 — `get_canonical_smiles` wrong parameter key

**Severity**: 🟡 Medium — Cat-A drug name → SMILES tasks always failed  
**File**: `chainmind/mcp/molecular_server.py`

```python
# BUG: model sends {"name": "Aspirin"}, tool reads {"query": ...}
query = args.get("query", "")  # ← empty string → PubChem crash

# FIX
query = args.get("query") or args.get("name") or args.get("compound", "")
```

---

## Root Cause #5 — Cat-C (KG) scored 0%: missing Mermaid instructions

**Severity**: 🟡 Medium — all knowledge graph tasks failed  
**File**: `chainmind/agents/base_agent.py`

```python
# FIX: detect KG task and inject Mermaid format into think_prompt
kg_keywords = ("knowledge graph", "drug network", "pathway", "interaction map")
if any(kw in task_text for kw in kg_keywords):
    think_prompt += "\n\n⚠️ IMPORTANT: Return a ```mermaid\ngraph TD\n...``` block."
```

---

## Root Cause #6 — SLURM `ENVIRONMENT=BATCH` crash

**Severity**: 🟡 Medium — all SLURM jobs crashed at startup  
**File**: `run_full_selfcontained.sh`

```bash
# FIX: override SLURM's ENVIRONMENT variable before Pydantic Settings reads it
export ENVIRONMENT=development
```

---

## Root Cause #7 — Reflexion LLMRequest missing system_prompt (NEW FIX)

**Severity**: 🟡 Medium — exp001_reflexion reflection steps used default Qwen prompt  
**File**: `chainmind/agents/base_agent.py` ~line 510

```python
# BUG: reflection call had no system_prompt
LLMRequest(messages=[...reflect_prompt...], temperature=0.2)

# FIX
LLMRequest(messages=[...], system_prompt=self._build_system_prompt(), temperature=0.2)
```

---

## Root Cause #8 — main vs master branch: removed Gemini/Ollama fallback

**Severity**: 🟢 Low — doesn't affect local-only setup but removes resilience  
**File**: `chainmind/llm/router.py`

The `main` branch had Gemini → OpenAI → Ollama fallback chain with circuit breakers.  
`master` simplified to local vLLM only. Acceptable for our use case.

---

## Root Cause #9 — specialists.py: domain mismatch (supply chain → drug discovery)

**Severity**: 🟢 Info only — correctly fixed in master  
The `main` branch had `DemandForecastingAgent`, `InventoryAgent` (supply chain).  
`master` correctly replaced with `ComputationalChemistAgent`, `WebResearchAgent`, `KnowledgeGraphAgent`.

---

## Jobs Timeline

```
Job 2620713: run_v3 (9/10 bugs fixed, system_prompt bug still present) — STILL RUNNING
Job 2620747: run_v4 (ALL 10 bugs fixed, including system_prompt) — QUEUED (Priority)
```

**Cancel run_v3 if you want to free the node for run_v4:**
```bash
scancel 2620713
```

---

## What to Do Next

1. **Wait for run_v4** (job 2620747) — this is the first truly correct benchmark run
2. **Cancel run_v3** (job 2620713) if you need the GPU sooner for run_v4
3. **Docker push**: Set `DOCKERHUB_TOKEN` in GitHub secrets → CI auto-builds image
4. **Fine-tuning**: After run_v4 results, submit `sbatch experiments/exp010_qlora_sft/train.sh`
