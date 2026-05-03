#!/usr/bin/env bash
# =============================================================================
# run_cat_a_pilot.sh — Fastest path to publishable Cat-A results
#
# Runs ChainMind-Bench Category A (40 Molecular Property tasks) against:
#   1. chainmind_qwen  — Full ChainMind with MCP tools (RDKit + PubChem)
#   2. qwen_direct     — Qwen-7B with NO tools (parametric knowledge only)
#
# The delta between these two systems on deterministic chemistry tasks is the
# paper's headline result: "MCP tool augmentation improves Cat-A TSR by X%."
#
# Prerequisites:
#   - vLLM running (started below automatically if not running)
#   - .venv activated
#   - RDKit installed (pip install rdkit)
#
# Output:
#   results/cat_a_pilot/bench_chainmind_qwen_*.json
#   results/cat_a_pilot/bench_qwen_direct_*.json
#   results/cat_a_pilot/comparison_*.md
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_ROOT/results/cat_a_pilot"
# Port 8100 — matches start_vllm_optimized.sh default
VLLM_URL="${VLLM_BASE_URL:-http://localhost:8100}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || true

mkdir -p "$RESULTS_DIR"
echo "================================================================"
echo "  ChainMind Cat-A Pilot Benchmark"
echo "  Results → $RESULTS_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Verify or start vLLM
# ---------------------------------------------------------------------------
echo ""
echo "[ 1/4 ] Checking vLLM server at $VLLM_URL ..."
if curl -sf "$VLLM_URL/v1/models" > /dev/null 2>&1; then
    echo "        ✓ vLLM already running"
else
    echo "        ✗ vLLM not detected — attempting to start on port 8100..."
    if [ -f "$SCRIPT_DIR/start_model_server.sh" ]; then
        bash "$SCRIPT_DIR/start_model_server.sh" qwen2.5-7b --background
        echo "        Waiting 50s for vLLM to initialize..."
        sleep 50
        if curl -sf "http://localhost:8100/health" > /dev/null 2>&1; then
            echo "        ✓ vLLM started successfully on port 8100"
            VLLM_URL="http://localhost:8100"
        else
            echo "        ✗ vLLM failed to start. Run manually:"
            echo "          bash scripts/start_model_server.sh qwen2.5-7b"
            exit 1
        fi
    else
        echo "        ✗ scripts/start_model_server.sh not found."
        echo "          Start vLLM manually:"
        echo "          bash scripts/start_vllm_optimized.sh"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Verify RDKit
# ---------------------------------------------------------------------------
echo ""
echo "[ 2/4 ] Checking RDKit ..."
python -c "from rdkit import Chem; print('        ✓ RDKit OK')" || {
    echo "        ✗ RDKit not found. Install: pip install rdkit"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 3: Run ChainMind (with tools)
# ---------------------------------------------------------------------------
echo ""
echo "[ 3/4 ] Running ChainMind (Qwen-7B + MCP tools) on Cat-A ..."
python -m chainmind.eval.bench_runner \
    --mode full \
    --system chainmind_qwen \
    --category A \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/chainmind_qwen_run.log"

echo ""
echo "        ✓ ChainMind run complete"

# ---------------------------------------------------------------------------
# Step 4: Run baseline (no tools)
# ---------------------------------------------------------------------------
echo ""
echo "[ 4/4 ] Running Qwen-7B Direct (NO tools) on Cat-A ..."
python -m chainmind.eval.bench_runner \
    --mode full \
    --system qwen_direct \
    --category A \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/qwen_direct_run.log"

echo ""
echo "        ✓ Baseline run complete"

# ---------------------------------------------------------------------------
# Step 5: Generate comparison table
# ---------------------------------------------------------------------------
echo ""
echo "[ 5/5 ] Generating comparison report ..."
python - <<'PYEOF'
import json, glob, pathlib, datetime

results_dir = pathlib.Path("results/cat_a_pilot")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

records = {}
for jf in results_dir.glob("*.json"):
    try:
        data = json.loads(jf.read_text())
        for r in data.get("results", []):
            sys = r["system"]
            records.setdefault(sys, []).append(r)
    except Exception:
        pass

lines = [
    f"# Cat-A Pilot Results — {ts}\n",
    "## Task Success Rate (TSR %)",
    "",
    "| System | N | TSR (%) | Tool Rate (%) | Avg Score | Avg Latency (ms) |",
    "|--------|---|---------|---------------|-----------|-----------------|",
]

for sys, recs in sorted(records.items()):
    n = len(recs)
    tsr = sum(1 for r in recs if r["passed"]) / max(n, 1) * 100
    tool_rate = sum(1 for r in recs if r.get("n_tools_called", 0) > 0) / max(n, 1) * 100
    avg_score = sum(r["score"] for r in recs) / max(n, 1)
    avg_lat = sum(r["latency_ms"] for r in recs) / max(n, 1)
    lines.append(f"| {sys:<30} | {n} | {tsr:>6.1f} | {tool_rate:>13.1f} | {avg_score:>9.3f} | {avg_lat:>16.0f} |")

lines += [
    "",
    "## Tool Invocation Evidence (ChainMind only)",
    "",
]

cm_recs = records.get("ChainMind (Qwen-7B)", [])
if cm_recs:
    total_calls = sum(r.get("n_tools_called", 0) for r in cm_recs)
    total_ok    = sum(r.get("n_tools_succeeded", 0) for r in cm_recs)
    lines += [
        f"- Total MCP tool calls: **{total_calls}** across {len(cm_recs)} tasks",
        f"- Tool success rate: **{total_ok / max(total_calls, 1) * 100:.1f}%**",
        f"- Most frequent tools: {_top_tools(cm_recs)}",
    ]
    def _top_tools(recs):
        from collections import Counter
        c = Counter()
        for r in recs:
            for t in r.get("tools_called", []):
                c[t] += 1
        return ", ".join(f"{k}×{v}" for k, v in c.most_common(5)) or "none recorded"

    lines[-1] = f"- Most frequent tools: {_top_tools(cm_recs)}"

report = "\n".join(lines)
out = results_dir / f"comparison_{ts}.md"
out.write_text(report)
print(report)
print(f"\n💾 Report saved → {out}")
PYEOF

echo ""
echo "================================================================"
echo "  PILOT COMPLETE"
echo "  Next steps:"
echo "  1. Review results/cat_a_pilot/comparison_*.md"
echo "  2. If TSR delta (ChainMind - Direct) >= 15pp → run full 100 tasks:"
echo "     python -m chainmind.eval.bench_runner --mode full --system all"
echo "  3. Run ablation: python -m chainmind.eval.ablation --mode full --n 40"
echo "================================================================"
