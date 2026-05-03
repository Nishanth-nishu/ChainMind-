#!/usr/bin/env python3
"""
ChainMind Interactive CLI — Feature-rich terminal interface.

Features:
  - Startup health check (API + vLLM server status)
  - Session memory (conversation continuity across turns)
  - Agent listing with capabilities
  - File browser for mol2/pdb directories
  - Token usage and latency display
  - Contextual error messages with fix suggestions
"""

import sys
import os
import time
import json
import glob
import httpx
import threading
import uuid
from typing import Dict, Any, Optional

# ─── ANSI Color Codes ────────────────────────────────────────────────
class C:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'

    @staticmethod
    def ok(msg):    return f"{C.GREEN}{msg}{C.ENDC}"
    @staticmethod
    def warn(msg):  return f"{C.YELLOW}{msg}{C.ENDC}"
    @staticmethod
    def err(msg):   return f"{C.RED}{msg}{C.ENDC}"
    @staticmethod
    def info(msg):  return f"{C.CYAN}{msg}{C.ENDC}"
    @staticmethod
    def bold(msg):  return f"{C.BOLD}{msg}{C.ENDC}"
    @staticmethod
    def dim(msg):   return f"{C.DIM}{msg}{C.ENDC}"


# ─── Spinner ──────────────────────────────────────────────────────────
class Spinner:
    def __init__(self, message="Thinking..."):
        self.message = message
        self.chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.running = False
        self.thread = None

    def _spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\r{C.CYAN}{self.chars[i % len(self.chars)]} {self.message}{C.ENDC}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stdout.flush()


# ─── API Client ───────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
TIMEOUT  = 300.0


def api_get(path: str, timeout: float = 5.0) -> Optional[dict]:
    """Safe GET request to ChainMind API."""
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: dict, timeout: float = TIMEOUT) -> Optional[dict]:
    """Safe POST request to ChainMind API."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        print(f"\n  {C.err('✗ Cannot connect to ChainMind API at')} {API_BASE}")
        print(f"    {C.dim('Fix: Run')} {C.bold('make dev')} {C.dim('in another terminal')}\n")
        return None
    except httpx.ReadTimeout:
        print(f"\n  {C.warn('⏱  Request timed out')} {C.dim(f'(>{timeout}s)')}\n")
        return None
    except Exception as e:
        print(f"\n  {C.err(f'✗ API Error: {e}')}\n")
        return None


# ─── Commands ─────────────────────────────────────────────────────────

def cmd_help():
    """Show help."""
    print(f"""
  {C.bold('Available Commands:')}
  {C.info('help')}              Show this help message
  {C.info('agents')}            List registered specialist agents
  {C.info('health')}            Check system health (API, vLLM, RAG)
  {C.info('files <path>')}      List mol2/pdb files in a directory
  {C.info('memory')}            Show current session memory status
  {C.info('clear')}             Clear screen
  {C.info('exit / quit')}       Exit the CLI

  {C.bold('Query Examples:')}
  {C.dim('>')} Does aspirin pass Lipinski's Rule of 5?
  {C.dim('>')} Parse the mol2 file at /path/to/mol_0001.mol2
  {C.dim('>')} Search ArXiv for GNN molecular property prediction
  {C.dim('>')} Generate a knowledge graph for PROTAC degradation
""")


def cmd_health():
    """Show system health."""
    print(f"\n  {C.bold('System Health')}")
    print(f"  {'─' * 45}")

    # API server
    data = api_get("/api/v1/health")
    if data:
        status = data.get("status", "unknown")
        healthy = data.get("is_healthy", False)
        icon = C.ok("✓") if healthy else C.err("✗")
        print(f"  {icon} API Server       {C.dim(f'({status})')}")

        # Provider details
        providers = data.get("providers", {})
        for name, info in providers.items():
            p_healthy = info.get("healthy", False)
            cb = info.get("circuit_breaker", {})
            cb_state = cb.get("state", "unknown")
            p_icon = C.ok("✓") if p_healthy else C.err("✗")
            print(f"  {p_icon} LLM: {name:<12} {C.dim(f'(circuit: {cb_state})')}")
    else:
        print(f"  {C.err('✗')} API Server       {C.dim('(unreachable)')}")
        print(f"    {C.dim('Fix: Run')} {C.bold('make dev')} {C.dim('in another terminal')}")

    # vLLM server
    try:
        r = httpx.get("http://localhost:8100/health", timeout=3.0)
        if r.status_code == 200:
            print(f"  {C.ok('✓')} vLLM Server      {C.dim('(localhost:8100)')}")
        else:
            print(f"  {C.err('✗')} vLLM Server      {C.dim(f'(status {r.status_code})')}")
    except Exception:
        print(f"  {C.err('✗')} vLLM Server      {C.dim('(not running)')}")
        print(f"    {C.dim('Fix: Run')} {C.bold('bash scripts/start_vllm_optimized.sh')}")

    print()


def cmd_agents():
    """List registered agents."""
    data = api_get("/api/v1/agents")
    if not data:
        print(f"  {C.err('✗ Cannot fetch agents. Is the API server running?')}\n")
        return

    agents = data.get("agents", [])
    total = data.get("total", 0)
    print(f"\n  {C.bold(f'Registered Agents ({total})')}")
    print(f"  {'─' * 55}")

    for agent in agents:
        name = agent.get("name", "Unknown")
        role = agent.get("role", "unknown")
        caps = agent.get("capabilities", [])
        tools = agent.get("tools", [])
        print(f"  {C.ok('●')} {C.bold(name)}")
        print(f"    Role: {C.info(role)}")
        print(f"    Capabilities: {C.dim(', '.join(caps))}")
        print(f"    Tools: {C.dim(', '.join(tools))}")
        print()


def cmd_files(path: str):
    """List mol2/pdb files in a directory."""
    path = path.strip()
    if not path:
        print(f"  {C.warn('Usage:')} files /path/to/directory\n")
        return

    if not os.path.isdir(path):
        print(f"  {C.err(f'✗ Directory not found: {path}')}\n")
        return

    mol2_files = sorted(glob.glob(os.path.join(path, "*.mol2")))
    pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
    all_files = mol2_files + pdb_files

    if not all_files:
        print(f"  {C.warn('No .mol2 or .pdb files found in')} {path}\n")
        return

    print(f"\n  {C.bold(f'Molecular Files ({len(all_files)} found)')}")
    print(f"  {'─' * 55}")

    for i, f in enumerate(all_files[:20]):
        basename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024
        ext = os.path.splitext(basename)[1]
        color = C.info if ext == ".mol2" else C.HEADER
        print(f"  {color}{'●'}{C.ENDC} {basename:<30} {C.dim(f'{size_kb:.1f} KB')}")

    if len(all_files) > 20:
        print(f"  {C.dim(f'  ... and {len(all_files) - 20} more')}")

    print(f"\n  {C.dim('Tip: Ask me to parse one, e.g.:')}")
    if mol2_files:
        print(f"  {C.dim(f'  > Parse {mol2_files[0]}')}")
    elif pdb_files:
        print(f"  {C.dim(f'  > Parse {pdb_files[0]}')}")
    print()


def cmd_memory(session_id: str):
    """Show current session info."""
    print(f"\n  {C.bold('Session Info')}")
    print(f"  {'─' * 45}")
    print(f"  Session ID: {C.dim(session_id)}")
    print(f"  {C.dim('Memory is stored per-session. Restart the CLI to start fresh.')}")
    print()


# ─── Reasoning Step Formatter ─────────────────────────────────────────

STEP_ICONS = {
    "think":   ("🧠", C.CYAN),
    "act":     ("🛠️ ", C.YELLOW),
    "observe": ("👁️ ", C.HEADER),
    "verify":  ("✅", C.BLUE),
    "reflect": ("🔄", C.RED),
}


def format_step(step: Dict[str, Any]):
    step_type = step.get("step_type", "").lower()
    content = step.get("content", "")
    icon, color = STEP_ICONS.get(step_type, ("▪️ ", C.DIM))

    # Truncate long content for readability
    if len(content) > 300:
        content = content[:300] + "..."

    print(f"  {color}{icon} [{step_type.upper()}] {content}{C.ENDC}")


# ─── Main Query ───────────────────────────────────────────────────────

def query_system(query: str, session_id: str):
    """Send query to the ChainMind API and display results."""
    payload = {"query": query, "session_id": session_id}

    spinner = Spinner("Agents working on your task...")
    spinner.start()

    data = api_post("/api/v1/query", payload)
    spinner.stop()

    if data is None:
        return  # Error already printed by api_post

    status = data.get("status", "")

    if status == "completed":
        # Show reasoning trace
        trace = data.get("reasoning_trace", [])
        if trace:
            print(f"\n  {C.bold('⚡ Reasoning Trace:')}")
            for step in trace:
                format_step(step)
                time.sleep(0.3)

        # Show final answer
        latency = data.get("latency_ms", 0)
        steps = data.get("reasoning_steps", 0)
        source = data.get("source_agent", "unknown")

        print(f"\n  {C.bold(C.ok(f'🎯 Answer'))} {C.dim(f'({latency:.0f}ms · {steps} steps · via {source})')}")
        print(f"  {'─' * 55}")

        result = data.get("result", "No result")
        # Indent the result for clean display
        for line in result.split("\n"):
            print(f"  {line}")
        print()

    elif status == "failed":
        error = data.get("error", "Unknown error")
        print(f"\n  {C.err('✗ Task Failed')}")

        # Provide specific fix suggestions
        if "All LLM providers failed" in error:
            if "local" in error and "context length" in error.lower():
                print(f"  {C.dim('The prompt was too large for the model context window.')}")
                print(f"  {C.dim('Try a simpler query or point to a specific file.')}")
            elif "keys exhausted" in error:
                print(f"  {C.dim('All Gemini API keys are rate-limited.')}")
                print(f"  {C.dim('Ensure vLLM is running:')} {C.bold('bash scripts/start_vllm_optimized.sh')}")
            elif "Cannot connect to Ollama" in error:
                print(f"  {C.dim('Ollama is not running. Start vLLM instead:')}")
                print(f"  {C.bold('bash scripts/start_vllm_optimized.sh')}")
            else:
                print(f"  {C.dim(error[:200])}")
        else:
            print(f"  {C.dim(error[:200])}")
        print()
    else:
        print(f"\n  {C.warn(f'Unexpected status: {status}')}")
        print(f"  {C.dim(json.dumps(data, indent=2)[:300])}\n")


# ─── Startup Health Check ─────────────────────────────────────────────

def startup_check():
    """Quick health check on startup."""
    print(f"  {C.dim('Checking system status...')}")

    api_ok = api_get("/api/v1/health") is not None
    vllm_ok = False
    try:
        r = httpx.get("http://localhost:8100/health", timeout=3.0)
        vllm_ok = (r.status_code == 200)
    except Exception:
        pass

    if api_ok and vllm_ok:
        print(f"  {C.ok('✓ API server')}  {C.ok('✓ vLLM GPU inference')}  {C.ok('✓ Ready!')}")
    elif api_ok and not vllm_ok:
        print(f"  {C.ok('✓ API server')}  {C.warn('⚠ vLLM not running')}  {C.dim('(using cloud LLM fallback)')}")
        print(f"  {C.dim('  For local GPU inference: bash scripts/start_vllm_optimized.sh')}")
    elif not api_ok:
        print(f"  {C.err('✗ API server not running')}")
        print(f"  {C.dim('  Start it with:')} {C.bold('make dev')}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────

def print_banner():
    print(f"""
{C.BOLD}{C.BLUE}  ╔══════════════════════════════════════════════════╗
  ║          🧬 ChainMind · D4 Agent CLI             ║
  ║    Drug Discovery · Molecular Analysis · RAG     ║
  ╚══════════════════════════════════════════════════╝{C.ENDC}
""")


def main():
    session_id = str(uuid.uuid4())

    print_banner()
    startup_check()

    print(f"  {C.dim('Type')} {C.info('help')} {C.dim('for commands, or ask a question directly.')}")
    print()

    while True:
        try:
            user_input = input(f"  {C.BOLD}{C.BLUE}ChainMind>{C.ENDC} ").strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("exit", "quit"):
                print(f"  {C.dim('Goodbye! 👋')}")
                break
            elif cmd == "help":
                cmd_help()
            elif cmd == "health":
                cmd_health()
            elif cmd == "agents":
                cmd_agents()
            elif cmd == "memory":
                cmd_memory(session_id)
            elif cmd == "clear":
                os.system("clear" if os.name != "nt" else "cls")
                print_banner()
            elif cmd.startswith("files "):
                cmd_files(user_input[6:])
            else:
                query_system(user_input, session_id)

        except KeyboardInterrupt:
            print(f"\n  {C.dim('Goodbye! 👋')}")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"\n  {C.err(f'Unexpected error: {e}')}\n")


if __name__ == "__main__":
    main()
