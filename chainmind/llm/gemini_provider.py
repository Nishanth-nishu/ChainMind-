"""
GeminiProvider — REMOVED.

ChainMind no longer uses Gemini. This file is kept as a stub to avoid
ImportError if any cached pyc references it. It raises immediately if instantiated.

Use local vLLM providers instead:
    bash scripts/start_model_server.sh qwen2.5-7b
"""


class GeminiProvider:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GeminiProvider has been removed from ChainMind. "
            "Use local vLLM providers: bash scripts/start_model_server.sh qwen2.5-7b"
        )
