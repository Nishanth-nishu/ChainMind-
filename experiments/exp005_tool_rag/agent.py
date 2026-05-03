"""
EXP005 — Dynamic Tool Selection (Tool-RAG)
Paper: Qin et al., "ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs"
       ICLR 2024. https://arxiv.org/abs/2307.16789

Hypothesis: Listing all 6 tools in every THINK prompt confuses Qwen-7B —
it sometimes calls tools irrelevant to the query. A lightweight BM25 retriever
surfaces only the top-2 most relevant tools per query, reducing selection errors.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import LLMMessage


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                k1: float = 1.5, b: float = 0.75, avg_dl: float = 20) -> float:
    dl = len(doc_tokens)
    tf_map = Counter(doc_tokens)
    idf_map = {t: math.log(1 + (1 - 0.5 + 1) / (0.5 + 1)) for t in query_tokens}
    score = 0.0
    for t in query_tokens:
        tf = tf_map.get(t, 0)
        idf = idf_map.get(t, 0.1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


class ToolRAGAgent(BaseAgent):
    """EXP005: BM25-based tool retriever — show only top-K tools per query."""

    TOP_K = 2

    def _select_top_tools(self, query: str) -> dict:
        """Return the top-K most relevant tools via BM25."""
        if len(self._tool_registry) <= self.TOP_K:
            return self._tool_registry

        q_tokens = re.findall(r"\b\w+\b", query.lower())
        scores = {}
        for name, (server, tool_def) in self._tool_registry.items():
            doc = f"{name} {tool_def.description}".lower()
            doc_tokens = re.findall(r"\b\w+\b", doc)
            scores[name] = _bm25_score(q_tokens, doc_tokens)

        top_names = sorted(scores, key=scores.get, reverse=True)[: self.TOP_K]
        return {n: self._tool_registry[n] for n in top_names}

    def _build_think_prompt(
        self,
        messages: list[LLMMessage],
        tool_descriptions: str,
        memory_context: str,
        step: int,
    ) -> str:
        """Inject only BM25-retrieved tools for this step's query."""
        # Pull the user query from the last user message
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        top_tools = self._select_top_tools(user_msg)

        # Re-format descriptions for only retrieved tools
        if top_tools:
            lines = []
            for name, (_, td) in top_tools.items():
                req = ", ".join(td.required_params) if td.required_params else "none"
                lines.append(f"- {name}: {td.description} (required: {req})")
            retrieved_descs = "\n".join(lines)
        else:
            retrieved_descs = tool_descriptions

        return super()._build_think_prompt(
            messages, retrieved_descs, memory_context, step
        )
