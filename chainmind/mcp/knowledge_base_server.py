"""
ChainMind Knowledge Base MCP Server — RAG knowledge base tools.

Provides MCP tools for searching, retrieving, and ingesting
documents into the hybrid RAG knowledge base.
"""

from __future__ import annotations

from typing import Any

from chainmind.config.constants import ToolCategory
from chainmind.core.types import MCPToolDefinition
from chainmind.mcp.base_server import BaseMCPServer


class KnowledgeBaseMCPServer(BaseMCPServer):
    """MCP server for knowledge base operations."""

    def __init__(self, knowledge_base: Any = None):
        super().__init__(name="knowledge_base")
        self._kb = knowledge_base
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        self._register_tool(
            MCPToolDefinition(
                name="search_knowledge_base",
                description="Search the supply chain knowledge base for relevant information. Supports keyword, semantic, and hybrid search modes.",
                category=ToolCategory.KNOWLEDGE_BASE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
                        "mode": {"type": "string", "enum": ["hybrid", "vector", "bm25"], "description": "Search mode"},
                    },
                },
                required_params=["query"],
            ),
            self._search_knowledge_base,
        )

        self._register_tool(
            MCPToolDefinition(
                name="get_document",
                description="Retrieve a specific document by its ID from the knowledge base",
                category=ToolCategory.KNOWLEDGE_BASE,
                parameters={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Document identifier"},
                    },
                },
                required_params=["doc_id"],
            ),
            self._get_document,
        )

        self._register_tool(
            MCPToolDefinition(
                name="ingest_document",
                description="Add a new document to the knowledge base",
                category=ToolCategory.KNOWLEDGE_BASE,
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content"},
                        "title": {"type": "string", "description": "Document title"},
                        "category": {"type": "string", "description": "Document category"},
                    },
                },
                required_params=["content"],
            ),
            self._ingest_document,
        )

    async def _search_knowledge_base(
        self, query: str, top_k: int = 5, mode: str = "hybrid"
    ) -> dict[str, Any]:
        """Search the knowledge base using the hybrid retriever."""
        if self._kb is None:
            return {"results": [], "message": "Knowledge base not initialized"}

        try:
            results = await self._kb.search(query, top_k=top_k, mode=mode)
            return {
                "query": query,
                "mode": mode,
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "content": r.content[:500],  # Truncate for context window
                        "score": round(r.score, 4),
                        "retriever": r.retriever,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "total_results": len(results),
            }
        except Exception as e:
            return {"error": f"Search failed: {e}", "results": []}

    async def _get_document(self, doc_id: str) -> dict[str, Any]:
        """Retrieve a specific document."""
        if self._kb is None:
            return {"error": "Knowledge base not initialized"}

        try:
            doc = await self._kb.get_document(doc_id)
            if doc:
                return {"doc_id": doc_id, "content": doc["content"], "metadata": doc.get("metadata", {})}
            return {"error": f"Document '{doc_id}' not found"}
        except Exception as e:
            return {"error": f"Retrieval failed: {e}"}

    async def _ingest_document(
        self, content: str, title: str = "", category: str = "general"
    ) -> dict[str, Any]:
        """Ingest a new document."""
        if self._kb is None:
            return {"error": "Knowledge base not initialized"}

        try:
            doc_id = await self._kb.ingest(content, {"title": title, "category": category})
            return {"doc_id": doc_id, "status": "ingested", "title": title}
        except Exception as e:
            return {"error": f"Ingestion failed: {e}"}
