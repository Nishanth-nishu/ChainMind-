import json
from typing import Any
from duckduckgo_search import DDGS

from chainmind.core.interfaces import IMCPServer
from chainmind.core.types import MCPToolDefinition, MCPToolResult

class ResearchMCPServer(IMCPServer):
    """MCP Server providing web research and literature search capabilities."""

    def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="search_duckduckgo",
                description="Search the web for scientific or general information.",
                parameters={"query": "The search query text"}
            )
        ]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> MCPToolResult:
        try:
            if tool_name == "search_duckduckgo":
                query = args.get("query", "")
                with DDGS() as ddgs:
                    # Fetch top 3 results
                    results = list(ddgs.text(query, max_results=3))
                
                if not results:
                    return MCPToolResult(result="", success=False, error=f"No results found for query: {query}")
                    
                formatted = [{"title": r.get("title"), "body": r.get("body")} for r in results]
                return MCPToolResult(result=json.dumps(formatted), success=True)
                
            else:
                return MCPToolResult(result="", success=False, error=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            return MCPToolResult(result="", success=False, error=str(e))
