"""
ChainMind MCP Base Server — Model Context Protocol implementation.

Implements the MCP server specification:
- Tool registration with JSON Schema parameter definitions
- Input validation against schemas
- Sandboxed execution with error boundary
- Automatic capability advertisement
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Callable, Awaitable

from chainmind.core.exceptions import MCPToolExecutionError, MCPToolNotFoundError
from chainmind.core.interfaces import IMCPServer
from chainmind.core.types import MCPToolDefinition, MCPToolResult

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[Any]]


class BaseMCPServer(IMCPServer):
    """
    Base MCP server with tool registration and sandboxed execution.

    Subclasses call self._register_tool() in __init__ to register
    their domain-specific tools.
    """

    def __init__(self, name: str):
        self._name = name
        self._tools: dict[str, MCPToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    @property
    def server_name(self) -> str:
        return self._name

    def _register_tool(
        self,
        definition: MCPToolDefinition,
        handler: ToolHandler,
    ) -> None:
        """Register a tool with its definition and handler."""
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        logger.debug(f"MCP[{self._name}] registered tool: {definition.name}")

    def list_tools(self) -> list[MCPToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """
        Execute a tool with input validation and error boundary.

        All tool executions are sandboxed — exceptions are caught and
        returned as error results, never propagated.
        """
        if tool_name not in self._handlers:
            raise MCPToolNotFoundError(
                f"Tool '{tool_name}' not found on server '{self._name}'. "
                f"Available: {list(self._tools.keys())}"
            )

        tool_def = self._tools[tool_name]

        # Validate required parameters
        for param in tool_def.required_params:
            if param not in arguments:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Missing required parameter: '{param}'",
                )

        # Execute in error boundary
        start = time.perf_counter()
        try:
            handler = self._handlers[tool_name]
            result = await handler(**arguments)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"MCP tool '{tool_name}' failed: {e}", exc_info=True)
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
