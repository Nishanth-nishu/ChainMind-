"""
ChainMind Base Agent — ReAct reasoning loop with Reflexion-style self-correction.

Implements the core Thought→Action→Observe→Verify→Reflect cycle from:
- ReAct (Yao et al., 2023) — interleaved reasoning and acting
- Reflexion (Shinn et al., 2023) — self-correcting agents via verbal reinforcement

Every specialist agent extends this base to inherit:
- ReAct loop with configurable max steps
- MCP tool execution via registered tools
- Guardrail hooks at input/output/action boundaries
- Memory read/write at each step
- Structured reasoning trace for observability
- Execution budget enforcement (tokens, steps, time)
"""

from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from typing import Any

from chainmind.config.constants import AgentRole, ReActStep, TaskStatus
from chainmind.core.exceptions import (
    AgentExecutionError,
    AgentMaxStepsError,
    AgentTimeoutError,
)
from chainmind.core.interfaces import IAgent, IGuardrail, IMCPServer, IMemoryStore
from chainmind.core.types import (
    AgentCard,
    AgentContext,
    LLMMessage,
    LLMRequest,
    MCPToolDefinition,
    MCPToolResult,
    ReasoningStep,
    TaskRequest,
    TaskResponse,
)
from chainmind.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class BaseAgent(IAgent):
    """
    Base ReAct agent with self-correction and guardrails.

    Subclasses MUST implement:
    - agent_card property
    - _build_system_prompt() method
    - _select_tools() method (which MCP tools to register)
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_servers: list[IMCPServer] | None = None,
        guardrails: list[IGuardrail] | None = None,
        memory_store: IMemoryStore | None = None,
        max_steps: int = 15,
        max_timeout_seconds: int = 120,
    ):
        self._llm_router = llm_router
        self._mcp_servers = mcp_servers or []
        self._guardrails = guardrails or []
        self._memory_store = memory_store
        self._max_steps = max_steps
        self._max_timeout = max_timeout_seconds

        # Build tool registry from MCP servers
        self._tool_registry: dict[str, tuple[IMCPServer, MCPToolDefinition]] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools from MCP servers."""
        for server in self._mcp_servers:
            for tool_def in server.list_tools():
                self._tool_registry[tool_def.name] = (server, tool_def)
        logger.info(
            f"Agent '{self.agent_card.name}' registered {len(self._tool_registry)} tools"
        )

    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        ...

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent's specialization."""
        ...

    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """
        Execute the ReAct loop to process a task.

        Loop: THINK → ACT → OBSERVE → VERIFY → (REFLECT on failure)
        """
        start_time = time.perf_counter()
        reasoning_trace: list[ReasoningStep] = []
        step_count = 0
        final_answer: str | None = None

        # Apply input guardrails
        guarded_query = await self._apply_input_guardrails(task.query)

        # Load memory context
        memory_context = await self._load_memory(context.session_id)

        # Build initial conversation
        messages: list[LLMMessage] = list(context.conversation_history)
        messages.append(LLMMessage(role="user", content=guarded_query))

        system_prompt = self._build_system_prompt()
        tool_descriptions = self._format_tool_descriptions()

        try:
            while step_count < self._max_steps:
                # Budget check
                elapsed = time.perf_counter() - start_time
                if elapsed > self._max_timeout:
                    raise AgentTimeoutError(
                        f"Agent exceeded timeout of {self._max_timeout}s"
                    )

                step_count += 1

                # === THINK ===
                think_prompt = self._build_think_prompt(
                    messages, tool_descriptions, memory_context, step_count
                )

                think_response = await self._llm_router.generate(
                    LLMRequest(
                        messages=[LLMMessage(role="user", content=think_prompt)],
                        system_prompt=system_prompt,
                        temperature=0.3,  # Lower temp for reasoning
                        max_tokens=2048,
                    )
                )

                thought = think_response.content
                reasoning_trace.append(ReasoningStep(
                    step_type=ReActStep.THINK, content=thought
                ))

                # Check if agent wants to provide final answer
                if self._is_final_answer(thought):
                    final_answer = self._extract_final_answer(thought)
                    break

                # === ACT ===
                tool_call = self._parse_tool_call(thought)
                if tool_call is None:
                    # No tool call parsed — treat as final answer
                    final_answer = thought
                    break

                tool_name, tool_args = tool_call
                reasoning_trace.append(ReasoningStep(
                    step_type=ReActStep.ACT,
                    content=f"Calling tool: {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_args,
                ))

                # Apply action guardrails
                await self._apply_action_guardrails(tool_name, tool_args)

                # Execute tool via MCP
                tool_result = await self._execute_tool(tool_name, tool_args)

                # === OBSERVE ===
                observation = (
                    f"Tool '{tool_name}' returned: {tool_result.result}"
                    if tool_result.success
                    else f"Tool '{tool_name}' FAILED: {tool_result.error}"
                )

                reasoning_trace.append(ReasoningStep(
                    step_type=ReActStep.OBSERVE,
                    content=observation,
                    tool_name=tool_name,
                    tool_output=str(tool_result.result) if tool_result.success else tool_result.error,
                ))

                # === VERIFY ===
                verified = await self._verify_step(tool_name, tool_result, task.query)
                reasoning_trace.append(ReasoningStep(
                    step_type=ReActStep.VERIFY,
                    content=f"Verification: {'PASS' if verified else 'FAIL'}",
                ))

                if not verified:
                    # === REFLECT ===
                    reflection = await self._reflect(
                        thought, tool_name, tool_result, task.query
                    )
                    reasoning_trace.append(ReasoningStep(
                        step_type=ReActStep.REFLECT,
                        content=reflection,
                    ))

                # Update messages with observation for next iteration
                messages.append(LLMMessage(role="assistant", content=thought))
                messages.append(LLMMessage(role="tool", content=observation))

            if final_answer is None:
                raise AgentMaxStepsError(
                    f"Agent did not converge after {self._max_steps} steps"
                )

            # Apply output guardrails
            final_answer = await self._apply_output_guardrails(final_answer)

            # Store to memory
            await self._store_memory(
                context.session_id, task.query, final_answer
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return TaskResponse(
                task_id=task.task_id,
                source_agent=self.agent_card.agent_id,
                status=TaskStatus.COMPLETED,
                result=final_answer,
                reasoning_trace=reasoning_trace,
                latency_ms=elapsed_ms,
            )

        except (AgentTimeoutError, AgentMaxStepsError) as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return TaskResponse(
                task_id=task.task_id,
                source_agent=self.agent_card.agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                reasoning_trace=reasoning_trace,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Agent {self.agent_card.name} error: {e}", exc_info=True)
            return TaskResponse(
                task_id=task.task_id,
                source_agent=self.agent_card.agent_id,
                status=TaskStatus.FAILED,
                error=f"Unexpected error: {e}",
                reasoning_trace=reasoning_trace,
                latency_ms=elapsed_ms,
            )

    # =========================================================================
    # ReAct Helpers
    # =========================================================================

    def _build_think_prompt(
        self,
        messages: list[LLMMessage],
        tool_descriptions: str,
        memory_context: str,
        step: int,
    ) -> str:
        """Build the thinking prompt for the current step."""
        history = "\n".join(
            f"[{m.role}]: {m.content}" for m in messages[-10:]  # Last 10 messages
        )

        return f"""You are a specialist AI agent. Use the ReAct pattern to solve the task.

## Available Tools
{tool_descriptions}

## Relevant Memory
{memory_context if memory_context else "No relevant memories."}

## Conversation History
{history}

## Instructions (Step {step}/{self._max_steps})
Think step-by-step about what to do next.

If you need to use a tool, respond with:
THOUGHT: [your reasoning]
ACTION: [tool_name]
ACTION_INPUT: [JSON arguments]

If you have the final answer, respond with:
THOUGHT: [your reasoning]
FINAL_ANSWER: [your complete answer to the user's question]

Respond now:"""

    def _format_tool_descriptions(self) -> str:
        """Format all available tools for the system prompt."""
        if not self._tool_registry:
            return "No tools available."

        descriptions = []
        for name, (_, tool_def) in self._tool_registry.items():
            params = json.dumps(tool_def.parameters, indent=2)
            descriptions.append(
                f"- **{name}**: {tool_def.description}\n  Parameters: {params}"
            )
        return "\n".join(descriptions)

    def _is_final_answer(self, thought: str) -> bool:
        """Check if the thought contains a final answer."""
        return "FINAL_ANSWER:" in thought

    def _extract_final_answer(self, thought: str) -> str:
        """Extract the final answer from the thought."""
        if "FINAL_ANSWER:" in thought:
            return thought.split("FINAL_ANSWER:", 1)[1].strip()
        return thought

    def _parse_tool_call(self, thought: str) -> tuple[str, dict[str, Any]] | None:
        """Parse tool name and arguments from thought."""
        try:
            if "ACTION:" not in thought:
                return None

            action_part = thought.split("ACTION:", 1)[1]

            if "ACTION_INPUT:" in action_part:
                tool_name = action_part.split("ACTION_INPUT:", 1)[0].strip()
                args_str = action_part.split("ACTION_INPUT:", 1)[1].strip()
                # Clean up any trailing text
                if "\n" in args_str:
                    args_str = args_str.split("\n")[0].strip()
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"input": args_str}
            else:
                tool_name = action_part.strip().split("\n")[0].strip()
                args = {}

            return tool_name, args

        except Exception as e:
            logger.warning(f"Failed to parse tool call from thought: {e}")
            return None

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> MCPToolResult:
        """Execute a tool via MCP server."""
        if tool_name not in self._tool_registry:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found. Available: {list(self._tool_registry.keys())}",
            )

        server, tool_def = self._tool_registry[tool_name]
        try:
            result = await server.execute_tool(tool_name, args)
            return result
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool execution error: {e}",
            )

    async def _verify_step(
        self, tool_name: str, result: MCPToolResult, original_query: str
    ) -> bool:
        """Post-condition verification (Reflexion pattern)."""
        if not result.success:
            return False

        # Simple verification: result is not empty/None
        if result.result is None or (isinstance(result.result, str) and not result.result.strip()):
            return False

        return True

    async def _reflect(
        self, thought: str, tool_name: str, result: MCPToolResult, query: str
    ) -> str:
        """Self-reflection on failure (Reflexion pattern)."""
        reflect_prompt = f"""The previous action failed or produced unsatisfactory results.

Original query: {query}
Thought: {thought}
Tool used: {tool_name}
Result: {result.error or result.result}

What went wrong and what should be done differently? Be concise."""

        try:
            response = await self._llm_router.generate(
                LLMRequest(
                    messages=[LLMMessage(role="user", content=reflect_prompt)],
                    temperature=0.2,
                    max_tokens=512,
                )
            )
            return response.content
        except Exception as e:
            return f"Reflection failed: {e}"

    # =========================================================================
    # Guardrail Hooks
    # =========================================================================

    async def _apply_input_guardrails(self, content: str) -> str:
        """Run input through all input guardrails."""
        result_content = content
        for guard in self._guardrails:
            if hasattr(guard, "check_input"):
                gr = await guard.check(result_content, {"type": "input"})
                if not gr.passed:
                    from chainmind.core.exceptions import GuardrailBlockedError
                    raise GuardrailBlockedError(gr.guardrail_name, gr.reason)
                if gr.modified_content:
                    result_content = gr.modified_content
        return result_content

    async def _apply_output_guardrails(self, content: str) -> str:
        """Run output through all output guardrails."""
        result_content = content
        for guard in self._guardrails:
            gr = await guard.check(result_content, {"type": "output"})
            if not gr.passed:
                logger.warning(f"Output guardrail {gr.guardrail_name} flagged: {gr.reason}")
            if gr.modified_content:
                result_content = gr.modified_content
        return result_content

    async def _apply_action_guardrails(
        self, tool_name: str, args: dict[str, Any]
    ) -> None:
        """Validate tool execution against action guardrails."""
        for guard in self._guardrails:
            gr = await guard.check(
                tool_name, {"type": "action", "tool_name": tool_name, "args": args}
            )
            if not gr.passed:
                from chainmind.core.exceptions import GuardrailBlockedError
                raise GuardrailBlockedError(gr.guardrail_name, gr.reason)

    # =========================================================================
    # Memory Hooks
    # =========================================================================

    async def _load_memory(self, session_id: str) -> str:
        """Load relevant memories for the current session."""
        if self._memory_store is None:
            return ""
        try:
            from chainmind.core.types import MemoryEntry
            entries = await self._memory_store.retrieve(session_id, top_k=5)
            if entries:
                return "\n".join(f"- {e.content}" for e in entries)
            return ""
        except Exception as e:
            logger.warning(f"Memory load failed: {e}")
            return ""

    async def _store_memory(
        self, session_id: str, query: str, answer: str
    ) -> None:
        """Store interaction in memory."""
        if self._memory_store is None:
            return
        try:
            from chainmind.core.types import MemoryEntry
            entry = MemoryEntry(
                session_id=session_id,
                agent_id=self.agent_card.agent_id,
                content=f"Q: {query[:200]}\nA: {answer[:500]}",
                memory_type="episodic",
            )
            await self._memory_store.store(entry)
        except Exception as e:
            logger.warning(f"Memory store failed: {e}")
