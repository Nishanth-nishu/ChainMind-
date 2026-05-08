import os
from typing import AsyncIterator, Any
from openai import AsyncOpenAI
import time

from chainmind.core.interfaces import ILLMProvider
from chainmind.core.types import LLMRequest, LLMResponse


class LocalProvider(ILLMProvider):
    """
    Local vLLM provider using the OpenAI-compatible API.
    Connects to http://0.0.0.0:8100/v1 by default.

    BUG FIX (2026-05-08): system_prompt was silently dropped.
    The OpenAI-compatible API expects the system prompt as the FIRST message
    with role="system". If request.system_prompt is set, we must prepend it.
    Without this fix, the model receives vLLM's default Qwen system prompt
    ("You are Qwen...") instead of the ReAct specialist prompt, causing:
      - Cat-A: hallucinated numerical values (no tool calls)
      - Cat-C: no Mermaid blocks (no format instructions)
      - Cat-D: no multi-step reasoning (no ReAct format)
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8100/v1",
        model_id: str = "chainmind-qwen",
        served_model_name: str = "chainmind-qwen",
    ):
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = served_model_name

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """
        Convert LLMRequest to the OpenAI messages format.

        Rule: if request.system_prompt is set, it becomes the FIRST message
        with role="system". All other messages follow in order.

        This is required for vLLM/OpenAI chat completions to respect our
        ReAct specialist prompt instead of the model's default system prompt.
        """
        messages = []

        # Inject system prompt as the first message if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Append remaining conversation messages
        for m in request.messages:
            # Skip any existing system messages — our system_prompt takes precedence
            if m.role == "system" and request.system_prompt:
                continue
            messages.append({"role": m.role, "content": m.content})

        return messages

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.perf_counter()

        messages = self._build_messages(request)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        elapsed = time.perf_counter() - start_time
        content = response.choices[0].message.content or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(content=content, usage=usage)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = self._build_messages(request)

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
