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
    """
    
    def __init__(self, base_url: str = "http://0.0.0.0:8100/v1", model_id: str = "chainmind-qwen", served_model_name: str = "chainmind-qwen"):
        # Uses an empty API key since vLLM does not require one formatably
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = served_model_name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.perf_counter()
        
        # Convert LLMMessages to dict for OpenAI spec
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages, # type: ignore
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )
        
        elapsed = time.perf_counter() - start_time
        content = response.choices[0].message.content or ""
        
        # Optional usage stats
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
        return LLMResponse(content=content, usage=usage)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages, # type: ignore
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        try:
            # Quick models endpoint check
            await self.client.models.list()
            return True
        except Exception:
            return False
