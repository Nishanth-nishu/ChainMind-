"""
ChainMind gRPC Client — Async client for inter-service communication.

Provides a clean async interface to the ChainMind gRPC service
with automatic retry, connection pooling, and health checking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import grpc

from chainmind.grpc_service import agent_service_pb2 as pb2
from chainmind.grpc_service import agent_service_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class ChainMindClient:
    """
    Async gRPC client for the ChainMind service.

    Usage:
        async with ChainMindClient("localhost:50051") as client:
            response = await client.query("What are inventory levels?")
            print(response.result)
    """

    def __init__(
        self,
        target: str = "localhost:50051",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        self._target = target
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._channel: grpc.aio.Channel | None = None
        self._stub: pb2_grpc.ChainMindServiceStub | None = None

    async def __aenter__(self) -> ChainMindClient:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish gRPC channel."""
        self._channel = grpc.aio.insecure_channel(
            self._target,
            options=[
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self._stub = pb2_grpc.ChainMindServiceStub(self._channel)
        logger.info(f"gRPC client connected to {self._target}")

    async def close(self) -> None:
        """Close gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("gRPC client disconnected")

    async def query(
        self,
        query: str,
        session_id: str = "",
        target_agent: str = "",
        max_steps: int = 15,
        temperature: float = 0.7,
    ) -> pb2.QueryResponse:
        """
        Send a query to the ChainMind service.

        Returns the full QueryResponse protobuf message.
        """
        if not self._stub:
            raise RuntimeError("Client not connected. Use 'async with' or call connect().")

        request = pb2.QueryRequest(
            query=query,
            session_id=session_id,
            target_agent=target_agent,
            max_steps=max_steps,
            temperature=temperature,
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._stub.ProcessQuery(
                    request,
                    timeout=self._timeout,
                )
                return response

            except grpc.aio.AioRpcError as e:
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    if attempt < self._max_retries:
                        wait = 2 ** attempt
                        logger.warning(
                            f"gRPC call failed (attempt {attempt}/{self._max_retries}), "
                            f"retrying in {wait}s: {e.code()}"
                        )
                        await asyncio.sleep(wait)
                        continue
                raise

    async def stream_query(
        self,
        query: str,
        session_id: str = "",
        target_agent: str = "",
    ) -> AsyncIterator[pb2.QueryChunk]:
        """
        Stream a query response token-by-token.

        Yields QueryChunk messages as they arrive.
        """
        if not self._stub:
            raise RuntimeError("Client not connected.")

        request = pb2.QueryRequest(
            query=query,
            session_id=session_id,
            target_agent=target_agent,
        )

        async for chunk in self._stub.StreamQuery(
            request,
            timeout=self._timeout,
        ):
            yield chunk

    async def health_check(self) -> pb2.HealthCheckResponse:
        """Check service health."""
        if not self._stub:
            raise RuntimeError("Client not connected.")

        request = pb2.HealthCheckRequest(service_name="chainmind")
        return await self._stub.HealthCheck(request, timeout=5.0)

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> pb2.KnowledgeSearchResponse:
        """Search the knowledge base."""
        if not self._stub:
            raise RuntimeError("Client not connected.")

        request = pb2.KnowledgeSearchRequest(
            query=query,
            top_k=top_k,
            mode=mode,
        )
        return await self._stub.SearchKnowledge(request, timeout=self._timeout)


async def main():
    """Quick client test."""
    async with ChainMindClient("localhost:50051") as client:
        # Health check
        health = await client.health_check()
        print(f"Health: {health.status} (healthy={health.is_healthy})")

        # Query
        response = await client.query("What are the inventory levels for SKU-001?")
        print(f"Response: {response.result}")
        print(f"Agent: {response.source_agent}")
        print(f"Latency: {response.latency_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
