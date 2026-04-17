"""
ChainMind gRPC Server — High-performance agent service.

Provides a gRPC interface for high-throughput clients.
Includes interceptors for logging, health checks, and graceful shutdown.

NOTE: Before running, compile the proto file:
    make proto
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from concurrent import futures
from typing import Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from chainmind.config.settings import get_settings

logger = logging.getLogger(__name__)


class ChainMindGRPCServer:
    """
    gRPC server wrapper with health checks, reflection, and graceful shutdown.

    This is a placeholder that demonstrates the gRPC setup pattern.
    The actual servicer implementation requires compiled protobuf stubs.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 50051, max_workers: int = 10):
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._server: grpc.Server | None = None

    def start(self) -> None:
        """Start the gRPC server."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
            ],
        )

        # Register health check service
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
        health_servicer.set(
            "chainmind.v1.ChainMindService",
            health_pb2.HealthCheckResponse.SERVING,
        )

        # Enable server reflection for grpcurl/clients
        service_names = (
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, self._server)

        # Bind address
        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self._server.start()
        logger.info(f"gRPC server started on {address}")
        logger.info(f"  Workers: {self._max_workers}")
        logger.info(f"  Health check: enabled")
        logger.info(f"  Reflection: enabled")

        self._server.wait_for_termination()

    def _handle_shutdown(self, signum, frame) -> None:
        """Graceful shutdown on SIGTERM/SIGINT."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self._server:
            # Grace period: finish in-flight requests
            event = self._server.stop(grace=10)
            event.wait()
            logger.info("gRPC server shut down gracefully")
        sys.exit(0)


def main() -> None:
    """Entry point for gRPC server."""
    settings = get_settings()

    from chainmind.observability.logger import setup_logging
    setup_logging(settings)

    server = ChainMindGRPCServer(
        host=settings.grpc_host,
        port=settings.grpc_port,
        max_workers=settings.grpc_max_workers,
    )
    server.start()


if __name__ == "__main__":
    main()
