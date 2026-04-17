"""
Unit tests for the Circuit Breaker — self-healing pattern.

Tests state transitions: CLOSED → OPEN → HALF_OPEN → CLOSED
"""

import asyncio
import pytest
from chainmind.llm.circuit_breaker import CircuitBreaker
from chainmind.config.constants import CircuitState
from chainmind.core.exceptions import LLMCircuitOpenError


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(
        name="test_provider",
        failure_threshold=3,
        recovery_timeout_seconds=1,
        success_threshold=2,
    )


@pytest.mark.unit
class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_starts_closed(self, circuit_breaker):
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self, circuit_breaker):
        async def success():
            return "ok"

        result = await circuit_breaker.call(success)
        assert result == "ok"
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, circuit_breaker):
        async def fail():
            raise Exception("Provider down")

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception, match="Provider down"):
                await circuit_breaker.call(fail)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, circuit_breaker):
        async def fail():
            raise Exception("fail")

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(fail)

        # Next call should be rejected immediately
        with pytest.raises(LLMCircuitOpenError):
            await circuit_breaker.call(fail)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self, circuit_breaker):
        async def fail():
            raise Exception("fail")

        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(fail)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_recovers_on_success(self, circuit_breaker):
        async def fail():
            raise Exception("fail")

        async def succeed():
            return "recovered"

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(fail)

        # Wait for half-open
        await asyncio.sleep(1.1)
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Successful probes
        await circuit_breaker.call(succeed)
        await circuit_breaker.call(succeed)

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self, circuit_breaker):
        async def fail():
            raise Exception("fail")

        # Trip
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(fail)

        await asyncio.sleep(1.1)
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Probe fails → back to OPEN
        with pytest.raises(Exception):
            await circuit_breaker.call(fail)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_to_dict_for_observability(self, circuit_breaker):
        result = circuit_breaker.to_dict()
        assert result["name"] == "test_provider"
        assert result["state"] == "closed"
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker):
        async def fail():
            raise Exception("fail")

        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(fail)

        assert circuit_breaker.state == CircuitState.OPEN

        await circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED
