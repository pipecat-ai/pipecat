#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Tests for the Service Resilience Framework.

These tests verify the retry policies, circuit breaker functionality,
and service resilience patterns work correctly.

NOTE: These are lightweight tests that don't require external dependencies.
Manual testing recommended for: Complex failure scenarios, external service integrations
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipecat.services.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    RetryPolicy,
    RetryableError,
    ServiceResilience,
)


def test_retry_policy():
    """Test retry policy functionality."""
    print("Testing retry policy...")
    
    # Test delay calculation
    policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=False)
    assert policy.calculate_delay(0) == 0.0  # No delay for first attempt
    assert policy.calculate_delay(1) == 1.0  # base_delay
    assert policy.calculate_delay(2) == 2.0  # base_delay * 2^1
    assert policy.calculate_delay(3) == 4.0  # base_delay * 2^2
    print("✓ Delay calculation test passed")
    
    # Test max delay limit
    policy = RetryPolicy(base_delay=10.0, max_delay=15.0, jitter=False)
    assert policy.calculate_delay(3) == 15.0  # Capped at max_delay
    print("✓ Max delay limit test passed")
    
    # Test retry decision logic
    policy = RetryPolicy(max_attempts=3)
    assert policy.should_retry(ConnectionError(), 0) is True
    assert policy.should_retry(TimeoutError(), 1) is True
    assert policy.should_retry(RetryableError("test"), 2) is True
    assert policy.should_retry(ConnectionError(), 3) is False
    assert policy.should_retry(ValueError(), 0) is False
    print("✓ Retry decision logic test passed")

async def test_retry_execution():
    """Test retry execution functionality."""
    print("Testing retry execution...")
    
    # Test successful execution with retries
    policy = RetryPolicy(max_attempts=3, base_delay=0.01)
    call_count = 0
    
    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = await policy.execute_with_retry(flaky_operation, "test-op")
    assert result == "success"
    assert call_count == 3
    print("✓ Retry execution success test passed")
    
    # Test retry exhaustion
    policy = RetryPolicy(max_attempts=2, base_delay=0.01)
    
    async def always_failing_operation():
        raise ConnectionError("Always fails")
    
    try:
        await policy.execute_with_retry(always_failing_operation, "test-op")
        assert False, "Should have raised ConnectionError"
    except ConnectionError:
        pass  # Expected
    print("✓ Retry exhaustion test passed")


# Additional tests are included in the comprehensive test functions above


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("Testing circuit breaker...")
    
    # Test normal operation in CLOSED state
    cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
    
    async def successful_operation():
        return "success"
    
    result = await cb.call(successful_operation)
    assert result == "success"
    assert cb.state == CircuitBreakerState.CLOSED
    print("✓ Circuit breaker closed state test passed")
    
    # Test circuit breaker opens after failures
    cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
    
    async def failing_operation():
        raise RuntimeError("Operation failed")
    
    # First failure
    try:
        await cb.call(failing_operation)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    assert cb.state == CircuitBreakerState.CLOSED
    
    # Second failure - should open circuit
    try:
        await cb.call(failing_operation)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    assert cb.state == CircuitBreakerState.OPEN
    
    # Third call should be rejected immediately
    try:
        await cb.call(failing_operation)
        assert False, "Should have raised CircuitBreakerOpenError"
    except CircuitBreakerOpenError:
        pass
    print("✓ Circuit breaker open state test passed")

async def test_service_resilience():
    """Test service resilience functionality."""
    print("Testing service resilience...")
    
    # Test successful operation
    resilience = ServiceResilience("test-service")
    
    async def successful_operation():
        return "success"
    
    result = await resilience.execute(successful_operation)
    assert result == "success"
    
    stats = resilience.get_stats()
    assert stats["total_calls"] == 1
    assert stats["successful_calls"] == 1
    assert stats["failed_calls"] == 0
    print("✓ Service resilience success test passed")
    
    # Test fallback functionality
    resilience = ServiceResilience("test-service")
    
    async def primary_operation():
        raise ConnectionError("Primary failed")
    
    async def fallback_operation():
        return "fallback-success"
    
    result = await resilience.execute_with_fallback(
        primary_operation,
        fallback_operation,
    )
    assert result == "fallback-success"
    print("✓ Service resilience fallback test passed")

if __name__ == "__main__":
    async def run_all_tests():
        """Run all functionality tests."""
        print("🧪 Testing Service Resilience Framework...\n")
        
        # Run all test functions
        test_retry_policy()
        await test_retry_execution()
        await test_circuit_breaker()
        await test_service_resilience()
        
        print("\n✅ All tests passed! Service Resilience Framework is working correctly.")
    
    # Run all tests
    asyncio.run(run_all_tests()) 