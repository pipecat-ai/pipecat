#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger


class CircuitBreakerState(Enum):
    """Circuit breaker states following the standard pattern."""
    
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing fast, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.
    
    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Seconds to wait before attempting recovery (default: 60)
        success_threshold: Successful calls needed to close circuit in half-open state (default: 3)
        timeout: Maximum time to wait for operation in seconds (default: 30)
        expected_exceptions: Exception types that count as failures
    """
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0
    expected_exceptions: tuple = (
        Exception,  # Count all exceptions as failures by default
    )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, failure_count: int):
        self.service_name = service_name
        self.failure_count = failure_count
        super().__init__(
            f"Circuit breaker is OPEN for {service_name} "
            f"(failures: {failure_count}). Service calls are being rejected."
        )


class CircuitBreaker:
    """Circuit breaker implementation for service resilience.
    
    The circuit breaker prevents cascading failures by monitoring service health
    and temporarily blocking requests to failing services, allowing them time to recover.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, requests are rejected immediately  
    - HALF_OPEN: Testing recovery, limited requests are allowed
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
        
        logger.debug(f"Circuit breaker '{name}' initialized in CLOSED state")

    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Current success count (only relevant in HALF_OPEN state)."""
        return self._success_count

    async def call(
        self,
        operation: Callable[[], Any],
        *args,
        **kwargs,
    ) -> Any:
        """Execute an operation through the circuit breaker.
        
        Args:
            operation: Async callable to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If operation fails and circuit remains closed
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitBreakerState.OPEN:
                if time.time() - self._last_failure_time >= self.config.recovery_timeout:
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(self.name, self._failure_count)
            
            # In HALF_OPEN state, only allow limited requests
            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
        
        # Execute the operation with timeout
        try:
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self.config.timeout
            )
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            timeout_error = TimeoutError(
                f"Operation timed out after {self.config.timeout}s"
            )
            await self._record_failure(timeout_error)
            raise timeout_error
            
        except Exception as e:
            await self._record_failure(e)
            raise

    async def _record_success(self):
        """Record a successful operation."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' recorded success "
                    f"({self._success_count}/{self.config.success_threshold})"
                )
                
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    logger.debug(
                        f"Circuit breaker '{self.name}' reset failure count "
                        f"after successful operation"
                    )
                    self._failure_count = 0

    async def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        async with self._lock:
            # Only count expected exceptions as failures
            if not isinstance(exception, self.config.expected_exceptions):
                logger.debug(
                    f"Circuit breaker '{self.name}' ignoring unexpected exception: "
                    f"{type(exception).__name__}"
                )
                return
            
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure "
                f"({self._failure_count}/{self.config.failure_threshold}): "
                f"{exception}"
            )
            
            # Transition to OPEN if failure threshold exceeded
            if (self._state == CircuitBreakerState.CLOSED and 
                self._failure_count >= self.config.failure_threshold):
                await self._transition_to_open()
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in HALF_OPEN state transitions back to OPEN
                await self._transition_to_open()

    async def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self._state = CircuitBreakerState.OPEN
        self._success_count = 0
        
        logger.error(
            f"Circuit breaker '{self.name}' transitioned to OPEN state "
            f"after {self._failure_count} failures. "
            f"Will attempt recovery in {self.config.recovery_timeout}s"
        )

    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._success_count = 0
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state. "
            f"Testing service recovery..."
        )

    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned to CLOSED state. "
            f"Service has recovered."
        )

    async def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        async with self._lock:
            await self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_stats(self) -> dict:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        } 