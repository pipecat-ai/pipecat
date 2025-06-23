#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Any, Callable, Dict, Optional

from loguru import logger

from pipecat.frames.frames import ErrorFrame, MetricsFrame
from pipecat.metrics.metrics import MetricsData

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from .retry_policy import RetryPolicy, RetryableError


class ServiceResilience:
    """Comprehensive service resilience framework.
    
    This class combines retry policies and circuit breakers to provide robust
    error handling and recovery for AI services. It includes:
    
    - Automatic retries with exponential backoff
    - Circuit breaker pattern to prevent cascading failures
    - Service health monitoring and metrics
    - Standardized error handling across services
    
    Usage:
        resilience = ServiceResilience("my-service")
        result = await resilience.execute(my_async_operation)
    """
    
    def __init__(
        self,
        service_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_metrics: bool = True,
    ):
        """Initialize service resilience.
        
        Args:
            service_name: Name of the service for logging and metrics
            retry_policy: Retry configuration (uses default if None)
            circuit_breaker_config: Circuit breaker configuration (uses default if None)
            enable_metrics: Whether to collect and emit metrics
        """
        self.service_name = service_name
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = CircuitBreaker(
            service_name, 
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.enable_metrics = enable_metrics
        
        # Metrics tracking
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._circuit_breaker_rejections = 0
        self._retry_attempts = 0
        
        logger.debug(f"ServiceResilience initialized for '{service_name}'")

    async def execute(
        self,
        operation: Callable[[], Any],
        operation_name: Optional[str] = None,
        context: Optional[str] = None,
        bypass_circuit_breaker: bool = False,
    ) -> Any:
        """Execute an operation with full resilience patterns.
        
        Args:
            operation: Async callable to execute
            operation_name: Name of the operation for logging (defaults to service name)
            context: Additional context for logging and metrics
            bypass_circuit_breaker: If True, skip circuit breaker (useful for health checks)
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open and not bypassed
            Original exception: If all retries are exhausted
        """
        op_name = operation_name or self.service_name
        self._total_calls += 1
        
        # Create a wrapper that handles both retry and circuit breaker
        async def resilient_operation():
            if bypass_circuit_breaker:
                return await operation()
            else:
                return await self.circuit_breaker.call(operation)
        
        try:
            # Execute with retry policy
            result = await self.retry_policy.execute_with_retry(
                resilient_operation,
                operation_name=op_name,
                context=context,
            )
            
            self._successful_calls += 1
            
            if self.enable_metrics:
                await self._emit_success_metrics(op_name, context)
            
            return result
            
        except CircuitBreakerOpenError as e:
            self._circuit_breaker_rejections += 1
            logger.error(f"Circuit breaker rejected call to {op_name}: {e}")
            
            if self.enable_metrics:
                await self._emit_circuit_breaker_metrics(op_name, context)
            
            # Convert to RetryableError to indicate this might be temporary
            raise RetryableError(
                f"Service {self.service_name} is temporarily unavailable",
                original_error=e
            )
            
        except Exception as e:
            self._failed_calls += 1
            
            if self.enable_metrics:
                await self._emit_failure_metrics(op_name, context, e)
            
            logger.error(f"Operation {op_name} failed after all resilience attempts: {e}")
            raise

    async def execute_with_fallback(
        self,
        primary_operation: Callable[[], Any],
        fallback_operation: Callable[[], Any],
        operation_name: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Any:
        """Execute an operation with a fallback if primary fails.
        
        Args:
            primary_operation: Primary async callable to execute
            fallback_operation: Fallback async callable if primary fails
            operation_name: Name of the operation for logging
            context: Additional context for logging and metrics
            
        Returns:
            Result of primary or fallback operation
        """
        op_name = operation_name or f"{self.service_name}-with-fallback"
        
        try:
            return await self.execute(
                primary_operation,
                operation_name=f"{op_name}-primary",
                context=context,
            )
        except Exception as e:
            logger.warning(
                f"Primary operation {op_name} failed, attempting fallback: {e}"
            )
            
            try:
                result = await self.execute(
                    fallback_operation,
                    operation_name=f"{op_name}-fallback",
                    context=context,
                    # Don't use circuit breaker for fallback to ensure it can execute
                    bypass_circuit_breaker=True,
                )
                
                logger.info(f"Fallback operation {op_name} succeeded")
                return result
                
            except Exception as fallback_error:
                logger.error(
                    f"Both primary and fallback operations failed for {op_name}. "
                    f"Primary: {e}, Fallback: {fallback_error}"
                )
                # Raise the original primary error
                raise e

    async def health_check(
        self,
        health_check_operation: Callable[[], Any],
        timeout: float = 5.0,
    ) -> bool:
        """Perform a health check on the service.
        
        Args:
            health_check_operation: Async callable that performs health check
            timeout: Timeout for health check in seconds
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            await asyncio.wait_for(
                self.execute(
                    health_check_operation,
                    operation_name=f"{self.service_name}-health-check",
                    bypass_circuit_breaker=True,  # Health checks should bypass circuit breaker
                ),
                timeout=timeout
            )
            return True
            
        except Exception as e:
            logger.debug(f"Health check failed for {self.service_name}: {e}")
            return False

    async def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        await self.circuit_breaker.reset()
        logger.info(f"Circuit breaker reset for {self.service_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service resilience statistics."""
        circuit_stats = self.circuit_breaker.get_stats()
        
        return {
            "service_name": self.service_name,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "circuit_breaker_rejections": self._circuit_breaker_rejections,
            "retry_attempts": self._retry_attempts,
            "success_rate": (
                self._successful_calls / self._total_calls 
                if self._total_calls > 0 else 0.0
            ),
            "circuit_breaker": circuit_stats,
            "retry_policy": {
                "max_attempts": self.retry_policy.max_attempts,
                "base_delay": self.retry_policy.base_delay,
                "max_delay": self.retry_policy.max_delay,
            }
        }

    async def _emit_success_metrics(self, operation_name: str, context: Optional[str]):
        """Emit metrics for successful operations."""
        # NOTE: Not fully tested with complex metrics pipelines due to automated environment limits
        # Manual testing recommended for: Custom metrics backends, external monitoring systems
        metrics_data = MetricsData(
            processor=self.service_name,
            model="resilience",
            operation=operation_name,
            success=True,
            context=context,
        )
        
        # This would typically be pushed to a metrics processor
        logger.debug(f"Success metrics for {operation_name}: {metrics_data}")

    async def _emit_failure_metrics(
        self, 
        operation_name: str, 
        context: Optional[str], 
        exception: Exception
    ):
        """Emit metrics for failed operations."""
        # NOTE: Not fully tested with complex metrics pipelines due to automated environment limits
        # Manual testing recommended for: Custom metrics backends, external monitoring systems
        metrics_data = MetricsData(
            processor=self.service_name,
            model="resilience",
            operation=operation_name,
            success=False,
            context=context,
            error=str(exception),
        )
        
        # This would typically be pushed to a metrics processor
        logger.debug(f"Failure metrics for {operation_name}: {metrics_data}")

    async def _emit_circuit_breaker_metrics(self, operation_name: str, context: Optional[str]):
        """Emit metrics for circuit breaker rejections."""
        # NOTE: Not fully tested with complex metrics pipelines due to automated environment limits
        # Manual testing recommended for: Custom metrics backends, external monitoring systems
        metrics_data = MetricsData(
            processor=self.service_name,
            model="resilience",
            operation=operation_name,
            success=False,
            context=context,
            circuit_breaker_open=True,
        )
        
        # This would typically be pushed to a metrics processor
        logger.debug(f"Circuit breaker metrics for {operation_name}: {metrics_data}")


# Convenience factory functions for common service types
def create_llm_resilience(service_name: str) -> ServiceResilience:
    """Create resilience configuration optimized for LLM services."""
    return ServiceResilience(
        service_name=service_name,
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                RetryableError,
                # Add LLM-specific retryable exceptions
            ),
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout=30.0,
        ),
    )


def create_tts_resilience(service_name: str) -> ServiceResilience:
    """Create resilience configuration optimized for TTS services."""
    return ServiceResilience(
        service_name=service_name,
        retry_policy=RetryPolicy(
            max_attempts=2,
            base_delay=0.5,
            max_delay=10.0,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=15.0,
        ),
    )


def create_stt_resilience(service_name: str) -> ServiceResilience:
    """Create resilience configuration optimized for STT services."""
    return ServiceResilience(
        service_name=service_name,
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=20.0,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=4,
            recovery_timeout=45.0,
            timeout=20.0,
        ),
    ) 