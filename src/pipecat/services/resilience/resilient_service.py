#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Callable, Optional, Union

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .circuit_breaker import CircuitBreakerConfig
from .retry_policy import RetryPolicy, RetryableError
from .service_resilience import ServiceResilience


class ResilientServiceMixin:
    """Mixin class to add resilience capabilities to existing services.
    
    This mixin can be added to any service class to provide standardized
    retry logic, circuit breaker functionality, and error handling.
    
    Usage:
        class MyService(ResilientServiceMixin, AIService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._setup_resilience("my-service")
            
            async def my_operation(self):
                return await self._execute_resilient(self._do_work)
    """
    
    def _setup_resilience(
        self,
        service_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_metrics: bool = True,
    ):
        """Setup resilience framework for this service.
        
        Args:
            service_name: Name of the service for logging and metrics
            retry_policy: Custom retry policy (uses default if None)
            circuit_breaker_config: Custom circuit breaker config (uses default if None)
            enable_metrics: Whether to collect and emit metrics
        """
        self._resilience = ServiceResilience(
            service_name=service_name,
            retry_policy=retry_policy,
            circuit_breaker_config=circuit_breaker_config,
            enable_metrics=enable_metrics,
        )
        
        logger.debug(f"Resilience framework setup for service: {service_name}")

    async def _execute_resilient(
        self,
        operation: Callable[[], Any],
        operation_name: Optional[str] = None,
        context: Optional[str] = None,
        bypass_circuit_breaker: bool = False,
    ) -> Any:
        """Execute an operation with resilience patterns.
        
        Args:
            operation: Async callable to execute
            operation_name: Name of the operation for logging
            context: Additional context for logging and metrics
            bypass_circuit_breaker: If True, skip circuit breaker
            
        Returns:
            Result of the operation
            
        Raises:
            RetryableError: If service is temporarily unavailable
            Original exception: If all retries are exhausted
        """
        if not hasattr(self, '_resilience'):
            raise RuntimeError(
                "Resilience framework not initialized. "
                "Call _setup_resilience() in your service's __init__ method."
            )
        
        try:
            return await self._resilience.execute(
                operation=operation,
                operation_name=operation_name,
                context=context,
                bypass_circuit_breaker=bypass_circuit_breaker,
            )
        except RetryableError as e:
            # Convert RetryableError to ErrorFrame for Pipecat's error handling
            if hasattr(self, 'push_error'):
                await self.push_error(ErrorFrame(str(e), fatal=False))
            raise
        except Exception as e:
            # Convert other exceptions to ErrorFrame
            if hasattr(self, 'push_error'):
                await self.push_error(ErrorFrame(str(e), fatal=True))
            raise

    async def _execute_with_fallback(
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
        if not hasattr(self, '_resilience'):
            raise RuntimeError(
                "Resilience framework not initialized. "
                "Call _setup_resilience() in your service's __init__ method."
            )
        
        try:
            return await self._resilience.execute_with_fallback(
                primary_operation=primary_operation,
                fallback_operation=fallback_operation,
                operation_name=operation_name,
                context=context,
            )
        except Exception as e:
            # Convert exceptions to ErrorFrame
            if hasattr(self, 'push_error'):
                await self.push_error(ErrorFrame(str(e), fatal=True))
            raise

    async def _health_check(
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
        if not hasattr(self, '_resilience'):
            return False
        
        return await self._resilience.health_check(
            health_check_operation=health_check_operation,
            timeout=timeout,
        )

    async def _reset_circuit_breaker(self):
        """Manually reset the circuit breaker for this service."""
        if hasattr(self, '_resilience'):
            await self._resilience.reset_circuit_breaker()

    def _get_resilience_stats(self) -> dict:
        """Get resilience statistics for this service."""
        if hasattr(self, '_resilience'):
            return self._resilience.get_stats()
        return {}


class ResilientFrameProcessor(ResilientServiceMixin, FrameProcessor):
    """Frame processor with built-in resilience capabilities.
    
    This class provides a convenient base for creating resilient frame processors
    that automatically handle retries, circuit breaking, and error recovery.
    
    Example:
        class MyResilientProcessor(ResilientFrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._setup_resilience("my-processor")
            
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, MyFrame):
                    result = await self._execute_resilient(
                        lambda: self._process_my_frame(frame)
                    )
                    await self.push_frame(result)
                else:
                    await super().process_frame(frame, direction)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resilience will be setup by subclasses calling _setup_resilience()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frame with resilience error handling."""
        try:
            await super().process_frame(frame, direction)
        except RetryableError as e:
            # Service is temporarily unavailable, push non-fatal error
            await self.push_error(ErrorFrame(str(e), fatal=False))
        except Exception as e:
            # Unexpected error, push fatal error
            await self.push_error(ErrorFrame(str(e), fatal=True))
            raise 