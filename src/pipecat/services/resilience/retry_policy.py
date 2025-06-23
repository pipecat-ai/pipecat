#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union

from loguru import logger


class RetryableError(Exception):
    """Exception that indicates an operation should be retried."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


@dataclass
class RetryPolicy:
    """Configurable retry policy with exponential backoff and jitter.
    
    This class provides a standardized way to handle retries across all Pipecat
    services, ensuring consistent behavior and reducing code duplication.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        retryable_exceptions: Tuple of exception types that should trigger retries
    """
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
        RetryableError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if attempt == 0:
            return 0.0
            
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            # Add random jitter of ±25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.0, delay)  # Ensure non-negative
            
        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-indexed)
            
        Returns:
            True if the operation should be retried
        """
        if attempt >= self.max_attempts:
            return False
            
        return isinstance(exception, self.retryable_exceptions)

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation",
        context: Optional[str] = None,
    ) -> Any:
        """Execute an operation with retry logic.
        
        Args:
            operation: Async callable to execute
            operation_name: Name of the operation for logging
            context: Additional context for logging
            
        Returns:
            Result of the operation
            
        Raises:
            The last exception if all retries are exhausted
        """
        last_exception = None
        context_str = f" ({context})" if context else ""
        
        for attempt in range(self.max_attempts):
            try:
                if attempt > 0:
                    delay = self.calculate_delay(attempt)
                    logger.debug(
                        f"Retrying {operation_name}{context_str} "
                        f"(attempt {attempt + 1}/{self.max_attempts}) "
                        f"after {delay:.2f}s delay"
                    )
                    await asyncio.sleep(delay)
                
                result = await operation()
                
                if attempt > 0:
                    logger.info(
                        f"Successfully executed {operation_name}{context_str} "
                        f"after {attempt} retries"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    if attempt >= self.max_attempts - 1:
                        logger.error(
                            f"Failed to execute {operation_name}{context_str} "
                            f"after {self.max_attempts} attempts: {e}"
                        )
                    else:
                        logger.error(
                            f"Non-retryable error in {operation_name}{context_str}: {e}"
                        )
                    raise
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed for "
                    f"{operation_name}{context_str}: {e}"
                )
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Unexpected error in retry logic for {operation_name}")


# Common retry policies for different scenarios
DEFAULT_RETRY_POLICY = RetryPolicy()

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
)

CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    base_delay=2.0,
    max_delay=120.0,
)

WEBSOCKET_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
        RetryableError,
        # Add websocket-specific exceptions
    ),
) 