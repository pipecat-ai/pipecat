#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service resilience framework for Pipecat.

This module provides standardized reliability patterns for AI services including:
- Retry mechanisms with exponential backoff
- Circuit breaker pattern for failing services
- Service health monitoring and metrics
- Centralized error handling and recovery
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .retry_policy import RetryPolicy, RetryableError
from .service_resilience import ServiceResilience, create_llm_resilience, create_tts_resilience, create_stt_resilience
from .resilient_service import ResilientServiceMixin, ResilientFrameProcessor

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState", 
    "RetryPolicy",
    "RetryableError",
    "ServiceResilience",
    "create_llm_resilience",
    "create_tts_resilience", 
    "create_stt_resilience",
    "ResilientServiceMixin",
    "ResilientFrameProcessor",
] 