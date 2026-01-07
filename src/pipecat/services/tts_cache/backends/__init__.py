#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cache backend implementations for TTS caching."""

from pipecat.services.tts_cache.backends.base import CacheBackend
from pipecat.services.tts_cache.backends.memory import MemoryCacheBackend

try:
    from pipecat.services.tts_cache.backends.redis import RedisCacheBackend

    REDIS_AVAILABLE = True
except ImportError:
    RedisCacheBackend = None  # type: ignore
    REDIS_AVAILABLE = False

__all__ = [
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "REDIS_AVAILABLE",
]
