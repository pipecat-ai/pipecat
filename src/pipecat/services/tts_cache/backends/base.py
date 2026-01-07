#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base class for TTS cache storage backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pipecat.services.tts_cache.models import CachedTTSResponse


class CacheBackend(ABC):
    """Abstract base class for TTS cache storage backends.

    All backends must be thread-safe, non-blocking, and fail-safe.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        """Retrieve cached response, or None if not found/expired."""

    @abstractmethod
    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        """Store a TTS response in cache. Returns True on success."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""

    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries. Returns number of entries deleted."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""

    @abstractmethod
    async def close(self) -> None:
        """Close backend connections and cleanup resources."""

    async def __aenter__(self) -> "CacheBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
