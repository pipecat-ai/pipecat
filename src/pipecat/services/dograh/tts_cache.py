#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TTS caching utilities for pre-warming and runtime cache lookup.

This module provides:
1. CachedTTSData - data structure for cached audio + alignment
2. TTSCacheManager - Redis-based cache manager for TTS data
3. TTSCacheWarmer - Pre-warming utility to cache common sentences
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import msgpack
except ModuleNotFoundError:
    logger.warning("msgpack not installed. TTS caching will not work.")
    msgpack = None

try:
    import redis.asyncio as aioredis
except ModuleNotFoundError:
    logger.warning("redis not installed. TTS caching will not work.")
    aioredis = None


@dataclass
class CachedTTSData:
    """Data structure for cached TTS audio and alignment data.

    Attributes:
        audio_chunks: List of raw audio byte chunks
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels
        word_timestamps: List of (word, timestamp_seconds) tuples
        total_duration_seconds: Total audio duration
    """

    audio_chunks: List[bytes]
    sample_rate: int
    num_channels: int
    word_timestamps: List[Tuple[str, float]]
    total_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audio_chunks": self.audio_chunks,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "word_timestamps": self.word_timestamps,
            "total_duration_seconds": self.total_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedTTSData":
        """Create from dictionary."""
        return cls(
            audio_chunks=data["audio_chunks"],
            sample_rate=data["sample_rate"],
            num_channels=data["num_channels"],
            word_timestamps=[tuple(wt) for wt in data["word_timestamps"]],
            total_duration_seconds=data["total_duration_seconds"],
        )


class TTSCacheManager:
    """Manages TTS cache storage and retrieval using Redis.

    This manager provides:
    - Cache key generation based on text + voice settings
    - Storage/retrieval of audio + alignment data
    - TTL-based cache expiration
    """

    # Cache key prefix for namespacing
    CACHE_PREFIX = "tts_cache:"

    def __init__(
        self,
        redis_url: str,
        cache_ttl_seconds: int = 86400,  # 24 hours default
    ):
        """Initialize the TTS cache manager.

        Args:
            redis_url: Redis connection URL
            cache_ttl_seconds: Cache entry TTL in seconds (default 24 hours)
        """
        self._redis_url = redis_url
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client: Optional[Any] = None

    async def connect(self):
        """Establish connection to Redis."""
        if aioredis is None:
            raise RuntimeError("redis package not installed")
        if not self._redis_client:
            self._redis_client = await aioredis.from_url(self._redis_url)
            logger.debug(f"Connected to Redis at {self._redis_url}")

    async def disconnect(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            logger.debug("Disconnected from Redis")

    def generate_cache_key(
        self,
        text: str,
        voice_id: str,
        model: str,
        sample_rate: int,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a unique cache key for the given parameters.

        Args:
            text: The text to be synthesized
            voice_id: Voice identifier
            model: TTS model name
            sample_rate: Audio sample rate
            settings: Additional voice settings that affect output

        Returns:
            SHA256-based cache key
        """
        # Normalize text: lowercase and strip whitespace
        normalized_text = text.strip().lower()

        key_data = {
            "text": normalized_text,
            "voice_id": voice_id,
            "model": model,
            "sample_rate": sample_rate,
            "settings": settings or {},
        }

        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return f"{self.CACHE_PREFIX}{key_hash}"

    async def get(self, cache_key: str) -> Optional[CachedTTSData]:
        """Retrieve cached TTS data.

        Args:
            cache_key: The cache key to look up

        Returns:
            CachedTTSData if found, None otherwise
        """
        if not self._redis_client:
            return None

        try:
            data = await self._redis_client.get(cache_key)
            if data:
                unpacked = msgpack.unpackb(data, raw=False)
                logger.debug(f"Cache hit for key: {cache_key[:20]}...")
                return CachedTTSData.from_dict(unpacked)
            return None
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None

    async def set(self, cache_key: str, data: CachedTTSData) -> bool:
        """Store TTS data in cache.

        Args:
            cache_key: The cache key to store under
            data: The TTS data to cache

        Returns:
            True if successful, False otherwise
        """
        if not self._redis_client:
            return False

        try:
            packed = msgpack.packb(data.to_dict(), use_bin_type=True)
            await self._redis_client.setex(cache_key, self._cache_ttl_seconds, packed)
            logger.debug(f"Cached TTS data for key: {cache_key[:20]}...")
            return True
        except Exception as e:
            logger.warning(f"Error storing in cache: {e}")
            return False

    async def delete(self, cache_key: str) -> bool:
        """Delete a cache entry.

        Args:
            cache_key: The cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self._redis_client:
            return False

        try:
            await self._redis_client.delete(cache_key)
            return True
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")
            return False

    async def exists(self, cache_key: str) -> bool:
        """Check if a cache entry exists.

        Args:
            cache_key: The cache key to check

        Returns:
            True if exists, False otherwise
        """
        if not self._redis_client:
            return False

        try:
            return await self._redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            return False
