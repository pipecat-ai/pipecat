#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Redis-based distributed cache backend for TTS caching."""

import pickle
from typing import Any, Dict, Optional

from loguru import logger

from pipecat.services.tts_cache.backends.base import CacheBackend
from pipecat.services.tts_cache.models import CachedTTSResponse

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache with TTL support."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "pipecat:tts:cache:",
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        **redis_kwargs,
    ):
        """Initialize Redis cache backend.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for all cache keys.
            max_connections: Maximum number of Redis connections.
            socket_timeout: Socket timeout in seconds.
            **redis_kwargs: Additional Redis client arguments.
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "RedisCacheBackend requires redis-py. "
                "Install with: pip install 'redis[asyncio]>=5.0.0'"
            )

        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._redis_kwargs = redis_kwargs
        self._client: Optional["aioredis.Redis"] = None
        logger.debug(f"Initialized RedisCacheBackend: prefix={key_prefix}")

    async def _get_client(self) -> "aioredis.Redis":
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=False,
                max_connections=self._max_connections,
                socket_timeout=self._socket_timeout,
                **self._redis_kwargs,
            )
            try:
                await self._client.ping()
                logger.debug("Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self._client = None
                raise
        return self._client

    def _make_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        """Retrieve cached response, or None if not found/expired."""
        try:
            client = await self._get_client()
            data = await client.get(self._make_key(key))

            if data is None:
                return None

            response = pickle.loads(data)
            if not isinstance(response, CachedTTSResponse):
                logger.error(f"Invalid cached data type: {type(response)}")
                return None
            return response

        except aioredis.ConnectionError as e:
            logger.error(f"Redis connection error on get: {e}")
            return None

        except (pickle.UnpicklingError, Exception) as e:
            logger.error(f"Redis get error: {e}")
            try:
                client = await self._get_client()
                await client.delete(self._make_key(key))
            except Exception:
                pass
            return None

    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        """Store a TTS response in cache. Returns True on success."""
        try:
            client = await self._get_client()
            data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)

            data_size_mb = len(data) / (1024 * 1024)
            if data_size_mb > 10:
                logger.warning(f"Large cache entry: {data_size_mb:.1f}MB for key {key[:16]}...")

            prefixed_key = self._make_key(key)
            if ttl and ttl > 0:
                await client.setex(prefixed_key, ttl, data)
            else:
                await client.set(prefixed_key, data)
            return True

        except pickle.PicklingError as e:
            logger.error(f"Redis pickle error: {e}")
            return False

        except aioredis.ConnectionError as e:
            logger.error(f"Redis connection error on set: {e}")
            return False

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries. Returns number of entries deleted."""
        try:
            client = await self._get_client()

            if namespace is None:
                search_pattern = f"{self._key_prefix}*"
            else:
                search_pattern = f"{self._key_prefix}{namespace}*"

            count = 0
            async for key in client.scan_iter(match=search_pattern, count=100):
                await client.delete(key)
                count += 1
            return count

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        try:
            client = await self._get_client()
            result = await client.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        try:
            client = await self._get_client()

            key_count = 0
            async for _ in client.scan_iter(match=f"{self._key_prefix}*", count=100):
                key_count += 1

            info = await client.info("stats")

            safe_url = self._redis_url
            if "@" in safe_url:
                safe_url = safe_url.split("@")[-1]

            return {
                "type": "redis",
                "url": safe_url,
                "prefix": self._key_prefix,
                "size": key_count,
                "redis_hits": info.get("keyspace_hits", 0),
                "redis_misses": info.get("keyspace_misses", 0),
                "redis_evicted_keys": info.get("evicted_keys", 0),
            }

        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"type": "redis", "error": str(e)}

    async def close(self) -> None:
        """Close backend connections and cleanup resources."""
        if self._client:
            try:
                await self._client.aclose()
                logger.debug("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
