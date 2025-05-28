# import json

# from config import settings
# from loguru import logger
# from redis import ConnectionError, Redis


# class RedisService:
#     """Redis service."""

#     def __init__(self):
#         try:
#             self.redis = Redis(host=settings.redis_host, port=settings.redis_port)
#             self.redis.ping()  # Test connection
#             logger.info(
#                 f"Successfully connected to Redis at {settings.redis_host}:{settings.redis_port}"
#             )
#         except ConnectionError as e:
#             logger.error(f"Failed to connect to Redis: {e}")
#             raise

#     def get_redis(self):
#         return self.redis

#     def set_call_details(self, call_id: str, call_details: dict):
#         self.redis.set(call_id, json.dumps(call_details), ex=60 * 60)

#     def get_call_details(self, call_id: str):
#         try:
#             data = self.redis.get(call_id)
#             if data is None:
#                 return None
#             return json.loads(data)
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to decode JSON for call_id {call_id}: {e}")
#             return None
#         except Exception as e:
#             logger.error(f"Failed to get call details for call_id {call_id}: {e}")
#             return None


# redis_service = RedisService()


import json
from typing import Optional

import redis.asyncio as redis
from config import settings
from loguru import logger


class RedisService:
    """Async Redis service."""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._connection_pool = None

    async def connect(
        self,
        retry_on_timeout: bool = True,
        decode_responses: bool = True,
    ):
        """Initialize async Redis connection."""
        try:
            logger.debug(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}")

            self.redis = await redis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}",
                decode_responses=decode_responses,
                retry_on_timeout=retry_on_timeout,
            )

            # Test connection
            await self.redis.ping()
            logger.info(
                f"Successfully connected to Redis at {settings.redis_host}:{settings.redis_port}"
            )
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection and cleanup resources."""
        try:
            if self.redis:
                await self.redis.aclose()
                logger.info("Redis connection closed")
            if self._connection_pool:
                await self._connection_pool.aclose()
                logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
        finally:
            self.redis = None
            self._connection_pool = None

    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self.redis is None or not await self.redis.ping():
            await self.connect()

    async def get_redis(self) -> redis.Redis:
        """Get the Redis client instance."""
        await self._ensure_connected()
        return self.redis

    async def set_call_details(self, call_id: str, call_details: dict, ttl: int = 3600):
        """Store call details in Redis with TTL.

        Args:
            call_id: Unique call identifier
            call_details: Dictionary containing call information
            ttl: Time to live in seconds (default: 1 hour)
        """
        try:
            await self._ensure_connected()
            serialized_data = json.dumps(call_details)
            await self.redis.set(call_id, serialized_data, ex=ttl)
            logger.debug(f"Stored call details for call_id: {call_id}")
        except json.JSONEncodeError as e:
            logger.error(f"Failed to serialize call details for call_id {call_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to set call details for call_id {call_id}: {e}")
            raise

    async def get_call_details(self, call_id: str) -> Optional[dict]:
        """Retrieve call details from Redis.

        Args:
            call_id: Unique call identifier

        Returns:
            Dictionary containing call details or None if not found
        """
        try:
            await self._ensure_connected()
            data = await self.redis.get(call_id)
            if data is None:
                logger.debug(f"No call details found for call_id: {call_id}")
                return None

            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for call_id {call_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get call details for call_id {call_id}: {e}")
            return None

    async def delete_call_details(self, call_id: str) -> bool:
        """Delete call details from Redis.

        Args:
            call_id: Unique call identifier

        Returns:
            True if deleted, False if key didn't exist
        """
        try:
            await self._ensure_connected()
            result = await self.redis.delete(call_id)
            if result:
                logger.debug(f"Deleted call details for call_id: {call_id}")
                return True
            else:
                logger.debug(f"No call details to delete for call_id: {call_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete call details for call_id {call_id}: {e}")
            return False

    async def exists(self, call_id: str) -> bool:
        """Check if call details exist in Redis.

        Args:
            call_id: Unique call identifier

        Returns:
            True if key exists, False otherwise
        """
        try:
            await self._ensure_connected()
            result = await self.redis.exists(call_id)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence for call_id {call_id}: {e}")
            return False

    async def extend_ttl(self, call_id: str, ttl: int = 3600) -> bool:
        """Extend the TTL of call details.

        Args:
            call_id: Unique call identifier
            ttl: New time to live in seconds

        Returns:
            True if TTL was extended, False if key doesn't exist
        """
        try:
            await self._ensure_connected()
            result = await self.redis.expire(call_id, ttl)
            if result:
                logger.debug(f"Extended TTL for call_id: {call_id} to {ttl} seconds")
                return True
            else:
                logger.debug(f"Cannot extend TTL - call_id not found: {call_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to extend TTL for call_id {call_id}: {e}")
            return False

    async def get_all_call_ids(self, pattern: str = "*") -> list[str]:
        """Get all call IDs matching a pattern.

        Args:
            pattern: Redis key pattern (default: all keys)

        Returns:
            List of call IDs
        """
        try:
            await self._ensure_connected()
            keys = await self.redis.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"Failed to get call IDs with pattern {pattern}: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if self.redis is None or not await self.redis.ping():
                return False
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


redis_service = RedisService()
