import json

from config import settings
from loguru import logger
from redis import ConnectionError, Redis


class RedisService:
    """Redis service."""

    def __init__(self):
        try:
            self.redis = Redis(host=settings.redis_host, port=settings.redis_port)
            self.redis.ping()  # Test connection
            logger.info(
                f"Successfully connected to Redis at {settings.redis_host}:{settings.redis_port}"
            )
        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get_redis(self):
        return self.redis

    def set_call_details(self, call_id: str, call_details: dict):
        self.redis.set(call_id, json.dumps(call_details), ex=60 * 60)

    def get_call_details(self, call_id: str):
        try:
            data = self.redis.get(call_id)
            if data is None:
                return None
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for call_id {call_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get call details for call_id {call_id}: {e}")
            return None


redis_service = RedisService()
