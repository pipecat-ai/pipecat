#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Redis pub/sub worker bus for distributed workers."""

import asyncio

from loguru import logger

from pipecat.bus.bus import WorkerBus
from pipecat.bus.messages import BusMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer

try:
    from redis.asyncio import Redis
    from redis.asyncio.client import PubSub
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use RedisBus, you need to `uv add "pipecat-ai[redis]"`.')
    raise ImportError(f"Missing module: {e}") from e


class RedisBus(WorkerBus):
    """Distributed worker bus backed by Redis pub/sub.

    Publishes serialized messages to a Redis channel for cross-process
    communication. ``BusLocalMessage`` messages bypass Redis and are
    delivered directly to local subscribers.

    Requires the ``redis[hiredis]`` package (``redis.asyncio``).

    Example::

        from redis.asyncio import Redis

        redis = Redis.from_url("redis://localhost:6379")
        bus = RedisBus(redis=redis, channel="my-session")
    """

    def __init__(
        self,
        *,
        redis: Redis,
        serializer: MessageSerializer | None = None,
        channel: str = "pipecat:bus",
        **kwargs,
    ):
        """Initialize the RedisBus.

        Args:
            redis: A ``redis.asyncio.Redis`` client instance.
            serializer: The `MessageSerializer` for encoding/decoding messages.
                Defaults to `JSONMessageSerializer`.
            channel: The Redis pub/sub channel name. Defaults to ``"pipecat:bus"``.
            **kwargs: Additional arguments passed to `WorkerBus`.
        """
        super().__init__(**kwargs)
        self._redis = redis
        self._serializer = serializer or JSONMessageSerializer()
        self._channel = channel
        self._pubsub: PubSub | None = None
        self._reader_task: asyncio.Task | None = None

    async def start(self):
        """Subscribe to Redis channel and start the reader task."""
        await super().start()
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        self._pubsub = pubsub
        self._reader_task = self.create_task(self._reader_loop())
        await asyncio.sleep(0)

    async def stop(self):
        """Stop the reader task and unsubscribe from Redis."""
        await super().stop()
        if self._reader_task:
            await self.cancel_task(self._reader_task)
            self._reader_task = None
        if self._pubsub:
            await self._pubsub.unsubscribe(self._channel)
            await self._pubsub.close()
            self._pubsub = None

    async def publish(self, message: BusMessage) -> None:
        """Publish a message to the Redis channel.

        Args:
            message: The bus message to publish.
        """
        logger.trace(f"{self}: publishing {message} to {self._channel}")
        data = self._serializer.serialize(message)
        await self._redis.publish(self._channel, data)

    async def _reader_loop(self) -> None:
        """Read messages from Redis pub/sub and deliver to subscribers."""
        assert self._pubsub is not None, "start() must be called before _reader_loop"
        async for raw_message in self._pubsub.listen():
            if raw_message["type"] != "message":
                continue
            try:
                message = self._serializer.deserialize(raw_message["data"])
                if message:
                    self.on_message_received(message)
            except Exception:
                logger.exception(f"{self}: failed to deserialize message")
