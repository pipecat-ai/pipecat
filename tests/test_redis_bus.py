#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import unittest

from pipecat.bus import (
    BusAddWorkerMessage,
    BusDataMessage,
    BusEndMessage,
    BusFrameMessage,
    BusJobRequestMessage,
    BusSubscriber,
)
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.base_worker import BaseWorker

try:
    from pipecat.bus.network.redis import RedisBus
except Exception:
    raise unittest.SkipTest('redis extra not installed (`uv add "pipecat-ai[redis]"`)')

_sub_counter = itertools.count()


def _make_sub(received: list) -> BusSubscriber:
    """Create a BusSubscriber that appends messages to the given list."""
    sub_name = f"test_sub_{next(_sub_counter)}"

    class _Sub(BusSubscriber):
        @property
        def name(self) -> str:
            return sub_name

        async def on_bus_message(self, message):
            received.append(message)

    return _Sub()


class FakePubSub:
    """In-memory fake Redis pub/sub for testing."""

    def __init__(self):
        self._subscriptions: dict[str, asyncio.Queue] = {}
        self._closed = False

    async def subscribe(self, channel: str):
        self._subscriptions[channel] = asyncio.Queue()

    async def unsubscribe(self, channel: str):
        self._subscriptions.pop(channel, None)

    async def close(self):
        self._closed = True

    async def listen(self):
        """Yield messages from the subscription queue."""
        # We only support one channel in tests
        channel = next(iter(self._subscriptions))
        queue = self._subscriptions[channel]
        while True:
            msg = await queue.get()
            yield msg

    def inject(self, channel: str, data: bytes):
        """Inject a raw message into the fake pub/sub."""
        if channel in self._subscriptions:
            self._subscriptions[channel].put_nowait(
                {"type": "message", "data": data, "channel": channel}
            )


class FakeRedis:
    """In-memory fake Redis client for testing."""

    def __init__(self):
        self._pubsubs: list[FakePubSub] = []
        self._published: list[tuple[str, bytes]] = []

    def pubsub(self):
        ps = FakePubSub()
        self._pubsubs.append(ps)
        return ps

    async def publish(self, channel: str, data: bytes):
        self._published.append((channel, data))
        # Fan out to all pubsub instances
        for ps in self._pubsubs:
            ps.inject(channel, data)


async def create_test_redis_bus():
    """Create a RedisBus with fake Redis and TaskManager for testing."""
    redis = FakeRedis()
    serializer = JSONMessageSerializer()
    bus = RedisBus(redis=redis, serializer=serializer, channel="test:bus")
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, redis, serializer


class TestRedisBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.redis, self.serializer = await create_test_redis_bus()

    async def test_send_publishes_to_redis(self):
        """send() serializes and publishes to the Redis channel."""
        msg = BusDataMessage(source="task_a", target="task_b")
        await self.bus.send(msg)

        self.assertEqual(len(self.redis._published), 1)
        channel, data = self.redis._published[0]
        self.assertEqual(channel, "test:bus")
        self.assertIsInstance(data, bytes)

        # Verify it deserializes back
        restored = self.serializer.deserialize(data)
        self.assertEqual(restored.source, "task_a")
        self.assertEqual(restored.target, "task_b")

    async def test_local_mixin_delivered_locally_not_to_redis(self):
        """BusLocalMessage messages are delivered to local subscribers but not published to Redis."""
        received = []
        await self.bus.subscribe(_make_sub(received))
        await self.bus.start()

        worker = BaseWorker("test")
        msg = BusAddWorkerMessage(source="parent", worker=worker)
        await self.bus.send(msg)

        await asyncio.sleep(0.05)
        await self.bus.stop()

        # Not published to Redis
        self.assertEqual(len(self.redis._published), 0)
        # But delivered locally
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], BusAddWorkerMessage)
        self.assertIs(received[0].worker, worker)

    async def test_round_trip_via_subscriber(self):
        """Messages published are received by subscribers."""
        received = []
        await self.bus.subscribe(_make_sub(received))
        await self.bus.start()

        msg = BusEndMessage(source="task_a", reason="done")
        await self.bus.send(msg)

        # Give the reader worker time to process
        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], BusEndMessage)
        self.assertEqual(received[0].source, "task_a")
        self.assertEqual(received[0].reason, "done")

    async def test_multiple_subscribers_receive(self):
        """Multiple subscribers each receive every message."""
        received_a = []
        received_b = []
        await self.bus.subscribe(_make_sub(received_a))
        await self.bus.subscribe(_make_sub(received_b))
        await self.bus.start()

        msg = BusDataMessage(source="x")
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    async def test_frame_message_round_trip(self):
        """BusFrameMessage with a frame adapter round-trips through Redis."""
        received = []
        await self.bus.subscribe(_make_sub(received))
        await self.bus.start()

        msg = BusFrameMessage(
            source="task_a",
            frame=TextFrame(text="hello"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusFrameMessage)
        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello")
        self.assertEqual(restored.direction, FrameDirection.DOWNSTREAM)

    async def test_job_request_round_trip(self):
        """BusJobRequestMessage round-trips through Redis."""
        received = []
        await self.bus.subscribe(_make_sub(received))
        await self.bus.start()

        msg = BusJobRequestMessage(
            source="parent",
            target="worker",
            job_id="t-1",
            payload={"key": "value"},
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusJobRequestMessage)
        self.assertEqual(restored.job_id, "t-1")
        self.assertEqual(restored.payload, {"key": "value"})

    async def test_custom_channel(self):
        """Messages are published to the configured channel."""
        bus = RedisBus(
            redis=self.redis,
            serializer=self.serializer,
            channel="custom:channel",
        )
        await bus.send(BusDataMessage(source="a"))
        self.assertEqual(self.redis._published[0][0], "custom:channel")

    async def test_stop_cleans_up(self):
        """stop() cancels the reader worker and unsubscribes from Redis."""
        await self.bus.start()
        self.assertIsNotNone(self.bus._pubsub)
        self.assertIsNotNone(self.bus._reader_task)

        await self.bus.stop()
        self.assertIsNone(self.bus._pubsub)
        self.assertIsNone(self.bus._reader_task)


if __name__ == "__main__":
    unittest.main()
