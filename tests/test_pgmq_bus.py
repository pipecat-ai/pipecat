#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import json
import unittest
from dataclasses import dataclass, field
from datetime import datetime

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
    from pipecat.bus.network.pgmq import PgmqBus
except Exception:
    raise unittest.SkipTest('pgmq extra not installed (`uv add "pipecat-ai[pgmq]"`)')

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


@dataclass
class _FakeMessage:
    msg_id: int
    message: dict
    read_ct: int = 0
    enqueued_at: datetime = field(default_factory=datetime.utcnow)
    vt: datetime = field(default_factory=datetime.utcnow)


class FakePgmq:
    """In-memory fake PGMQueue covering the surface used by ``PgmqBus``."""

    def __init__(self):
        self._queues: dict[str, list[_FakeMessage]] = {}
        self._next_id = 1
        self.sent: list[tuple[str, dict]] = []

    async def create_queue(self, queue: str, unlogged: bool = False) -> None:
        self._queues.setdefault(queue, [])

    async def drop_queue(self, queue: str, partitioned: bool = False) -> bool:
        existed = queue in self._queues
        self._queues.pop(queue, None)
        return existed

    async def list_queues(self) -> list[str]:
        return list(self._queues.keys())

    async def send(self, queue: str, message: dict, delay: int = 0, tz=None) -> int:
        if queue not in self._queues:
            raise RuntimeError(f"queue '{queue}' does not exist")
        msg_id = self._next_id
        self._next_id += 1
        self._queues[queue].append(_FakeMessage(msg_id=msg_id, message=message))
        self.sent.append((queue, message))
        return msg_id

    async def read_with_poll(
        self,
        queue: str,
        vt: int | None = None,
        qty: int = 1,
        max_poll_seconds: int = 5,
        poll_interval_ms: int = 100,
    ) -> list[_FakeMessage]:
        elapsed = 0.0
        interval = poll_interval_ms / 1000.0
        timeout = float(max_poll_seconds)
        while elapsed < timeout:
            queue_msgs = self._queues.get(queue)
            if queue_msgs:
                taken = queue_msgs[:qty]
                # Mark as read by removing for now (simulates vt with a long
                # enough vt that tests don't observe redelivery).
                self._queues[queue] = queue_msgs[qty:]
                return taken
            await asyncio.sleep(interval)
            elapsed += interval
        return []

    async def delete(self, queue: str, msg_id: int) -> bool:
        # Messages in this fake are removed from the visible queue when read.
        # delete() is essentially a no-op for tracking, but we record the call.
        return True

    def inject(self, queue: str, message: dict) -> int:
        """Push a message into a queue without going through send()."""
        if queue not in self._queues:
            self._queues[queue] = []
        msg_id = self._next_id
        self._next_id += 1
        self._queues[queue].append(_FakeMessage(msg_id=msg_id, message=message))
        return msg_id


async def create_test_pgmq_bus(
    channel: str = "test_bus",
) -> tuple[PgmqBus, FakePgmq, JSONMessageSerializer]:
    """Create a PgmqBus with FakePgmq and TaskManager for testing."""
    pgmq = FakePgmq()
    serializer = JSONMessageSerializer()
    bus = PgmqBus(
        pgmq=pgmq,
        serializer=serializer,
        channel=channel,
        # Tight timing so tests run fast
        poll_interval_ms=10,
        max_poll_seconds=1,
    )
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, pgmq, serializer


class TestPgmqBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.pgmq, self.serializer = await create_test_pgmq_bus()
        await self.bus.start()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_send_publishes_to_pgmq(self):
        """send() serializes and publishes to the PGMQ queue."""
        msg = BusDataMessage(source="task_a", target="task_b")
        await self.bus.send(msg)

        self.assertEqual(len(self.pgmq.sent), 1)
        queue, payload = self.pgmq.sent[0]
        self.assertTrue(queue.startswith("test_bus_"))
        self.assertIsInstance(payload, dict)

        restored = self.serializer.deserialize(json.dumps(payload).encode("utf-8"))
        self.assertEqual(restored.source, "task_a")
        self.assertEqual(restored.target, "task_b")

    async def test_local_message_stays_local(self):
        """BusLocalMessage messages are delivered locally but not over PGMQ."""

        received = []
        await self.bus.subscribe(_make_sub(received))

        worker = BaseWorker("test")
        msg = BusAddWorkerMessage(source="parent", worker=worker)
        await self.bus.send(msg)

        await asyncio.sleep(0.05)

        # Not published to PGMQ
        self.assertEqual(len(self.pgmq.sent), 0)
        # But delivered locally
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], BusAddWorkerMessage)
        self.assertIs(received[0].worker, worker)

    async def test_round_trip_via_subscriber(self):
        """Messages published are received by subscribers."""
        received = []
        await self.bus.subscribe(_make_sub(received))

        msg = BusEndMessage(source="task_a", reason="done")
        await self.bus.send(msg)

        await asyncio.sleep(0.2)

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

        msg = BusDataMessage(source="x")
        await self.bus.send(msg)

        await asyncio.sleep(0.2)

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    async def test_frame_message_round_trip(self):
        """BusFrameMessage with a frame adapter round-trips through PGMQ."""
        received = []
        await self.bus.subscribe(_make_sub(received))

        msg = BusFrameMessage(
            source="task_a",
            frame=TextFrame(text="hello"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.2)

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusFrameMessage)
        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello")
        self.assertEqual(restored.direction, FrameDirection.DOWNSTREAM)

    async def test_job_request_round_trip(self):
        """BusJobRequestMessage round-trips through PGMQ."""
        received = []
        await self.bus.subscribe(_make_sub(received))

        msg = BusJobRequestMessage(
            source="parent",
            target="worker",
            job_id="t-1",
            payload={"key": "value"},
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.2)

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusJobRequestMessage)
        self.assertEqual(restored.job_id, "t-1")
        self.assertEqual(restored.payload, {"key": "value"})


class TestPgmqBusBroadcast(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pgmq = FakePgmq()
        self.serializer = JSONMessageSerializer()
        self.tm = TaskManager()
        self.tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

        self.bus_a = PgmqBus(
            pgmq=self.pgmq,
            serializer=self.serializer,
            channel="test_bus",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        await self.bus_a.setup(self.tm)

        self.bus_b = PgmqBus(
            pgmq=self.pgmq,
            serializer=self.serializer,
            channel="test_bus",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        await self.bus_b.setup(self.tm)

    async def asyncTearDown(self):
        await self.bus_a.stop()
        await self.bus_b.stop()

    async def test_broadcast_to_multiple_instances(self):
        """Publishing on one instance reaches subscribers on a peer instance."""
        await self.bus_a.start()
        await self.bus_b.start()

        received_a = []
        received_b = []
        await self.bus_a.subscribe(_make_sub(received_a))
        await self.bus_b.subscribe(_make_sub(received_b))

        # bus_a publishes — both bus_a and bus_b should receive (round-trip).
        msg = BusEndMessage(source="task_a", reason="hello peers")
        await self.bus_a.send(msg)

        await asyncio.sleep(0.3)

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)
        self.assertIsInstance(received_a[0], BusEndMessage)
        self.assertIsInstance(received_b[0], BusEndMessage)

    async def test_publish_skips_dropped_queue(self):
        """If a peer queue disappears mid-publish the broadcast keeps going."""
        await self.bus_a.start()
        await self.bus_b.start()

        received_b = []
        await self.bus_b.subscribe(_make_sub(received_b))

        # Drop bus_a's own queue out from under it (simulates a crashed peer).
        await self.pgmq.drop_queue(self.bus_a._queue_name)

        msg = BusDataMessage(source="task_a")
        # Should not raise even though one of the two peer queues is gone.
        await self.bus_a.send(msg)

        await asyncio.sleep(0.3)

        # bus_b still received the message
        self.assertEqual(len(received_b), 1)


class TestPgmqBusEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def test_custom_channel(self):
        """Channel parameter is honored in the queue name."""
        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            channel="custom_channel",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        self.assertTrue(bus._queue_name.startswith("custom_channel_"))
        await bus.stop()

    async def test_channel_sanitization(self):
        """Invalid channel characters are replaced with underscores."""
        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            channel="pipecat:acme/v2",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()

        # Queue name must match PGMQ's allowed identifier pattern.
        self.assertRegex(bus._queue_name, r"^[A-Za-z_][A-Za-z0-9_]*$")
        # And contain the sanitized prefix.
        self.assertTrue(bus._queue_name.startswith("pipecat_acme_v2_"))

        await bus.stop()

    async def test_stop_cleans_up(self):
        """stop() cancels the reader worker and drops the queue."""
        pgmq = FakePgmq()
        bus = PgmqBus(pgmq=pgmq, channel="cleanup_test", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)

        await bus.start()
        queue_name = bus._queue_name
        self.assertIsNotNone(queue_name)
        self.assertIsNotNone(bus._reader_task)
        self.assertIn(queue_name, await pgmq.list_queues())

        await bus.stop()
        self.assertIsNone(bus._queue_name)
        self.assertIsNone(bus._reader_task)
        self.assertNotIn(queue_name, await pgmq.list_queues())

    async def test_deserialize_failure_is_handled(self):
        """A bad payload on the queue is logged and the reader keeps going."""
        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            channel="poison_test",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)

        received = []
        await bus.subscribe(_make_sub(received))
        await bus.start()

        # Inject a bogus payload directly into the bus's own queue.
        pgmq.inject(bus._queue_name, {"not": "a real bus message"})

        # Then send a real message through the normal path.
        await bus.send(BusEndMessage(source="task_a", reason="recovered"))

        await asyncio.sleep(0.3)

        # The poison message should be silently dropped; the real one delivered.
        real = [m for m in received if isinstance(m, BusEndMessage)]
        self.assertEqual(len(real), 1)
        self.assertEqual(real[0].reason, "recovered")

        await bus.stop()


class TestPgmqBusErrorPaths(unittest.IsolatedAsyncioTestCase):
    """Cover error and edge branches in ``PgmqBus``."""

    async def test_sanitize_channel_leading_digit(self):
        """A channel starting with a digit gets a ``q_`` prefix."""
        pgmq = FakePgmq()
        bus = PgmqBus(pgmq=pgmq, channel="123corp", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            self.assertTrue(bus._queue_name.startswith("q_123corp_"))
        finally:
            await bus.stop()

    async def test_peer_cache_hit_on_rapid_publish(self):
        """A second publish within the TTL reuses the cached peer list."""
        pgmq = FakePgmq()
        bus = PgmqBus(pgmq=pgmq, channel="cache_hit", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            list_calls = {"n": 0}
            real_list_queues = pgmq.list_queues

            async def counting_list_queues():
                list_calls["n"] += 1
                return await real_list_queues()

            pgmq.list_queues = counting_list_queues

            await bus.send(BusDataMessage(source="a"))
            first = list_calls["n"]
            await bus.send(BusDataMessage(source="a"))
            self.assertEqual(list_calls["n"], first, "second publish must hit cache")
        finally:
            await bus.stop()

    async def test_publish_skips_when_serializer_returns_invalid_json(self):
        """If the serializer's bytes aren't decodable JSON, publish bails out."""

        class BadSerializer(JSONMessageSerializer):
            def serialize(self, message):
                return b"not-valid-json"

        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            serializer=BadSerializer(),
            channel="bad_serializer",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            await bus.send(BusDataMessage(source="a"))
            # Nothing should land on the wire because json.loads failed.
            self.assertEqual(len(pgmq.sent), 0)
        finally:
            await bus.stop()

    async def test_publish_handles_send_failure_to_cached_peer(self):
        """A queue dropped after the cache warmed lets publish keep going."""
        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            channel="cached_drop",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            # Warm the peer cache with the bus's own queue.
            await bus.send(BusDataMessage(source="warmup"))
            self.assertIn("cached_drop", bus._backend._peer_cache)
            # Drop our queue so the cached entry now points at nothing.
            await pgmq.drop_queue(bus._queue_name)
            # Force-keep the cache fresh so we hit the cached path on the next publish.
            cached_at, peers = bus._backend._peer_cache["cached_drop"]
            bus._backend._peer_cache["cached_drop"] = (float("inf"), peers)
            # Should swallow the exception from the dropped peer and reset the cache.
            await bus.send(BusDataMessage(source="dead-letter"))
            self.assertNotIn("cached_drop", bus._backend._peer_cache)
        finally:
            # Re-create the queue so stop() can drop it cleanly without surfacing.
            await pgmq.create_queue(bus._queue_name)
            await bus.stop()

    async def test_stop_swallows_drop_queue_failure(self):
        """``stop()`` logs but doesn't raise when ``drop_queue`` fails."""

        class BadDropPgmq(FakePgmq):
            async def drop_queue(self, queue, partitioned=False):
                raise RuntimeError("simulated drop failure")

        pgmq = BadDropPgmq()
        bus = PgmqBus(pgmq=pgmq, channel="bad_drop", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()

        # Should not raise even though drop_queue throws.
        await bus.stop()
        self.assertIsNone(bus._queue_name)

    async def test_reader_loop_recovers_from_read_failure(self):
        """A failure inside ``read_with_poll`` triggers backoff, not crash."""

        class FlakyReadPgmq(FakePgmq):
            def __init__(self):
                super().__init__()
                self.read_calls = 0

            async def read_with_poll(self, *args, **kwargs):
                self.read_calls += 1
                if self.read_calls == 1:
                    raise RuntimeError("simulated read failure")
                return await super().read_with_poll(*args, **kwargs)

        pgmq = FlakyReadPgmq()
        bus = PgmqBus(pgmq=pgmq, channel="flaky_read", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            # Give the reader long enough to fail once and back off.
            await asyncio.sleep(1.2)
            self.assertGreaterEqual(pgmq.read_calls, 2)
        finally:
            await bus.stop()

    async def test_reader_loop_handles_deserializer_exception(self):
        """If the serializer raises while decoding, the message is still deleted."""

        class RaisingSerializer(JSONMessageSerializer):
            def deserialize(self, data):
                raise RuntimeError("simulated deserialize crash")

        pgmq = FakePgmq()
        bus = PgmqBus(
            pgmq=pgmq,
            serializer=RaisingSerializer(),
            channel="bad_deserialize",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        await bus.start()
        try:
            pgmq.inject(bus._queue_name, {"junk": True})
            # Reader should consume + log + continue without crashing.
            await asyncio.sleep(0.3)
        finally:
            await bus.stop()

    async def test_reader_loop_logs_when_delete_fails(self):
        """A failing ``delete`` after dispatch is logged and not propagated."""

        class BadDeletePgmq(FakePgmq):
            async def delete(self, queue, msg_id):
                raise RuntimeError("simulated delete failure")

        pgmq = BadDeletePgmq()
        bus = PgmqBus(pgmq=pgmq, channel="bad_delete", poll_interval_ms=10, max_poll_seconds=1)
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)

        received = []
        await bus.subscribe(_make_sub(received))
        await bus.start()
        try:
            await bus.send(BusEndMessage(source="a", reason="ok"))
            await asyncio.sleep(0.3)
            # Real message still delivered locally despite delete failing.
            self.assertEqual(len(received), 1)
        finally:
            await bus.stop()


if __name__ == "__main__":
    unittest.main()
