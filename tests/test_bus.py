#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import unittest

from pipecat.bus import (
    AsyncQueueBus,
    BusCancelMessage,
    BusCancelWorkerMessage,
    BusDataMessage,
    BusJobCancelMessage,
    BusSubscriber,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


async def create_test_bus():
    """Create an AsyncQueueBus with a TaskManager for testing."""
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, tm


class _CollectorSub(BusSubscriber):
    """Subscriber that collects messages into a list."""

    _counter = itertools.count()

    def __init__(self):
        self._name = f"collector_{next(self._counter)}"
        self.received = []

    @property
    def name(self) -> str:
        return self._name

    async def on_bus_message(self, message):
        self.received.append(message)


class TestAsyncQueueBus(unittest.IsolatedAsyncioTestCase):
    async def test_send_delivers_to_subscriber(self):
        """send() delivers a message to a subscriber."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        msg = BusDataMessage(source="task_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(sub.received), 1)
        self.assertIs(sub.received[0], msg)

    async def test_multiple_messages_in_order(self):
        """Messages are dispatched in FIFO order."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        msgs = [BusDataMessage(source=f"task_{i}") for i in range(5)]
        for m in msgs:
            await bus.send(m)
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(len(sub.received), 5)
        for sent, got in zip(msgs, sub.received):
            self.assertIs(sent, got)

    async def test_start_stop_lifecycle(self):
        """start() begins dispatch tasks, stop() cancels them cleanly."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        await bus.send(BusDataMessage(source="a"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(sub.received), 1)

        await bus.stop()

        await bus.send(BusDataMessage(source="b"))
        await asyncio.sleep(0.05)
        # After stop, messages are not dispatched
        self.assertEqual(len(sub.received), 1)


class TestBusSubscriber(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_calls_on_bus_message(self):
        """subscribe() delivers messages to subscriber's on_bus_message."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        msg = BusDataMessage(source="task_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(sub.received), 1)
        self.assertIs(sub.received[0], msg)

    async def test_multiple_subscribers_independent(self):
        """Two subscribers each get every message on their own task."""
        bus, _ = await create_test_bus()
        sub1 = _CollectorSub()
        sub2 = _CollectorSub()
        await bus.subscribe(sub1)
        await bus.subscribe(sub2)
        await bus.start()

        msg = BusDataMessage(source="task_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(sub1.received), 1)
        self.assertEqual(len(sub2.received), 1)
        self.assertIs(sub1.received[0], msg)
        self.assertIs(sub2.received[0], msg)

    async def test_unsubscribe_stops_delivery(self):
        """unsubscribe() prevents further message delivery."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        await bus.send(BusDataMessage(source="a"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(sub.received), 1)

        await bus.unsubscribe(sub)
        await bus.send(BusDataMessage(source="b"))
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(sub.received), 1)

    async def test_slow_subscriber_does_not_block_others(self):
        """A slow subscriber does not block a fast subscriber."""
        bus, _ = await create_test_bus()
        fast_received = []
        fast_done = asyncio.Event()

        class SlowSub(BusSubscriber):
            @property
            def name(self) -> str:
                return "slow"

            async def on_bus_message(self, message):
                await asyncio.sleep(0.5)

        class FastSub(BusSubscriber):
            @property
            def name(self) -> str:
                return "fast"

            async def on_bus_message(self, message):
                fast_received.append(message)
                fast_done.set()

        await bus.subscribe(SlowSub())
        await bus.subscribe(FastSub())
        await bus.start()

        await bus.send(BusDataMessage(source="a"))
        await asyncio.wait_for(fast_done.wait(), timeout=0.1)
        await bus.stop()

        self.assertEqual(len(fast_received), 1)


class TestBusMessagePriority(unittest.IsolatedAsyncioTestCase):
    async def test_system_message_preempts_data_messages(self):
        """System messages are delivered before data messages queued earlier."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)

        # Queue messages before starting dispatch
        for i in range(5):
            await bus.send(BusDataMessage(source=f"data_{i}"))
        await bus.send(BusCancelMessage(source="runner", reason="urgent"))

        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        # System message should be first
        self.assertIsInstance(sub.received[0], BusCancelMessage)
        self.assertEqual(sub.received[0].source, "runner")

    async def test_data_messages_preserve_fifo_order(self):
        """Data messages maintain FIFO order among themselves."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        for i in range(5):
            await bus.send(BusDataMessage(source=f"task_{i}"))
        await asyncio.sleep(0.1)
        await bus.stop()

        sources = [m.source for m in sub.received]
        self.assertEqual(sources, [f"task_{i}" for i in range(5)])

    async def test_system_messages_preserve_fifo_order(self):
        """Multiple system messages maintain FIFO order among themselves."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)
        await bus.start()

        await bus.send(BusCancelMessage(source="first", reason="a"))
        await bus.send(BusCancelMessage(source="second", reason="b"))
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(sub.received[0].source, "first")
        self.assertEqual(sub.received[1].source, "second")

    async def test_mixed_messages_system_first(self):
        """When data and system messages are queued, all system come first."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)

        # Queue before starting so all messages are pending
        await bus.send(BusDataMessage(source="data_1"))
        await bus.send(BusDataMessage(source="data_2"))
        await bus.send(BusCancelMessage(source="cancel_1"))
        await bus.send(BusDataMessage(source="data_3"))
        await bus.send(BusCancelWorkerMessage(source="cancel_2", target="task"))

        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        sources = [m.source for m in sub.received]
        self.assertEqual(sources, ["cancel_1", "cancel_2", "data_1", "data_2", "data_3"])

    async def test_task_cancel_is_system_priority(self):
        """BusJobCancelMessage has system priority."""
        bus, _ = await create_test_bus()
        sub = _CollectorSub()
        await bus.subscribe(sub)

        await bus.send(BusDataMessage(source="data"))
        await bus.send(BusJobCancelMessage(source="parent", target="worker", job_id="t1"))

        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertIsInstance(sub.received[0], BusJobCancelMessage)


class _FailingSub(BusSubscriber):
    """Subscriber that raises on its first message, then records the rest.

    Models a subscriber whose lifecycle hook (e.g. ``on_job_response``) throws:
    the bus dispatch task must survive so later messages are still delivered.
    """

    _counter = itertools.count()

    def __init__(self):
        self._name = f"failing_{next(self._counter)}"
        self.received = []
        self._raised = False

    @property
    def name(self) -> str:
        return self._name

    async def on_bus_message(self, message):
        if not self._raised:
            self._raised = True
            raise RuntimeError("simulated subscriber failure")
        self.received.append(message)


class TestSubscriberException(unittest.IsolatedAsyncioTestCase):
    """A subscriber exception must not stop future delivery to that subscriber."""

    async def test_data_dispatch_survives_subscriber_exception(self):
        """A raise in on_bus_message must not tear down the data dispatch task."""
        bus, _ = await create_test_bus()
        sub = _FailingSub()
        await bus.subscribe(sub)
        await bus.start()

        # First data message raises inside _data_dispatch_task.
        await bus.send(BusDataMessage(source="first"))
        await asyncio.sleep(0.05)
        # Second data message must still be delivered (was lost before the fix).
        await bus.send(BusDataMessage(source="second"))
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual([m.source for m in sub.received], ["second"])

    async def test_router_survives_subscriber_exception_on_system_message(self):
        """A raise on a system message must not tear down the router task."""
        bus, _ = await create_test_bus()
        sub = _FailingSub()
        await bus.subscribe(sub)
        await bus.start()

        # First system message raises inline inside _router_task.
        await bus.send(BusCancelMessage(source="first"))
        await asyncio.sleep(0.05)
        # Second system message must still be delivered (was lost before the fix).
        await bus.send(BusCancelMessage(source="second"))
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual([m.source for m in sub.received], ["second"])


if __name__ == "__main__":
    unittest.main()
