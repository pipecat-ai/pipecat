#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.bus import (
    AsyncQueueBus,
    BusDataMessage,
    BusFrameMessage,
    BusSubscriber,
)
from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestBusMessageRouting(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_message_no_target(self):
        """A BusMessage with no target is broadcast (target is None)."""
        msg = BusDataMessage(source="task_a")
        self.assertIsNone(msg.target)

    async def test_targeted_message(self):
        """A BusMessage with target set is only for that task."""
        msg = BusDataMessage(source="task_a", target="task_b")
        self.assertEqual(msg.target, "task_b")

    async def test_broadcast_reaches_all_subscribers(self):
        """Broadcast messages (no target) reach all subscribers."""
        bus = AsyncQueueBus()
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)
        received_a = []
        received_b = []

        class SubA(BusSubscriber):
            @property
            def name(self) -> str:
                return "sub_a"

            async def on_bus_message(self, message):
                received_a.append(message)

        class SubB(BusSubscriber):
            @property
            def name(self) -> str:
                return "sub_b"

            async def on_bus_message(self, message):
                received_b.append(message)

        await bus.subscribe(SubA())
        await bus.subscribe(SubB())

        await bus.start()
        msg = BusDataMessage(source="task_x")  # no target
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        # Both subscribers see the broadcast
        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    async def test_bus_frame_message_wraps_frame(self):
        """BusFrameMessage wraps a frame with source and direction."""
        frame = TextFrame(text="hello")
        msg = BusFrameMessage(
            source="task_a",
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
        )
        self.assertIs(msg.frame, frame)
        self.assertEqual(msg.direction, FrameDirection.DOWNSTREAM)
        self.assertEqual(msg.source, "task_a")
        self.assertIsNone(msg.target)

    async def test_bus_frame_message_with_target(self):
        """BusFrameMessage can carry a target for directed delivery."""
        frame = TextFrame(text="hello")
        msg = BusFrameMessage(
            source="task_a",
            target="task_b",
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
        )
        self.assertEqual(msg.target, "task_b")
        self.assertIs(msg.frame, frame)


if __name__ == "__main__":
    unittest.main()
