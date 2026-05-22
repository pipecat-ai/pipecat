#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.bus import AsyncQueueBus, BusBridgeProcessor, BusFrameMessage
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import run_test


class TestBusBridgeProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_frames_sent_to_bus_not_passed_through(self):
        """Non-lifecycle frames are sent to the bus, not passed through."""
        bus = AsyncQueueBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="test_task",
        )
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="hello")]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=[],
        )

        # Frame NOT passed through downstream
        text_frames = [f for f in down if isinstance(f, TextFrame)]
        self.assertEqual(len(text_frames), 0)

        # Frame sent to bus
        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 1)
        self.assertEqual(bus_frame_msgs[0].frame.text, "hello")
        self.assertEqual(bus_frame_msgs[0].source, "test_task")
        self.assertEqual(bus_frame_msgs[0].direction, FrameDirection.DOWNSTREAM)

    async def test_lifecycle_frames_pass_through_not_sent_to_bus(self):
        """Lifecycle frames pass through but are never sent to the bus."""
        bus = AsyncQueueBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="test_task",
        )
        pipeline = Pipeline([processor])

        # run_test sends StartFrame + frames_to_send + EndFrame
        # TextFrame goes to bus (not downstream), lifecycle frames pass through
        frames_to_send = [TextFrame(text="hello")]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=[],
        )

        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        # Only the TextFrame, not StartFrame or EndFrame
        self.assertEqual(len(bus_frame_msgs), 1)
        self.assertIsInstance(bus_frame_msgs[0].frame, TextFrame)

    async def test_exclude_frames_not_sent_to_bus(self):
        """Excluded frame types pass through but are not sent to the bus."""
        bus = AsyncQueueBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="test_task",
            exclude_frames=(TextFrame,),
        )
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="excluded")]
        expected_down_frames = [TextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Frame passed through
        self.assertEqual(len(down), 1)
        self.assertEqual(down[0].text, "excluded")

        # But NOT sent to bus
        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 0)

    async def test_bus_frame_injected_at_bridge(self):
        """Frames from the bus are injected at the bridge position and
        travel downstream alongside frames from later pipeline processors."""
        from pipecat.frames.frames import EndFrame
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.frame_processor import FrameProcessor

        class AppendFrameProcessor(FrameProcessor):
            """Appends a TextFrame for every TextFrame it sees."""

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.push_frame(TextFrame(text="after_bridge"), direction)

        bus = AsyncQueueBus()
        bridge = BusBridgeProcessor(
            bus=bus,
            worker_name="main_task",
        )
        pipeline = Pipeline([bridge, AppendFrameProcessor()])
        worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(worker, frame):
            received.append(frame)

        msg = BusFrameMessage(
            source="child_task",
            frame=TextFrame(text="from_child"),
            direction=FrameDirection.DOWNSTREAM,
        )

        async def inject_and_end():
            await asyncio.sleep(0.02)
            await bridge.on_bus_message(msg)
            await asyncio.sleep(0.02)
            await worker.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_and_end())

        texts = [f.text for f in received if isinstance(f, TextFrame)]
        self.assertIn("from_child", texts)
        self.assertIn("after_bridge", texts)

    async def test_skips_own_frames(self):
        """Bridge ignores bus frames from its own worker."""
        bus = AsyncQueueBus()
        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="test_task",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        # Own frame should be ignored
        msg = BusFrameMessage(
            source="test_task",
            frame=TextFrame(text="self"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(msg)

        # Should not have injected anything
        self.assertEqual(len(injected), 0)

    async def test_target_task_filtering(self):
        """Bridge with target_task only accepts frames from that worker."""
        bus = AsyncQueueBus()
        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="main_task",
            target_task="specific_child",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        # Frame from wrong worker — should be ignored
        wrong_msg = BusFrameMessage(
            source="other_child",
            frame=TextFrame(text="wrong"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(wrong_msg)
        self.assertEqual(len(injected), 0)

        # Frame from correct worker — should be injected
        right_msg = BusFrameMessage(
            source="specific_child",
            frame=TextFrame(text="right"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(right_msg)
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0].text, "right")

    async def test_targeted_message_for_other_task_skipped(self):
        """Bridge skips bus messages targeted at a different worker."""
        bus = AsyncQueueBus()
        processor = BusBridgeProcessor(
            bus=bus,
            worker_name="main_task",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        msg = BusFrameMessage(
            source="child",
            target="other_task",
            frame=TextFrame(text="not_for_me"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(msg)
        self.assertEqual(len(injected), 0)


if __name__ == "__main__":
    unittest.main()
