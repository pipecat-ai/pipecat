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
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.workers.runner import WorkerRunner

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

        runner = WorkerRunner()
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

    async def test_duplicate_frame_id_from_sibling_dropped(self):
        """Sibling re-broadcast of the same frame.id is dropped at the receiver.

        In a multi-worker setup, sibling workers briefly co-existing as
        ``active=True`` during a handoff transient re-broadcast frames
        originally emitted by another worker. Without dedup the host
        bridge delivers the same ``frame.id`` once per re-broadcasting
        sibling, which pollutes the assistant aggregator's LLM context.

        Regression: same ``frame.id`` received twice (once from the
        original emitter, once from a sibling) is pushed only once.
        """
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

        # Same TextFrame instance (same ``frame.id``) re-broadcast by a
        # sibling worker during the handoff transient.
        original_frame = TextFrame(text="result")

        first_delivery = BusFrameMessage(
            source="child_a",
            frame=original_frame,
            direction=FrameDirection.DOWNSTREAM,
        )
        sibling_rebroadcast = BusFrameMessage(
            source="child_b",
            frame=original_frame,
            direction=FrameDirection.DOWNSTREAM,
        )

        await processor.on_bus_message(first_delivery)
        await processor.on_bus_message(sibling_rebroadcast)

        # Only the first delivery reaches the pipeline.
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0].text, "result")

    async def test_dedup_distinct_frame_ids_all_delivered(self):
        """Distinct ``frame.id``s are always delivered (dedup is by id).

        Sanity check: the dedup key is the frame id, not the frame type
        or content. Two frames with different ids must both pass through.
        """
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

        # Two distinct frames (different ``frame.id``) with identical text.
        first = TextFrame(text="same_text")
        second = TextFrame(text="same_text")
        self.assertNotEqual(first.id, second.id)

        await processor.on_bus_message(
            BusFrameMessage(
                source="child",
                frame=first,
                direction=FrameDirection.DOWNSTREAM,
            )
        )
        await processor.on_bus_message(
            BusFrameMessage(
                source="child",
                frame=second,
                direction=FrameDirection.DOWNSTREAM,
            )
        )

        self.assertEqual(len(injected), 2)

    async def test_dedup_lru_eviction_rebuilds_set_from_history(self):
        """LRU eviction path: bounded deque + set stay consistent.

        The dedup state is a ``set[int]`` (O(1) membership) backed by a
        ``deque(maxlen=512)`` history. When the 513th distinct id is
        appended, the deque silently evicts its oldest entry. The set
        does not get the eviction signal, so it must be rebuilt from the
        history to keep memory bounded and to allow very old ids to be
        delivered again if they ever recur.

        Regression: after pushing more than 512 distinct frames, the
        oldest evicted id is no longer in the dedup set (so a hypothetical
        re-delivery would pass through), and the most recent 512 ids
        remain deduplicated.
        """
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

        # Push 513 distinct frames → deque is full (maxlen=512) and the
        # first frame's id has been evicted from the history. The set
        # rebuild branch must fire on the 513th delivery.
        frames = [TextFrame(text=f"f{i}") for i in range(513)]
        for f in frames:
            await processor.on_bus_message(
                BusFrameMessage(
                    source="child",
                    frame=f,
                    direction=FrameDirection.DOWNSTREAM,
                )
            )

        # All 513 distinct ids were delivered exactly once.
        self.assertEqual(len(injected), 513)

        # Dedup state is now bounded to the deque length (set was
        # rebuilt from history when it overgrew the deque).
        self.assertEqual(len(processor._seen_bus_history), 512)
        self.assertEqual(len(processor._seen_bus_ids), 512)
        self.assertEqual(processor._seen_bus_ids, set(processor._seen_bus_history))

        # The oldest frame's id has been evicted from both structures —
        # if the same id ever recurred, it would no longer be dropped.
        self.assertNotIn(frames[0].id, processor._seen_bus_ids)
        # The most recent 512 ids are still tracked.
        self.assertIn(frames[-1].id, processor._seen_bus_ids)
        self.assertIn(frames[1].id, processor._seen_bus_ids)


if __name__ == "__main__":
    unittest.main()
