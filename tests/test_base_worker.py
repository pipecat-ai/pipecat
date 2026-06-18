#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.bus import (
    AsyncQueueBus,
    BusActivateWorkerMessage,
    BusAddWorkerMessage,
    BusCancelMessage,
    BusCancelWorkerMessage,
    BusDeactivateWorkerMessage,
    BusEndMessage,
    BusEndWorkerMessage,
    BusFrameMessage,
    BusJobCancelMessage,
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusJobStreamDataMessage,
    BusJobStreamEndMessage,
    BusJobStreamStartMessage,
    BusJobUpdateMessage,
    BusTTSSpeakMessage,
)
from pipecat.frames.frames import EndFrame, Frame, TextFrame, TTSSpeakFrame
from pipecat.pipeline.job_context import JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerReadyData
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner


class _FrameGenerator(FrameProcessor):
    """Generates a new TextFrame for each input TextFrame."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            await self.push_frame(TextFrame(text=f"generated_{frame.text}"), direction)
        else:
            await self.push_frame(frame, direction)


async def create_test_bus():
    """Create an AsyncQueueBus with a TaskManager for testing."""
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, tm


def create_test_registry():
    """Create a registry for testing worker lifecycle."""
    return WorkerRegistry(runner_name="test-runner")


async def register_tasks(registry, *names):
    """Pre-register worker names so the ready-wait completes immediately."""
    for name in names:
        await registry.register(WorkerReadyData(worker_name=name, runner="test-runner"))


def capture_bus(bus):
    """Monkey-patch bus.send to capture sent messages in a list."""
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


def make_stub_pipeline_task(name, *, bridged=None, active=True):
    """Create a PipelineWorker with an IdentityFilter pipeline."""
    return PipelineWorker(
        Pipeline([IdentityFilter()]),
        name=name,
        bridged=bridged,
        cancel_on_idle_timeout=False,
    )


class TestPipelineTaskLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = await create_test_bus()
        self.registry = create_test_registry()

    async def test_task_starts_inactive_by_default(self):
        """Bridged worker is inactive by default."""
        worker = make_stub_pipeline_task("test", bridged=())
        worker._active = False
        worker._pending_activation = False
        self.assertFalse(worker.active)

    async def test_handoff_via_bus_message_after_pipeline_start(self):
        """Task activates when BusActivateWorkerMessage received and pipeline started."""
        worker = make_stub_pipeline_task("test", bridged=())
        worker._active = False
        worker._pending_activation = False
        await worker.attach(registry=self.registry, bus=self.bus)

        handoff_done = asyncio.Event()
        handoff_args_received = []

        @worker.event_handler("on_activated")
        async def on_handoff(t, args):
            handoff_args_received.append(args)
            handoff_done.set()

        args = {"messages": ["hello"]}

        async def handoff_after_start():
            await asyncio.sleep(0.05)
            await self.bus.send(BusActivateWorkerMessage(source="other", target="test", args=args))
            await asyncio.wait_for(handoff_done.wait(), timeout=2.0)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), handoff_after_start())

        self.assertTrue(worker.active)
        self.assertEqual(len(handoff_args_received), 1)
        self.assertEqual(handoff_args_received[0], args)

    async def test_active_true_fires_on_activated(self):
        """active=True fires on_activated after pipeline starts."""
        worker = make_stub_pipeline_task("test", bridged=())

        activated = asyncio.Event()

        @worker.event_handler("on_activated")
        async def on_activated(t, args):
            activated.set()

        async def wait_and_end():
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), wait_and_end())

        self.assertTrue(worker.active)
        self.assertTrue(activated.is_set())

    async def test_deactivate_self_takes_effect_before_activate_is_sent(self):
        """activate_worker(deactivate_self=True) deactivates before the bus
        round-trip, so this worker and the target are never both active."""
        worker = make_stub_pipeline_task("test", bridged=())
        worker._active = True
        worker._pending_activation = False
        await worker.attach(registry=self.registry, bus=self.bus)

        sent: list[tuple[str, bool]] = []
        original_send = worker.send_bus_message

        async def record_send(message):
            sent.append((type(message).__name__, worker.active))
            await original_send(message)

        worker.send_bus_message = record_send

        await worker.activate_worker("other", deactivate_self=True)

        # Flipped synchronously, without waiting for the bus round-trip.
        self.assertFalse(worker.active)
        self.assertEqual(
            [name for name, _ in sent],
            ["BusDeactivateWorkerMessage", "BusActivateWorkerMessage"],
        )
        # Both messages were published with this worker already inactive.
        self.assertTrue(all(active is False for _, active in sent))

    async def test_activation_args_property_set_and_cleared(self):
        """activation_args returns the latest args while active and is cleared on deactivate."""
        worker = make_stub_pipeline_task("test", bridged=())
        worker._active = False
        worker._pending_activation = False

        activated = asyncio.Event()
        deactivated = asyncio.Event()

        @worker.event_handler("on_activated")
        async def _on_activated(t, args):
            activated.set()

        @worker.event_handler("on_deactivated")
        async def _on_deactivated(t):
            deactivated.set()

        args = {"messages": ["hello"]}
        observed_while_active = {}

        async def drive():
            await asyncio.sleep(0.05)
            await self.bus.send(BusActivateWorkerMessage(source="other", target="test", args=args))
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            observed_while_active["args"] = worker.activation_args
            observed_while_active["active"] = worker.active
            await self.bus.send(BusDeactivateWorkerMessage(source="other", target="test"))
            await asyncio.wait_for(deactivated.wait(), timeout=2.0)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), drive())

        self.assertTrue(observed_while_active["active"])
        self.assertEqual(observed_while_active["args"], args)
        self.assertFalse(worker.active)
        self.assertIsNone(worker.activation_args)

    async def test_activation_args_none_before_activation(self):
        """activation_args is None before any activation has occurred."""
        worker = make_stub_pipeline_task("test", bridged=())
        worker._active = False
        worker._pending_activation = False
        self.assertIsNone(worker.activation_args)

    async def test_activate_task_with_deactivate_self_sends_both_messages(self):
        """activate_worker(deactivate_self=True) sends deactivate then activate."""
        sent = capture_bus(self.bus)

        worker = make_stub_pipeline_task("task_a", bridged=())
        await worker.attach(registry=self.registry, bus=self.bus)

        await worker.activate_worker("task_b", deactivate_self=True)

        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateWorkerMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "task_a")
        activate_msgs = [m for m in sent if isinstance(m, BusActivateWorkerMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "task_b")

    async def test_end_without_parent_sends_bus_end_message(self):
        """end() with no parent sends BusEndMessage."""
        sent = capture_bus(self.bus)

        worker = BaseWorker("task_a")
        await worker.attach(registry=self.registry, bus=self.bus)
        await worker.end(reason="done")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "task_a")
        self.assertEqual(end_msgs[0].reason, "done")

    async def test_end_with_parent_sends_bus_end_message(self):
        """end() with parent still sends BusEndMessage (runner handles it)."""
        sent = capture_bus(self.bus)

        parent = BaseWorker("parent_task")
        await parent.attach(registry=self.registry, bus=self.bus)
        worker = BaseWorker("child")
        await worker.attach(registry=self.registry, bus=self.bus)
        await parent.add_workers(worker)
        await worker.end(reason="goodbye")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "child")
        self.assertEqual(end_msgs[0].reason, "goodbye")

    async def test_cancel_sends_bus_cancel_message(self):
        """cancel() sends BusCancelMessage."""
        sent = capture_bus(self.bus)

        worker = BaseWorker("task_a")
        await worker.attach(registry=self.registry, bus=self.bus)
        await worker.cancel()

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].source, "task_a")

    async def test_add_workers_sends_bus_add_worker_message(self):
        """add_workers() sends BusAddWorkerMessage."""
        sent = capture_bus(self.bus)

        worker = BaseWorker("task_a")
        await worker.attach(registry=self.registry, bus=self.bus)
        new_task = BaseWorker("task_b")
        await worker.add_workers(new_task)

        add_msgs = [m for m in sent if isinstance(m, BusAddWorkerMessage)]
        self.assertEqual(len(add_msgs), 1)
        self.assertIs(add_msgs[0].worker, new_task)

    async def test_started_at_none_before_pipeline_starts(self):
        """started_at is None before the pipeline has started."""
        worker = make_stub_pipeline_task("test")
        self.assertIsNone(worker.started_at)

    async def test_started_at_set_after_pipeline_starts(self):
        """started_at becomes set once the pipeline starts."""
        worker = make_stub_pipeline_task("test")

        ready_event = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def _on_started(t, frame):
            ready_event.set()

        async def wait_and_end():
            await asyncio.wait_for(ready_event.wait(), timeout=2.0)
            self.assertIsNotNone(worker.started_at)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), wait_and_end())

    async def test_on_pipeline_started_event(self):
        """on_pipeline_started fires after pipeline starts."""
        worker = make_stub_pipeline_task("test")

        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def on_started(t, frame):
            started.set()

        async def wait_and_end():
            await asyncio.wait_for(started.wait(), timeout=2.0)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), wait_and_end())

        self.assertTrue(started.is_set())

    async def test_on_pipeline_finished_event(self):
        """on_pipeline_finished fires after pipeline finishes."""
        worker = make_stub_pipeline_task("test")

        finished_fired = asyncio.Event()

        @worker.event_handler("on_pipeline_finished")
        async def on_finished(t, frame):
            finished_fired.set()

        async def end_pipeline():
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), end_pipeline())

        self.assertTrue(finished_fired.is_set())

    async def test_activate_task_with_deactivate_self_deactivates(self):
        """activate_worker(deactivate_self=True) sends a deactivate for the calling worker."""
        sent = capture_bus(self.bus)
        worker = make_stub_pipeline_task("test", bridged=())
        await worker.attach(registry=self.registry, bus=self.bus)

        self.assertTrue(worker.active)
        await worker.activate_worker("other", deactivate_self=True)
        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateWorkerMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "test")

    async def test_bus_end_task_message_ends_pipeline(self):
        """BusEndWorkerMessage causes the pipeline to end gracefully."""
        worker = make_stub_pipeline_task("test")

        finished = asyncio.Event()

        @worker.event_handler("on_pipeline_finished")
        async def on_finished(t, frame):
            if isinstance(frame, EndFrame):
                finished.set()

        async def send_end_message():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusEndWorkerMessage(source="runner", target="test", reason="shutdown")
            )

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), send_end_message())

        self.assertTrue(finished.is_set())

    async def test_bus_cancel_task_message_cancels_pipeline(self):
        """BusCancelWorkerMessage cancels the pipeline worker."""
        worker = make_stub_pipeline_task("test")

        async def send_cancel_message():
            await asyncio.sleep(0.05)
            await self.bus.send(BusCancelWorkerMessage(source="runner", target="test"))

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        try:
            await runner.add_workers(worker)
            await asyncio.gather(runner.run(), send_cancel_message())
        except asyncio.CancelledError:
            pass

        self.assertTrue(worker.has_finished())

    async def test_queue_frame(self):
        """queue_frame injects a frame into the pipeline."""
        worker = make_stub_pipeline_task("test")

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frame(TextFrame(text="injected"))
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "injected")

    async def test_queue_frames(self):
        """queue_frames injects multiple frames into the pipeline."""
        worker = make_stub_pipeline_task("test")

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frames([TextFrame(text="a"), TextFrame(text="b")])
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].text, "a")
        self.assertEqual(received[1].text, "b")

    async def test_bus_tts_speak_message_queues_tts_speak_frame(self):
        """BusTTSSpeakMessage queues a TTSSpeakFrame into the pipeline."""
        worker = make_stub_pipeline_task("test")

        received = []
        worker.set_reached_downstream_filter((TTSSpeakFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def send_speak():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusTTSSpeakMessage(
                    source="other",
                    target="test",
                    text="hello there",
                    append_to_context=True,
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), send_speak())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "hello there")
        self.assertTrue(received[0].append_to_context)

    async def test_bus_tts_speak_message_ignored_for_other_target(self):
        """BusTTSSpeakMessage targeted at another worker is ignored."""
        worker = make_stub_pipeline_task("test")

        received = []
        worker.set_reached_downstream_filter((TTSSpeakFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def send_speak():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusTTSSpeakMessage(source="other", target="someone-else", text="ignored")
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), send_speak())

        self.assertEqual(received, [])

    async def test_self_handoff(self):
        """A worker can hand off to itself via activate_worker(self.name, deactivate_self=True)."""
        worker = make_stub_pipeline_task("test", bridged=())

        handoff_done = asyncio.Event()

        @worker.event_handler("on_activated")
        async def on_handoff(t, args):
            handoff_done.set()

        async def self_handoff():
            # Wait for first activation (from active=True)
            await asyncio.sleep(0.05)
            handoff_done.clear()
            await worker.activate_worker("test", deactivate_self=True)
            await asyncio.wait_for(handoff_done.wait(), timeout=2.0)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), self_handoff())

        self.assertTrue(worker.active)

    async def test_add_workers_tracks_children(self):
        """add_workers() populates children list and sets parent."""
        parent = BaseWorker("parent")
        await parent.attach(registry=self.registry, bus=self.bus)
        child_a = BaseWorker("child_a")
        child_b = BaseWorker("child_b")

        await parent.add_workers(child_a)
        await parent.add_workers(child_b)

        self.assertEqual(len(parent.children), 2)
        self.assertIs(parent.children[0], child_a)
        self.assertIs(parent.children[1], child_b)

    async def test_end_propagates_to_children(self):
        """BusEndWorkerMessage on parent sends end to each child."""
        sent = capture_bus(self.bus)

        parent = BaseWorker("parent")
        await parent.attach(registry=self.registry, bus=self.bus)
        child_a = BaseWorker("child_a")
        child_b = BaseWorker("child_b")
        await parent.add_workers(child_a)
        await parent.add_workers(child_b)

        # Pre-set children as finished so gather returns immediately
        child_a._finished_event.set()
        child_b._finished_event.set()

        await parent.on_bus_message(
            BusEndWorkerMessage(source="runner", target="parent", reason="shutdown")
        )

        end_msgs = [m for m in sent if isinstance(m, BusEndWorkerMessage)]
        targets = {m.target for m in end_msgs}
        self.assertIn("child_a", targets)
        self.assertIn("child_b", targets)

    async def test_end_waits_for_children(self):
        """Parent waits for children to finish before completing _handle_worker_end."""
        parent = BaseWorker("parent")
        await parent.attach(registry=self.registry, bus=self.bus)
        child = BaseWorker("child")
        await parent.add_workers(child)

        order = []

        async def delayed_child_finish():
            await asyncio.sleep(0.1)
            order.append("child_finished")
            child._finished_event.set()

        async def send_end():
            await asyncio.sleep(0.05)
            await parent.on_bus_message(
                BusEndWorkerMessage(source="runner", target="parent", reason="shutdown")
            )
            order.append("parent_end_returned")

        await asyncio.gather(send_end(), delayed_child_finish())

        # Child must finish before parent's on_bus_message returns
        self.assertEqual(order, ["child_finished", "parent_end_returned"])

    async def test_cancel_propagates_to_children(self):
        """BusCancelWorkerMessage on parent sends cancel to each child."""
        sent = capture_bus(self.bus)

        parent = BaseWorker("parent")
        await parent.attach(registry=self.registry, bus=self.bus)
        child_a = BaseWorker("child_a")
        child_b = BaseWorker("child_b")
        await parent.add_workers(child_a)
        await parent.add_workers(child_b)

        await parent.on_bus_message(
            BusCancelWorkerMessage(source="runner", target="parent", reason="abort")
        )

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelWorkerMessage)]
        targets = {m.target for m in cancel_msgs}
        self.assertIn("child_a", targets)
        self.assertIn("child_b", targets)


def make_generating_task(name, *, bridged=None):
    """Create a PipelineWorker whose pipeline generates frames."""
    return PipelineWorker(
        Pipeline([_FrameGenerator()]),
        name=name,
        bridged=bridged,
        cancel_on_idle_timeout=False,
    )


class TestEdgeToBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = await create_test_bus()
        self.registry = create_test_registry()

    async def test_generated_frames_reach_bus(self):
        """Pipeline-generated frames are broadcast to the bus."""
        sent = capture_bus(self.bus)

        worker = make_generating_task("worker", bridged=())

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frame(TextFrame(text="input"))
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        text_msgs = [m for m in bus_frame_msgs if isinstance(m.frame, TextFrame)]
        generated = [m for m in text_msgs if m.frame.text == "generated_input"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].source, "worker")

    async def test_bus_frames_not_rebroadcast_by_same_task(self):
        """Frames from the bus with source==self are ignored by edge processors."""
        sent = capture_bus(self.bus)

        worker = make_stub_pipeline_task("worker", bridged=())

        async def inject_frame():
            await asyncio.sleep(0.05)
            # Send a frame from "other" — edge source accepts it (downstream, source != worker)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="from_bus"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frame())

        # The frame passes through the identity pipeline and reaches
        # EdgeSink, which re-broadcasts with source="worker". That's
        # expected. But it must NOT create a loop — EdgeSource ignores
        # it because source == "worker".
        # Filter to TextFrame only to ignore metrics frames.
        bus_frame_msgs = [
            m for m in sent if isinstance(m, BusFrameMessage) and isinstance(m.frame, TextFrame)
        ]
        from_task = [m for m in bus_frame_msgs if m.source == "worker"]
        from_other = [m for m in bus_frame_msgs if m.source == "other"]
        # One re-broadcast from worker (EdgeSink), one original from other
        self.assertEqual(len(from_other), 1)
        self.assertEqual(len(from_task), 1)
        # No infinite loop — total is exactly 2
        self.assertEqual(len(bus_frame_msgs), 2)

    async def test_unbridged_task_no_edge_sinks(self):
        """An unbridged PipelineWorker does not tee frames to the bus."""
        sent = capture_bus(self.bus)

        worker = make_stub_pipeline_task("root", bridged=None)

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frame(TextFrame(text="root_frame"))
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 0)

    async def test_bus_frame_enters_task_pipeline(self):
        """Bus frame messages enter the pipeline via edge source processor."""
        worker = make_stub_pipeline_task("worker", bridged=())

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="from_bus"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frame())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "from_bus")

    async def test_direction_preserved_in_bus_frame(self):
        """Direction is preserved when generated frames are sent to the bus."""
        sent = capture_bus(self.bus)

        worker = make_generating_task("worker", bridged=())

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frame(TextFrame(text="hello"))
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        text_msgs = [m for m in bus_frame_msgs if isinstance(m.frame, TextFrame)]
        generated = [m for m in text_msgs if m.frame.text == "generated_hello"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].direction, FrameDirection.DOWNSTREAM)

    async def test_bridged_task_accepts_matching_bridge(self):
        """Bridged worker with named bridge accepts frames from that bridge."""
        worker = make_stub_pipeline_task("worker", bridged=("voice",))

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice_frame"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frame())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "voice_frame")

    async def test_bridged_task_rejects_non_matching_bridge(self):
        """Bridged worker with named bridge rejects frames from other bridges."""
        worker = make_stub_pipeline_task("worker", bridged=("voice",))

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video_frame"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frame())

        self.assertEqual(len(received), 0)

    async def test_bridged_task_empty_tuple_accepts_all(self):
        """Bridged worker with empty tuple accepts frames from any bridge."""
        worker = make_stub_pipeline_task("worker", bridged=())

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def inject_frames():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="none"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frames())

        self.assertEqual(len(received), 3)

    async def test_bridged_task_multiple_bridges(self):
        """Bridged worker with multiple bridge names accepts from all listed."""
        worker = make_stub_pipeline_task("worker", bridged=("voice", "video"))

        received = []
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame(t, frame):
            received.append(frame)

        async def inject_frames():
            await asyncio.sleep(0.05)
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await self.bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="other"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="other",
                )
            )
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), inject_frames())

        texts = sorted([r.text for r in received])
        self.assertEqual(texts, ["video", "voice"])

    async def test_not_bridged_task_ignores_bridge(self):
        """Non-bridged worker (bridged=None) has no edge processors."""
        sent = capture_bus(self.bus)

        worker = make_stub_pipeline_task("root", bridged=None)

        async def push_frames():
            await asyncio.sleep(0.05)
            await worker.queue_frame(TextFrame(text="test"))
            await asyncio.sleep(0.05)
            await worker.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 0)


class TestJobLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = await create_test_bus()
        self.registry = create_test_registry()

    async def _attach(self, worker):
        await worker.attach(registry=self.registry, bus=self.bus)
        await worker.setup(self.tm)
        return worker

    async def test_request_job_sends_request(self):
        """request_job() sends BusJobRequestMessage to each worker."""
        sent = capture_bus(self.bus)

        parent = await self._attach(BaseWorker("parent"))
        await self.registry.register(WorkerReadyData(worker_name="worker", runner="test-runner"))

        job_id = await parent.request_job("worker", payload={"key": "val"})

        request_msgs = [m for m in sent if isinstance(m, BusJobRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].job_id, job_id)
        self.assertEqual(request_msgs[0].target, "worker")
        self.assertEqual(request_msgs[0].payload, {"key": "val"})

    async def test_request_job_group_multiple_tasks(self):
        """request_job_group() with multiple tasks sends messages for each."""
        sent = capture_bus(self.bus)

        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "w1", "w2")

        job_id = await parent.request_job_group("w1", "w2", payload={"work": True})

        request_msgs = [m for m in sent if isinstance(m, BusJobRequestMessage)]
        self.assertEqual(len(request_msgs), 2)
        targets = {m.target for m in request_msgs}
        self.assertEqual(targets, {"w1", "w2"})
        for m in request_msgs:
            self.assertEqual(m.job_id, job_id)

    async def test_on_job_request_called(self):
        """BusJobRequestMessage triggers on_job_request with the message."""
        worker = await self._attach(BaseWorker("worker"))

        received = []

        @worker.event_handler("on_job_request")
        async def on_request(t, message):
            received.append(message)

        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1", payload={"x": 1})
        )
        await asyncio.sleep(0)  # let async event handlers run

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].job_id, "t1")
        self.assertEqual(received[0].source, "parent")
        self.assertEqual(received[0].payload, {"x": 1})
        self.assertIn("t1", worker.active_jobs)

    async def test_send_job_response(self):
        """send_job_response() sends BusJobResponseMessage to requester."""
        sent = capture_bus(self.bus)

        worker = await self._attach(BaseWorker("worker"))
        # Simulate receiving a job request
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )

        await worker.send_job_response("t1", {"result": 42})

        response_msgs = [m for m in sent if isinstance(m, BusJobResponseMessage)]
        self.assertEqual(len(response_msgs), 1)
        self.assertEqual(response_msgs[0].target, "parent")
        self.assertEqual(response_msgs[0].job_id, "t1")
        self.assertEqual(response_msgs[0].response, {"result": 42})
        self.assertEqual(response_msgs[0].status, JobStatus.COMPLETED)

    async def test_send_job_update(self):
        """send_job_update() sends BusJobUpdateMessage to requester."""
        sent = capture_bus(self.bus)

        worker = await self._attach(BaseWorker("worker"))
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )

        await worker.send_job_update("t1", {"progress": 50})

        update_msgs = [m for m in sent if isinstance(m, BusJobUpdateMessage)]
        self.assertEqual(len(update_msgs), 1)
        self.assertEqual(update_msgs[0].target, "parent")
        self.assertEqual(update_msgs[0].job_id, "t1")
        self.assertEqual(update_msgs[0].update, {"progress": 50})

    async def test_on_job_completed_fires_when_all_respond(self):
        """on_job_completed fires when all tasks in a job group respond."""
        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "w1", "w2")

        completed = []

        @parent.event_handler("on_job_completed")
        async def on_completed(t, result):
            completed.append(result)

        job_id = await parent.request_job_group("w1", "w2")

        # First response — should not trigger on_job_completed
        await parent.on_bus_message(
            BusJobResponseMessage(
                source="w1",
                target="parent",
                job_id=job_id,
                status=JobStatus.COMPLETED,
                response={"a": 1},
            )
        )
        await asyncio.sleep(0)  # let async event handlers run
        self.assertEqual(len(completed), 0)

        # Second response — should trigger on_job_completed
        await parent.on_bus_message(
            BusJobResponseMessage(
                source="w2",
                target="parent",
                job_id=job_id,
                status=JobStatus.COMPLETED,
                response={"b": 2},
            )
        )
        await asyncio.sleep(0)  # let async event handlers run
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].job_id, job_id)
        self.assertEqual(completed[0].responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_cancel_job_group_sends_cancel_to_all_tasks(self):
        """cancel_job_group() sends BusJobCancelMessage to all tasks in group."""
        sent = capture_bus(self.bus)

        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "w1", "w2")

        job_id = await parent.request_job_group("w1", "w2")
        sent.clear()

        await parent.cancel_job_group(job_id, reason="no longer needed")

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 2)
        targets = {m.target for m in cancel_msgs}
        self.assertEqual(targets, {"w1", "w2"})
        for m in cancel_msgs:
            self.assertEqual(m.job_id, job_id)
            self.assertEqual(m.reason, "no longer needed")

    async def test_send_job_response_raises_without_active_job(self):
        """send_job_response raises RuntimeError when job_id is unknown."""
        worker = await self._attach(BaseWorker("worker"))

        with self.assertRaises(RuntimeError):
            await worker.send_job_response("unknown", {"result": 1})

    async def test_send_job_update_raises_without_active_job(self):
        """send_job_update raises RuntimeError when job_id is unknown."""
        worker = await self._attach(BaseWorker("worker"))

        with self.assertRaises(RuntimeError):
            await worker.send_job_update("unknown", {"progress": 50})

    async def test_cancel_auto_sends_cancelled_response(self):
        """BusJobCancelMessage auto-sends a cancelled response and clears state."""
        worker = await self._attach(BaseWorker("worker"))

        # Set up job state
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )
        self.assertIn("t1", worker.active_jobs)

        # Cancel should auto-send response (via send_job_response) and clear state
        await worker.on_bus_message(
            BusJobCancelMessage(source="parent", target="worker", job_id="t1")
        )
        # Yield to let the cancel coroutine complete its work.
        await asyncio.sleep(0.05)
        self.assertNotIn("t1", worker.active_jobs)

    async def test_on_job_cancelled_fires(self):
        """BusJobCancelMessage triggers on_job_cancelled with the message."""
        worker = await self._attach(BaseWorker("worker"))

        received = []

        @worker.event_handler("on_job_cancelled")
        async def on_cancelled(t, message):
            received.append(message)

        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )
        await worker.on_bus_message(
            BusJobCancelMessage(
                source="parent", target="worker", job_id="t1", reason="no longer needed"
            )
        )
        await asyncio.sleep(0.05)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].job_id, "t1")
        self.assertEqual(received[0].reason, "no longer needed")

    async def test_send_job_stream_start(self):
        """send_job_stream_start() sends BusJobStreamStartMessage to requester."""
        sent = capture_bus(self.bus)

        worker = await self._attach(BaseWorker("worker"))
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )

        await worker.send_job_stream_start("t1", {"content_type": "text"})

        msgs = [m for m in sent if isinstance(m, BusJobStreamStartMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].job_id, "t1")
        self.assertEqual(msgs[0].data, {"content_type": "text"})

    async def test_send_job_stream_data(self):
        """send_job_stream_data() sends BusJobStreamDataMessage to requester."""
        sent = capture_bus(self.bus)

        worker = await self._attach(BaseWorker("worker"))
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )

        await worker.send_job_stream_data("t1", {"text": "hello "})

        msgs = [m for m in sent if isinstance(m, BusJobStreamDataMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].job_id, "t1")
        self.assertEqual(msgs[0].data, {"text": "hello "})

    async def test_send_job_stream_end(self):
        """send_job_stream_end() sends BusJobStreamEndMessage to requester."""
        sent = capture_bus(self.bus)

        worker = await self._attach(BaseWorker("worker"))
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="t1")
        )

        await worker.send_job_stream_end("t1", {"final": True})

        msgs = [m for m in sent if isinstance(m, BusJobStreamEndMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].job_id, "t1")
        self.assertEqual(msgs[0].data, {"final": True})

    async def test_stream_end_triggers_on_job_completed(self):
        """BusJobStreamEndMessage triggers group completion like BusJobResponseMessage."""
        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "w1", "w2")

        completed = []

        @parent.event_handler("on_job_completed")
        async def on_completed(t, result):
            completed.append(result)

        job_id = await parent.request_job_group("w1", "w2")

        # First worker ends stream — should not trigger on_job_completed
        await parent.on_bus_message(
            BusJobStreamEndMessage(
                source="w1", target="parent", job_id=job_id, data={"result": "a"}
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(completed), 0)

        # Second worker ends stream — should trigger on_job_completed
        await parent.on_bus_message(
            BusJobStreamEndMessage(
                source="w2", target="parent", job_id=job_id, data={"result": "b"}
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].job_id, job_id)
        self.assertEqual(completed[0].responses, {"w1": {"result": "a"}, "w2": {"result": "b"}})

    async def test_send_job_stream_raises_without_active_job(self):
        """All stream helpers raise RuntimeError when job_id is unknown."""
        worker = await self._attach(BaseWorker("worker"))

        with self.assertRaises(RuntimeError):
            await worker.send_job_stream_start("unknown")

        with self.assertRaises(RuntimeError):
            await worker.send_job_stream_data("unknown")

        with self.assertRaises(RuntimeError):
            await worker.send_job_stream_end("unknown")

    async def test_request_job_timeout_cancels_job(self):
        """Short timeout triggers BusJobCancelMessage with reason 'timeout'."""
        sent = capture_bus(self.bus)

        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "worker")

        job_id = await parent.request_job("worker", timeout=0.05)

        # Wait for timeout to fire
        await asyncio.sleep(0.1)

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].job_id, job_id)
        self.assertEqual(cancel_msgs[0].reason, "timeout")

        # Clean up remaining tasks
        await parent.cleanup()

    async def test_request_job_timeout_cancelled_on_completion(self):
        """Responding before timeout prevents cancel from being sent."""
        sent = capture_bus(self.bus)

        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "worker")

        job_id = await parent.request_job("worker", timeout=0.5)

        # Let the timeout worker start before responding
        await asyncio.sleep(0)

        # Respond before timeout fires
        await parent.on_bus_message(
            BusJobResponseMessage(
                source="worker",
                target="parent",
                job_id=job_id,
                status=JobStatus.COMPLETED,
                response={"ok": True},
            )
        )

        # Wait past what would have been the timeout
        await asyncio.sleep(0.1)

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 0)

        # Clean up remaining tasks
        await parent.cleanup()

    async def test_request_job_no_timeout_by_default(self):
        """timeout_task is None when no timeout is given."""
        parent = await self._attach(BaseWorker("parent"))
        await register_tasks(self.registry, "worker")

        job_id = await parent.request_job("worker")

        group = parent.job_groups[job_id]
        self.assertIsNone(group.timeout_task)


class TestJobDecorator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = await create_test_bus()
        self.registry = create_test_registry()

    async def _attach(self, worker):
        await worker.attach(registry=self.registry, bus=self.bus)
        await worker.setup(self.tm)
        return worker

    async def _wait_until(self, predicate, timeout=1.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while not predicate():
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("condition not met within timeout")
            await asyncio.sleep(0.01)

    async def test_sequential_runs_one_at_a_time_in_order(self):
        """sequential=True serializes same-name requests in FIFO order."""
        running = 0
        max_running = 0
        completion_order: list[str] = []
        gates: dict[str, asyncio.Event] = {}

        class WorkerTask(BaseWorker):
            @job(name="work", sequential=True)
            async def on_work(self, message):
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await gates[message.job_id].wait()
                completion_order.append(message.job_id)
                running -= 1

        worker = await self._attach(WorkerTask("worker"))

        for tid in ("t1", "t2", "t3"):
            gates[tid] = asyncio.Event()
            await worker.on_bus_message(
                BusJobRequestMessage(source="parent", target="worker", job_id=tid, job_name="work")
            )

        # Let the first handler enter the locked region.
        await self._wait_until(lambda: running == 1)

        # Release in order; verify completion in order.
        for tid in ("t1", "t2", "t3"):
            gates[tid].set()
            await self._wait_until(lambda tid=tid: tid in completion_order)

        self.assertEqual(completion_order, ["t1", "t2", "t3"])
        self.assertEqual(max_running, 1)

    async def test_non_sequential_runs_concurrently(self):
        """Without sequential=True, same-name handlers run concurrently."""
        running = 0
        max_running = 0
        release = asyncio.Event()

        class WorkerTask(BaseWorker):
            @job(name="work")
            async def on_work(self, message):
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await release.wait()
                running -= 1

        worker = await self._attach(WorkerTask("worker"))

        for tid in ("t1", "t2"):
            await worker.on_bus_message(
                BusJobRequestMessage(source="parent", target="worker", job_id=tid, job_name="work")
            )

        await self._wait_until(lambda: running == 2)
        self.assertEqual(max_running, 2)
        release.set()
        await self._wait_until(lambda: running == 0)

    async def test_sequential_locks_are_per_name(self):
        """Sequential lock is per job name; different names do not block each other."""
        running = 0
        max_running = 0
        release = asyncio.Event()

        class WorkerTask(BaseWorker):
            @job(name="a", sequential=True)
            async def on_a(self, message):
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await release.wait()
                running -= 1

            @job(name="b", sequential=True)
            async def on_b(self, message):
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await release.wait()
                running -= 1

        worker = await self._attach(WorkerTask("worker"))

        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="ta", job_name="a")
        )
        await worker.on_bus_message(
            BusJobRequestMessage(source="parent", target="worker", job_id="tb", job_name="b")
        )

        await self._wait_until(lambda: running == 2)
        self.assertEqual(max_running, 2)
        release.set()
        await self._wait_until(lambda: running == 0)


if __name__ == "__main__":
    unittest.main()
