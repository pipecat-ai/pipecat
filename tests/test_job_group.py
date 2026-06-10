#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.bus import (
    AsyncQueueBus,
    BusJobCancelMessage,
    BusJobRequestMessage,
    BusJobResponseMessage,
)
from pipecat.pipeline.job_context import (
    JobError,
    JobEvent,
    JobGroupError,
    JobGroupEvent,
    JobStatus,
)
from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerReadyData
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.base_worker import BaseWorker


class StubTask(BaseWorker):
    pass


class JobWorkerTask(BaseWorker):
    """Worker that automatically responds to job requests via the bus."""

    def __init__(self, name, *, response=None, status=JobStatus.COMPLETED):
        super().__init__(name)
        self._auto_response = response
        self._auto_status = status

    async def on_job_request(self, message):
        await super().on_job_request(message)
        await self.send_job_response(message.job_id, self._auto_response, status=self._auto_status)


class UrgentJobWorkerTask(BaseWorker):
    """Worker that responds urgently to job requests."""

    def __init__(self, name, *, response=None, status=JobStatus.COMPLETED):
        super().__init__(name)
        self._auto_response = response
        self._auto_status = status

    async def on_job_request(self, message):
        await super().on_job_request(message)
        await self.send_job_response(
            message.job_id, self._auto_response, status=self._auto_status, urgent=True
        )


class UpdatingWorkerTask(BaseWorker):
    """Worker that sends updates before responding."""

    def __init__(self, name, *, updates, response=None):
        super().__init__(name)
        self._updates = updates
        self._auto_response = response

    async def on_job_request(self, message):
        await super().on_job_request(message)
        for update in self._updates:
            await self.send_job_update(message.job_id, update)
        await self.send_job_response(message.job_id, self._auto_response)


class StreamingWorkerTask(BaseWorker):
    """Worker that streams data before responding."""

    def __init__(self, name, *, chunks, response=None):
        super().__init__(name)
        self._chunks = chunks
        self._auto_response = response

    async def on_job_request(self, message):
        await super().on_job_request(message)
        await self.send_job_stream_start(message.job_id, {"content_type": "text"})
        for chunk in self._chunks:
            await self.send_job_stream_data(message.job_id, chunk)
        await self.send_job_stream_end(message.job_id, self._auto_response)


class SlowWorkerTask(BaseWorker):
    """Worker that blocks during job execution until cancelled."""

    def __init__(self, name):
        super().__init__(name)
        self.started = asyncio.Event()
        self.was_cancelled = False

    async def on_job_request(self, message):
        await super().on_job_request(message)
        self.started.set()
        try:
            # Block until cancelled
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.was_cancelled = True


async def create_test_env():
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    await bus.start()
    registry = WorkerRegistry(runner_name="test-runner")
    return bus, tm, registry


async def setup_task(bus, registry, task):
    """Subscribe a task to the bus and register it as ready."""
    await task.attach(registry=registry, bus=bus)
    await task.setup(bus.task_manager)
    await bus.subscribe(task)
    await registry.register(WorkerReadyData(worker_name=task.name, runner="test-runner"))


def capture_bus(bus):
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


class TestJobGroupContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await create_test_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_job_group_collects_responses(self):
        """job_group() context manager collects all responses."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        w1 = JobWorkerTask("w1", response={"a": 1})
        w2 = JobWorkerTask("w2", response={"b": 2})
        await setup_task(self.bus, self.registry, w1)
        await setup_task(self.bus, self.registry, w2)

        async with parent.job_group("w1", "w2", payload={"work": True}) as tg:
            pass

        self.assertEqual(tg.responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_job_group_sends_request(self):
        """job_group() sends BusJobRequestMessage to each task."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        w1 = JobWorkerTask("w1", response={"ok": True})
        await setup_task(self.bus, self.registry, w1)

        async with parent.job_group("w1", payload={"data": 1}) as tg:
            pass

        request_msgs = [m for m in sent if isinstance(m, BusJobRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].payload, {"data": 1})

    async def test_job_group_raises_on_cancel(self):
        """job_group() raises JobGroupError when group is cancelled."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        # StubTask doesn't auto-respond, so we can cancel manually
        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(JobGroupError) as ctx:
            async with parent.job_group("worker") as tg:
                asyncio.ensure_future(parent.cancel_job_group(tg.job_id, reason="manual cancel"))

        # Let the event loop schedule handler tasks spawned by the bus
        await asyncio.sleep(0)
        self.assertIn("manual cancel", str(ctx.exception))

    async def test_job_group_raises_on_timeout(self):
        """job_group() raises JobGroupError on timeout."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        # StubTask doesn't auto-respond, so timeout fires
        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(JobGroupError) as ctx:
            async with parent.job_group("worker", timeout=0.05) as tg:
                pass

        self.assertIn("timeout", str(ctx.exception))

    async def test_job_group_raises_on_ready_timeout(self):
        """job_group() raises JobGroupError when tasks aren't ready in time."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        # "ghost" is never registered, so the ready-wait times out
        with self.assertRaises(JobGroupError) as ctx:
            async with parent.job_group("ghost", timeout=0.05) as tg:
                pass

        self.assertIn("not ready", str(ctx.exception))

    async def test_job_group_cancels_on_block_exception(self):
        """job_group() cancels remaining jobs when the block raises."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(ValueError):
            async with parent.job_group("worker") as tg:
                raise ValueError("something went wrong")

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")

    async def test_job_group_raises_on_worker_error(self):
        """job_group() raises JobGroupError when a worker errors with cancel_on_error."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"error": "failed"}, status=JobStatus.ERROR)
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(JobGroupError):
            async with parent.job_group("worker") as tg:
                pass

        # Error response is tracked in partial responses
        self.assertEqual(tg.responses, {"worker": {"error": "failed"}})

    async def test_job_group_on_job_error_fires(self):
        """on_job_error fires when a worker errors with cancel_on_error."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"error": "boom"}, status=JobStatus.ERROR)
        await setup_task(self.bus, self.registry, worker)

        errors = []

        @parent.event_handler("on_job_error")
        async def on_error(task, message):
            errors.append(message)

        with self.assertRaises(JobGroupError):
            async with parent.job_group("worker") as tg:
                pass

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].source, "worker")
        self.assertEqual(errors[0].response, {"error": "boom"})
        self.assertEqual(errors[0].status, JobStatus.ERROR)

    async def test_job_group_partial_responses_on_error(self):
        """Partial responses from successful workers are available after error."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        # w1 responds successfully, w2 responds with error.
        # Order depends on bus dispatch, but both are registered.
        w1 = JobWorkerTask("w1", response={"ok": True})
        w2 = JobWorkerTask("w2", response={"error": "fail"}, status=JobStatus.ERROR)
        await setup_task(self.bus, self.registry, w1)
        await setup_task(self.bus, self.registry, w2)

        with self.assertRaises(JobGroupError):
            async with parent.job_group("w1", "w2") as tg:
                pass

        # w2's error response should be in partial responses
        self.assertIn("w2", tg.responses)
        self.assertEqual(tg.responses["w2"], {"error": "fail"})

    async def test_job_group_job_id_available(self):
        """job_id is available inside the async with block."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={})
        await setup_task(self.bus, self.registry, worker)

        captured_job_id = None

        async with parent.job_group("worker") as tg:
            captured_job_id = tg.job_id

        self.assertIsNotNone(captured_job_id)
        self.assertEqual(captured_job_id, tg.job_id)

    async def test_job_group_on_job_completed_still_fires(self):
        """on_job_completed callback still fires when using job_group()."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("w1", response={"ok": True})
        await setup_task(self.bus, self.registry, worker)

        completed = []

        @parent.event_handler("on_job_completed")
        async def on_completed(task, result):
            completed.append(result)

        async with parent.job_group("w1") as tg:
            pass

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].responses, {"w1": {"ok": True}})

    async def test_job_group_iterates_updates(self):
        """async for yields update events from workers."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = UpdatingWorkerTask(
            "worker",
            updates=[{"progress": 25}, {"progress": 75}],
            response={"result": "done"},
        )
        await setup_task(self.bus, self.registry, worker)

        events = []
        async with parent.job_group("worker", payload={"work": True}) as tg:
            async for event in tg:
                events.append(event)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, JobGroupEvent.UPDATE)
        self.assertEqual(events[0].worker_name, "worker")
        self.assertEqual(events[0].data, {"progress": 25})
        self.assertEqual(events[1].data, {"progress": 75})
        self.assertEqual(tg.responses, {"worker": {"result": "done"}})

    async def test_job_group_iterates_stream_events(self):
        """async for yields stream_start, stream_data, and stream_end events."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StreamingWorkerTask(
            "worker",
            chunks=[{"text": "hello "}, {"text": "world"}],
            response={"final": True},
        )
        await setup_task(self.bus, self.registry, worker)

        events = []
        async with parent.job_group("worker") as tg:
            async for event in tg:
                events.append(event)

        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                JobGroupEvent.STREAM_START,
                JobGroupEvent.STREAM_DATA,
                JobGroupEvent.STREAM_DATA,
                JobGroupEvent.STREAM_END,
            ],
        )
        self.assertEqual(events[0].data, {"content_type": "text"})
        self.assertEqual(events[1].data, {"text": "hello "})
        self.assertEqual(events[2].data, {"text": "world"})
        self.assertEqual(events[3].data, {"final": True})

    async def test_job_group_iterates_mixed_events(self):
        """async for yields events from multiple workers interleaved."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        w1 = UpdatingWorkerTask("w1", updates=[{"status": "working"}], response={"a": 1})
        w2 = JobWorkerTask("w2", response={"b": 2})
        await setup_task(self.bus, self.registry, w1)
        await setup_task(self.bus, self.registry, w2)

        events = []
        async with parent.job_group("w1", "w2") as tg:
            async for event in tg:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, JobGroupEvent.UPDATE)
        self.assertEqual(events[0].worker_name, "w1")
        self.assertEqual(tg.responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_job_group_no_iteration_still_works(self):
        """job_group() works without iterating (pass body)."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = UpdatingWorkerTask("worker", updates=[{"progress": 50}], response={"ok": True})
        await setup_task(self.bus, self.registry, worker)

        async with parent.job_group("worker") as tg:
            pass

        self.assertEqual(tg.responses, {"worker": {"ok": True}})

    async def test_urgent_job_response_collected(self):
        """Urgent job responses are collected like normal responses."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = UrgentJobWorkerTask("worker", response={"urgent": True})
        await setup_task(self.bus, self.registry, worker)

        async with parent.job_group("worker") as tg:
            pass

        self.assertEqual(tg.responses, {"worker": {"urgent": True}})

    async def test_urgent_job_response_triggers_on_job_error(self):
        """Urgent error response triggers on_job_error and cancels group."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = UrgentJobWorkerTask(
            "worker", response={"error": "critical"}, status=JobStatus.ERROR
        )
        await setup_task(self.bus, self.registry, worker)

        errors = []

        @parent.event_handler("on_job_error")
        async def on_error(task, message):
            errors.append(message)

        with self.assertRaises(JobGroupError):
            async with parent.job_group("worker") as tg:
                pass

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].response, {"error": "critical"})

    async def test_urgent_response_has_system_priority(self):
        """Urgent job response is delivered before normal data messages."""
        from pipecat.bus import BusDataMessage, BusJobResponseUrgentMessage

        bus = self.bus

        # Queue data messages first, then an urgent response
        parent = StubTask("parent")
        await setup_task(bus, self.registry, parent)

        received = []

        @parent.event_handler("on_bus_message")
        async def on_msg(task, message):
            received.append(message)

        # Send data messages before starting dispatch
        for i in range(3):
            await bus.send(BusDataMessage(source=f"data_{i}"))
        await bus.send(
            BusJobResponseUrgentMessage(
                source="worker", target="parent", job_id="t1", status=JobStatus.COMPLETED
            )
        )

        # Start dispatch — urgent should arrive first
        # bus is already started in setUp; restart by stopping and starting
        await bus.stop()
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        # First message should be the urgent response
        self.assertIsInstance(received[0], BusJobResponseUrgentMessage)


class TestJobContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await create_test_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_job_collects_response(self):
        """job() context manager collects the worker's response."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"result": 42})
        await setup_task(self.bus, self.registry, worker)

        async with parent.job("worker", payload={"x": 1}) as t:
            pass

        self.assertEqual(t.response, {"result": 42})

    async def test_job_sends_request(self):
        """job() sends a BusJobRequestMessage to the task."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"ok": True})
        await setup_task(self.bus, self.registry, worker)

        async with parent.job("worker") as t:
            pass

        request_msgs = [m for m in sent if isinstance(m, BusJobRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].target, "worker")
        self.assertEqual(request_msgs[0].source, "parent")

    async def test_job_iterates_events(self):
        """job() yields intermediate events via async for."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = UpdatingWorkerTask("worker", updates=[{"progress": 50}], response={"done": True})
        await setup_task(self.bus, self.registry, worker)

        events = []
        async with parent.job("worker") as t:
            async for event in t:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, JobEvent.UPDATE)
        self.assertIsInstance(events[0], JobEvent)
        self.assertEqual(t.response, {"done": True})

    async def test_job_streams_data(self):
        """job() yields stream events from a streaming worker."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StreamingWorkerTask(
            "worker",
            chunks=[{"text": "hello"}, {"text": "world"}],
            response={"ok": True},
        )
        await setup_task(self.bus, self.registry, worker)

        events = []
        async with parent.job("worker") as t:
            async for event in t:
                events.append(event)

        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                JobEvent.STREAM_START,
                JobEvent.STREAM_DATA,
                JobEvent.STREAM_DATA,
                JobEvent.STREAM_END,
            ],
        )
        self.assertEqual(t.response, {"ok": True})

    async def test_job_cancels_on_exception(self):
        """job() cancels the job if the block raises."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(ValueError):
            async with parent.job("worker") as t:
                raise ValueError("something went wrong")

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")

    async def test_job_raises_on_worker_error(self):
        """job() raises JobError when the worker errors."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"error": "boom"}, status=JobStatus.ERROR)
        await setup_task(self.bus, self.registry, worker)

        with self.assertRaises(JobError):
            async with parent.job("worker") as t:
                pass

    async def test_job_exposes_job_id(self):
        """job() exposes the job_id inside the context."""
        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = JobWorkerTask("worker", response={"ok": True})
        await setup_task(self.bus, self.registry, worker)

        async with parent.job("worker") as t:
            self.assertIsInstance(t.job_id, str)
            self.assertTrue(len(t.job_id) > 0)

    async def test_job_group_cancels_on_cancelled_error(self):
        """job_group() cancels workers when CancelledError is raised (e.g. tool interruption)."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        async def run_job_group():
            async with parent.job_group("worker") as tg:
                # Simulate tool cancellation while waiting
                raise asyncio.CancelledError()

        task = asyncio.create_task(run_job_group())
        with self.assertRaises(asyncio.CancelledError):
            await task

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")
        # Job group should be cleaned up
        self.assertEqual(len(parent.job_groups), 0)

    async def test_job_cancels_on_cancelled_error(self):
        """job() cancels the worker when CancelledError is raised (e.g. tool interruption)."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = StubTask("worker")
        await setup_task(self.bus, self.registry, worker)

        async def run_job():
            async with parent.job("worker") as t:
                raise asyncio.CancelledError()

        task = asyncio.create_task(run_job())
        with self.assertRaises(asyncio.CancelledError):
            await task

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")
        self.assertEqual(len(parent.job_groups), 0)

    async def test_fire_and_forget_jobs_cancelled_manually(self):
        """User cancels fire-and-forget jobs in a CancelledError handler."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        w1 = StubTask("w1")
        w2 = StubTask("w2")
        await setup_task(self.bus, self.registry, w1)
        await setup_task(self.bus, self.registry, w2)

        # Simulate what a user would do in on_function_calls_cancelled:
        # track job IDs and cancel only the ones started here
        try:
            job_ids = []
            job_ids.append(await parent.request_job("w1", payload={"job": 1}))
            job_ids.append(await parent.request_job("w2", payload={"job": 2}))
            self.assertEqual(len(parent.job_groups), 2)
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            for jid in job_ids:
                await parent.cancel_job_group(jid, reason="tool cancelled")

        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 2)
        cancelled_targets = {m.target for m in cancel_msgs}
        self.assertEqual(cancelled_targets, {"w1", "w2"})
        for m in cancel_msgs:
            self.assertEqual(m.reason, "tool cancelled")
        self.assertEqual(len(parent.job_groups), 0)

    async def test_cancel_interrupts_running_handler(self):
        """Cancelling a job interrupts a handler that is currently executing."""
        sent = capture_bus(self.bus)

        parent = StubTask("parent")
        await setup_task(self.bus, self.registry, parent)

        worker = SlowWorkerTask("worker")
        await setup_task(self.bus, self.registry, worker)

        job_id = await parent.request_job("worker", payload={"job": 1})

        # Wait for the worker to start executing
        await asyncio.wait_for(worker.started.wait(), timeout=2.0)

        # Cancel while the handler is blocked
        await parent.cancel_job_group(job_id, reason="no longer needed")

        # Give the event loop time to process the cancellation
        await asyncio.sleep(0.1)

        # The handler should have been cancelled
        self.assertTrue(worker.was_cancelled)

        # A cancel message should have been sent
        cancel_msgs = [m for m in sent if isinstance(m, BusJobCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "no longer needed")

        # A CANCELLED response should have been sent back
        response_msgs = [m for m in sent if isinstance(m, BusJobResponseMessage)]
        self.assertEqual(len(response_msgs), 1)
        self.assertEqual(response_msgs[0].status, JobStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
