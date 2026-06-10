#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the UIWorker user-job-group lifecycle.

Covers:
- ``UIWorker.on_bus_message`` forwarding of worker job updates/responses
  for registered user job groups as ``BusUIJob*`` carriers.
- The reserved ``__cancel_job_group`` client event routing to
  ``cancel_job_group``.
- ``UIJobGroupContext`` publishing ``group_started`` / ``group_completed``
  envelopes and (de)registering the group on the worker.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.bus.messages import BusJobResponseMessage, BusJobUpdateMessage
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.job_context import JobGroup, JobStatus
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.ui import UIWorker


async def _make_solo_worker(**kwargs) -> UIWorker:
    """A UIWorker with a task manager and a ``queue_frame`` spy.

    Suitable for testing forwarding logic by directly invoking
    ``on_bus_message`` and asserting on captured ``send_bus_message``
    calls.
    """
    worker = UIWorker("ui", llm=MagicMock(), **kwargs)
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    worker._task_manager = tm

    recorded: list = []

    async def _record(frame, direction=FrameDirection.DOWNSTREAM):
        recorded.append(frame)

    worker.queue_frame = _record  # type: ignore[method-assign]
    worker._recorded = recorded  # type: ignore[attr-defined]
    return worker


class TestUIWorkerForwarding(unittest.IsolatedAsyncioTestCase):
    async def test_unregistered_job_update_is_not_forwarded(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusJobUpdateMessage(
                source="worker", target=worker.name, job_id="t-unknown", update={"x": 1}
            )
        )

        forwarded = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if isinstance(c.args[0], BusUIJobUpdateMessage)
        ]
        self.assertEqual(forwarded, [])

    async def test_registered_job_update_is_forwarded(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(
            job_id="t1", worker_names=["worker"], label="hello", cancellable=True
        )
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusJobUpdateMessage(
                source="worker",
                target=worker.name,
                job_id="t1",
                update={"kind": "tool_call", "tool": "WebSearch"},
            )
        )

        forwarded = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if isinstance(c.args[0], BusUIJobUpdateMessage)
        ]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0].job_id, "t1")
        self.assertEqual(forwarded[0].worker_name, "worker")
        self.assertEqual(forwarded[0].data, {"kind": "tool_call", "tool": "WebSearch"})

    async def test_registered_job_response_is_forwarded(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(
            job_id="t1", worker_names=["worker"], label=None, cancellable=True
        )
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusJobResponseMessage(
                source="worker",
                target=worker.name,
                job_id="t1",
                status=JobStatus.COMPLETED,
                response={"answer": 42},
            )
        )

        forwarded = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if isinstance(c.args[0], BusUIJobCompletedMessage)
        ]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0].job_id, "t1")
        self.assertEqual(forwarded[0].worker_name, "worker")
        self.assertEqual(forwarded[0].status, "completed")
        self.assertEqual(forwarded[0].response, {"answer": 42})

    async def test_response_status_serializes_for_cancelled_and_error(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(job_id="t1", worker_names=["w"], label=None, cancellable=True)
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusJobResponseMessage(
                source="w", target=worker.name, job_id="t1", status=JobStatus.CANCELLED
            )
        )
        await worker.on_bus_message(
            BusJobResponseMessage(
                source="w", target=worker.name, job_id="t1", status=JobStatus.ERROR
            )
        )

        statuses = [
            c.args[0].status
            for c in worker.send_bus_message.await_args_list
            if isinstance(c.args[0], BusUIJobCompletedMessage)
        ]
        self.assertEqual(statuses, ["cancelled", "error"])


class TestCancelJobEvent(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_event_routes_to_cancel_job_group(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(job_id="t1", worker_names=["w"], label=None, cancellable=True)
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "t1", "reason": "user clicked cancel"},
            )
        )

        worker.cancel_job_group.assert_awaited_once_with("t1", reason="user clicked cancel")

    async def test_cancel_event_default_reason_when_omitted(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(job_id="t1", worker_names=["w"], label=None, cancellable=True)
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "t1"},
            )
        )

        worker.cancel_job_group.assert_awaited_once()
        self.assertEqual(worker.cancel_job_group.await_args.kwargs["reason"], "cancelled by user")

    async def test_non_cancellable_group_is_ignored(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(
            job_id="t1", worker_names=["w"], label=None, cancellable=False
        )
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "t1"},
            )
        )

        worker.cancel_job_group.assert_not_awaited()

    async def test_unknown_job_id_is_ignored(self):
        worker = await _make_solo_worker()
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "nope"},
            )
        )

        worker.cancel_job_group.assert_not_awaited()

    async def test_missing_or_bad_payload_is_ignored(self):
        worker = await _make_solo_worker()
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]

        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload=None,
            )
        )
        await worker.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=worker.name,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": 42},
            )
        )

        worker.cancel_job_group.assert_not_awaited()


class TestForwardingDoesNotInjectLLMContext(unittest.IsolatedAsyncioTestCase):
    async def test_job_update_forwarding_does_not_queue_append_frames(self):
        worker = await _make_solo_worker()
        worker._register_ui_job_group(job_id="t1", worker_names=["w"], label=None, cancellable=True)

        await worker.on_bus_message(
            BusJobUpdateMessage(source="w", target=worker.name, job_id="t1", update={"x": 1})
        )

        appends = [f for f in worker._recorded if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(appends, [])


def _stub_job_group(worker, job_id="t1", worker_names=("w1",)):
    """Make ``create_job_group_and_request_job`` return a self-completing group.

    The group completes on the next loop tick so both the context-manager
    and fire-and-forget paths terminate without a running bus or workers.
    """

    async def _fake_create(names, *, name=None, payload=None, timeout=None, cancel_on_error=True):
        group = JobGroup(job_id=job_id, worker_names=set(names))

        async def _finish():
            # Yield so JobGroupContext.__aenter__ can set event_queue first.
            await asyncio.sleep(0)
            group.complete()

        asyncio.create_task(_finish())
        return group

    worker.create_job_group_and_request_job = _fake_create  # type: ignore[method-assign]


class TestUIJobGroupContext(unittest.IsolatedAsyncioTestCase):
    async def test_context_publishes_started_and_completed(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.ui_job_group("w1", label="My research") as tg:
            self.assertEqual(tg.job_id, "t1")
            self.assertIn("t1", worker._ui_job_groups)

        self.assertNotIn("t1", worker._ui_job_groups)

        kinds = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertEqual(
            kinds,
            ["BusUIJobGroupStartedMessage", "BusUIJobGroupCompletedMessage"],
        )

        started = worker.send_bus_message.await_args_list[0].args[0]
        self.assertIsInstance(started, BusUIJobGroupStartedMessage)
        self.assertEqual(started.job_id, "t1")
        self.assertEqual(started.workers, ["w1"])
        self.assertEqual(started.label, "My research")
        self.assertTrue(started.cancellable)

        completed = worker.send_bus_message.await_args_list[1].args[0]
        self.assertIsInstance(completed, BusUIJobGroupCompletedMessage)
        self.assertEqual(completed.job_id, "t1")

    async def test_non_cancellable_group_sets_flag_in_started_message(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.ui_job_group("w1", cancellable=False):
            pass

        started = worker.send_bus_message.await_args_list[0].args[0]
        self.assertFalse(started.cancellable)

    async def test_unregisters_on_exit(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.ui_job_group("w1") as tg:
            pass

        self.assertNotIn(tg.job_id, worker._ui_job_groups)

    async def test_start_ui_job_group_returns_id_and_publishes(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        job_id = await worker.start_ui_job_group("w1", label="Background work")
        self.assertEqual(job_id, "t1")

        started = worker.send_bus_message.await_args_list[0].args[0]
        self.assertIsInstance(started, BusUIJobGroupStartedMessage)
        self.assertEqual(started.label, "Background work")

        # The background runner drains the group and publishes completion.
        for _ in range(50):
            await asyncio.sleep(0)
            if any(
                isinstance(c.args[0], BusUIJobGroupCompletedMessage)
                for c in worker.send_bus_message.await_args_list
            ):
                break
        else:
            self.fail("group_completed envelope was not published")

        self.assertNotIn("t1", worker._ui_job_groups)


if __name__ == "__main__":
    unittest.main()
