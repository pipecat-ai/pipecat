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
import warnings
from unittest.mock import AsyncMock, MagicMock

from pipecat.bus.messages import (
    BusJobResponseMessage,
    BusJobStreamEndMessage,
    BusJobUpdateMessage,
)
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.job_context import JobGroup, JobGroupParams, JobParams, JobStatus
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.base_ui_worker import BaseUIWorker
from pipecat.workers.ui import UIWorker
from pipecat.workers.ui.ui_job_context import UIJobGroupContext


async def _make_solo_worker(**kwargs) -> UIWorker:
    """A UIWorker with a task manager and a ``queue_frame`` spy.

    Suitable for testing forwarding logic by directly invoking
    ``on_bus_message`` and asserting on captured ``send_bus_message``
    calls.
    """
    worker = UIWorker("ui", llm=MagicMock(), **kwargs)
    tm = TaskManager()
    worker._task_manager = tm

    recorded: list = []

    async def _record(frame, direction=FrameDirection.DOWNSTREAM):
        recorded.append(frame)

    worker.queue_frame = _record  # type: ignore[method-assign]
    worker._recorded = recorded  # type: ignore[attr-defined]
    return worker


def _register(worker, *, job_id, worker_names, label=None, cancellable=True):
    """Put a live job group on the worker, as dispatch would."""
    group = JobGroup(
        job_id=job_id,
        worker_names=list(worker_names),
        label=label,
        cancellable=cancellable,
    )
    worker._job_groups[job_id] = group
    return group


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
        _register(worker, job_id="t1", worker_names=["worker"], label="hello", cancellable=True)
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
        _register(worker, job_id="t1", worker_names=["worker"], label=None, cancellable=True)
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
        # Two workers, so the group stays live across both responses
        # (source "w" responds twice here, and "other" never does).
        _register(worker, job_id="t1", worker_names=["w", "other"])
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
        # The error cancels the group, so "other", which never reached a
        # terminal state of its own, is reported cancelled.
        self.assertEqual(statuses, ["cancelled", "error", "cancelled"])


class TestLateMessagesAfterTeardown(unittest.IsolatedAsyncioTestCase):
    """Messages that arrive for a group that is already gone."""

    async def test_late_cancelled_response_is_not_forwarded(self):
        worker = await _make_solo_base_worker()
        _register(worker, job_id="t1", worker_names=["w1", "w2"])
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.cancel_job_group("t1", reason="user")
        before = len(worker.send_bus_message.await_args_list)

        # Each worker answers the cancel it was sent, after the group is gone.
        for name in ("w1", "w2"):
            await worker.on_bus_message(
                BusJobResponseMessage(
                    source=name,
                    target=worker.name,
                    job_id="t1",
                    status=JobStatus.CANCELLED,
                )
            )

        self.assertEqual(len(worker.send_bus_message.await_args_list), before)

    async def test_late_update_is_not_forwarded(self):
        worker = await _make_solo_base_worker()
        _register(worker, job_id="t1", worker_names=["w1"])
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]

        await worker.cancel_job_group("t1", reason="user")
        before = len(worker.send_bus_message.await_args_list)

        await worker.on_bus_message(
            BusJobUpdateMessage(source="w1", target=worker.name, job_id="t1", update={"x": 1})
        )

        self.assertEqual(len(worker.send_bus_message.await_args_list), before)


class TestCancelJobEvent(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_event_routes_to_cancel_job_group(self):
        worker = await _make_solo_worker()
        _register(worker, job_id="t1", worker_names=["w"], label=None, cancellable=True)
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
        _register(worker, job_id="t1", worker_names=["w"], label=None, cancellable=True)
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
        _register(worker, job_id="t1", worker_names=["w"], label=None, cancellable=False)
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
        _register(worker, job_id="t1", worker_names=["w"], label=None, cancellable=True)

        await worker.on_bus_message(
            BusJobUpdateMessage(source="w", target=worker.name, job_id="t1", update={"x": 1})
        )

        appends = [f for f in worker._recorded if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(appends, [])


def _stub_job_group(worker, job_id="t1", worker_names=("w1",)):
    """Stub the transport under ``create_job_group_and_request_job``.

    Patches the ready-wait, the request send, and group creation (for a
    deterministic ``job_id``) while leaving the real
    ``create_job_group_and_request_job`` — and therefore the UI
    registration and ``group_started`` emission — in place. Each worker
    "responds" on the next loop tick through the real
    ``_track_job_group_response`` path, so teardown (including
    ``group_completed`` emission) is the production code too.
    """

    async def _ready(names):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(True)
        return fut

    async def _send(worker_name, jid, job_name=None, payload=None):
        pass

    def _fake_create(names, *, params=None, **kwargs):
        params = params or JobGroupParams()
        group = JobGroup(
            job_id=job_id,
            worker_names=list(names),
            cancel_on_error=params.cancel_on_error,
            label=params.label,
            cancellable=params.cancellable,
        )
        worker._job_groups[job_id] = group

        async def _finish():
            # Yield so JobGroupContext.__aenter__ can set event_queue first.
            await asyncio.sleep(0)
            # Each worker "responds" through the real bus-message path, so
            # response recording, group teardown, and (for BaseUIWorker)
            # envelope forwarding all run production code.
            for n in names:
                await worker.on_bus_message(
                    BusJobResponseMessage(
                        source=n,
                        target=worker.name,
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        response={},
                    )
                )

        asyncio.create_task(_finish())
        return group

    worker._wait_workers_ready = _ready  # type: ignore[method-assign]
    worker._send_job_request = _send  # type: ignore[method-assign]
    worker._create_job_group = _fake_create  # type: ignore[method-assign]


class TestUIJobGroupContext(unittest.IsolatedAsyncioTestCase):
    async def test_label_and_cancellable_read_the_group_params(self):
        """The deprecated context still answers for how it was built."""
        worker = await _make_solo_worker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            context = UIJobGroupContext(worker, ("w1",), label="My research", cancellable=False)

        self.assertEqual(context.label, "My research")
        self.assertFalse(context.cancellable)

    async def test_context_publishes_started_and_completed(self):
        worker = await _make_solo_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.ui_job_group("w1", label="My research") as tg:
            self.assertEqual(tg.job_id, "t1")
            self.assertIn("t1", worker._job_groups)

        self.assertNotIn("t1", worker._job_groups)

        kinds = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertEqual(
            kinds,
            [
                "BusUIJobGroupStartedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )

        started = worker.send_bus_message.await_args_list[0].args[0]
        self.assertIsInstance(started, BusUIJobGroupStartedMessage)
        self.assertEqual(started.job_id, "t1")
        self.assertEqual(started.workers, ["w1"])
        self.assertEqual(started.label, "My research")
        self.assertTrue(started.cancellable)

        completed = worker.send_bus_message.await_args_list[2].args[0]
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

        self.assertNotIn(tg.job_id, worker._job_groups)

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

        self.assertNotIn("t1", worker._job_groups)


async def _make_solo_base_worker() -> BaseUIWorker:
    """A plain BaseUIWorker (no LLM) with a task manager attached."""
    worker = BaseUIWorker("plain")
    worker._task_manager = TaskManager()
    return worker


class TestBaseUIWorkerJobGroups(unittest.IsolatedAsyncioTestCase):
    """A BaseUIWorker dispatches client-visible job groups without any LLM."""

    async def test_context_publishes_started_and_completed(self):
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.job_group("w1", params=JobGroupParams(label="Research: SMRs")) as tg:
            self.assertEqual(tg.job_id, "t1")
            self.assertIn("t1", worker._job_groups)

        self.assertNotIn("t1", worker._job_groups)
        kinds = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertEqual(
            kinds,
            [
                "BusUIJobGroupStartedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )
        self.assertEqual(worker.send_bus_message.await_args_list[0].args[0].label, "Research: SMRs")

    async def test_registered_update_and_response_are_forwarded(self):
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1"], label=None, cancellable=True)

        await worker.on_bus_message(
            BusJobUpdateMessage(source="w1", target=worker.name, job_id="t1", update={"p": 1})
        )
        await worker.on_bus_message(
            BusJobResponseMessage(
                source="w1",
                target=worker.name,
                job_id="t1",
                status=JobStatus.COMPLETED,
                response={"ok": True},
            )
        )

        forwarded = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertIn("BusUIJobUpdateMessage", forwarded)
        self.assertIn("BusUIJobCompletedMessage", forwarded)

    async def test_cancel_event_routes_to_cancel_job_group(self):
        worker = await _make_solo_base_worker()
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1"], label=None, cancellable=True)

        await worker.on_bus_message(
            BusUIEventMessage(
                source="main",
                target=None,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "t1", "reason": "user clicked cancel"},
            )
        )

        worker.cancel_job_group.assert_awaited_once_with("t1", reason="user clicked cancel")

    async def test_non_cancellable_group_ignores_cancel_event(self):
        worker = await _make_solo_base_worker()
        worker.cancel_job_group = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1"], label=None, cancellable=False)

        await worker.on_bus_message(
            BusUIEventMessage(
                source="main",
                target=None,
                event_name=_UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
                payload={"job_id": "t1"},
            )
        )

        worker.cancel_job_group.assert_not_awaited()

    async def test_single_job_with_ui_publishes_envelopes(self):
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        async with worker.job("w1", params=JobParams(label="one job")) as t:
            self.assertEqual(t.job_id, "t1")
            self.assertIn("t1", worker._job_groups)

        self.assertNotIn("t1", worker._job_groups)
        kinds = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertEqual(
            kinds,
            [
                "BusUIJobGroupStartedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )

    async def test_error_response_forwards_worker_envelope_before_group_completes(self):
        # Regression: with the default cancel_on_error=True, a worker ERROR
        # cancels the group inside base handling. The client must still get
        # that worker's job_completed envelope, before group_completed.
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1", "w2"])

        await worker.on_bus_message(
            BusJobResponseMessage(
                source="w1",
                target=worker.name,
                job_id="t1",
                status=JobStatus.ERROR,
                response={"error": "boom"},
            )
        )

        ui_messages = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if type(c.args[0]).__name__.startswith("BusUIJob")
        ]
        self.assertEqual(
            [type(m).__name__ for m in ui_messages],
            [
                "BusUIJobCompletedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )
        # The erroring worker keeps its status; the other is synthesized
        # as cancelled (its own CANCELLED response would arrive too late).
        self.assertEqual(ui_messages[0].worker_name, "w1")
        self.assertEqual(ui_messages[0].status, "error")
        self.assertEqual(ui_messages[1].worker_name, "w2")
        self.assertEqual(ui_messages[1].status, "cancelled")
        self.assertNotIn("t1", worker._job_groups)

    async def test_cancel_synthesizes_cancelled_envelopes_for_unreported_workers(self):
        # The workers' own CANCELLED responses arrive after unregistration,
        # so cancellation reports them deterministically instead.
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1", "w2"])

        await worker.cancel_job_group("t1", reason="user clicked cancel")

        ui_messages = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if type(c.args[0]).__name__.startswith("BusUIJob")
        ]
        self.assertEqual(
            [type(m).__name__ for m in ui_messages],
            [
                "BusUIJobCompletedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )
        self.assertEqual(
            {(m.worker_name, m.status) for m in ui_messages[:2]},
            {("w1", "cancelled"), ("w2", "cancelled")},
        )
        self.assertNotIn("t1", worker._job_groups)

    async def test_stream_end_completion_completes_group(self):
        # Regression: a worker may finish via send_job_stream_end instead of
        # a response; that terminal path must also complete the card and
        # release the registration.
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _register(worker, job_id="t1", worker_names=["w1"])

        await worker.on_bus_message(
            BusJobStreamEndMessage(
                source="w1", target=worker.name, job_id="t1", data={"final": True}
            )
        )

        ui_messages = [
            c.args[0]
            for c in worker.send_bus_message.await_args_list
            if type(c.args[0]).__name__.startswith("BusUIJob")
        ]
        self.assertEqual(
            [type(m).__name__ for m in ui_messages],
            ["BusUIJobCompletedMessage", "BusUIJobGroupCompletedMessage"],
        )
        self.assertEqual(ui_messages[0].worker_name, "w1")
        self.assertEqual(ui_messages[0].status, "completed")
        self.assertEqual(ui_messages[0].response, {"final": True})
        self.assertNotIn("t1", worker._job_groups)

    async def test_request_job_group_returns_id_and_publishes(self):
        worker = await _make_solo_base_worker()
        worker.send_bus_message = AsyncMock()  # type: ignore[method-assign]
        _stub_job_group(worker)

        job_id = await worker.request_job_group("w1", params=JobGroupParams(label="bg work"))
        self.assertEqual(job_id, "t1")
        await asyncio.sleep(0.05)  # let the background drainer finish

        kinds = [type(c.args[0]).__name__ for c in worker.send_bus_message.await_args_list]
        self.assertEqual(
            kinds,
            [
                "BusUIJobGroupStartedMessage",
                "BusUIJobCompletedMessage",
                "BusUIJobGroupCompletedMessage",
            ],
        )


if __name__ == "__main__":
    unittest.main()
