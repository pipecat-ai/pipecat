#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the native RTVI⇄bus UI bridge built into PipelineWorker.

Inbound: typed RTVI UI messages from the client (fired via the RTVI
processor's ``on_ui_message`` event) are republished onto the bus as a
broadcast ``BusUIEventMessage``. Outbound: ``BusUICommandMessage`` and
the four ``BusUIJob*`` lifecycle carriers are translated into the
matching RTVI frames and queued downstream. The bridge is active only
when RTVI is enabled.
"""

import asyncio
import unittest

from pipecat.bus import BusTTSSpeakMessage
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIUICommandFrame, RTVIUIJobGroupFrame
from pipecat.processors.frameworks.rtvi.models import (
    A11yNode,
    A11ySnapshot,
    UICancelJobGroupData,
    UICancelJobGroupMessage,
    UIEventData,
    UIEventMessage,
    UISnapshotData,
    UISnapshotMessage,
)


def _make_root(*, enable_rtvi=True):
    """A PipelineWorker with bus + frame spies installed."""
    worker = PipelineWorker(
        Pipeline([IdentityFilter()]),
        name="root",
        enable_rtvi=enable_rtvi,
        cancel_on_idle_timeout=False,
    )
    sent: list = []
    frames: list = []

    async def _record_bus(message):
        sent.append(message)

    async def _record_frame(frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)

    worker.send_bus_message = _record_bus  # type: ignore[method-assign]
    worker.queue_frame = _record_frame  # type: ignore[method-assign]
    return worker, sent, frames


async def _fire_ui_message(worker, message):
    """Fire the RTVI processor's ``on_ui_message`` event and drain it."""
    await worker.rtvi._call_event_handler("on_ui_message", message)
    tasks = [t for (_name, t) in list(worker.rtvi._event_tasks)]
    if tasks:
        await asyncio.gather(*tasks)


class TestUIBridgeInbound(unittest.IsolatedAsyncioTestCase):
    async def test_republishes_ui_event_as_broadcast_bus_message(self):
        worker, sent, _frames = _make_root()

        await _fire_ui_message(
            worker,
            UIEventMessage(id="m1", data=UIEventData(event="nav_click", payload={"view": "home"})),
        )

        events = [m for m in sent if isinstance(m, BusUIEventMessage)]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "root")
        self.assertIsNone(events[0].target)
        self.assertEqual(events[0].event_name, "nav_click")
        self.assertEqual(events[0].payload, {"view": "home"})

    async def test_snapshot_message_routes_to_internal_event_name(self):
        worker, sent, _frames = _make_root()
        tree = A11ySnapshot(root=A11yNode(ref="root", role="document"), captured_at=1)

        await _fire_ui_message(worker, UISnapshotMessage(id="m2", data=UISnapshotData(tree=tree)))

        events = [m for m in sent if isinstance(m, BusUIEventMessage)]
        self.assertEqual(events[0].event_name, _UI_SNAPSHOT_BUS_EVENT_NAME)
        self.assertEqual(events[0].payload, tree.model_dump(exclude_none=True))

    async def test_cancel_task_message_routes_to_internal_event_name(self):
        worker, sent, _frames = _make_root()

        await _fire_ui_message(
            worker,
            UICancelJobGroupMessage(
                id="m3", data=UICancelJobGroupData(job_id="t-1", reason="user")
            ),
        )

        events = [m for m in sent if isinstance(m, BusUIEventMessage)]
        self.assertEqual(events[0].event_name, _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME)
        self.assertEqual(events[0].payload, {"job_id": "t-1", "reason": "user"})

    async def test_missing_payload_becomes_none(self):
        worker, sent, _frames = _make_root()

        await _fire_ui_message(worker, UIEventMessage(id="m1", data=UIEventData(event="hello")))

        events = [m for m in sent if isinstance(m, BusUIEventMessage)]
        self.assertEqual(events[0].event_name, "hello")
        self.assertIsNone(events[0].payload)

    async def test_unknown_message_type_is_ignored(self):
        worker, sent, _frames = _make_root()

        await _fire_ui_message(worker, object())

        self.assertEqual([m for m in sent if isinstance(m, BusUIEventMessage)], [])


class TestUIBridgeOutbound(unittest.IsolatedAsyncioTestCase):
    async def test_command_becomes_rtvi_ui_command_frame(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusUICommandMessage(source="ui", target=None, command_name="toast", payload={"t": "Hi"})
        )

        ui_frames = [f for f in frames if isinstance(f, RTVIUICommandFrame)]
        self.assertEqual(len(ui_frames), 1)
        self.assertEqual(ui_frames[0].command, "toast")
        self.assertEqual(ui_frames[0].payload, {"t": "Hi"})

    async def test_group_started_envelope(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusUIJobGroupStartedMessage(
                source="ui",
                target=None,
                job_id="t1",
                workers=["w1", "w2"],
                label="Doing stuff",
                cancellable=True,
                at=1700,
            )
        )

        frame = next(f for f in frames if isinstance(f, RTVIUIJobGroupFrame))
        self.assertEqual(frame.data.kind, "group_started")
        self.assertEqual(frame.data.job_id, "t1")
        self.assertEqual(frame.data.workers, ["w1", "w2"])
        self.assertEqual(frame.data.label, "Doing stuff")
        self.assertTrue(frame.data.cancellable)
        self.assertEqual(frame.data.at, 1700)

    async def test_job_update_envelope(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusUIJobUpdateMessage(
                source="ui",
                target=None,
                job_id="t1",
                worker_name="w1",
                data={"kind": "tool_call", "tool": "WebSearch"},
                at=1701,
            )
        )

        frame = next(f for f in frames if isinstance(f, RTVIUIJobGroupFrame))
        self.assertEqual(frame.data.kind, "job_update")
        self.assertEqual(frame.data.job_id, "t1")
        self.assertEqual(frame.data.worker_name, "w1")
        self.assertEqual(frame.data.data, {"kind": "tool_call", "tool": "WebSearch"})
        self.assertEqual(frame.data.at, 1701)

    async def test_job_completed_envelope(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusUIJobCompletedMessage(
                source="ui",
                target=None,
                job_id="t1",
                worker_name="w1",
                status="completed",
                response={"answer": 42},
                at=1702,
            )
        )

        frame = next(f for f in frames if isinstance(f, RTVIUIJobGroupFrame))
        self.assertEqual(frame.data.kind, "job_completed")
        self.assertEqual(frame.data.job_id, "t1")
        self.assertEqual(frame.data.worker_name, "w1")
        self.assertEqual(frame.data.status, "completed")
        self.assertEqual(frame.data.response, {"answer": 42})
        self.assertEqual(frame.data.at, 1702)

    async def test_group_completed_envelope(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusUIJobGroupCompletedMessage(source="ui", target=None, job_id="t1", at=1703)
        )

        frame = next(f for f in frames if isinstance(f, RTVIUIJobGroupFrame))
        self.assertEqual(frame.data.kind, "group_completed")
        self.assertEqual(frame.data.job_id, "t1")
        self.assertEqual(frame.data.at, 1703)

    async def test_non_ui_bus_message_queues_no_frame(self):
        worker, _sent, frames = _make_root()

        # A plain BusUIEventMessage (inbound carrier) is not an outbound
        # command/job-group, so the outbound translation must ignore it.
        await worker.on_bus_message(
            BusUIEventMessage(source="x", target=None, event_name="e", payload={})
        )

        self.assertEqual(
            [f for f in frames if isinstance(f, (RTVIUICommandFrame, RTVIUIJobGroupFrame))], []
        )

    async def test_worker_without_rtvi_does_not_translate(self):
        worker, _sent, frames = _make_root(enable_rtvi=False)
        self.assertIsNone(worker._rtvi)

        await worker.on_bus_message(
            BusUICommandMessage(source="ui", target=None, command_name="toast", payload={})
        )

        self.assertEqual(frames, [])


class TestUISpeakBridge(unittest.IsolatedAsyncioTestCase):
    async def test_speak_message_becomes_tts_speak_frame(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusTTSSpeakMessage(source="ui", target="root", text="hello there")
        )

        speak = [f for f in frames if isinstance(f, TTSSpeakFrame)]
        self.assertEqual(len(speak), 1)
        self.assertEqual(speak[0].text, "hello there")

    async def test_speak_message_works_without_rtvi(self):
        # TTS does not require RTVI; the speak branch must run regardless.
        worker, _sent, frames = _make_root(enable_rtvi=False)
        self.assertIsNone(worker._rtvi)

        await worker.on_bus_message(
            BusTTSSpeakMessage(source="ui", target="root", text="hello there")
        )

        speak = [f for f in frames if isinstance(f, TTSSpeakFrame)]
        self.assertEqual(len(speak), 1)
        self.assertEqual(speak[0].text, "hello there")

    async def test_speak_message_for_other_target_is_ignored(self):
        worker, _sent, frames = _make_root()

        await worker.on_bus_message(
            BusTTSSpeakMessage(source="ui", target="someone-else", text="hello")
        )

        self.assertEqual([f for f in frames if isinstance(f, TTSSpeakFrame)], [])


if __name__ == "__main__":
    unittest.main()
