#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smoke tests for the UI Agent Protocol wire format.

The module under test is data only (constants, payload models, and
envelope classes), so the goal is to pin the shapes: any accidental
rename or type change to a wire-format field would break compatibility
with existing client code, and we want a test that fails loudly.
"""

import unittest

from pipecat.processors.frameworks.rtvi.models import (
    Click,
    Focus,
    Highlight,
    Navigate,
    ScrollTo,
    SelectText,
    SetInputValue,
    Toast,
    UICancelTaskData,
    UICancelTaskMessage,
    UICommandData,
    UICommandMessage,
    UIEventData,
    UIEventMessage,
    UISnapshotData,
    UISnapshotMessage,
    UITaskCompletedData,
    UITaskGroupCompletedData,
    UITaskGroupStartedData,
    UITaskMessage,
    UITaskUpdateData,
)


class TestEnvelopeMessages(unittest.TestCase):
    """Pin the on-the-wire envelope shapes for each first-class UI message."""

    def test_ui_event_envelope(self):
        msg = UIEventMessage(id="m1", data=UIEventData(name="nav_click", payload={"view": "home"}))
        self.assertEqual(
            msg.model_dump(),
            {
                "label": "rtvi-ai",
                "type": "ui-event",
                "id": "m1",
                "data": {"name": "nav_click", "payload": {"view": "home"}},
            },
        )

    def test_ui_command_envelope_no_id(self):
        # Server-to-client push: no id field on the envelope (matches
        # ServerMessage / LLMFunctionCallMessage shape).
        msg = UICommandMessage(data=UICommandData(name="toast", payload={"title": "Saved"}))
        self.assertEqual(
            msg.model_dump(),
            {
                "label": "rtvi-ai",
                "type": "ui-command",
                "data": {"name": "toast", "payload": {"title": "Saved"}},
            },
        )

    def test_ui_snapshot_envelope(self):
        msg = UISnapshotMessage(id="m2", data=UISnapshotData(tree={"root": "..."}))
        self.assertEqual(
            msg.model_dump(),
            {
                "label": "rtvi-ai",
                "type": "ui-snapshot",
                "id": "m2",
                "data": {"tree": {"root": "..."}},
            },
        )

    def test_ui_cancel_task_envelope(self):
        msg = UICancelTaskMessage(id="m3", data=UICancelTaskData(task_id="t-99", reason="user"))
        self.assertEqual(
            msg.model_dump(),
            {
                "label": "rtvi-ai",
                "type": "ui-cancel-task",
                "id": "m3",
                "data": {"task_id": "t-99", "reason": "user"},
            },
        )

    def test_ui_task_group_started(self):
        msg = UITaskMessage(
            data=UITaskGroupStartedData(task_id="t-1", agents=["a", "b"], label="Search", at=42)
        )
        self.assertEqual(msg.type, "ui-task")
        self.assertEqual(msg.data.kind, "group_started")
        self.assertEqual(msg.data.task_id, "t-1")

    def test_ui_task_update(self):
        msg = UITaskMessage(
            data=UITaskUpdateData(task_id="t-1", agent_name="a", data={"progress": 0.5}, at=43)
        )
        self.assertEqual(msg.data.kind, "task_update")
        self.assertEqual(msg.data.agent_name, "a")

    def test_ui_task_completed(self):
        msg = UITaskMessage(
            data=UITaskCompletedData(
                task_id="t-1", agent_name="a", status="completed", response={"ok": True}, at=44
            )
        )
        self.assertEqual(msg.data.kind, "task_completed")
        self.assertEqual(msg.data.status, "completed")

    def test_ui_task_group_completed(self):
        msg = UITaskMessage(data=UITaskGroupCompletedData(task_id="t-1", at=45))
        self.assertEqual(msg.data.kind, "group_completed")


class TestPayloadShapes(unittest.TestCase):
    """Pin the on-the-wire dict shape of each command payload."""

    def test_toast_required_only(self):
        self.assertEqual(
            dict(Toast(title="Saved")),
            {
                "title": "Saved",
                "subtitle": None,
                "description": None,
                "image_url": None,
                "duration_ms": None,
            },
        )

    def test_navigate(self):
        self.assertEqual(
            dict(Navigate(view="home", params={"id": "42"})),
            {"view": "home", "params": {"id": "42"}},
        )

    def test_scroll_to_with_ref(self):
        self.assertEqual(
            dict(ScrollTo(ref="e42", behavior="smooth")),
            {"ref": "e42", "target_id": None, "behavior": "smooth"},
        )

    def test_highlight_with_target_id(self):
        self.assertEqual(
            dict(Highlight(target_id="cta", duration_ms=2000)),
            {"ref": None, "target_id": "cta", "duration_ms": 2000},
        )

    def test_focus(self):
        self.assertEqual(
            dict(Focus(ref="e7")),
            {"ref": "e7", "target_id": None},
        )

    def test_click(self):
        self.assertEqual(
            dict(Click(ref="e9")),
            {"ref": "e9", "target_id": None},
        )

    def test_set_input_value_default_replace_true(self):
        self.assertEqual(
            dict(SetInputValue(ref="e3", value="Marie Curie")),
            {"value": "Marie Curie", "ref": "e3", "target_id": None, "replace": True},
        )

    def test_set_input_value_append(self):
        payload = dict(SetInputValue(ref="e3", value="more", replace=False))
        self.assertFalse(payload["replace"])

    def test_select_text_full(self):
        self.assertEqual(
            dict(SelectText(ref="e15", start_offset=4, end_offset=12)),
            {"ref": "e15", "target_id": None, "start_offset": 4, "end_offset": 12},
        )

    def test_select_text_whole_target_when_offsets_omitted(self):
        payload = dict(SelectText(ref="e15"))
        self.assertIsNone(payload["start_offset"])
        self.assertIsNone(payload["end_offset"])


if __name__ == "__main__":
    unittest.main()
