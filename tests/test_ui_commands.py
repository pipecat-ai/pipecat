#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for UIWorker.send_command and standard command payload models."""

import unittest
from unittest.mock import MagicMock

from pipecat.bus.ui.messages import BusUICommandMessage
from pipecat.processors.frameworks.rtvi.models import (
    Click,
    Focus,
    Highlight,
    Navigate,
    ScrollTo,
    SelectText,
    SetInputValue,
    Toast,
)
from pipecat.workers.ui import UIWorker


def _make_worker():
    """A UIWorker whose ``send_bus_message`` captures sent commands.

    ``send_command`` publishes via ``send_bus_message``; replacing it
    avoids needing an attached bus for these unit tests.
    """
    worker = UIWorker("ui", llm=MagicMock())
    sent: list[BusUICommandMessage] = []

    async def _record(message):
        if isinstance(message, BusUICommandMessage):
            sent.append(message)

    worker.send_bus_message = _record  # type: ignore[method-assign]
    return worker, sent


class TestSendCommand(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_pydantic_payload_via_model_dump(self):
        worker, sent = _make_worker()

        await worker.send_command("toast", Toast(title="Saved", subtitle="Favorites"))

        self.assertEqual(len(sent), 1)
        cmd = sent[0]
        self.assertEqual(cmd.source, "ui")
        self.assertIsNone(cmd.target)
        self.assertEqual(cmd.command_name, "toast")
        self.assertEqual(
            cmd.payload,
            {
                "title": "Saved",
                "subtitle": "Favorites",
                "description": None,
                "image_url": None,
                "duration_ms": None,
            },
        )

    async def test_forwards_dict_payload_as_is(self):
        worker, sent = _make_worker()

        await worker.send_command("app_specific", {"foo": 1, "bar": [1, 2, 3]})

        self.assertEqual(sent[0].command_name, "app_specific")
        self.assertEqual(sent[0].payload, {"foo": 1, "bar": [1, 2, 3]})

    async def test_none_payload_becomes_empty_dict(self):
        worker, sent = _make_worker()

        await worker.send_command("ping")

        self.assertEqual(sent[0].payload, {})

    async def test_dict_payload_for_apps_with_custom_command_names(self):
        worker, sent = _make_worker()

        await worker.send_command("navigate", {"view": "home"})
        self.assertEqual(sent[0].payload, {"view": "home"})


class TestStandardCommands(unittest.IsolatedAsyncioTestCase):
    async def test_toast_payload_shape(self):
        worker, sent = _make_worker()
        await worker.send_command(
            "toast",
            Toast(
                title="Now playing",
                subtitle="Nirvana",
                description="Smells Like Teen Spirit",
                image_url="https://example.com/cover.jpg",
                duration_ms=3000,
            ),
        )
        self.assertEqual(
            sent[0].payload,
            {
                "title": "Now playing",
                "subtitle": "Nirvana",
                "description": "Smells Like Teen Spirit",
                "image_url": "https://example.com/cover.jpg",
                "duration_ms": 3000,
            },
        )

    async def test_navigate_payload_shape(self):
        worker, sent = _make_worker()
        await worker.send_command("navigate", Navigate(view="detail", params={"id": "42"}))
        self.assertEqual(sent[0].payload, {"view": "detail", "params": {"id": "42"}})

    async def test_scroll_to_payload_shape_by_target_id(self):
        worker, sent = _make_worker()
        await worker.send_command(
            "scroll_to", ScrollTo(target_id="new_releases", behavior="smooth")
        )
        self.assertEqual(
            sent[0].payload,
            {"ref": None, "target_id": "new_releases", "behavior": "smooth"},
        )

    async def test_scroll_to_payload_shape_by_ref(self):
        worker, sent = _make_worker()
        await worker.send_command("scroll_to", ScrollTo(ref="e42", behavior="smooth"))
        self.assertEqual(
            sent[0].payload,
            {"ref": "e42", "target_id": None, "behavior": "smooth"},
        )

    async def test_highlight_payload_shape(self):
        worker, sent = _make_worker()
        await worker.send_command("highlight", Highlight(target_id="play_btn", duration_ms=1000))
        self.assertEqual(
            sent[0].payload,
            {"ref": None, "target_id": "play_btn", "duration_ms": 1000},
        )

    async def test_focus_payload_shape(self):
        worker, sent = _make_worker()
        await worker.send_command("focus", Focus(target_id="search_input"))
        self.assertEqual(
            sent[0].payload,
            {"ref": None, "target_id": "search_input"},
        )

    async def test_focus_payload_shape_by_ref(self):
        worker, sent = _make_worker()
        await worker.send_command("focus", Focus(ref="e7"))
        self.assertEqual(
            sent[0].payload,
            {"ref": "e7", "target_id": None},
        )

    async def test_select_text_payload_whole_element(self):
        worker, sent = _make_worker()
        await worker.send_command("select_text", SelectText(ref="e42"))
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": None,
                "end_offset": None,
            },
        )

    async def test_select_text_payload_with_offsets(self):
        worker, sent = _make_worker()
        await worker.send_command(
            "select_text",
            SelectText(ref="e42", start_offset=5, end_offset=12),
        )
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": 5,
                "end_offset": 12,
            },
        )

    async def test_set_input_value_payload_replace_default(self):
        worker, sent = _make_worker()
        await worker.send_command(
            "set_input_value",
            SetInputValue(ref="e7", value="hello"),
        )
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e7",
                "target_id": None,
                "value": "hello",
                "replace": True,
            },
        )

    async def test_set_input_value_payload_append(self):
        worker, sent = _make_worker()
        await worker.send_command(
            "set_input_value",
            SetInputValue(ref="e7", value=" world", replace=False),
        )
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e7",
                "target_id": None,
                "value": " world",
                "replace": False,
            },
        )

    async def test_click_payload_by_ref(self):
        worker, sent = _make_worker()
        await worker.send_command("click", Click(ref="e42"))
        self.assertEqual(sent[0].payload, {"ref": "e42", "target_id": None})

    async def test_click_payload_by_target_id(self):
        worker, sent = _make_worker()
        await worker.send_command("click", Click(target_id="submit"))
        self.assertEqual(sent[0].payload, {"ref": None, "target_id": "submit"})


if __name__ == "__main__":
    unittest.main()
