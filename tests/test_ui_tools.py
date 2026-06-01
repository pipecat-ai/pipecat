#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``ReplyToolMixin`` and the action helper methods on ``UIWorker``.

The mixin exposes a single bundled ``reply(answer, scroll_to,
highlight, ...)`` LLM tool whose ``answer`` argument is required. The
helper methods (``scroll_to``, ``highlight``, ...) are plain instance
methods on ``UIWorker`` that wrap ``send_command`` with the standard
payload models; apps call them inside custom ``@tool`` bodies when the
canonical ``reply`` shape doesn't fit.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.bus.ui.messages import BusUICommandMessage
from pipecat.workers.llm.tool_decorator import _collect_tools
from pipecat.workers.ui import ReplyToolMixin, UIWorker


class _WorkerWithReply(ReplyToolMixin, UIWorker):
    pass


class _PlainWorker(UIWorker):
    pass


def _new(cls: type) -> UIWorker:
    return cls("ui", llm=MagicMock())


def _capture(worker: UIWorker) -> list[BusUICommandMessage]:
    sent: list[BusUICommandMessage] = []

    async def _record(message):
        sent.append(message)

    worker.send_bus_message = _record  # type: ignore[method-assign]
    return sent


class TestUIWorkerActionHelpers(unittest.IsolatedAsyncioTestCase):
    """The helper methods are plain methods, not LLM tools."""

    async def test_scroll_to_helper_dispatches_command(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.scroll_to("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "scroll_to")
        self.assertEqual(
            sent[0].payload,
            {"ref": "e42", "target_id": None, "behavior": None},
        )

    async def test_highlight_helper_dispatches_command(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.highlight("e7")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "highlight")
        self.assertEqual(
            sent[0].payload,
            {"ref": "e7", "target_id": None, "duration_ms": None},
        )

    async def test_select_text_helper_whole_element(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.select_text("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "select_text")
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": None,
                "end_offset": None,
            },
        )

    async def test_select_text_helper_with_offsets(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.select_text("e42", start_offset=10, end_offset=25)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": 10,
                "end_offset": 25,
            },
        )

    async def test_click_helper_dispatches_command(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.click("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "click")
        self.assertEqual(sent[0].payload, {"ref": "e42", "target_id": None})

    async def test_set_input_value_helper_default_replace(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.set_input_value("e42", "hello world")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "set_input_value")
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "value": "hello world",
                "replace": True,
            },
        )

    async def test_set_input_value_helper_append_mode(self):
        worker = _new(_PlainWorker)
        sent = _capture(worker)

        await worker.set_input_value("e42", "more text", replace=False)

        self.assertEqual(sent[0].payload["replace"], False)

    async def test_helpers_are_not_llm_tools(self):
        worker = _new(_PlainWorker)
        tool_names = [t.__name__ for t in _collect_tools(worker)]
        for name in ("scroll_to", "highlight", "select_text", "click", "set_input_value"):
            self.assertNotIn(name, tool_names)


class TestReplyToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_reply_tool(self):
        worker = _new(_WorkerWithReply)
        tool_names = [t.__name__ for t in _collect_tools(worker)]
        self.assertEqual(tool_names, ["reply"])

    async def test_plain_uiworker_has_no_reply_tool(self):
        worker = _new(_PlainWorker)
        tool_names = [t.__name__ for t in _collect_tools(worker)]
        self.assertNotIn("reply", tool_names)

    async def test_reply_with_answer_only_terminates(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(params, answer="The Pixel 9 is from Google.")

        self.assertEqual(sent, [])
        worker.respond_to_job.assert_awaited_once_with(
            "The Pixel 9 is from Google.", tts_speak=True
        )
        params.result_callback.assert_awaited_once_with(None)

    async def test_reply_with_highlight_only(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="This one, the Nothing Phone 3.",
            highlight=["e29"],
        )

        self.assertEqual([m.command_name for m in sent], ["highlight"])
        self.assertEqual(sent[0].payload["ref"], "e29")
        worker.respond_to_job.assert_awaited_once_with(
            "This one, the Nothing Phone 3.", tts_speak=True
        )

    async def test_reply_with_multiple_highlights(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="Here are the Apple phones.",
            highlight=["e5", "e8", "e47"],
        )

        self.assertEqual([m.command_name for m in sent], ["highlight"] * 3)
        self.assertEqual([m.payload["ref"] for m in sent], ["e5", "e8", "e47"])
        worker.respond_to_job.assert_awaited_once_with("Here are the Apple phones.", tts_speak=True)

    async def test_reply_with_scroll_and_highlight_runs_in_order(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="Here's the iPhone 17.",
            scroll_to="e5",
            highlight=["e5"],
        )

        self.assertEqual([m.command_name for m in sent], ["scroll_to", "highlight"])
        worker.respond_to_job.assert_awaited_once_with("Here's the iPhone 17.", tts_speak=True)

    async def test_reply_with_select_text_only(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="Here, in this paragraph.",
            select_text="e11",
        )

        self.assertEqual([m.command_name for m in sent], ["select_text"])
        self.assertEqual(sent[0].payload["ref"], "e11")
        worker.respond_to_job.assert_awaited_once_with("Here, in this paragraph.", tts_speak=True)

    async def test_reply_with_scroll_and_select_text(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="Here, in this paragraph.",
            scroll_to="e11",
            select_text="e11",
        )

        self.assertEqual([m.command_name for m in sent], ["scroll_to", "select_text"])

    async def test_reply_with_fills_writes_each_input(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="Got it.",
            fills=[
                {"ref": "e5", "value": "Mark"},
                {"ref": "e7", "value": "Backman"},
            ],
        )

        self.assertEqual(
            [m.command_name for m in sent],
            ["set_input_value", "set_input_value"],
        )
        self.assertEqual(sent[0].payload["ref"], "e5")
        self.assertEqual(sent[0].payload["value"], "Mark")
        self.assertEqual(sent[1].payload["ref"], "e7")
        self.assertEqual(sent[1].payload["value"], "Backman")

    async def test_reply_with_click_clicks_each_in_order(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(params, answer="Submitted.", click=["e22", "e26"])

        self.assertEqual([m.command_name for m in sent], ["click", "click"])
        self.assertEqual([m.payload["ref"] for m in sent], ["e22", "e26"])

    async def test_reply_with_fills_skips_invalid_entries(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="x",
            fills=[
                {"ref": "e5", "value": "Mark"},
                {"ref": None, "value": "missing ref"},
                {"value": "no ref"},
                {"ref": "e7"},
            ],
        )

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].payload["ref"], "e5")

    async def test_reply_with_non_dict_fill_entries_does_not_crash(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="x",
            fills=[None, "e5", 42, {"ref": "e9", "value": "ok"}],  # type: ignore[list-item]
        )

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].payload["ref"], "e9")
        worker.respond_to_job.assert_awaited_once_with("x", tts_speak=True)
        params.result_callback.assert_awaited_once_with(None)

    async def test_reply_with_non_string_highlight_refs_skipped(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="x",
            highlight=[None, "e1", 42, "e2"],  # type: ignore[list-item]
        )

        self.assertEqual([m.payload["ref"] for m in sent], ["e1", "e2"])
        worker.respond_to_job.assert_awaited_once_with("x", tts_speak=True)

    async def test_reply_with_non_string_click_refs_skipped(self):
        worker = _new(_WorkerWithReply)
        sent = _capture(worker)
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="x",
            click=[None, "e1", {"ref": "e2"}, "e3"],  # type: ignore[list-item]
        )

        self.assertEqual([m.payload["ref"] for m in sent], ["e1", "e3"])
        worker.respond_to_job.assert_awaited_once_with("x", tts_speak=True)

    async def test_reply_dispatches_via_helper_methods(self):
        worker = _new(_WorkerWithReply)
        worker.scroll_to = AsyncMock()  # type: ignore[method-assign]
        worker.highlight = AsyncMock()  # type: ignore[method-assign]
        worker.select_text = AsyncMock()  # type: ignore[method-assign]
        worker.set_input_value = AsyncMock()  # type: ignore[method-assign]
        worker.click = AsyncMock()  # type: ignore[method-assign]
        worker.respond_to_job = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await worker.reply(
            params,
            answer="x",
            scroll_to="e1",
            highlight=["e2", "e3"],
            select_text="e4",
            fills=[{"ref": "e5", "value": "v"}],
            click=["e6", "e7"],
        )

        worker.scroll_to.assert_awaited_once_with("e1")
        self.assertEqual(
            worker.highlight.await_args_list,
            [unittest.mock.call("e2"), unittest.mock.call("e3")],
        )
        worker.select_text.assert_awaited_once_with("e4")
        worker.set_input_value.assert_awaited_once_with("e5", "v")
        self.assertEqual(
            worker.click.await_args_list,
            [unittest.mock.call("e6"), unittest.mock.call("e7")],
        )


if __name__ == "__main__":
    unittest.main()
