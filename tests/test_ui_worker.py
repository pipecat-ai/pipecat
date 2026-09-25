#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for UIWorker dispatch, LLM-context injection, and single-flight respond."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.bus.messages import BusJobCancelMessage, BusJobRequestMessage, BusTTSSpeakMessage
from pipecat.bus.ui.messages import (
    UI_SNAPSHOT_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
)
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    YesNoResult,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
)
from pipecat.pipeline.job_context import JobError, JobStatus
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.workers.ui import UI_STATE_PROMPT_GUIDE, UIWorker, ui_event
from pipecat.workers.ui.ui_tools import screen_tools


class _StubUIWorker(UIWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: list[BusUIEventMessage] = []

    @ui_event("nav_click")
    async def _on_nav(self, message: BusUIEventMessage) -> None:
        self.captured.append(message)


class _PlainWorker(UIWorker):
    pass


async def _make_worker(cls=_StubUIWorker, **kwargs) -> UIWorker:
    """A UIWorker wired with a task manager and a ``queue_frame`` spy.

    ``queue_frame`` is replaced with a recorder so tests can assert the
    frames the UI logic produces without running the pipeline. The
    respond handler sends its own response, so ``send_job_response`` is
    mocked to let ``_run_llm_turn`` run without an active job entry.
    """
    worker = cls("ui", llm=MagicMock(), **kwargs)
    tm = TaskManager()
    worker._task_manager = tm

    recorded: list = []

    async def _record(frame, direction=FrameDirection.DOWNSTREAM):
        recorded.append(frame)

    worker.queue_frame = _record  # type: ignore[method-assign]
    worker._recorded = recorded  # type: ignore[attr-defined]
    worker.send_job_response = AsyncMock()  # type: ignore[method-assign]
    return worker


async def _settle() -> None:
    """Yield enough times for spawned task handlers to run/park."""
    for _ in range(10):
        await asyncio.sleep(0)


async def _start(worker: UIWorker, message: BusJobRequestMessage) -> asyncio.Task:
    """Start a respond turn; it parks at ``await self._pending`` until resolved."""
    t = asyncio.create_task(worker._run_llm_turn(message))
    await _settle()
    return t


def _respond_msg(job_id: str, query: str = "hi") -> BusJobRequestMessage:
    return BusJobRequestMessage(
        source="voice",
        target="ui",
        job_name="respond",
        job_id=job_id,
        payload={"query": query},
    )


async def _dispatch(worker: UIWorker, message: BusUIEventMessage) -> None:
    await worker.on_bus_message(message)
    for _ in range(5):
        await asyncio.sleep(0)


def _append_frames(worker) -> list[LLMMessagesAppendFrame]:
    return [f for f in worker._recorded if isinstance(f, LLMMessagesAppendFrame)]


def _update_frames(worker) -> list[LLMMessagesUpdateFrame]:
    return [f for f in worker._recorded if isinstance(f, LLMMessagesUpdateFrame)]


class TestUIWorkerDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_to_matching_ui_event_handler(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="nav_click", payload={"view": "home"}
            ),
        )

        self.assertEqual(len(worker.captured), 1)
        self.assertEqual(worker.captured[0].event_name, "nav_click")
        self.assertEqual(worker.captured[0].payload, {"view": "home"})

    async def test_unknown_event_name_does_not_raise(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="never_registered", payload={"x": 1}
            ),
        )

        self.assertEqual(worker.captured, [])

    async def test_ignores_events_targeted_at_other_workers(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music",
                target="someone_else",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        self.assertEqual(worker.captured, [])
        self.assertEqual(_append_frames(worker), [])

    async def test_broadcast_event_with_no_target_is_handled(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target=None, event_name="nav_click", payload={"view": "home"}
            ),
        )

        self.assertEqual(len(worker.captured), 1)

    async def test_handler_runs_in_separate_task_so_bus_is_not_blocked(self):
        gate = asyncio.Event()
        observed: list[bool] = []

        class _BlockingWorker(_StubUIWorker):
            @ui_event("slow")
            async def _slow(self, message):
                await gate.wait()
                observed.append(True)

        worker = await _make_worker(cls=_BlockingWorker)

        await worker.on_bus_message(
            BusUIEventMessage(source="music", target="ui", event_name="slow", payload={})
        )
        self.assertEqual(observed, [])

        gate.set()
        await _settle()
        self.assertEqual(observed, [True])

    async def test_duplicate_handler_names_raise_at_init(self):
        with self.assertRaises(ValueError):

            class _Bad(UIWorker):
                @ui_event("nav")
                async def a(self, message):
                    pass

                @ui_event("nav")
                async def b(self, message):
                    pass

            _Bad("ui", llm=MagicMock())

    async def test_default_construction_unaffected(self):
        worker = _PlainWorker("ui", llm=MagicMock())
        self.assertTrue(worker._auto_inject_ui_state)
        self.assertTrue(worker.active)


class TestUIWorkerInjection(unittest.IsolatedAsyncioTestCase):
    async def test_injects_xml_developer_message_by_default(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="nav_click", payload={"view": "home"}
            ),
        )

        frames = _append_frames(worker)
        self.assertEqual(len(frames), 1)

        frame = frames[0]
        self.assertFalse(frame.run_llm)
        self.assertEqual(len(frame.messages), 1)
        msg = frame.messages[0]
        self.assertEqual(msg["role"], "developer")

        content = msg["content"]
        self.assertIn('<ui_event name="nav_click">', content)
        self.assertIn("</ui_event>", content)
        inner = content[len('<ui_event name="nav_click">') : -len("</ui_event>")]
        self.assertEqual(json.loads(inner), {"view": "home"})

    async def test_inject_events_false_disables_injection(self):
        worker = await _make_worker(inject_events=False)

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="nav_click", payload={"view": "home"}
            ),
        )

        self.assertEqual(_append_frames(worker), [])
        self.assertEqual(len(worker.captured), 1)

    async def test_render_override_replaces_default_xml(self):
        class _CustomRender(_StubUIWorker):
            def render_ui_event(self, message):
                return f"[UI] {message.event_name}"

        worker = await _make_worker(cls=_CustomRender)

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="nav_click", payload={"view": "home"}
            ),
        )

        frames = _append_frames(worker)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].messages[0]["content"], "[UI] nav_click")

    async def test_empty_render_skips_injection(self):
        class _NoRender(_StubUIWorker):
            def render_ui_event(self, message):
                return ""

        worker = await _make_worker(cls=_NoRender)

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music", target="ui", event_name="nav_click", payload={"view": "home"}
            ),
        )

        self.assertEqual(_append_frames(worker), [])


_SAMPLE_SNAPSHOT = {
    "root": {
        "ref": "e1",
        "role": "generic",
        "children": [
            {
                "ref": "e2",
                "role": "main",
                "children": [
                    {"ref": "e3", "role": "heading", "name": "Home", "level": 1},
                    {
                        "ref": "e4",
                        "role": "region",
                        "name": "Trending artists",
                        "children": [
                            {"ref": "e5", "role": "button", "name": "Bad Bunny"},
                            {
                                "ref": "e6",
                                "role": "button",
                                "name": "Taylor Swift",
                                "state": ["focused"],
                            },
                        ],
                    },
                ],
            },
        ],
    },
    "captured_at": 1700000000000,
}


class TestUIWorkerSnapshot(unittest.IsolatedAsyncioTestCase):
    async def test_reserved_snapshot_event_stored_without_dispatch(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name=UI_SNAPSHOT_EVENT_NAME,
                payload=_SAMPLE_SNAPSHOT,
            ),
        )

        self.assertEqual(worker._latest_snapshot, _SAMPLE_SNAPSHOT)
        self.assertEqual(worker.captured, [])
        self.assertEqual(_append_frames(worker), [])

    async def test_non_dict_snapshot_payload_is_ignored(self):
        worker = await _make_worker()

        await _dispatch(
            worker,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name=UI_SNAPSHOT_EVENT_NAME,
                payload="not a snapshot",
            ),
        )

        self.assertIsNone(worker._latest_snapshot)

    async def test_render_ui_state_empty_without_snapshot(self):
        worker = await _make_worker()
        self.assertEqual(worker.render_ui_state(), "")

    async def test_render_ui_state_produces_indented_block(self):
        worker = await _make_worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        rendered = worker.render_ui_state()

        self.assertTrue(rendered.startswith("<ui_state>\n"))
        self.assertTrue(rendered.endswith("\n</ui_state>"))

        self.assertIn("- generic [ref=e1]:", rendered)
        self.assertIn("- main [ref=e2]:", rendered)
        self.assertIn('- heading "Home" [level=1] [ref=e3]', rendered)
        self.assertIn('- region "Trending artists" [ref=e4]:', rendered)
        self.assertIn('- button "Bad Bunny" [ref=e5]', rendered)
        self.assertIn('- button "Taylor Swift" [focused] [ref=e6]', rendered)

        self.assertIn("  - main", rendered)
        self.assertIn("    - heading", rendered)
        self.assertIn("      - button", rendered)

    async def test_inject_ui_state_queues_expected_frame(self):
        worker = await _make_worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        await worker.inject_ui_state()

        frames = _append_frames(worker)
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertFalse(frame.run_llm)
        self.assertEqual(len(frame.messages), 1)
        msg = frame.messages[0]
        self.assertEqual(msg["role"], "developer")
        self.assertTrue(msg["content"].startswith("<ui_state>"))
        self.assertTrue(msg["content"].endswith("</ui_state>"))

    async def test_inject_ui_state_no_op_without_snapshot(self):
        worker = await _make_worker()
        await worker.inject_ui_state()
        self.assertEqual(_append_frames(worker), [])

    async def test_render_emits_grid_dims(self):
        worker = await _make_worker()
        worker._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "generic",
                "children": [
                    {
                        "ref": "e2",
                        "role": "grid",
                        "name": "Trending artists",
                        "colcount": 8,
                        "rowcount": 2,
                        "children": [{"ref": "e3", "role": "button", "name": "Bad Bunny"}],
                    },
                ],
            },
            "captured_at": 1700000000000,
        }

        rendered = worker.render_ui_state()
        self.assertIn('- grid "Trending artists" [cols=8] [rows=2] [ref=e2]', rendered)

    async def test_render_preserves_offscreen_tag(self):
        worker = await _make_worker()
        worker._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "generic",
                "children": [
                    {"ref": "e2", "role": "button", "name": "Visible"},
                    {"ref": "e3", "role": "button", "name": "Below fold", "state": ["offscreen"]},
                ],
            },
            "captured_at": 1700000000000,
        }

        rendered = worker.render_ui_state()
        self.assertIn('- button "Visible" [ref=e2]', rendered)
        self.assertIn('- button "Below fold" [offscreen] [ref=e3]', rendered)

    async def test_render_emits_selection_block_when_present(self):
        worker = await _make_worker()
        worker._latest_snapshot = {
            "root": {"ref": "e1", "role": "generic", "children": [{"ref": "e2", "role": "main"}]},
            "captured_at": 1700000000000,
            "selection": {"ref": "e2", "text": "the highlighted passage"},
        }

        rendered = worker.render_ui_state()
        self.assertIn('<selection ref="e2">', rendered)
        self.assertIn("the highlighted passage", rendered)
        self.assertIn("</selection>", rendered)
        self.assertTrue(rendered.endswith("</ui_state>"))
        sel_idx = rendered.index('<selection ref="e2">')
        close_idx = rendered.index("</ui_state>")
        self.assertLess(sel_idx, close_idx)

    async def test_render_omits_selection_when_missing(self):
        worker = await _make_worker()
        worker._latest_snapshot = {
            "root": {"ref": "e1", "role": "generic"},
            "captured_at": 1700000000000,
        }

        rendered = worker.render_ui_state()
        self.assertNotIn("<selection", rendered)

    async def test_render_skips_selection_with_missing_ref_or_text(self):
        worker = await _make_worker()
        worker._latest_snapshot = {
            "root": {"ref": "e1", "role": "generic"},
            "captured_at": 1,
            "selection": {"ref": "e2"},
        }
        self.assertNotIn("<selection", worker.render_ui_state())

        worker._latest_snapshot = {
            "root": {"ref": "e1", "role": "generic"},
            "captured_at": 1,
            "selection": {"text": "stuff"},
        }
        self.assertNotIn("<selection", worker.render_ui_state())


class TestUIWorkerSnapshotInjection(unittest.IsolatedAsyncioTestCase):
    """The <ui_state> snapshot is injected just before inference via the LLM's
    on_before_process_frame hook (e.g. during a respond job).
    """

    def _worker(self, cls=_PlainWorker, **kwargs) -> UIWorker:
        # A real LLM service so the on_before_process_frame event actually fires.
        llm = OpenAILLMService(api_key="sk-test")
        return cls("ui", llm=llm, **kwargs)

    async def _fire(self, worker: UIWorker, context: LLMContext) -> None:
        await worker.llm._call_event_handler(
            "on_before_process_frame", LLMContextFrame(context=context)
        )

    def _developer_messages(self, context: LLMContext) -> list:
        return [m for m in context.messages if isinstance(m, dict) and m.get("role") == "developer"]

    async def test_injects_ui_state_on_user_turn(self):
        worker = self._worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        devs = self._developer_messages(ctx)
        self.assertEqual(len(devs), 1)
        self.assertTrue(devs[0]["content"].startswith("<ui_state>"))

    async def test_skips_tool_result_continuation(self):
        # The follow-up inference after a tool result must not stack a second
        # <ui_state> within the same turn.
        worker = self._worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "result"},
            ]
        )
        before = len(ctx.messages)
        await self._fire(worker, ctx)
        self.assertEqual(len(ctx.messages), before)

    async def test_no_injection_when_disabled(self):
        worker = self._worker(auto_inject_ui_state=False)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        self.assertEqual(len(ctx.messages), 1)

    async def test_no_injection_without_snapshot(self):
        worker = self._worker()
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        self.assertEqual(len(ctx.messages), 1)

    async def test_render_ui_state_override_amends_injected_content(self):
        # Apps override render_ui_state to amend what the LLM sees of the screen
        # (e.g. prepend app context or trim the tree). The auto-inject hook uses
        # the override's output, not the default rendering.
        class _CustomState(_PlainWorker):
            def render_ui_state(self) -> str:
                base = super().render_ui_state()
                return f"<app_note>cart has 2 items</app_note>\n{base}" if base else base

        worker = self._worker(cls=_CustomState)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        devs = self._developer_messages(ctx)
        self.assertEqual(len(devs), 1)
        self.assertTrue(devs[0]["content"].startswith("<app_note>cart has 2 items</app_note>"))
        self.assertIn("<ui_state>", devs[0]["content"])

    async def test_render_ui_state_override_empty_skips_injection(self):
        # An override returning "" suppresses injection even with a snapshot.
        class _NoState(_PlainWorker):
            def render_ui_state(self) -> str:
                return ""

        worker = self._worker(cls=_NoState)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        self.assertEqual(len(ctx.messages), 1)

    async def test_injects_once_per_turn(self):
        # After injecting, the tail is the developer message, so a re-fire on the
        # same context is a no-op (no accumulation within a turn).
        worker = self._worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        ctx = LLMContext([{"role": "user", "content": "hi"}])
        await self._fire(worker, ctx)
        await self._fire(worker, ctx)
        self.assertEqual(len(self._developer_messages(ctx)), 1)


class TestUIWorkerKeepHistory(unittest.IsolatedAsyncioTestCase):
    async def test_default_resets_context_per_job(self):
        worker = await _make_worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        t = await _start(
            worker,
            BusJobRequestMessage(source="voice", target="ui", job_id="t1", payload={"query": "hi"}),
        )

        updates = _update_frames(worker)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].messages, [])
        self.assertFalse(updates[0].run_llm)

        await worker.respond_to_job()
        await t

    async def test_reset_runs_before_inject(self):
        worker = await _make_worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        t = await _start(
            worker,
            BusJobRequestMessage(source="voice", target="ui", job_id="t1", payload={"query": "hi"}),
        )

        frame_types = [type(f).__name__ for f in worker._recorded]
        update_idx = frame_types.index("LLMMessagesUpdateFrame")
        append_idx = frame_types.index("LLMMessagesAppendFrame")
        self.assertLess(update_idx, append_idx)

        await worker.respond_to_job()
        await t

    async def test_keep_history_true_skips_reset(self):
        worker = await _make_worker(keep_history=True)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        t = await _start(
            worker,
            BusJobRequestMessage(source="voice", target="ui", job_id="t1", payload={"query": "hi"}),
        )

        # Injection moved to the on_before_process_frame hook, so _run_llm_turn
        # itself queues only the query append.
        self.assertEqual(_update_frames(worker), [])
        appends = _append_frames(worker)
        self.assertEqual(len(appends), 1)
        self.assertEqual(appends[0].messages[0]["content"], "hi")

        await worker.respond_to_job()
        await t

    async def test_reset_context_method_emits_update_frame(self):
        worker = await _make_worker(keep_history=True)

        await worker._reset_context()

        updates = _update_frames(worker)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].messages, [])
        self.assertFalse(updates[0].run_llm)


class TestUIWorkerRespondToJob(unittest.IsolatedAsyncioTestCase):
    async def test_current_job_tracks_in_flight_request(self):
        worker = await _make_worker()
        self.assertIsNone(worker.current_job)
        message = BusJobRequestMessage(
            source="voice", target="ui", job_id="t1", payload={"query": "hi"}
        )
        t = await _start(worker, message)
        self.assertIs(worker.current_job, message)

        await worker.respond_to_job()
        await t

    async def test_respond_to_job_responds_with_answer(self):
        worker = await _make_worker()
        message = BusJobRequestMessage(source="voice", target="ui", job_name="respond", job_id="t1")
        t = await _start(worker, message)

        await worker.respond_to_job("hello")
        await t

        worker.send_job_response.assert_awaited_once()
        call = worker.send_job_response.await_args
        self.assertEqual(call.args[0], "t1")
        self.assertEqual(call.kwargs["response"], {"answer": "hello"})
        self.assertIsNone(worker.current_job)

    async def test_respond_to_job_no_op_when_idle(self):
        worker = await _make_worker()
        await worker.respond_to_job("hello")
        worker.send_job_response.assert_not_awaited()

    async def test_respond_to_job_none_answer_responds_none(self):
        worker = await _make_worker()
        message = BusJobRequestMessage(source="voice", target="ui", job_name="respond", job_id="t1")
        t = await _start(worker, message)

        await worker.respond_to_job()
        await t

        self.assertIsNone(worker.send_job_response.await_args.kwargs["response"])

    async def test_respond_to_job_tts_speak_speaks_and_responds_none(self):
        worker = await _make_worker()
        worker.send_bus_message = AsyncMock()
        message = BusJobRequestMessage(source="voice", target="ui", job_name="respond", job_id="t1")
        t = await _start(worker, message)

        await worker.respond_to_job("spoken phrase", tts_speak=True)
        await t

        # Responds None so the requester's voice LLM does not run...
        self.assertIsNone(worker.send_job_response.await_args.kwargs["response"])
        # ...and publishes a BusTTSSpeakMessage to the requester for its TTS.
        worker.send_bus_message.assert_awaited_once()
        sent = worker.send_bus_message.await_args.args[0]
        self.assertIsInstance(sent, BusTTSSpeakMessage)
        self.assertEqual(sent.text, "spoken phrase")
        self.assertEqual(sent.target, "voice")

    async def test_respond_to_job_tts_speak_empty_answer_does_not_speak(self):
        worker = await _make_worker()
        worker.send_bus_message = AsyncMock()
        message = BusJobRequestMessage(source="voice", target="ui", job_name="respond", job_id="t1")
        t = await _start(worker, message)

        await worker.respond_to_job(tts_speak=True)
        await t

        worker.send_bus_message.assert_not_awaited()
        self.assertIsNone(worker.send_job_response.await_args.kwargs["response"])

    async def test_cancellation_frees_lock_for_subsequent_jobs(self):
        worker = await _make_worker()
        msg_a = _respond_msg("a")
        msg_b = _respond_msg("b")

        await worker._handle_job_request(msg_a)
        await _settle()
        self.assertIs(worker.current_job, msg_a)
        self.assertTrue(worker._job_locks["respond"].locked())

        await worker._handle_job_cancel(
            BusJobCancelMessage(source="voice", target="ui", job_id="a")
        )
        await _settle()
        self.assertIsNone(worker.current_job)
        self.assertFalse(worker._job_locks["respond"].locked())

        await worker._handle_job_request(msg_b)
        await _settle()
        self.assertIs(worker.current_job, msg_b)

        await worker.respond_to_job("B done")
        await _settle()

    async def test_cancel_unknown_job_id_is_noop(self):
        worker = await _make_worker()
        msg_a = _respond_msg("a")

        await worker._handle_job_request(msg_a)
        await _settle()

        await worker._handle_job_cancel(
            BusJobCancelMessage(source="voice", target="ui", job_id="unrelated")
        )
        await _settle()

        self.assertIs(worker.current_job, msg_a)
        self.assertTrue(worker._job_locks["respond"].locked())

        await worker.respond_to_job("A done")
        await _settle()

    async def test_concurrent_same_name_jobs_serialize(self):
        worker = await _make_worker()
        msg_a = _respond_msg("a")
        msg_b = _respond_msg("b")

        await worker._handle_job_request(msg_a)
        await _settle()
        self.assertIs(worker.current_job, msg_a)

        await worker._handle_job_request(msg_b)
        await _settle()
        self.assertIs(worker.current_job, msg_a)

        await worker.respond_to_job("A done")
        await _settle()
        self.assertIs(worker.current_job, msg_b)

        await worker.respond_to_job("B done")
        await _settle()
        self.assertIsNone(worker.current_job)


class TestUIWorkerRespondJob(unittest.IsolatedAsyncioTestCase):
    async def test_respond_handler_runs_after_setup(self):
        worker = await _make_worker()
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        t = await _start(
            worker,
            BusJobRequestMessage(
                source="voice", target="ui", job_id="t1", payload={"query": "hello"}
            ),
        )

        # The snapshot is injected by the hook at inference time, so the handler
        # itself queues only the rendered query.
        appends = _append_frames(worker)
        self.assertEqual(len(appends), 1)
        self.assertEqual(appends[0].messages[0]["content"], "hello")
        self.assertTrue(appends[0].run_llm)
        self.assertEqual(worker.current_job.job_id, "t1")

        await worker.respond_to_job("done")
        await t
        self.assertIsNone(worker.current_job)
        worker.send_job_response.assert_awaited_once()

    async def test_render_query_override(self):
        class _Custom(_StubUIWorker):
            def render_query(self, message):
                return f"Q: {message.payload['q']}"

        worker = await _make_worker(cls=_Custom)
        t = await _start(
            worker,
            BusJobRequestMessage(source="voice", target="ui", job_id="t1", payload={"q": "hi"}),
        )

        query_frames = [f for f in _append_frames(worker) if f.run_llm]
        self.assertEqual(query_frames[0].messages[0]["content"], "Q: hi")

        await worker.respond_to_job()
        await t

    async def test_handler_failure_clears_state(self):
        class _Boom(_StubUIWorker):
            def render_query(self, message):
                raise RuntimeError("boom")

        worker = await _make_worker(cls=_Boom)

        with self.assertRaises(RuntimeError):
            await worker._run_llm_turn(
                BusJobRequestMessage(
                    source="voice", target="ui", job_id="t1", payload={"query": "x"}
                )
            )

        self.assertIsNone(worker.current_job)
        self.assertIsNone(worker._pending)


class TestUIWorkerPromptGuide(unittest.IsolatedAsyncioTestCase):
    async def test_default_appends_ui_state_prompt_guide(self):
        worker = UIWorker("ui", llm=MagicMock())
        worker.llm.append_system_instruction.assert_called_once_with(UI_STATE_PROMPT_GUIDE)

    async def test_custom_prompt_guide_overrides(self):
        worker = UIWorker("ui", llm=MagicMock(), prompt_guide="MY GUIDE")
        worker.llm.append_system_instruction.assert_called_once_with("MY GUIDE")

    async def test_none_disables_injection(self):
        worker = UIWorker("ui", llm=MagicMock(), prompt_guide=None)
        worker.llm.append_system_instruction.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class _FakeClassifier(BaseClassifier):
    """Answers from a scripted probability and records what it was asked."""

    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability
        self.choice_label = ""
        self.asked: list = []
        self.setup_task_manager = None
        self.cleaned_up = False

    async def setup(self, task_manager):
        self.setup_task_manager = task_manager

    async def cleanup(self):
        self.cleaned_up = True

    async def _ask(self, state, questions):
        results = {}
        for name, question in questions.items():
            self.asked.append((state, question))
            if isinstance(question, ChoiceQuestion):
                options = question.options
                label = self.choice_label if self.choice_label in options else next(iter(options))
                results[name] = ChoiceResult(
                    choice=label,
                    probabilities={o: float(o == label) for o in options},
                    confidence=self.probability,
                )
            else:
                results[name] = YesNoResult(probability=self.probability)
        return results, None


class TestUIWorkerClassifier(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_to_an_llm_classifier_over_the_llm(self):
        worker = await _make_worker()
        self.assertIsInstance(worker.classifier, LLMClassifier)
        self.assertIs(worker.classifier.llm, worker.llm)

    async def test_activation_sets_the_classifier_up_and_cleanup_cleans_it(self):
        classifier = _FakeClassifier(0.9)
        worker = await _make_worker(classifier=classifier)

        await worker.on_activated(None)
        self.assertIs(classifier.setup_task_manager, worker.task_manager)
        self.assertIs(worker.classifier, classifier)

        await worker.cleanup()
        self.assertTrue(classifier.cleaned_up)

    async def test_should_respond_asks_with_the_event_and_the_screen(self):
        classifier = _FakeClassifier(0.9)
        worker = await _make_worker(classifier=classifier)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        event = BusUIEventMessage(
            source="music", target="ui", event_name="nav_click", payload={"view": "home"}
        )

        self.assertTrue(await worker.should_respond(event))

        state, question = classifier.asked[0]
        self.assertEqual(state["event"], {"name": "nav_click", "payload": {"view": "home"}})
        self.assertTrue(state["screen"].startswith("<ui_state>"))
        self.assertIn("say something", question.instructions)
        self.assertIn("routine click", question.no)

    async def test_should_respond_is_false_when_no_is_likelier(self):
        worker = await _make_worker(classifier=_FakeClassifier(0.3))
        event = BusUIEventMessage(source="music", target="ui", event_name="scroll", payload={})
        self.assertFalse(await worker.should_respond(event))

    async def test_should_respond_without_a_snapshot_sends_only_the_event(self):
        classifier = _FakeClassifier(0.9)
        worker = await _make_worker(classifier=classifier)
        event = BusUIEventMessage(source="music", target="ui", event_name="scroll", payload={})

        await worker.should_respond(event)

        state, _ = classifier.asked[0]
        self.assertNotIn("screen", state)

    async def test_which_element_asks_over_the_named_elements(self):
        classifier = _FakeClassifier(0.9)
        classifier.choice_label = "e6"
        worker = await _make_worker(classifier=classifier)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT

        ref = await worker.which_element("the Taylor Swift one")

        self.assertEqual(ref, "e6")
        state, question = classifier.asked[0]
        self.assertEqual(state["utterance"], "the Taylor Swift one")
        self.assertIn("<ui_state>", state["screen"])
        self.assertEqual(question.options["e5"], 'button "Bad Bunny"')
        self.assertEqual(question.options["e6"], 'button "Taylor Swift"')
        self.assertNotIn("e1", question.options)
        self.assertIn("referring to", question.instructions)

    async def test_which_element_returns_none_below_the_threshold(self):
        classifier = _FakeClassifier(0.2)
        classifier.choice_label = "e5"
        worker = await _make_worker(classifier=classifier)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        self.assertIsNone(await worker.which_element("that one"))

    async def test_which_element_without_a_snapshot_asks_nothing(self):
        classifier = _FakeClassifier(0.9)
        worker = await _make_worker(classifier=classifier)
        self.assertIsNone(await worker.which_element("that one"))
        self.assertEqual(classifier.asked, [])

    async def test_say_asks_the_pipeline_to_speak(self):
        worker = await _make_worker()
        worker.send_bus_message = AsyncMock()

        await worker.say("Two items in your cart.")

        sent = worker.send_bus_message.await_args.args[0]
        self.assertIsInstance(sent, BusTTSSpeakMessage)
        self.assertEqual(sent.text, "Two items in your cart.")
        self.assertIsNone(sent.target)
        self.assertTrue(sent.append_to_context)

        await worker.say("Done.", target="voice")
        self.assertEqual(worker.send_bus_message.await_args.args[0].target, "voice")


def _job(name: str, payload: dict) -> BusJobRequestMessage:
    return BusJobRequestMessage(
        source="voice", target="ui", job_name=name, job_id="j1", payload=payload
    )


class TestUIWorkerScreenJobs(unittest.IsolatedAsyncioTestCase):
    async def _worker(self, probability: float = 0.9, choice: str = "e6"):
        classifier = _FakeClassifier(probability)
        classifier.choice_label = choice
        worker = await _make_worker(classifier=classifier)
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        worker.send_bus_message = AsyncMock()
        return worker, classifier

    def _response(self, worker):
        args, kwargs = worker.send_job_response.await_args
        return args[1], kwargs.get("status", JobStatus.COMPLETED)

    async def test_find_answers_with_the_element_and_its_confidence(self):
        worker, _ = await self._worker()
        await worker._screen_job(
            _job("screen", {"action": "find", "target": "the Taylor Swift one"})
        )
        response, status = self._response(worker)
        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(response, {"label": "Taylor Swift", "confidence": 0.9})

    async def test_find_without_named_elements_answers_nothing(self):
        worker, classifier = await self._worker()
        worker._latest_snapshot = None
        await worker._screen_job(_job("screen", {"action": "find", "target": "anything"}))
        response, _ = self._response(worker)
        self.assertEqual(response, {"label": None, "confidence": 0.0})
        self.assertEqual(classifier.asked, [])

    async def test_check_answers_yes_or_no_about_the_screen(self):
        worker, classifier = await self._worker(probability=0.8)
        await worker._screen_job(
            _job("screen", {"action": "check", "target": "is an artist focused?"})
        )
        response, _ = self._response(worker)
        self.assertEqual(response, {"yes": True, "probability": 0.8})
        state, question = classifier.asked[0]
        self.assertTrue(state["screen"].startswith("<ui_state>"))
        self.assertEqual(question.instructions, "is an artist focused?")

    async def test_select_asks_one_question_per_element_in_one_call(self):
        worker, classifier = await self._worker(probability=0.7)
        await worker._screen_job(_job("screen", {"action": "select", "target": "musicians"}))
        response, _ = self._response(worker)
        self.assertEqual(
            response,
            {
                "matches": [
                    {"label": "Home", "probability": 0.7},
                    {"label": "Trending artists", "probability": 0.7},
                    {"label": "Bad Bunny", "probability": 0.7},
                    {"label": "Taylor Swift", "probability": 0.7},
                ]
            },
        )
        # One call, four questions, keyed by ref.
        self.assertEqual(len({id(state) for state, _ in classifier.asked}), 1)
        self.assertEqual(len(classifier.asked), 4)

    async def test_act_finds_the_element_and_sends_the_command(self):
        worker, _ = await self._worker(choice="e5")
        await worker._screen_job(_job("screen", {"action": "click", "target": "Bad Bunny"}))
        response, _ = self._response(worker)
        self.assertEqual(response, {"done": True, "label": "Bad Bunny"})
        sent = worker.send_bus_message.await_args.args[0]
        self.assertIsInstance(sent, BusUICommandMessage)
        self.assertEqual(sent.command_name, "click")
        self.assertEqual(sent.payload["ref"], "e5")

    async def test_act_below_the_threshold_does_nothing(self):
        worker, _ = await self._worker(probability=0.2, choice="e5")
        await worker._screen_job(_job("screen", {"action": "click", "target": "that"}))
        response, _ = self._response(worker)
        self.assertEqual(response, {"done": False, "label": None})
        worker.send_bus_message.assert_not_awaited()

    async def test_act_writes_a_value_into_an_input(self):
        worker, _ = await self._worker(choice="e6")
        await worker._screen_job(
            _job("screen", {"action": "fill", "target": "Taylor", "value": "hi"})
        )
        sent = worker.send_bus_message.await_args.args[0]
        self.assertEqual(sent.command_name, "set_input_value")
        self.assertEqual((sent.payload["ref"], sent.payload["value"]), ("e6", "hi"))

    async def test_act_with_an_unknown_action_is_a_value_error(self):
        worker, _ = await self._worker()
        with self.assertRaises(ValueError):
            await worker.act("explode", "the button")

    async def test_elements_lists_names_roles_and_state_without_refs(self):
        worker, _ = await self._worker()
        await worker._screen_job(_job("screen", {"action": "list", "target": "button"}))
        response, _ = self._response(worker)
        self.assertEqual(
            response,
            {
                "elements": [
                    {"role": "button", "name": "Bad Bunny", "state": [], "value": None},
                    {"role": "button", "name": "Taylor Swift", "state": ["focused"], "value": None},
                ]
            },
        )

    async def test_elements_carry_an_input_value(self):
        worker, _ = await self._worker()
        worker._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "form",
                "children": [
                    {"ref": "e2", "role": "textbox", "name": "Email", "value": "a@b.c"},
                    {"ref": "e3", "role": "textbox", "name": "Phone"},
                ],
            }
        }
        await worker._screen_job(_job("screen", {"action": "list", "target": "textbox"}))
        response, _ = self._response(worker)
        self.assertEqual(
            response,
            {
                "elements": [
                    {"role": "textbox", "name": "Email", "state": [], "value": "a@b.c"},
                    {"role": "textbox", "name": "Phone", "state": [], "value": None},
                ]
            },
        )

    async def test_an_unknown_action_answers_with_an_error_status(self):
        worker, _ = await self._worker()
        await worker._screen_job(_job("screen", {"action": "explode", "target": "the button"}))
        response, status = self._response(worker)
        self.assertEqual(status, JobStatus.ERROR)
        self.assertIn("explode", response["error"])

    async def test_a_classifier_failure_answers_with_an_error_status(self):
        from pipecat.classifiers.base_classifier import ClassifierError

        class _Broken(_FakeClassifier):
            async def _ask(self, state, questions):
                raise ClassifierError("down")

        worker = await _make_worker(classifier=_Broken(0.9))
        worker._latest_snapshot = _SAMPLE_SNAPSHOT
        await worker._screen_job(_job("screen", {"action": "check", "target": "?"}))
        response, status = self._response(worker)
        self.assertEqual(status, JobStatus.ERROR)
        self.assertEqual(response, {"error": "down"})


class _FakeJob:
    """What ``pipeline_worker.job(...)`` returns: a context whose response is scripted."""

    def __init__(self, response=None, error: str | None = None):
        self.response = response
        self._error = error

    async def __aenter__(self):
        if self._error:
            raise JobError(self._error)
        return self

    async def __aexit__(self, *exc):
        return False


class TestScreenTool(unittest.IsolatedAsyncioTestCase):
    def test_the_tool_describes_itself_to_the_llm(self):
        (tool,) = screen_tools("ui")
        schema = DirectFunctionWrapper(tool)
        self.assertEqual(schema.name, "screen")
        self.assertEqual(sorted(schema.properties), ["action", "target", "value"])
        self.assertEqual(schema.required, ["action"])
        self.assertIn("fill", schema.properties["action"]["description"])
        self.assertIn("checkout button", schema.description)

    async def test_the_tool_sends_the_job_and_returns_its_answer(self):
        (tool,) = screen_tools("ui", timeout=5)
        params = MagicMock()
        params.pipeline_worker.job = MagicMock(return_value=_FakeJob({"label": "Taylor Swift"}))
        params.result_callback = AsyncMock()

        await tool(params, action="find", target="the Taylor Swift one")

        params.pipeline_worker.job.assert_called_once_with(
            "ui",
            name="screen",
            payload={"action": "find", "target": "the Taylor Swift one"},
            timeout=5,
        )
        params.result_callback.assert_awaited_once_with({"label": "Taylor Swift"})

    async def test_a_failed_job_returns_the_error_as_data(self):
        (tool,) = screen_tools("ui")
        params = MagicMock()
        params.pipeline_worker.job = MagicMock(return_value=_FakeJob(error="no such worker"))
        params.result_callback = AsyncMock()

        await tool(params, action="check", target="anything?")

        params.result_callback.assert_awaited_once_with({"error": "no such worker"})
