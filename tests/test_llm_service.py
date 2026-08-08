#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.utils.async_tool_cancellation import CANCEL_ASYNC_TOOL_NAME
from pipecat.utils.asyncio.task_manager import TaskManager


def _expected_missing_tool_message(name: str) -> str:
    return LLMService.MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE.format(function_name=name)


class MockLLMService(LLMService):
    """Minimal LLM service for testing function call execution."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=kwargs.pop("system_instruction", None),
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=kwargs.pop("filter_incomplete_user_turns", None),
            user_turn_completion_config=kwargs.pop("user_turn_completion_config", None),
        )
        super().__init__(settings=settings, **kwargs)
        # Stub the pipeline task so FunctionCallParams can be constructed.
        self._pipeline_worker = SimpleNamespace(app_resources=None)


class TestUnparameterizedSubclass(unittest.TestCase):
    """Backward-compat coverage: third-party providers subclass LLMService
    without specifying a generic adapter parameter. That should keep working
    after LLMService became `Generic[TAdapter]`.
    """

    def test_unparameterized_subclass_instantiates(self):
        # MockLLMService is declared as `class MockLLMService(LLMService):`
        # — no generic bracket. The TypeVar's `bound=BaseLLMAdapter` should
        # resolve TAdapter to BaseLLMAdapter for callers that don't opt in.
        service = MockLLMService()
        adapter = service.get_llm_adapter()

        # Default adapter_class is OpenAILLMAdapter; the runtime instance
        # should reflect that, regardless of how generics are erased.
        self.assertIsInstance(adapter, OpenAILLMAdapter)
        self.assertIsInstance(adapter, BaseLLMAdapter)


class TestLLMService(unittest.IsolatedAsyncioTestCase):
    async def _run_function_calls_inline(self, service: MockLLMService):
        async def run_inline(runner_items):
            for runner_item in runner_items:
                await service._run_function_call(runner_item)

        service._run_parallel_function_calls = run_inline
        service._run_sequential_function_calls = run_inline

    async def test_missing_function_call_emits_terminal_result(self):
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        with patch("pipecat.services.llm_service.logger") as mock_logger:
            await service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="missing_tool",
                        tool_call_id="call_1",
                        arguments={"query": "weather"},
                        context=LLMContext(),
                    )
                ]
            )

        self.assertEqual(
            [type(frame) for frame in recorded_frames],
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ],
        )
        self.assertEqual(recorded_frames[1].function_name, "missing_tool")
        self.assertEqual(
            recorded_frames[2].result,
            _expected_missing_tool_message("missing_tool"),
        )

        # The tool was not advertised, so this is treated as a hallucination
        # (warning at queue time). The execution-time "just unregistered"
        # warning must not double-log.
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("not in the currently advertised tool set" in w for w in warnings))
        self.assertFalse(any("just unregistered" in w for w in warnings))

    async def test_function_call_timeout_cancels_handler(self):
        """A timed-out function call must not keep running or emit a late result."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        results = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            if frame_cls is FunctionCallResultFrame:
                results.append(kwargs["result"])

        service.broadcast_frame = mock_broadcast_frame

        handler_cancelled = asyncio.Event()
        side_effects = []

        async def slow_handler(params):
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

            side_effects.append("completed")
            await params.result_callback("late-success")

        service.register_function("slow_tool", slow_handler, timeout_secs=0.01)

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="slow_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        await asyncio.wait_for(handler_cancelled.wait(), timeout=1.0)
        self.assertEqual(side_effects, [])
        self.assertEqual(results, [None])

    async def test_function_call_timeout_does_not_wait_for_cancellation_cleanup(self):
        """The timeout result is terminal even while cancellation cleanup is still running."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        results = []
        timeout_result = asyncio.Event()

        async def mock_broadcast_frame(frame_cls, **kwargs):
            if frame_cls is FunctionCallResultFrame:
                results.append(kwargs["result"])
                timeout_result.set()

        service.broadcast_frame = mock_broadcast_frame

        handler_cancelled = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cleanup_completed = asyncio.Event()

        async def handler_with_slow_cancellation_cleanup(params):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                await finish_cleanup.wait()
                await params.result_callback("late-success")
                cleanup_completed.set()

        service.register_function(
            "slow_cleanup_tool", handler_with_slow_cancellation_cleanup, timeout_secs=0.01
        )

        call_task = asyncio.create_task(
            service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="slow_cleanup_tool",
                        tool_call_id="call_1",
                        arguments={},
                        context=LLMContext(),
                    )
                ]
            )
        )

        await asyncio.wait_for(timeout_result.wait(), timeout=1.0)
        await asyncio.wait_for(call_task, timeout=1.0)

        self.assertTrue(handler_cancelled.is_set())
        self.assertTrue(call_task.done())
        self.assertFalse(cleanup_completed.is_set())
        self.assertEqual(results, [None])

        finish_cleanup.set()
        await asyncio.wait_for(cleanup_completed.wait(), timeout=1.0)
        self.assertEqual(results, [None])

    async def test_cleanup_drains_timed_out_handler_cleanup(self):
        """A detached timeout handler remains owned by the service until cleanup."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        timeout_result = asyncio.Event()

        async def mock_broadcast_frame(frame_cls, **kwargs):
            if frame_cls is FunctionCallResultFrame:
                timeout_result.set()

        service.broadcast_frame = mock_broadcast_frame

        first_cancellation = asyncio.Event()
        cleanup_exited = asyncio.Event()

        async def cancellation_suppressing_handler(params):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancellation.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_exited.set()

        service.register_function(
            "cleanup_owned_tool", cancellation_suppressing_handler, timeout_secs=0.01
        )

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="cleanup_owned_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        await asyncio.wait_for(timeout_result.wait(), timeout=1.0)
        await asyncio.wait_for(first_cancellation.wait(), timeout=1.0)
        self.assertEqual(len(service._timed_out_function_call_tasks), 1)

        await asyncio.wait_for(service.cleanup(), timeout=1.0)

        self.assertTrue(cleanup_exited.is_set())
        self.assertEqual(service._timed_out_function_call_tasks, set())

    async def test_function_call_timeout_stops_after_result_callback(self):
        """A valid result callback satisfies the timeout while handler cleanup continues."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        results = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            if frame_cls is FunctionCallResultFrame:
                results.append(kwargs["result"])

        service.broadcast_frame = mock_broadcast_frame

        cleanup_completed = asyncio.Event()

        async def handler_with_cleanup(params):
            await params.result_callback("success")
            await asyncio.sleep(0.05)
            cleanup_completed.set()

        service.register_function("cleanup_tool", handler_with_cleanup, timeout_secs=0.01)

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="cleanup_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        self.assertTrue(cleanup_completed.is_set())
        self.assertEqual(results, ["success"])

    async def test_function_call_interruption_cancels_owned_timeout_handler(self):
        """External cancellation still propagates to a handler with a global timeout."""
        service = MockLLMService(function_call_timeout_secs=1.0)
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        results = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            if frame_cls is FunctionCallResultFrame:
                results.append(kwargs["result"])

        service.broadcast_frame = mock_broadcast_frame

        handler_started = asyncio.Event()
        handler_cancelled = asyncio.Event()

        async def interrupted_handler(params):
            handler_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        service.register_function("interrupted_tool", interrupted_handler)

        call_task = asyncio.create_task(
            service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="interrupted_tool",
                        tool_call_id="call_1",
                        arguments={},
                        context=LLMContext(),
                    )
                ]
            )
        )

        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        call_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call_task

        self.assertTrue(handler_cancelled.is_set())
        self.assertEqual(results, [])

    async def test_timeout_terminal_claim_wins_over_interruption(self):
        """Interruption cannot replace a timeout result whose broadcast has started."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()

        result_broadcast_started = asyncio.Event()
        finish_result_broadcast = asyncio.Event()
        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls)
            if frame_cls is FunctionCallResultFrame:
                result_broadcast_started.set()
                await finish_result_broadcast.wait()

        service.broadcast_frame = mock_broadcast_frame

        async def slow_handler(params):
            await asyncio.Event().wait()

        service.register_function("racing_tool", slow_handler, timeout_secs=0.01)

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="racing_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        await asyncio.wait_for(result_broadcast_started.wait(), timeout=1.0)
        function_call_task = next(iter(service._function_call_tasks))

        await service._cancel_function_call("racing_tool")
        self.assertNotIn(FunctionCallCancelFrame, recorded_frames)

        finish_result_broadcast.set()
        await asyncio.wait_for(function_call_task, timeout=1.0)
        self.assertEqual(recorded_frames.count(FunctionCallResultFrame), 1)

    async def test_interruption_terminal_claim_drops_cleanup_result(self):
        """A handler cannot replace a claimed cancellation during cleanup."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls)

        service.broadcast_frame = mock_broadcast_frame

        handler_started = asyncio.Event()

        async def cleanup_result_handler(params):
            handler_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await params.result_callback("cleanup-result")

        service.register_function("cleanup_result_tool", cleanup_result_handler)

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="cleanup_result_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        await service._cancel_function_call("cleanup_result_tool")

        terminal_frames = [
            frame_cls
            for frame_cls in recorded_frames
            if frame_cls in (FunctionCallResultFrame, FunctionCallCancelFrame)
        ]
        self.assertEqual(terminal_frames, [FunctionCallCancelFrame])

    async def test_function_call_preserves_handler_timeout_error(self):
        """TimeoutError from application code follows the normal error path."""
        service = MockLLMService()
        service._task_manager = TaskManager()
        service._call_event_handler = AsyncMock()
        service.broadcast_frame = AsyncMock()
        service.push_error = AsyncMock()
        await self._run_function_calls_inline(service)

        async def failing_handler(params):
            raise TimeoutError("downstream request timed out")

        service.register_function("failing_tool", failing_handler, timeout_secs=1.0)

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="failing_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        service.push_error.assert_awaited_once()
        self.assertIn(
            "downstream request timed out", service.push_error.await_args.kwargs["error_msg"]
        )
        self.assertFalse(
            any(
                call.args and call.args[0] is FunctionCallResultFrame
                for call in service.broadcast_frame.await_args_list
            )
        )

    async def test_function_unregistered_between_queue_and_execute(self):
        """Function unregistered between queuing and execution still terminates."""
        service = MockLLMService()
        service._call_event_handler = AsyncMock()

        async def real_handler(params):
            await params.result_callback("should not be called")

        service.register_function("doomed_tool", real_handler)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        async def run_inline(runner_items):
            # Simulate the function being unregistered after queuing but before execution.
            service.unregister_function("doomed_tool")
            for runner_item in runner_items:
                await service._run_function_call(runner_item)

        service._run_parallel_function_calls = run_inline
        service._run_sequential_function_calls = run_inline

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="doomed_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        self.assertEqual(
            [type(frame) for frame in recorded_frames],
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ],
        )
        self.assertEqual(
            recorded_frames[2].result,
            _expected_missing_tool_message("doomed_tool"),
        )

    async def test_missing_function_call_dev_error_logged_as_error(self):
        """Tool advertised to the LLM but missing a handler → logger.error."""
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)
        service.broadcast_frame = AsyncMock()

        context = LLMContext(
            tools=ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="advertised_but_unhandled",
                        description="",
                        properties={},
                        required=[],
                    )
                ]
            )
        )

        with patch("pipecat.services.llm_service.logger") as mock_logger:
            await service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="advertised_but_unhandled",
                        tool_call_id="call_1",
                        arguments={},
                        context=context,
                    )
                ]
            )

        errors = [c.args[0] for c in mock_logger.error.call_args_list]
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        self.assertTrue(
            any(
                "advertised" in e and "register_function" in e and "advertised_but_unhandled" in e
                for e in errors
            ),
            f"expected dev-error log; got errors={errors}, warnings={warnings}",
        )
        self.assertFalse(any("not in the currently advertised tool set" in w for w in warnings))

    async def test_missing_function_call_hallucination_logged_as_warning(self):
        """Tool not advertised to the LLM → logger.warning (hallucination)."""
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)
        service.broadcast_frame = AsyncMock()

        context = LLMContext(
            tools=ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="something_else",
                        description="",
                        properties={},
                        required=[],
                    )
                ]
            )
        )

        with patch("pipecat.services.llm_service.logger") as mock_logger:
            await service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="never_advertised",
                        tool_call_id="call_1",
                        arguments={},
                        context=context,
                    )
                ]
            )

        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        errors = [c.args[0] for c in mock_logger.error.call_args_list]
        self.assertTrue(
            any(
                "not in the currently advertised tool set" in w and "never_advertised" in w
                for w in warnings
            ),
            f"expected hallucination warning; got warnings={warnings}, errors={errors}",
        )
        self.assertFalse(any("advertised" in e and "register_function" in e for e in errors))

    async def test_catch_all_handler_suppresses_missing_warnings(self):
        """register_function(None, ...) suppresses both dev-error and hallucination logs."""
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)
        service.broadcast_frame = AsyncMock()

        async def catch_all(params):
            await params.result_callback("handled")

        service.register_function(None, catch_all)

        with patch("pipecat.services.llm_service.logger") as mock_logger:
            await service.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name="anything",
                        tool_call_id="call_1",
                        arguments={},
                        context=LLMContext(),
                    )
                ]
            )

        errors = [c.args[0] for c in mock_logger.error.call_args_list]
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        self.assertFalse(any("register_function" in e for e in errors))
        self.assertFalse(any("not in the currently advertised tool set" in w for w in warnings))

    async def test_missing_function_call_allows_user_mute_cleanup(self):
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="missing_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        strategy = FunctionCallUserMuteStrategy()
        muted = False
        for frame in recorded_frames:
            muted = await strategy.process_frame(frame)

        self.assertFalse(muted)

    async def test_builtin_cancel_tool_allows_user_mute_cleanup(self):
        """The built-in cancel tool is excluded from FunctionCallsStartedFrame,
        so the mute strategy sees a result frame for a tool call id it is not
        tracking.
        """
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        async def async_tool(params: FunctionCallParams):
            await params.result_callback("done")

        service.register_function("async_tool", async_tool, cancel_on_interruption=False)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=CANCEL_ASYNC_TOOL_NAME,
                    tool_call_id="cancel_1",
                    arguments={"tool_call_id": "call_1"},
                    context=LLMContext(),
                )
            ]
        )

        self.assertNotIn(FunctionCallsStartedFrame, [type(frame) for frame in recorded_frames])

        strategy = FunctionCallUserMuteStrategy()
        muted = False
        for frame in recorded_frames:
            muted = await strategy.process_frame(frame)

        self.assertFalse(muted)

    async def test_intermediate_results_allow_user_mute_cleanup(self):
        """An async tool reporting intermediate updates emits a result frame per
        update, so the mute strategy sees the same tool call id finish twice.
        """
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        await self._run_function_calls_inline(service)

        async def async_tool(params: FunctionCallParams):
            await params.result_callback(
                "working", properties=FunctionCallResultProperties(is_final=False)
            )
            await params.result_callback("done")

        service.register_function("async_tool", async_tool, cancel_on_interruption=False)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="async_tool",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        self.assertEqual(
            len([f for f in recorded_frames if isinstance(f, FunctionCallResultFrame)]), 2
        )

        strategy = FunctionCallUserMuteStrategy()
        muted = False
        for frame in recorded_frames:
            muted = await strategy.process_frame(frame)

        self.assertFalse(muted)


class TestAppendSystemInstruction(unittest.IsolatedAsyncioTestCase):
    """Coverage for `LLMService.append_system_instruction`."""

    def _service(self, system_instruction: str | None = None) -> MockLLMService:
        # Construct with the prompt so the base snapshot happens the real way
        # (in __init__), rather than poking _base_system_instruction directly.
        return MockLLMService(system_instruction=system_instruction)

    def test_append_preserves_existing_prompt(self):
        service = self._service("APP")
        service.append_system_instruction("GUIDE")
        self.assertEqual(service._settings.system_instruction, "APP\n\nGUIDE")

    def test_append_with_no_base_uses_text_alone(self):
        service = self._service(None)
        service.append_system_instruction("GUIDE")
        self.assertEqual(service._settings.system_instruction, "GUIDE")

    def test_multiple_appends_join_in_order(self):
        service = self._service("APP")
        service.append_system_instruction("G1")
        service.append_system_instruction("G2")
        self.assertEqual(service._settings.system_instruction, "APP\n\nG1\n\nG2")

    async def test_appended_guide_survives_turn_completion_toggle(self):
        service = self._service("APP")
        service.append_system_instruction("GUIDE")

        # Enabling turn completion composes after the appended guide, once.
        await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
        composed = service._settings.system_instruction
        self.assertTrue(composed.startswith("APP\n\nGUIDE\n\n"))
        self.assertEqual(composed.count("GUIDE"), 1)

        # Disabling restores base + guide (without the turn instructions).
        await service._update_settings(LLMSettings(filter_incomplete_user_turns=False))
        self.assertEqual(service._settings.system_instruction, "APP\n\nGUIDE")

    async def test_runtime_system_instruction_update_preserves_appended(self):
        service = self._service("APP")
        service.append_system_instruction("GUIDE")

        # A runtime system_instruction change replaces the base but keeps the
        # appended guide composed onto the end.
        await service._update_settings(LLMSettings(system_instruction="NEW"))
        self.assertEqual(service._settings.system_instruction, "NEW\n\nGUIDE")

    async def test_base_set_after_append_composes(self):
        # No base at construction; the guide is appended first, then the user
        # sets a system_instruction at runtime. The guide is retained.
        service = self._service(None)
        service.append_system_instruction("GUIDE")
        self.assertEqual(service._settings.system_instruction, "GUIDE")

        await service._update_settings(LLMSettings(system_instruction="APP"))
        self.assertEqual(service._settings.system_instruction, "APP\n\nGUIDE")

    async def test_appended_guide_survives_async_tool_cancellation_toggle(self):
        service = self._service("APP")
        service.append_system_instruction("GUIDE")

        # Enabling async tool cancellation composes after the appended guide,
        # without duplicating it.
        service._setup_async_tool_cancellation()
        composed = service._settings.system_instruction
        self.assertTrue(composed.startswith("APP\n\nGUIDE\n\n"))
        self.assertEqual(composed.count("GUIDE"), 1)
        self.assertNotEqual(composed, "APP\n\nGUIDE")  # async instructions appended

        # Disabling recomposes back to base + guide.
        service._teardown_async_tool_cancellation()
        self.assertEqual(service._settings.system_instruction, "APP\n\nGUIDE")


class TestProcessFrameToolWiring(unittest.IsolatedAsyncioTestCase):
    """process_frame syncs handlers from the context frame's advertised tools."""

    async def test_context_frame_syncs_registered_direct_functions(self):
        service = MockLLMService()
        service._sync_registered_tool_handlers = Mock()
        ctx = LLMContext(tools=NOT_GIVEN)
        await service.process_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
        service._sync_registered_tool_handlers.assert_called_once_with(ctx.tools)

    async def test_base_service_does_not_handle_set_tools_frame(self):
        # The base service syncs handlers only from the context frame. An
        # LLMSetToolsFrame is a pure aggregator concern here; only realtime
        # services that run continuously handle it for handler sync.
        service = MockLLMService()
        service._sync_registered_tool_handlers = Mock()
        await service.process_frame(LLMSetToolsFrame(tools=NOT_GIVEN), FrameDirection.DOWNSTREAM)
        service._sync_registered_tool_handlers.assert_not_called()


class TestTurnCompletionSettingsDeprecation(unittest.IsolatedAsyncioTestCase):
    """The turn-completion settings belong to the user turn strategy.

    ``LLMTurnCompletionUserTurnStopStrategy`` enables them over an
    ``LLMUpdateSettingsFrame`` and is also what finalizes the user turn from the
    resulting verdict, so an application that sets them on the service directly
    gets the marker protocol with nothing waiting on it.
    """

    def test_filter_incomplete_user_turns_at_construction_warns(self):
        with self.assertWarns(DeprecationWarning) as ctx:
            MockLLMService(filter_incomplete_user_turns=True)
        self.assertIn("FilterIncompleteUserTurnStrategies", str(ctx.warning))

    def test_user_turn_completion_config_at_construction_warns(self):
        with self.assertWarns(DeprecationWarning) as ctx:
            MockLLMService(user_turn_completion_config=UserTurnCompletionConfig())
        self.assertIn("FilterIncompleteUserTurnStrategies(config=...)", str(ctx.warning))

    def test_construction_without_the_settings_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            MockLLMService()
            # Spelling the fields out with their off values, the way concrete
            # services build their defaults, must not warn either.
            MockLLMService(filter_incomplete_user_turns=False, user_turn_completion_config=None)

    async def test_strategy_settings_update_is_silent(self):
        """The strategy's own update is how the setting is meant to arrive."""
        service = MockLLMService()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await service.process_frame(
                LLMUpdateSettingsFrame(delta=LLMSettings(filter_incomplete_user_turns=True)),
                FrameDirection.DOWNSTREAM,
            )
        self.assertTrue(service._filter_incomplete_user_turns)
