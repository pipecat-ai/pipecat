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

import httpx

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
    InterruptionFrame,
    LLMContextFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallRunnerItem,
    LLMService,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.utils.async_tool_cancellation import cancel_tool_name
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory
from tests.frame_processor_helpers import frame_processor_setup


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
        # Stub the pipeline worker so FunctionCallParams can be constructed.
        self._setup = frame_processor_setup(
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None)
        )


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

        service.register_function(
            "async_tool", async_tool, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service._sync_registered_tool_handlers(NOT_GIVEN)

        recorded_frames = []

        async def mock_broadcast_frame(frame_cls, **kwargs):
            recorded_frames.append(frame_cls(**kwargs))

        service.broadcast_frame = mock_broadcast_frame

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=cancel_tool_name("async_tool"),
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

        service.register_function(
            "async_tool", async_tool, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service._sync_registered_tool_handlers(NOT_GIVEN)

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


class TestFunctionCallTimeout(unittest.IsolatedAsyncioTestCase):
    """A function call that runs past its deadline is cancelled, not abandoned."""

    # Deadlines and handler durations are kept small so the suite stays quick;
    # the gap between them is what each test actually depends on.
    TIMEOUT = 0.1
    HANDLER_DURATION = 0.4
    SETTLE = 0.6

    def _service(self, **kwargs) -> tuple[MockLLMService, list]:
        service = MockLLMService(**kwargs)
        # These tests exercise real function call tasks, so the service needs
        # the task manager a pipeline would otherwise hand it.
        service._task_manager = TaskManager()

        broadcast_frames = []

        async def mock_broadcast_frame(frame_cls, **frame_kwargs):
            broadcast_frames.append(frame_cls(**frame_kwargs))

        service.broadcast_frame = mock_broadcast_frame
        return service, broadcast_frames

    async def _run_call(self, service: MockLLMService, function_name: str, tool_call_id="call_1"):
        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

    async def _assert_timeout_cancels_handler(self, run_in_parallel: bool):
        service, frames = self._service(
            function_call_timeout_secs=self.TIMEOUT, run_in_parallel=run_in_parallel
        )
        if not run_in_parallel:
            await service._create_sequential_runner_task()

        side_effects = []
        rolled_back = []

        async def slow(params: FunctionCallParams):
            try:
                await asyncio.sleep(self.HANDLER_DURATION)
                side_effects.append(params.tool_call_id)
                await params.result_callback({"ok": True})
            except asyncio.CancelledError:
                rolled_back.append(params.tool_call_id)
                raise

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(side_effects, [])
        self.assertEqual(rolled_back, ["call_1"])
        self.assertEqual(
            [type(frame) for frame in frames],
            [
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallCancelFrame,
            ],
        )

        if not run_in_parallel:
            await service._cancel_sequential_runner_task()

    async def test_timeout_cancels_handler_before_its_side_effect(self):
        await self._assert_timeout_cancels_handler(run_in_parallel=True)

    async def test_timeout_cancels_handler_before_its_side_effect_sequentially(self):
        await self._assert_timeout_cancels_handler(run_in_parallel=False)

    async def test_timeout_notifies_application_code(self):
        service, _ = self._service(function_call_timeout_secs=self.TIMEOUT)
        cancelled = []

        @service.event_handler("on_function_calls_cancelled")
        async def on_cancelled(service, function_calls):
            cancelled.extend(f.tool_call_id for f in function_calls)

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(cancelled, ["call_1"])

    async def test_timeout_asks_for_inference(self):
        """Nothing else follows up on a deadline, so it has to run the LLM."""
        service, frames = self._service(function_call_timeout_secs=self.TIMEOUT)

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.SETTLE)

        self.assertIsInstance(frames[-1], FunctionCallCancelFrame)
        self.assertTrue(frames[-1].run_llm)

    async def test_interruption_does_not_ask_for_inference(self):
        """The user is talking; a cancelled call must not answer over them."""
        service, frames = self._service()

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.TIMEOUT)
        await service._handle_interruptions(InterruptionFrame())
        await asyncio.sleep(self.SETTLE)

        self.assertIsInstance(frames[-1], FunctionCallCancelFrame)
        self.assertFalse(frames[-1].run_llm)

    async def test_llm_requested_cancellation_does_not_ask_for_inference(self):
        """The cancelling tool's own result runs the LLM; a second run would double up."""
        service, frames = self._service()

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)

        service.register_function("slow", slow, cancel_on_interruption=False)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.TIMEOUT)
        await service._cancel_function_calls_by_tool_call_id("call_1")
        await asyncio.sleep(self.SETTLE)

        self.assertIsInstance(frames[-1], FunctionCallCancelFrame)
        self.assertFalse(frames[-1].run_llm)

    async def test_sequential_runner_survives_a_timeout(self):
        """A cancelled call must not take the sequential runner down with it."""
        service, frames = self._service(
            function_call_timeout_secs=self.TIMEOUT, run_in_parallel=False
        )
        await service._create_sequential_runner_task()

        completed = []

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)

        async def quick(params: FunctionCallParams):
            completed.append(params.tool_call_id)
            await params.result_callback({"ok": True})

        service.register_function("slow", slow)
        service.register_function("quick", quick)

        await self._run_call(service, "slow", tool_call_id="call_1")
        await asyncio.sleep(self.SETTLE)
        await self._run_call(service, "quick", tool_call_id="call_2")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(completed, ["call_2"])
        self.assertFalse(service._sequential_runner_task.done())

        await service._cancel_sequential_runner_task()

    async def test_result_delivered_after_a_timeout_is_rejected(self):
        """A handler that outlives its deadline can't settle the call late."""
        service, frames = self._service(function_call_timeout_secs=self.TIMEOUT)

        async def stubborn(params: FunctionCallParams):
            try:
                await asyncio.sleep(self.HANDLER_DURATION)
            except asyncio.CancelledError:
                # The call is already settled, so this result must not reach
                # the pipeline.
                await params.result_callback({"too": "late"})
                raise

        service.register_function("stubborn", stubborn)
        await self._run_call(service, "stubborn")
        await asyncio.sleep(self.SETTLE)

        self.assertFalse(any(isinstance(frame, FunctionCallResultFrame) for frame in frames))

    async def test_result_delivered_after_an_interruption_is_rejected(self):
        """Cancellation settles a call whatever triggered it, not just a timeout."""
        service, frames = self._service()

        async def stubborn(params: FunctionCallParams):
            try:
                await asyncio.sleep(self.HANDLER_DURATION)
            except asyncio.CancelledError:
                await params.result_callback({"too": "late"})
                raise

        service.register_function("stubborn", stubborn)
        await self._run_call(service, "stubborn")
        await asyncio.sleep(self.TIMEOUT)
        await service._handle_interruptions(InterruptionFrame())
        await asyncio.sleep(self.SETTLE)

        self.assertFalse(any(isinstance(frame, FunctionCallResultFrame) for frame in frames))
        self.assertIsInstance(frames[-1], FunctionCallCancelFrame)

    async def test_handler_finishing_within_the_deadline_is_untouched(self):
        service, frames = self._service(function_call_timeout_secs=self.HANDLER_DURATION)

        async def quick(params: FunctionCallParams):
            await asyncio.sleep(self.TIMEOUT)
            await params.result_callback({"ok": True})

        service.register_function("quick", quick)
        await self._run_call(service, "quick")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(
            [type(frame) for frame in frames],
            [FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
        )
        self.assertEqual(frames[-1].result, {"ok": True})

    async def test_deferred_result_after_the_handler_returns_is_delivered(self):
        """The deadline covers the handler; a call it deferred still settles."""
        service, frames = self._service(function_call_timeout_secs=self.TIMEOUT)
        deferred = {}

        async def defer(params: FunctionCallParams):
            deferred["result_callback"] = params.result_callback

        service.register_function("defer", defer, cancel_on_interruption=False)
        await self._run_call(service, "defer")
        await asyncio.sleep(self.SETTLE)

        await deferred["result_callback"]({"ok": True})

        self.assertEqual(
            [type(frame) for frame in frames],
            [FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
        )
        self.assertEqual(frames[-1].result, {"ok": True})

    async def test_timeout_settles_a_call_with_no_task_left_to_cancel(self):
        """The handler can finish while the deadline is being processed.

        Its task is gone by the time the cancellation lands, but the deadline
        has already settled the call and rejected its result, so the pipeline
        still has to be told the call is over.
        """
        service, frames = self._service(function_call_timeout_secs=self.TIMEOUT)
        cancelled = []

        @service.event_handler("on_function_calls_cancelled")
        async def on_cancelled(service, function_calls):
            cancelled.extend(f.tool_call_id for f in function_calls)

        async def quick(params: FunctionCallParams):
            pass

        service.register_function("quick", quick)
        runner_item = FunctionCallRunnerItem(
            registry_item=service._functions["quick"],
            function_name="quick",
            tool_call_id="call_1",
            arguments={},
            context=LLMContext(),
        )

        # No task is tracked for this call, exactly as when the handler beat
        # the timeout to the finish.
        self.assertEqual(service._function_call_tasks, {})
        await service._timeout_function_call(runner_item)
        # Event handlers run in their own task.
        await asyncio.sleep(self.TIMEOUT)

        self.assertEqual([type(frame) for frame in frames], [FunctionCallCancelFrame])
        self.assertTrue(frames[-1].run_llm)
        self.assertEqual(cancelled, ["call_1"])

    async def test_cancelling_the_sequential_runner_cancels_its_call(self):
        """A runner going away takes the call it was running with it."""
        service, frames = self._service(run_in_parallel=False)
        await service._create_sequential_runner_task()

        rolled_back = []

        async def slow(params: FunctionCallParams):
            try:
                await asyncio.sleep(self.HANDLER_DURATION)
                await params.result_callback({"ok": True})
            except asyncio.CancelledError:
                rolled_back.append(params.tool_call_id)
                raise

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.TIMEOUT)

        await service._cancel_sequential_runner_task()
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(rolled_back, ["call_1"])
        self.assertFalse(any(isinstance(frame, FunctionCallResultFrame) for frame in frames))

    async def test_per_tool_timeout_overrides_the_global_one(self):
        service, frames = self._service(function_call_timeout_secs=10.0)

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)
            await params.result_callback({"ok": True})

        service.register_function("slow", slow, timeout_secs=self.TIMEOUT)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.SETTLE)

        self.assertIsInstance(frames[-1], FunctionCallCancelFrame)

    async def test_no_timeout_configured_leaves_the_call_running(self):
        service, frames = self._service()

        async def slow(params: FunctionCallParams):
            await asyncio.sleep(self.HANDLER_DURATION)
            await params.result_callback({"ok": True})

        service.register_function("slow", slow)
        await self._run_call(service, "slow")
        await asyncio.sleep(self.SETTLE)

        self.assertIsInstance(frames[-1], FunctionCallResultFrame)
        self.assertEqual(frames[-1].result, {"ok": True})


class TestFunctionCallError(unittest.IsolatedAsyncioTestCase):
    """A handler that raises settles its call instead of holding it open."""

    TIMEOUT = 0.1
    SETTLE = 0.6

    def _service(self, **kwargs) -> tuple[MockLLMService, list, list]:
        service = MockLLMService(**kwargs)
        service._task_manager = TaskManager()

        broadcast_frames = []
        errors = []

        async def mock_broadcast_frame(frame_cls, **frame_kwargs):
            broadcast_frames.append(frame_cls(**frame_kwargs))

        async def mock_push_error(**kwargs):
            errors.append(kwargs)

        service.broadcast_frame = mock_broadcast_frame
        service.push_error = mock_push_error
        return service, broadcast_frames, errors

    async def _run_call(self, service: MockLLMService, function_name: str, tool_call_id="call_1"):
        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

    async def test_a_raising_handler_settles_its_call(self):
        service, frames, errors = self._service()

        async def boom(params: FunctionCallParams):
            raise RuntimeError("kaboom")

        service.register_function("boom", boom)
        await self._run_call(service, "boom")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(
            [type(frame) for frame in frames],
            [FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame],
        )
        self.assertEqual(frames[-1].result, "The function `boom` failed and returned no result.")

    async def test_a_raising_handler_still_reports_the_error_upstream(self):
        service, _, errors = self._service()

        async def boom(params: FunctionCallParams):
            raise RuntimeError("kaboom")

        service.register_function("boom", boom)
        await self._run_call(service, "boom")
        await asyncio.sleep(self.SETTLE)

        self.assertEqual(len(errors), 1)
        self.assertIn("kaboom", errors[0]["error_msg"])
        self.assertIsInstance(errors[0]["exception"], RuntimeError)

    async def test_the_exception_is_kept_out_of_the_llm_context(self):
        """Exception text reaches the user through the LLM; the ErrorFrame carries it instead."""
        service, frames, _ = self._service()

        async def boom(params: FunctionCallParams):
            raise RuntimeError("connection refused: token=sk-secret")

        service.register_function("boom", boom)
        await self._run_call(service, "boom")
        await asyncio.sleep(self.SETTLE)

        self.assertNotIn("sk-secret", frames[-1].result)
        self.assertNotIn("connection refused", frames[-1].result)

    async def test_a_handler_that_reports_then_raises_keeps_its_result(self):
        service, frames, _ = self._service()

        async def report_then_boom(params: FunctionCallParams):
            await params.result_callback({"ok": True})
            raise RuntimeError("kaboom")

        service.register_function("report_then_boom", report_then_boom)
        await self._run_call(service, "report_then_boom")
        await asyncio.sleep(self.SETTLE)

        results = [f for f in frames if isinstance(f, FunctionCallResultFrame)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, {"ok": True})

    async def test_a_raising_handler_does_not_also_time_out(self):
        """The deadline is cancelled on the way out, so the call settles once."""
        service, frames, _ = self._service(function_call_timeout_secs=self.TIMEOUT)

        async def boom(params: FunctionCallParams):
            raise RuntimeError("kaboom")

        service.register_function("boom", boom)
        await self._run_call(service, "boom")
        await asyncio.sleep(self.SETTLE)

        self.assertFalse(any(isinstance(f, FunctionCallCancelFrame) for f in frames))

    async def test_the_sequential_runner_survives_a_raising_handler(self):
        service, frames, _ = self._service(run_in_parallel=False)
        await service._create_sequential_runner_task()

        async def boom(params: FunctionCallParams):
            raise RuntimeError("kaboom")

        async def fine(params: FunctionCallParams):
            await params.result_callback({"ok": True})

        service.register_function("boom", boom)
        service.register_function("fine", fine)
        await self._run_call(service, "boom")
        await self._run_call(service, "fine", tool_call_id="call_2")
        await asyncio.sleep(self.SETTLE)

        results = [f for f in frames if isinstance(f, FunctionCallResultFrame)]
        self.assertEqual([f.tool_call_id for f in results], ["call_1", "call_2"])
        self.assertEqual(results[-1].result, {"ok": True})

        await service._cancel_sequential_runner_task()


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

    async def test_appended_guide_survives_async_tool_cancellation(self):
        async def handler(params):
            pass

        service = self._service("APP")
        service.append_system_instruction("GUIDE")

        # A cancellable tool composes its instructions after the appended guide,
        # without duplicating it.
        service.register_function(
            "write_report", handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service._sync_registered_tool_handlers(NOT_GIVEN)
        composed = service._settings.system_instruction
        self.assertTrue(composed.startswith("APP\n\nGUIDE\n\n"))
        self.assertEqual(composed.count("GUIDE"), 1)
        self.assertNotEqual(composed, "APP\n\nGUIDE")


class TestCancellationToolBehaviour(unittest.IsolatedAsyncioTestCase):
    """What the built-in tools report and refuse."""

    @staticmethod
    async def _handler(params):
        pass

    def _service_with_running_calls(self) -> MockLLMService:
        """A service holding two in-flight calls, one cancellable and one not."""
        service = MockLLMService()
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service.register_function(
            "get_current_weather", self._handler, cancel_on_interruption=False
        )
        service._sync_registered_tool_handlers(NOT_GIVEN)
        # A None task stands for a call with no live asyncio task behind it, which
        # is what lets these run without a task manager.
        for task, (name, call_id) in zip(
            (None, object()),
            (("write_report", "call-1"), ("get_current_weather", "call-2")),
        ):
            service._function_call_tasks[task] = FunctionCallRunnerItem(  # type: ignore[index]
                registry_item=service._functions[name],
                function_name=name,
                tool_call_id=call_id,
                arguments={},
                context=None,
            )
        return service

    async def _invoke(self, service, name: str, arguments: dict):
        results = []

        async def result_callback(result, *, properties=None):
            results.append(result)

        await service._functions[name].handler(
            FunctionCallParams(
                function_name=name,
                tool_call_id="builtin-call",
                arguments=arguments,
                llm=service,
                pipeline_worker=service.pipeline_worker,
                context=None,
                result_callback=result_callback,
                app_resources=None,
            )
        )
        return results[0]

    async def test_cancelling_falls_back_to_the_only_running_call(self):
        # Omitting tool_call_id with one call running means that call, which is
        # why the schema doesn't ask for it.
        service = self._service_with_running_calls()
        result = await self._invoke(service, cancel_tool_name("write_report"), {})
        self.assertEqual(result["cancelled"], "call-1")

    async def test_cancelling_by_tool_call_id(self):
        service = self._service_with_running_calls()
        result = await self._invoke(
            service, cancel_tool_name("write_report"), {"tool_call_id": "call-1"}
        )
        self.assertEqual(result["cancelled"], "call-1")

    async def test_refused_when_no_such_call_is_running(self):
        service = self._service_with_running_calls()
        service._function_call_tasks.clear()
        result = await self._invoke(service, cancel_tool_name("write_report"), {})
        self.assertIsNone(result["cancelled"])

    async def test_ambiguous_call_asks_for_a_tool_call_id(self):
        service = self._service_with_running_calls()
        # A second call of the same tool: the tool name no longer picks one out.
        service._function_call_tasks[object()] = FunctionCallRunnerItem(  # type: ignore[index]
            registry_item=service._functions["write_report"],
            function_name="write_report",
            tool_call_id="call-3",
            arguments={},
            context=None,
        )
        result = await self._invoke(service, cancel_tool_name("write_report"), {})
        self.assertIsNone(result["cancelled"])
        self.assertEqual({r["tool_call_id"] for r in result["running"]}, {"call-1", "call-3"})

    async def test_a_call_covered_only_by_the_deprecated_flag_is_cancelled(self):
        # The flag advertises a cancel tool for every async tool, so the same
        # calls it advertises against have to be the ones it can stop.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            service = MockLLMService(enable_async_tool_cancellation=True)
        service.register_function("write_report", self._handler, cancel_on_interruption=False)
        service._sync_registered_tool_handlers(NOT_GIVEN)
        service._function_call_tasks[None] = FunctionCallRunnerItem(  # type: ignore[index]
            registry_item=service._functions["write_report"],
            function_name="write_report",
            tool_call_id="call-1",
            arguments={},
            context=None,
        )

        result = await self._invoke(service, cancel_tool_name("write_report"), {})

        self.assertEqual(result["cancelled"], "call-1")


class TestAsyncToolCancellationExposure(unittest.IsolatedAsyncioTestCase):
    """A cancellable tool brings its own cancel tool, named for it."""

    @staticmethod
    async def _handler(params):
        pass

    def _sync(self, service):
        service._sync_registered_tool_handlers(NOT_GIVEN)

    def _advertised(self, service) -> set:
        return set(service.get_llm_adapter().builtin_tools)

    def test_a_cancellable_tool_gets_its_own_cancel_tool(self):
        service = MockLLMService()
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        self._sync(service)
        self.assertEqual(self._advertised(service), {cancel_tool_name("write_report")})
        self.assertIn(cancel_tool_name("write_report"), service._functions)

    def test_the_cancel_tool_does_not_ask_for_a_tool_call_id(self):
        # Requiring it would make every cancellation go looking for an id first,
        # including the common case of one call running.
        service = MockLLMService()
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        self._sync(service)
        schema = service._adapter.builtin_tools[cancel_tool_name("write_report")]
        self.assertEqual(schema.required, [])
        self.assertIn("tool_call_id", schema.properties)

    def test_a_tool_that_did_not_opt_in_gets_none(self):
        service = MockLLMService()
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service.register_function(
            "get_current_weather", self._handler, cancel_on_interruption=False
        )
        self._sync(service)
        # The weather tool has no cancel tool, so there is nothing to call against it.
        self.assertNotIn(cancel_tool_name("get_current_weather"), self._advertised(service))

    def test_nothing_advertised_without_a_cancellable_tool(self):
        service = MockLLMService()
        service.register_function(
            "get_current_weather", self._handler, cancel_on_interruption=False
        )
        self._sync(service)
        self.assertEqual(self._advertised(service), set())

    def test_withdrawn_when_the_cancellable_tool_goes(self):
        service = MockLLMService()
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        self._sync(service)
        service.unregister_function("write_report")
        self._sync(service)
        self.assertEqual(self._advertised(service), set())

    def test_cancellable_by_llm_ignored_on_a_synchronous_tool(self):
        # There is no moment at which the LLM could cancel a call it is waiting on.
        service = MockLLMService()
        service.register_function("lookup", self._handler, cancellable_by_llm=True)
        self._sync(service)
        self.assertFalse(service._functions["lookup"].cancellable_by_llm)
        self.assertEqual(self._advertised(service), set())

    def test_deprecated_flag_covers_every_async_tool(self):
        with self.assertWarns(DeprecationWarning):
            service = MockLLMService(enable_async_tool_cancellation=True)
        service.register_function(
            "get_current_weather", self._handler, cancel_on_interruption=False
        )
        self._sync(service)
        self.assertIn(cancel_tool_name("get_current_weather"), self._advertised(service))

    def test_deprecated_flag_needs_an_async_tool(self):
        with self.assertWarns(DeprecationWarning):
            service = MockLLMService(enable_async_tool_cancellation=True)
        self._sync(service)
        self.assertEqual(self._advertised(service), set())


class TestAsyncToolInstructions(unittest.IsolatedAsyncioTestCase):
    """The async-tool guidance follows the registry, however a tool got there."""

    @staticmethod
    async def _handler(params):
        pass

    def _composed(self, service) -> str:
        return service._settings.system_instruction or ""

    def test_manual_async_registration(self):
        service = MockLLMService(system_instruction="BASE")
        service.register_function("weather", self._handler, cancel_on_interruption=False)
        # A manual registration never goes through the advertised-tool path, and a
        # context can advertise nothing at all; the sync still settles the registry.
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertIn("ASYNC TOOLS:", self._composed(service))

    def test_advertised_async_tool(self):
        service = MockLLMService(system_instruction="BASE")
        service.register_function("weather", self._handler, cancel_on_interruption=False)
        service._sync_registered_tool_handlers(LLMContext(tools=NOT_GIVEN).tools)
        self.assertIn("ASYNC TOOLS:", self._composed(service))

    def test_absent_without_async_tools(self):
        service = MockLLMService(system_instruction="BASE")
        service.register_function("weather", self._handler)
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertNotIn("ASYNC TOOLS:", self._composed(service))
        self.assertEqual(self._composed(service), "BASE")

    def test_absent_for_a_cancel_tool_alone(self):
        # A cancel tool registers as synchronous, so it never brings the async
        # guidance in on its own.
        service = MockLLMService(system_instruction="BASE")
        service.register_function(
            "write_report", self._handler, cancel_on_interruption=False, cancellable_by_llm=True
        )
        service._sync_registered_tool_handlers(NOT_GIVEN)
        service.unregister_function("write_report")
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertNotIn("ASYNC TOOLS:", self._composed(service))


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


class TestToolHandlerErrorClassification(unittest.IsolatedAsyncioTestCase):
    """A tool handler's own failures say nothing about the LLM service."""

    async def _run_failing_tool(self, exception: Exception) -> list:
        service = MockLLMService()
        service._call_event_handler = AsyncMock()
        service.broadcast_frame = AsyncMock()

        async def failing_handler(params):
            raise exception

        service.register_function("call_some_api", failing_handler)

        errors = []

        async def capture_error(error):
            errors.append(error)

        service.push_error_frame = capture_error

        async def run_inline(runner_items):
            for runner_item in runner_items:
                await service._run_function_call(runner_item)

        service._run_parallel_function_calls = run_inline
        service._run_sequential_function_calls = run_inline

        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name="call_some_api",
                    tool_call_id="call_1",
                    arguments={},
                    context=LLMContext(),
                )
            ]
        )

        return service, errors

    async def test_http_error_from_a_tool_does_not_misconfigure_the_service(self):
        # A tool calls an unrelated API that answers 404. The LLM's own
        # credentials are fine, so it must stay usable.
        request = httpx.Request("GET", "https://weather.example.com/forecast")
        tool_error = httpx.HTTPStatusError(
            "Not Found", request=request, response=httpx.Response(404, request=request)
        )

        service, errors = await self._run_failing_tool(tool_error)

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].category, ErrorCategory.APPLICATION)
        self.assertTrue(service.is_usable)

    async def test_rejected_credentials_from_a_tool_leave_the_service_usable(self):
        # The same holds for a 401 from whatever the tool called.
        request = httpx.Request("GET", "https://weather.example.com/forecast")
        tool_error = httpx.HTTPStatusError(
            "Unauthorized", request=request, response=httpx.Response(401, request=request)
        )

        service, errors = await self._run_failing_tool(tool_error)

        self.assertEqual(errors[0].category, ErrorCategory.APPLICATION)
        self.assertTrue(service.is_usable)

    async def test_plain_tool_failures_are_still_reported(self):
        service, errors = await self._run_failing_tool(ValueError("bad arguments"))

        self.assertEqual(len(errors), 1)
        self.assertIn("call_some_api", errors[0].error)
        self.assertEqual(errors[0].category, ErrorCategory.APPLICATION)
