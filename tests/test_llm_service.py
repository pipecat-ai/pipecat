#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, patch

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy


def _expected_missing_tool_message(name: str) -> str:
    return LLMService.MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE.format(function_name=name)


class MockLLMService(LLMService):
    """Minimal LLM service for testing function call execution."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)


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
