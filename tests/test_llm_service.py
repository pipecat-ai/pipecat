#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMSetToolsFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.frame_processor import FrameDirection
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
            system_instruction=kwargs.pop("system_instruction", None),
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
