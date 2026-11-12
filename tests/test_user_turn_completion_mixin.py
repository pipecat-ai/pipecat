#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import unittest.mock
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    UserTurnCompletionFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_COMPLETION_INSTRUCTIONS,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    TurnMarker,
    UserTurnCompletionConfig,
    UserTurnCompletionLLMServiceMixin,
)


class MockProcessor(UserTurnCompletionLLMServiceMixin, FrameProcessor):
    """Simple mock processor using the turn completion mixin, as LLMService does."""

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await self._track_pushed_frame(frame)
        await super().push_frame(frame, direction)


class TestUserUserTurnCompletionLLMServiceMixin(unittest.IsolatedAsyncioTestCase):
    """Tests for UserUserTurnCompletionLLMServiceMixin functionality."""

    async def test_complete_marker_pushes_text(self):
        """Test that ● marker is detected and text after it is pushed normally."""
        processor = MockProcessor()

        # Capture frames that get pushed
        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        # Simulate LLM generating: "● Hello there!"
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello there!")

        # The marker rides as LLMMarkerFrame(append_to_context_immediately=False);
        # only the spoken text is pushed as an LLMTextFrame.
        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(len(text_frames), 1)
        self.assertEqual(text_frames[0].text, "Hello there!")
        self.assertFalse(text_frames[0].skip_tts)

        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_COMPLETE_MARKER)
        self.assertFalse(marker_frames[0].append_to_context_immediately)

        # UserTurnInferenceCompletedFrame broadcast in both directions.
        completed = [f for f in pushed_frames if isinstance(f, UserTurnInferenceCompletedFrame)]
        self.assertEqual(len(completed), 2)

    async def test_incomplete_short_marker_suppresses_text(self):
        """Test that ◐ marker suppresses text and is emitted as a stand-alone marker frame."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        await processor._push_turn_text(USER_TURN_INCOMPLETE_SHORT_MARKER)

        # No LLMTextFrame: response is suppressed.
        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(len(text_frames), 0)

        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_INCOMPLETE_SHORT_MARKER)
        self.assertTrue(marker_frames[0].append_to_context_immediately)

        # Incomplete markers do not emit UserTurnInferenceCompletedFrame.
        completed = [f for f in pushed_frames if isinstance(f, UserTurnInferenceCompletedFrame)]
        self.assertEqual(len(completed), 0)

    async def test_incomplete_long_marker_suppresses_text(self):
        """Test that ○ marker suppresses text and is emitted as a stand-alone marker frame."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        await processor._push_turn_text(USER_TURN_INCOMPLETE_LONG_MARKER)

        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(len(text_frames), 0)

        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_INCOMPLETE_LONG_MARKER)
        self.assertTrue(marker_frames[0].append_to_context_immediately)

        completed = [f for f in pushed_frames if isinstance(f, UserTurnInferenceCompletedFrame)]
        self.assertEqual(len(completed), 0)

    async def test_text_buffered_until_marker_found(self):
        """Test that text is buffered until a marker is detected."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        # Simulate token-by-token streaming without marker
        await processor._push_turn_text("Hello")
        await processor._push_turn_text(" there")

        # No frames should be pushed yet (buffering)
        self.assertEqual(len(pushed_frames), 0)

        # Now send the complete marker
        await processor._push_turn_text(f" {USER_TURN_COMPLETE_MARKER} How are you?")

        # One LLMTextFrame for the spoken portion; one LLMMarkerFrame for
        # the marker.
        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(len(text_frames), 1)
        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual(len(marker_frames), 1)

    async def test_turn_state_reset_after_llm_full_response_end_frame(self):
        """Test that the turn marker is reset when LLMFullResponseEndFrame is pushed."""
        processor = MockProcessor()

        # Mock push_frame on the instance so _push_turn_text can call it without
        # a live pipeline, but keep _turn_reset as the real implementation.
        processor.push_frame = AsyncMock()

        # Simulate first LLM response: complete marker sets the marker to COMPLETE
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello!")
        self.assertEqual(processor._turn_marker, TurnMarker.COMPLETE)

        # Restore the real push_frame so the mixin override runs, then call it
        # with LLMFullResponseEndFrame as the LLM service would.
        del processor.push_frame  # removes instance mock, restores class method

        # Patch only the FrameProcessor-level send so no live pipeline is needed.
        with unittest.mock.patch.object(FrameProcessor, "push_frame", AsyncMock()):
            end_frame = LLMFullResponseEndFrame()
            await processor.push_frame(end_frame)

        # The marker must now be cleared — ready for the next response
        self.assertIsNone(processor._turn_marker)
        self.assertEqual(processor._turn_text_buffer, "")

    async def test_response_end_reports_the_raw_text_and_marker(self):
        """When the response ends, the mixin reports what the LLM wrote and the marker it read."""
        processor = MockProcessor()
        pushed_frames = []
        with unittest.mock.patch.object(
            FrameProcessor,
            "push_frame",
            AsyncMock(side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)),
        ):
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor._push_turn_text("Well, ")
            await processor._push_turn_text(f"{USER_TURN_INCOMPLETE_SHORT_MARKER} go on")
            await processor._push_turn_text(" please")
            await processor.push_frame(LLMFullResponseEndFrame())

        reports = [f for f in pushed_frames if isinstance(f, LLMMarkerResponseFrame)]
        self.assertEqual(len(reports), 1)
        report = reports[0]
        # Everything the LLM produced, including what the mixin held back.
        self.assertEqual(report.raw, f"Well, {USER_TURN_INCOMPLETE_SHORT_MARKER} go on please")
        self.assertEqual((report.marker, report.kind), (USER_TURN_INCOMPLETE_SHORT_MARKER, "short"))
        self.assertEqual(
            report.markers,
            [
                USER_TURN_COMPLETE_MARKER,
                USER_TURN_INCOMPLETE_SHORT_MARKER,
                USER_TURN_INCOMPLETE_LONG_MARKER,
            ],
        )
        # The report precedes the end-of-response frame.
        self.assertLess(
            pushed_frames.index(report),
            next(i for i, f in enumerate(pushed_frames) if isinstance(f, LLMFullResponseEndFrame)),
        )

    async def test_a_response_without_a_marker_is_reported_as_such(self):
        processor = MockProcessor()
        pushed_frames = []
        with unittest.mock.patch.object(
            FrameProcessor,
            "push_frame",
            AsyncMock(side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)),
        ):
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor._push_turn_text("Hello there")
            await processor.push_frame(LLMFullResponseEndFrame())
        (report,) = [f for f in pushed_frames if isinstance(f, LLMMarkerResponseFrame)]
        self.assertEqual((report.raw, report.marker, report.kind), ("Hello there", None, None))

    async def test_a_response_the_mixin_never_saw_is_not_reported(self):
        processor = MockProcessor()
        pushed_frames = []
        with unittest.mock.patch.object(
            FrameProcessor,
            "push_frame",
            AsyncMock(side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)),
        ):
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor.push_frame(LLMFullResponseEndFrame())
        self.assertEqual([f for f in pushed_frames if isinstance(f, LLMMarkerResponseFrame)], [])

    async def test_text_after_a_complete_marker_streams_straight_through(self):
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello")
        await processor._push_turn_text(" there!")

        text = [f.text for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(text, ["Hello", " there!"])


class MockLLMService(LLMService):
    """Minimal LLM service for testing system_instruction composition."""

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


class TestSystemInstructionComposition(unittest.IsolatedAsyncioTestCase):
    """Tests for turn completion system_instruction composition in LLMService."""

    async def test_enable_turn_completion_sets_system_instruction(self):
        """Enabling turn completion should set system_instruction to completion instructions."""
        service = MockLLMService()
        self.assertIsNone(service._settings.system_instruction)

        delta = LLMSettings(filter_incomplete_user_turns=True)
        await service._update_settings(delta)

        self.assertEqual(service._settings.system_instruction, USER_TURN_COMPLETION_INSTRUCTIONS)
        self.assertIsNone(service._base_system_instruction)

    async def test_enable_turn_completion_appends_to_existing_system_instruction(self):
        """Enabling turn completion should append instructions to existing system_instruction."""
        service = MockLLMService(system_instruction="You are a helpful assistant.")

        delta = LLMSettings(filter_incomplete_user_turns=True)
        await service._update_settings(delta)

        expected = f"You are a helpful assistant.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)
        self.assertEqual(service._base_system_instruction, "You are a helpful assistant.")

    async def test_disable_turn_completion_restores_system_instruction(self):
        """Disabling turn completion should restore the original system_instruction."""
        service = MockLLMService(system_instruction="You are a helpful assistant.")

        # Enable
        await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
        self.assertIn(USER_TURN_COMPLETION_INSTRUCTIONS, service._settings.system_instruction)

        # Disable
        await service._update_settings(LLMSettings(filter_incomplete_user_turns=False))
        self.assertEqual(service._settings.system_instruction, "You are a helpful assistant.")
        # The base prompt is retained — it's the single source of truth that
        # composition rebuilds from; disabling just recomposes without the
        # turn-completion addon.
        self.assertEqual(service._base_system_instruction, "You are a helpful assistant.")

    async def test_disable_turn_completion_restores_none(self):
        """Disabling turn completion when original was None should restore None."""
        service = MockLLMService()

        await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
        self.assertEqual(service._settings.system_instruction, USER_TURN_COMPLETION_INSTRUCTIONS)

        await service._update_settings(LLMSettings(filter_incomplete_user_turns=False))
        self.assertIsNone(service._settings.system_instruction)

    async def test_update_system_instruction_while_turn_completion_active(self):
        """Changing system_instruction while turn completion is active should recompose."""
        service = MockLLMService(system_instruction="Original prompt.")

        await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
        expected = f"Original prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)

        # Now update system_instruction
        await service._update_settings(LLMSettings(system_instruction="New prompt."))
        expected = f"New prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)
        self.assertEqual(service._base_system_instruction, "New prompt.")

    async def test_update_config_recomposes_with_custom_instructions(self):
        """Updating turn completion config should recompose with new instructions."""
        service = MockLLMService(system_instruction="Base prompt.")

        await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))

        custom_config = UserTurnCompletionConfig(instructions="Custom turn instructions.")
        await service._update_settings(LLMSettings(user_turn_completion_config=custom_config))

        expected = "Base prompt.\n\nCustom turn instructions."
        self.assertEqual(service._settings.system_instruction, expected)

    async def test_simultaneous_enable_and_system_instruction_change(self):
        """Enabling turn completion and changing system_instruction in the same delta
        should use the new system_instruction as the base."""
        service = MockLLMService(system_instruction="Original prompt.")

        await service._update_settings(
            LLMSettings(
                filter_incomplete_user_turns=True,
                system_instruction="New prompt.",
            )
        )

        # apply_update sets system_instruction to "New prompt." before _update_settings
        # runs, so the base should be the new value the user explicitly set.
        self.assertEqual(service._base_system_instruction, "New prompt.")
        expected = f"New prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)


class TestConfigurableMarkers(unittest.IsolatedAsyncioTestCase):
    """Markers are configurable, and the prompts follow whatever is configured."""

    def test_default_markers_are_the_fill_gradient(self):
        """Full is a finished turn, half is cut off mid-thought, empty is not started."""
        self.assertEqual(USER_TURN_COMPLETE_MARKER, "●")
        self.assertEqual(USER_TURN_INCOMPLETE_SHORT_MARKER, "◐")
        self.assertEqual(USER_TURN_INCOMPLETE_LONG_MARKER, "○")

    def test_prompts_follow_configured_markers(self):
        """Custom markers reach the instructions and both re-prompts."""
        config = UserTurnCompletionConfig(
            complete_marker="Y", incomplete_short_marker="N", incomplete_long_marker="W"
        )

        for prompt in (config.completion_instructions, config.short_prompt, config.long_prompt):
            for default in (
                USER_TURN_COMPLETE_MARKER,
                USER_TURN_INCOMPLETE_SHORT_MARKER,
                USER_TURN_INCOMPLETE_LONG_MARKER,
            ):
                self.assertNotIn(default, prompt)

        self.assertIn("Write Y, a space, then your full reply", config.completion_instructions)
        self.assertIn("Respond with only N", config.completion_instructions)
        self.assertIn("Respond with only W", config.completion_instructions)
        self.assertIn("respond with Y", config.short_prompt)
        self.assertIn("respond with Y", config.long_prompt)

    def test_custom_instructions_still_win(self):
        """An explicit instructions string overrides the rendered default."""
        config = UserTurnCompletionConfig(instructions="do it my way")
        self.assertEqual(config.completion_instructions, "do it my way")

    async def test_parser_detects_configured_markers(self):
        """The detector matches the configured markers, not the defaults."""
        processor = MockProcessor()
        processor.set_user_turn_completion_config(
            UserTurnCompletionConfig(
                complete_marker="Y", incomplete_short_marker="N", incomplete_long_marker="W"
            )
        )

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        await processor._push_turn_text("Y Hello there!")

        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual([f.text for f in text_frames], ["Hello there!"])

        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual([f.marker for f in marker_frames], ["Y"])


class TestUserTurnCompletionReports(unittest.IsolatedAsyncioTestCase):
    """The turn-completion judgment pushed upstream as UserTurnCompletionFrame."""

    def _processor(self, *, filtering: bool = True) -> MockProcessor:
        processor = MockProcessor()
        processor._filter_incomplete_user_turns = filtering
        return processor

    def _recorder(self, pushed: list) -> AsyncMock:
        return AsyncMock(
            side_effect=lambda f, direction=FrameDirection.DOWNSTREAM: pushed.append((f, direction))
        )

    def _reports(self, pushed: list) -> list[UserTurnCompletionFrame]:
        return [f for f, _ in pushed if isinstance(f, UserTurnCompletionFrame)]

    async def test_each_marker_is_reported_upstream(self):
        for marker, kind in [
            (USER_TURN_COMPLETE_MARKER, "complete"),
            (USER_TURN_INCOMPLETE_SHORT_MARKER, "short"),
            (USER_TURN_INCOMPLETE_LONG_MARKER, "long"),
        ]:
            with self.subTest(kind=kind):
                processor = self._processor()
                pushed = []
                processor.push_frame = self._recorder(pushed)

                await processor._push_turn_text(f"{marker} Hello")

                reports = [(f, d) for f, d in pushed if isinstance(f, UserTurnCompletionFrame)]
                self.assertEqual(len(reports), 1)
                report, direction = reports[0]
                self.assertEqual((report.kind, report.marker), (kind, marker))
                self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_reports_carry_the_context_frame_id(self):
        processor = self._processor()
        context_frame = LLMContextFrame(context=LLMContext())
        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(context_frame, FrameDirection.DOWNSTREAM)
        pushed = []
        processor.push_frame = self._recorder(pushed)

        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello")

        (report,) = self._reports(pushed)
        self.assertEqual(report.context_frame_id, context_frame.id)

    async def test_a_tool_call_is_reported_as_complete(self):
        processor = self._processor()
        pushed = []
        with unittest.mock.patch.object(FrameProcessor, "push_frame", self._recorder(pushed)):
            await processor.push_frame(FunctionCallsStartedFrame(function_calls=[]))

        (report,) = self._reports(pushed)
        self.assertEqual((report.kind, report.marker), ("complete", None))
        completed = [f for f, _ in pushed if isinstance(f, UserTurnInferenceCompletedFrame)]
        self.assertEqual(len(completed), 2)

    async def test_a_response_without_a_marker_is_reported_with_no_kind(self):
        processor = self._processor()
        pushed = []
        with unittest.mock.patch.object(FrameProcessor, "push_frame", self._recorder(pushed)):
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor._push_turn_text("Hello there")
            await processor.push_frame(LLMFullResponseEndFrame())

        (report,) = self._reports(pushed)
        self.assertEqual((report.kind, report.marker), (None, None))

    async def test_each_inference_is_reported_once(self):
        processor = self._processor()
        pushed = []
        with (
            unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()),
            unittest.mock.patch.object(FrameProcessor, "push_frame", self._recorder(pushed)),
        ):
            await processor.process_frame(
                LLMContextFrame(context=LLMContext()), FrameDirection.DOWNSTREAM
            )
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Let me check")
            await processor.push_frame(FunctionCallsStartedFrame(function_calls=[]))
            await processor.push_frame(LLMFullResponseEndFrame())

            # The next inference is reported afresh.
            await processor.process_frame(
                LLMContextFrame(context=LLMContext()), FrameDirection.DOWNSTREAM
            )
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor._push_turn_text(USER_TURN_INCOMPLETE_SHORT_MARKER)
            await processor.push_frame(LLMFullResponseEndFrame())

        self.assertEqual([r.kind for r in self._reports(pushed)], ["complete", "short"])

    async def test_nothing_is_reported_while_filtering_is_off(self):
        processor = self._processor(filtering=False)
        pushed = []
        with unittest.mock.patch.object(FrameProcessor, "push_frame", self._recorder(pushed)):
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor.push_frame(FunctionCallsStartedFrame(function_calls=[]))
            await processor.push_frame(LLMFullResponseEndFrame())

        self.assertEqual(self._reports(pushed), [])


if __name__ == "__main__":
    unittest.main()
