#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import unittest.mock
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMarkerFrame,
    LLMTextFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_COMPLETION_INSTRUCTIONS,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    UserTurnCompletionConfig,
    UserTurnCompletionLLMServiceMixin,
)


class MockProcessor(UserTurnCompletionLLMServiceMixin, FrameProcessor):
    """Simple mock processor using the turn completion mixin."""

    pass


class TestUserUserTurnCompletionLLMServiceMixin(unittest.IsolatedAsyncioTestCase):
    """Tests for UserUserTurnCompletionLLMServiceMixin functionality."""

    async def test_complete_marker_pushes_text(self):
        """Test that ✓ marker is detected and text after it is pushed normally."""
        processor = MockProcessor()

        # Capture frames that get pushed
        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        # Simulate LLM generating: "✓ Hello there!"
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
        """Test that ○ marker suppresses text and is emitted as a stand-alone marker frame."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )
        # Mock timeout to avoid needing task manager
        processor._start_incomplete_timeout = AsyncMock()

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
        """Test that ◐ marker suppresses text and is emitted as a stand-alone marker frame."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )
        # Mock timeout to avoid needing task manager
        processor._start_incomplete_timeout = AsyncMock()

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
        # the marker; UserTurnInferenceCompletedFrame broadcast in both directions.
        text_frames = [f for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(len(text_frames), 1)
        marker_frames = [f for f in pushed_frames if isinstance(f, LLMMarkerFrame)]
        self.assertEqual(len(marker_frames), 1)

    async def test_turn_state_reset_after_llm_full_response_end_frame(self):
        """Test that _turn_complete_found is reset when LLMFullResponseEndFrame is pushed."""
        processor = MockProcessor()

        # Mock push_frame on the instance so _push_turn_text can call it without
        # a live pipeline, but keep _turn_reset as the real implementation.
        processor.push_frame = AsyncMock()

        # Simulate first LLM response: complete marker sets _turn_complete_found = True
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello!")
        self.assertTrue(processor._turn_complete_found)

        # Restore the real push_frame so the mixin override runs, then call it
        # with LLMFullResponseEndFrame as the LLM service would.
        del processor.push_frame  # removes instance mock, restores class method

        # Patch only the FrameProcessor-level send so no live pipeline is needed.
        with unittest.mock.patch.object(FrameProcessor, "push_frame", AsyncMock()):
            end_frame = LLMFullResponseEndFrame()
            await processor.push_frame(end_frame)

        # _turn_complete_found must now be False — ready for the next response
        self.assertFalse(processor._turn_complete_found)
        self.assertEqual(processor._turn_text_buffer, "")
        self.assertFalse(processor._turn_suppressed)

    async def test_incomplete_timeout_cancelled_on_resumed_speech(self):
        """A VADUserStartedSpeakingFrame cancels a pending incomplete timeout.

        Reproduces the resumed-speech half of #4707. Under
        ``FilterIncompleteUserTurnStrategies`` the user can pause (the LLM
        emits ○/◐, arming a re-prompt timer) and then resume speaking within
        the SAME user turn. Stock behavior only cancels the timer on
        ``InterruptionFrame``, which fires on a new turn — not on resumption
        inside an already-open turn — so the timer runs to expiry and fires a
        stale forced-✓ nudge while the user is mid-utterance. The mixin must
        cancel the pending timeout when the user resumes speaking.
        """
        processor = MockProcessor()
        processor._cancel_incomplete_timeout = AsyncMock()

        # Patch the FrameProcessor-level handler so no live pipeline is needed.
        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        processor._cancel_incomplete_timeout.assert_awaited_once()

    async def test_resumed_speech_cancels_timeout_without_resetting_turn(self):
        """Resumed speech cancels the timer but keeps turn state, unlike an interruption.

        A ``VADUserStartedSpeakingFrame`` within an open turn means the user
        resumed mid-turn: cancel the pending ○/◐ re-prompt but leave the rest
        of the turn-completion state intact for the next inference. By
        contrast an ``InterruptionFrame`` (a new turn) both cancels the timer
        and resets turn state.
        """
        processor = MockProcessor()
        processor._cancel_incomplete_timeout = AsyncMock()
        processor._turn_reset = AsyncMock()

        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            # Resumed speech within the open turn: cancel the timer only.
            await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            self.assertEqual(processor._cancel_incomplete_timeout.await_count, 1)
            processor._turn_reset.assert_not_awaited()

            # A new turn / interruption: cancel the timer and reset turn state.
            await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            self.assertEqual(processor._cancel_incomplete_timeout.await_count, 2)
            processor._turn_reset.assert_awaited_once()


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


if __name__ == "__main__":
    unittest.main()
