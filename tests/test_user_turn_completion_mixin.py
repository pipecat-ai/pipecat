#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import unittest.mock
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
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
    TurnMarker,
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

    async def test_new_response_cancels_pending_incomplete_timeout(self):
        """A new LLM response starting must cancel a pending incomplete timeout.

        This closes the race where the timeout fires at the same time a new
        (completed) inference arrives: whichever response starts first cancels
        the timeout before its text is parsed, so only one inference runs.
        """
        processor = MockProcessor()

        # Arm an incomplete timeout via an ○ marker.
        processor.push_frame = AsyncMock()
        processor._start_incomplete_timeout = AsyncMock()
        await processor._push_turn_text(USER_TURN_INCOMPLETE_SHORT_MARKER)
        self.assertEqual(processor._turn_marker, TurnMarker.INCOMPLETE)

        # Simulate a live pending timeout task.
        processor._incomplete_timeout_task = object()
        processor._cancel_incomplete_timeout = AsyncMock()

        # A new response begins (either the user's completed turn or the
        # timeout's own re-prompt): the pending timeout must be cancelled.
        del processor.push_frame  # restore the mixin override
        with unittest.mock.patch.object(FrameProcessor, "push_frame", AsyncMock()):
            await processor.push_frame(LLMFullResponseStartFrame())

        processor._cancel_incomplete_timeout.assert_awaited_once()

    async def test_vad_resume_cancels_pending_incomplete_timeout(self):
        """The user resuming speech mid-turn cancels the pending re-prompt timeout.

        A resume inside an already-open turn produces a VADUserStartedSpeakingFrame
        but no InterruptionFrame, so the incomplete (○/◐) re-prompt timeout would
        otherwise expire and talk over the user.
        """
        processor = MockProcessor()
        processor._incomplete_timeout_task = object()
        processor._cancel_incomplete_timeout = AsyncMock()

        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        processor._cancel_incomplete_timeout.assert_awaited_once()

    async def test_only_first_completion_voiced_per_user_turn(self):
        """A second ✓ inference within the same user turn is not voiced again.

        The acoustic detector can trigger several inferences per user turn, each
        producing its own ✓; only the first should be spoken. This holds as long
        as the user hasn't resumed speaking in between — a VADUserStartedSpeakingFrame
        resets the latch instead (see test_resumed_speech_does_not_permanently_silence_the_turn).
        """
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        # First inference: ✓ is voiced.
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} How are you?")
        # End of that response resets per-response state but not the per-turn latch.
        await processor._turn_reset()

        text_before = [f.text for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(text_before, ["How are you?"])

        # Second inference in the same user turn: identical ✓ must be dropped.
        pushed_frames.clear()
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} How are you?")
        self.assertEqual([f for f in pushed_frames if isinstance(f, LLMTextFrame)], [])
        completed = [f for f in pushed_frames if isinstance(f, UserTurnInferenceCompletedFrame)]
        self.assertEqual(completed, [])

    async def test_voiced_response_keeps_streaming_after_latch(self):
        """The response that set the latch keeps streaming its own continuation."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Hello")
        await processor._push_turn_text(" there!")

        text = [f.text for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(text, ["Hello", " there!"])

    async def test_new_user_turn_resets_completion_latch(self):
        """UserStartedSpeakingFrame lets the next user turn voice a completion again."""
        processor = MockProcessor()
        processor._user_turn_completion_voiced = True

        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        self.assertFalse(processor._user_turn_completion_voiced)

    async def test_vad_resume_resets_completion_latch(self):
        """VADUserStartedSpeakingFrame lets the same open turn voice a completion again.

        A resume inside an already-open turn produces no UserStartedSpeakingFrame
        (the controller only fires that for a genuinely new turn), so without this
        reset the latch would stay tripped for the rest of the turn. Since the
        controller drops any in-flight completion as stale once the user resumes
        (see UserTurnController._trigger_user_turn_stop), voicing it would only
        repeat/talk over the user anyway — so the next ✓, for the turn the user is
        now continuing, should get to speak instead.
        """
        processor = MockProcessor()
        processor._user_turn_completion_voiced = True

        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        self.assertFalse(processor._user_turn_completion_voiced)

    async def test_resumed_speech_does_not_permanently_silence_the_turn(self):
        """Regression test for the confirmed double-inference-fix interaction bug.

        Sequence: a first ✓ is voiced mid-turn (latch set); the user resumes
        speaking within the same still-open turn (no UserStartedSpeakingFrame
        fires, since the controller only emits that for a brand new turn); a
        second, legitimate ✓ then arrives once the user pauses again. Without
        resetting the latch on the resume, the second response would be
        silently dropped and the bot would never reply.
        """
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )

        # First (premature) inference: LLM says ✓, voiced, latch set.
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} First answer")
        await processor._turn_reset()
        self.assertTrue(processor._user_turn_completion_voiced)

        # The user resumes speaking within the same still-open turn (no new
        # UserStartedSpeakingFrame, since the turn never closed).
        with unittest.mock.patch.object(FrameProcessor, "process_frame", AsyncMock()):
            await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        # A second, legitimate inference completes once the user pauses again.
        await processor._push_turn_text(f"{USER_TURN_COMPLETE_MARKER} Second answer")
        await processor._turn_reset()

        texts = [f.text for f in pushed_frames if isinstance(f, LLMTextFrame)]
        self.assertEqual(texts, ["First answer", "Second answer"])


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
