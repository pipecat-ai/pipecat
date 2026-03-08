#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import unittest.mock
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_turn_completion_mixin import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
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

        # Should have 2 text frames: marker (skip_tts) and content (normal)
        self.assertEqual(len(pushed_frames), 2)

        # First frame should be the marker with skip_tts=True
        self.assertIsInstance(pushed_frames[0], LLMTextFrame)
        self.assertEqual(pushed_frames[0].text, USER_TURN_COMPLETE_MARKER)
        self.assertTrue(pushed_frames[0].skip_tts)

        # Second frame should be the actual text without skip_tts
        self.assertIsInstance(pushed_frames[1], LLMTextFrame)
        self.assertEqual(pushed_frames[1].text, "Hello there!")
        self.assertFalse(pushed_frames[1].skip_tts)

    async def test_incomplete_short_marker_suppresses_text(self):
        """Test that ○ marker suppresses text with skip_tts."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )
        # Mock timeout to avoid needing task manager
        processor._start_incomplete_timeout = AsyncMock()

        await processor._push_turn_text(USER_TURN_INCOMPLETE_SHORT_MARKER)

        # Should have 1 text frame with skip_tts=True
        self.assertEqual(len(pushed_frames), 1)
        self.assertIsInstance(pushed_frames[0], LLMTextFrame)
        self.assertEqual(pushed_frames[0].text, USER_TURN_INCOMPLETE_SHORT_MARKER)
        self.assertTrue(pushed_frames[0].skip_tts)

    async def test_incomplete_long_marker_suppresses_text(self):
        """Test that ◐ marker suppresses text with skip_tts."""
        processor = MockProcessor()

        pushed_frames = []
        processor.push_frame = AsyncMock(
            side_effect=lambda f, *args, **kwargs: pushed_frames.append(f)
        )
        # Mock timeout to avoid needing task manager
        processor._start_incomplete_timeout = AsyncMock()

        await processor._push_turn_text(USER_TURN_INCOMPLETE_LONG_MARKER)

        # Should have 1 text frame with skip_tts=True
        self.assertEqual(len(pushed_frames), 1)
        self.assertIsInstance(pushed_frames[0], LLMTextFrame)
        self.assertEqual(pushed_frames[0].text, USER_TURN_INCOMPLETE_LONG_MARKER)
        self.assertTrue(pushed_frames[0].skip_tts)

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

        # Now frames should be pushed
        self.assertEqual(len(pushed_frames), 2)

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


if __name__ == "__main__":
    unittest.main()
