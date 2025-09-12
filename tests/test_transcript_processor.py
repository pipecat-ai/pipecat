#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import unittest
from datetime import datetime, timezone
from typing import List, Tuple, cast

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    TTSTextFrame,
)
from pipecat.processors.transcript_processor import (
    AssistantTranscriptProcessor,
    UserTranscriptProcessor,
)
from pipecat.tests.utils import SleepFrame, run_test


class TestUserTranscriptProcessor(unittest.IsolatedAsyncioTestCase):
    """Tests for UserTranscriptProcessor"""

    async def test_basic_transcription(self):
        """Test basic transcription frame processing"""
        # Create processor
        processor = UserTranscriptProcessor()

        # Create test timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create frames to send
        frames_to_send = [
            TranscriptionFrame(text="Hello, world!", user_id="test_user", timestamp=timestamp)
        ]

        # Expected frames downstream - note the order:
        # 1. TranscriptionUpdateFrame (processor emits the update first)
        # 2. TranscriptionFrame (original frame is passed through)
        expected_down_frames = [TranscriptionUpdateFrame, TranscriptionFrame]

        # Run test
        received_frames, _ = await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify the content of the TranscriptionUpdateFrame
        update_frame = cast(
            TranscriptionUpdateFrame, received_frames[0]
        )  # Note: now checking first frame
        self.assertIsInstance(update_frame, TranscriptionUpdateFrame)
        self.assertEqual(len(update_frame.messages), 1)
        message = update_frame.messages[0]
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.user_id, "test_user")
        self.assertEqual(message.timestamp, timestamp)

    async def test_event_handler(self):
        """Test that event handlers are called with transcript updates"""
        # Create processor
        processor = UserTranscriptProcessor()

        # Track received updates
        received_updates: List[TranscriptionMessage] = []

        # Register event handler
        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.extend(frame.messages)

        # Create test data
        timestamp = datetime.now(timezone.utc).isoformat()
        frames_to_send = [
            TranscriptionFrame(text="First message", user_id="test_user", timestamp=timestamp),
            TranscriptionFrame(text="Second message", user_id="test_user", timestamp=timestamp),
        ]

        expected_down_frames = [
            TranscriptionUpdateFrame,
            TranscriptionFrame,  # First message
            TranscriptionUpdateFrame,
            TranscriptionFrame,  # Second message
        ]

        # Run test
        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify event handler received updates
        self.assertEqual(len(received_updates), 2)

        # Check first message
        self.assertEqual(received_updates[0].role, "user")
        self.assertEqual(received_updates[0].content, "First message")
        self.assertEqual(received_updates[0].timestamp, timestamp)

        # Check second message
        self.assertEqual(received_updates[1].role, "user")
        self.assertEqual(received_updates[1].content, "Second message")
        self.assertEqual(received_updates[1].timestamp, timestamp)

    async def test_text_aggregation(self):
        """Test that TTSTextFrames are properly aggregated into a single message"""
        # Create processor
        processor = AssistantTranscriptProcessor()

        # Track received updates
        received_updates: List[TranscriptionUpdateFrame] = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.append(frame)

        # Create test frames simulating bot speaking multiple text chunks
        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),  # Wait for StartedSpeaking to process
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="world!"),
            TTSTextFrame(text="How"),
            TTSTextFrame(text="are"),
            TTSTextFrame(text="you?"),
            SleepFrame(),  # Wait for text frames to queue
            BotStoppedSpeakingFrame(),
        ]

        # Expected order:
        # 1. BotStartedSpeakingFrame (system frame, immediate)
        # 2. All queued TTSTextFrames
        # 3. BotStoppedSpeakingFrame (system frame, immediate)
        # 4. TranscriptionUpdateFrame (after aggregation)
        expected_down_frames = [
            BotStartedSpeakingFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TranscriptionUpdateFrame,
            BotStoppedSpeakingFrame,
        ]

        # Run test
        received_frames, _ = await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify update was received
        self.assertEqual(len(received_updates), 1)

        # Get the update frame
        update_frame = received_updates[0]

        # Should have one aggregated message
        self.assertEqual(len(update_frame.messages), 1)

        message = update_frame.messages[0]
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Hello world! How are you?")

        # Verify timestamp exists
        self.assertIsNotNone(message.timestamp)

        # All frames should be passed through in order, with update at end
        downstream_update = cast(TranscriptionUpdateFrame, received_frames[-2])
        self.assertEqual(downstream_update.messages[0].content, "Hello world! How are you?")

    async def test_empty_text_handling(self):
        """Test that empty messages are not emitted"""
        processor = AssistantTranscriptProcessor()

        received_updates: List[TranscriptionUpdateFrame] = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.append(frame)

        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text=""),  # Empty text
            TTSTextFrame(text="   "),  # Just whitespace
            TTSTextFrame(text="\n"),  # Just newline
            BotStoppedSpeakingFrame(),
            # Pipeline ends here; run_test will automatically send EndFrame
        ]

        # From our earlier tests, we know BotStoppedSpeakingFrame comes before TTSTextFrames
        expected_down_frames = [
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            TTSTextFrame,  # empty
            TTSTextFrame,  # whitespace
            TTSTextFrame,  # newline
            # No TranscriptionUpdateFrame since content is empty after stripping
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        self.assertEqual(len(received_updates), 0, "No updates should be emitted for empty content")

    async def test_interruption_handling(self):
        """Test that messages are properly captured when bot is interrupted"""
        processor = AssistantTranscriptProcessor()

        # Track received updates
        received_updates: List[TranscriptionUpdateFrame] = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.append(frame)

        # Simulate bot being interrupted mid-sentence
        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="world!"),
            SleepFrame(),
            InterruptionFrame(),  # User interrupts here
            SleepFrame(),
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="New"),
            TTSTextFrame(text="response"),
            SleepFrame(),
            BotStoppedSpeakingFrame(),
        ]

        # Actual order of frames:
        expected_down_frames = [
            BotStartedSpeakingFrame,
            TTSTextFrame,  # "Hello"
            TTSTextFrame,  # "world!"
            InterruptionFrame,
            TranscriptionUpdateFrame,  # First message (emitted due to interruption)
            BotStartedSpeakingFrame,
            TTSTextFrame,  # "New"
            TTSTextFrame,  # "response"
            TranscriptionUpdateFrame,  # Second message
            BotStoppedSpeakingFrame,
        ]

        # Run test
        received_frames, _ = await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Should have received two updates
        self.assertEqual(len(received_updates), 2)

        # First update should be interrupted message
        first_message = received_updates[0].messages[0]
        self.assertEqual(first_message.role, "assistant")
        self.assertEqual(first_message.content, "Hello world!")
        self.assertIsNotNone(first_message.timestamp)

        # Second update should be new response
        second_message = received_updates[1].messages[0]
        self.assertEqual(second_message.role, "assistant")
        self.assertEqual(second_message.content, "New response")
        self.assertIsNotNone(second_message.timestamp)

        # Verify timestamps are different
        self.assertNotEqual(first_message.timestamp, second_message.timestamp)

    async def test_end_frame_handling(self):
        """Test that final messages are captured when pipeline ends normally"""
        processor = AssistantTranscriptProcessor()

        received_updates: List[TranscriptionUpdateFrame] = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.append(frame)

        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="world"),
            # Pipeline ends here; run_test will automatically send EndFrame
        ]

        expected_down_frames = [
            BotStartedSpeakingFrame,
            TTSTextFrame,
            TTSTextFrame,
            TranscriptionUpdateFrame,  # Final message emitted due to EndFrame
        ]

        # Run test - EndFrame will be sent automatically
        received_frames, _ = await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        self.assertEqual(len(received_updates), 1)
        message = received_updates[0].messages[0]
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Hello world")

    async def test_cancel_frame_handling(self):
        """Test that messages are properly captured when pipeline is cancelled"""
        processor = AssistantTranscriptProcessor()

        # Track updates with timestamps to verify order
        received_updates: List[Tuple[str, float]] = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            # Record message content and time received
            received_updates.append((frame.messages[0].content, asyncio.get_event_loop().time()))

        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="world"),
            SleepFrame(),  # Ensure messages are processed
            CancelFrame(),
        ]

        # We don't need to verify frame order, just that CancelFrame triggers message emission
        expected_down_frames = [
            BotStartedSpeakingFrame,
            TTSTextFrame,
            TTSTextFrame,
            CancelFrame,
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )

        # Verify that we received an update
        self.assertEqual(len(received_updates), 1, "Should receive one update before cancellation")
        content, _ = received_updates[0]
        self.assertEqual(content, "Hello world")

    async def test_transcript_processor_factory(self):
        """Test that factory properly manages processors and event handlers"""
        from pipecat.processors.transcript_processor import TranscriptProcessor

        factory = TranscriptProcessor()
        received_updates: List[TranscriptionMessage] = []

        # Register handler with factory
        @factory.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.extend(frame.messages)

        # Get processors and verify they're reused
        user_proc1 = factory.user()
        user_proc2 = factory.user()
        self.assertIs(user_proc1, user_proc2, "User processor should be reused")

        asst_proc1 = factory.assistant()
        asst_proc2 = factory.assistant()
        self.assertIs(asst_proc1, asst_proc2, "Assistant processor should be reused")

        # Test user processor
        timestamp = datetime.now(timezone.utc).isoformat()
        frames_to_send = [
            TranscriptionFrame(text="User message", user_id="user1", timestamp=timestamp)
        ]

        await run_test(
            user_proc1,
            frames_to_send=frames_to_send,
            expected_down_frames=[TranscriptionUpdateFrame, TranscriptionFrame],
        )

        # Test assistant processor
        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text="Assistant"),
            TTSTextFrame(text="message"),
            BotStoppedSpeakingFrame(),
        ]

        # The actual order we see in the output:
        await run_test(
            asst_proc1,
            frames_to_send=frames_to_send,
            expected_down_frames=[
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                TTSTextFrame,
                TTSTextFrame,
                TranscriptionUpdateFrame,
            ],
        )

        # Verify both processors triggered the same handler
        self.assertEqual(len(received_updates), 2)
        self.assertEqual(received_updates[0].role, "user")
        self.assertEqual(received_updates[0].content, "User message")
        self.assertEqual(received_updates[1].role, "assistant")
        self.assertEqual(received_updates[1].content, "Assistant message")

    async def test_text_fragments_with_spaces(self):
        """Test aggregating text fragments with various spacing patterns"""
        processor = AssistantTranscriptProcessor()

        # Track received updates
        received_updates = []

        @processor.event_handler("on_transcript_update")
        async def handle_update(proc, frame: TranscriptionUpdateFrame):
            received_updates.append(frame)

        # Test the specific pattern shared
        frames_to_send = [
            BotStartedSpeakingFrame(),
            SleepFrame(),
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text=" there"),
            TTSTextFrame(text="!"),
            TTSTextFrame(text=" How"),
            TTSTextFrame(text="'s"),
            TTSTextFrame(text=" it"),
            TTSTextFrame(text=" going"),
            TTSTextFrame(text="?"),
            BotStoppedSpeakingFrame(),
        ]

        expected_down_frames = [
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TTSTextFrame,
            TranscriptionUpdateFrame,
        ]

        # Run test
        received_frames, _ = await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify result
        self.assertEqual(len(received_updates), 1)
        message = received_updates[0].messages[0]
        self.assertEqual(message.role, "assistant")
        # Should be properly joined without extra spaces
        self.assertEqual(message.content, "Hello there! How's it going?")
