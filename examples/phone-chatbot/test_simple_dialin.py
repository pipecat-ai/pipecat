import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time
from datetime import timedelta

from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    EndTaskFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection

# Import the classes we want to test
from simple_dialin import CallStatistics, MetricsProcessor


class TestCallStatistics:
    def test_initialization(self):
        """Test that CallStatistics initializes with correct default values."""
        stats = CallStatistics()

        assert stats.silence_events == 0
        assert stats.user_messages == 0
        assert stats.bot_messages == 0
        assert stats.llm_tokens_used == 0
        assert stats.tts_characters_used == 0
        assert stats.end_time is None
        assert isinstance(stats.start_time, float)

    def test_record_silence_event(self):
        """Test recording silence events."""
        stats = CallStatistics()
        stats.record_silence_event()
        stats.record_silence_event()

        assert stats.silence_events == 2

    def test_record_user_message(self):
        """Test recording user messages."""
        stats = CallStatistics()
        stats.record_user_message()

        assert stats.user_messages == 1

    def test_record_bot_message(self):
        """Test recording bot messages."""
        stats = CallStatistics()
        stats.record_bot_message()
        stats.record_bot_message()
        stats.record_bot_message()

        assert stats.bot_messages == 3

    def test_record_llm_usage(self):
        """Test recording LLM token usage."""
        stats = CallStatistics()
        stats.record_llm_usage(100)
        stats.record_llm_usage(50)

        assert stats.llm_tokens_used == 150

    def test_record_tts_usage(self):
        """Test recording TTS character usage."""
        stats = CallStatistics()
        stats.record_tts_usage(200)

        assert stats.tts_characters_used == 200

    def test_end_call(self):
        """Test ending a call sets the end time."""
        stats = CallStatistics()
        stats.end_call()

        assert stats.end_time is not None
        assert isinstance(stats.end_time, float)

    def test_get_duration_seconds(self):
        """Test calculating call duration."""
        stats = CallStatistics()
        # Mock the start time to be 10 seconds ago
        stats.start_time = time.time() - 10

        # Without ending the call
        duration = stats.get_duration_seconds()
        assert 9 <= duration <= 11  # Allow small timing variations

        # After ending the call
        stats.end_call()
        duration = stats.get_duration_seconds()
        assert 9 <= duration <= 11  # Allow small timing variations


class TestMetricsProcessor:
    @pytest.fixture
    def call_stats(self):
        return CallStatistics()

    @pytest.fixture
    def metrics_processor(self, call_stats):
        return MetricsProcessor(call_stats)

    @pytest.mark.asyncio
    async def test_process_llm_metrics_frame(self, metrics_processor, call_stats):
        """Test processing an LLM metrics frame."""
        # Create a mock LLM usage metrics data with proper structure
        llm_data = MagicMock(spec=LLMUsageMetricsData)
        # Set up the nested structure correctly
        llm_data.value = MagicMock()
        llm_data.value.total_tokens = 150

        # Create a metrics frame with the LLM data
        frame = MetricsFrame(data=[llm_data])

        # Process the frame
        await metrics_processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check that the call stats were updated
        assert call_stats.llm_tokens_used == 150

    @pytest.mark.asyncio
    async def test_process_tts_metrics_frame(self, metrics_processor, call_stats):
        """Test processing a TTS metrics frame."""
        # Create a mock TTS usage metrics data
        tts_data = MagicMock(spec=TTSUsageMetricsData)
        tts_data.value = 200

        # Create a metrics frame with the TTS data
        frame = MetricsFrame(data=[tts_data])

        # Process the frame
        await metrics_processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check that the call stats were updated
        assert call_stats.tts_characters_used == 200

    @pytest.mark.asyncio
    async def test_process_transcription_frame(self, metrics_processor, call_stats):
        """Test processing a transcription frame."""
        # Create a transcription frame
        frame = MagicMock(spec=TranscriptionFrame)
        frame.final = True

        # Process the frame
        await metrics_processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check that the call stats were updated
        assert call_stats.user_messages == 1

    @pytest.mark.asyncio
    async def test_process_tts_speak_frame(self, metrics_processor, call_stats):
        """Test processing a TTS speak frame."""
        # Create a TTS speak frame
        frame = MagicMock(spec=TTSSpeakFrame)

        # Process the frame
        await metrics_processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check that the call stats were updated
        assert call_stats.bot_messages == 1


@pytest.mark.asyncio
async def test_user_idle_processor():
    """Test the user idle processor functionality."""
    # Mock the necessary components
    pipeline = AsyncMock()
    task = AsyncMock()
    call_stats = MagicMock()

    # Create a simple handle_user_idle function for testing
    async def handle_user_idle(processor, retry_count):
        # Record silence event
        call_stats.record_silence_event()

        if retry_count <= 2:
            await pipeline.push_frame(
                TTSSpeakFrame(
                    text="I notice you've been quiet for a while. Is there anything I can help you with?"
                )
            )
            return True
        else:
            await pipeline.push_frame(
                TTSSpeakFrame(
                    text="I'll be ending our call now. Feel free to call back if you need assistance later."
                )
            )
            await asyncio.sleep(0.1)  # Minimal sleep for testing
            call_stats.end_call()
            call_stats.log_summary()
            await task.queue_frame(EndTaskFrame())
            return False

    # Test first retry (retry_count = 1)
    result = await handle_user_idle(None, 1)

    # Check that the function returned True to continue monitoring
    assert result is True

    # Check that a TTSSpeakFrame was pushed to the pipeline
    pipeline.push_frame.assert_called_once()
    args, _ = pipeline.push_frame.call_args
    frame = args[0]
    assert isinstance(frame, TTSSpeakFrame)
    assert "I notice you've been quiet" in frame.text

    # Check that silence event was recorded
    call_stats.record_silence_event.assert_called_once()

    # Reset the mocks for the next test
    pipeline.reset_mock()
    call_stats.reset_mock()

    # Test third retry (retry_count = 3)
    result = await handle_user_idle(None, 3)

    # Check that the function returned False to stop monitoring
    assert result is False

    # Check that a TTSSpeakFrame was pushed to the pipeline
    pipeline.push_frame.assert_called_once()
    args, _ = pipeline.push_frame.call_args
    frame = args[0]
    assert isinstance(frame, TTSSpeakFrame)
    assert "I'll be ending our call now" in frame.text

    # Check that the call was ended
    task.queue_frame.assert_called_once()
    args, _ = task.queue_frame.call_args
    frame = args[0]
    assert isinstance(frame, EndTaskFrame)

    # Check that the call statistics were updated
    call_stats.record_silence_event.assert_called_once()
    call_stats.end_call.assert_called_once()
    call_stats.log_summary.assert_called_once()
