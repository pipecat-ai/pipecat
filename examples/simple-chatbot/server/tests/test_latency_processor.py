import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Import from tests.conftest instead of directly from bot_openai
from tests.conftest import LatencyTrackerProcessor
# Use our mock frames instead of actual Pipecat frames
from tests.mock_frames import (
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    BotInterruptionFrame,
    FrameDirection,
)


@pytest.fixture
def processor():
    """Create a fresh LatencyTrackerProcessor instance for each test."""
    return LatencyTrackerProcessor()


@pytest.fixture
def mock_tracker():
    """Create a mock for the LatencyTracker inside the processor."""
    with patch('bot_openai.LatencyTracker') as mock_tracker_class:
        mock_tracker_instance = MagicMock()
        mock_tracker_class.return_value = mock_tracker_instance
        yield mock_tracker_instance


@pytest.mark.asyncio
async def test_process_user_started_speaking_frame(processor):
    """Test processing UserStartedSpeakingFrame."""
    # Mock the tracker.user_start method
    processor.tracker.user_start = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a UserStartedSpeakingFrame
    frame = UserStartedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that tracker.user_start was called
    processor.tracker.user_start.assert_called_once()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_process_user_stopped_speaking_frame(processor):
    """Test processing UserStoppedSpeakingFrame."""
    # Mock the tracker.user_stop method
    processor.tracker.user_stop = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a UserStoppedSpeakingFrame
    frame = UserStoppedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that tracker.user_stop was called
    processor.tracker.user_stop.assert_called_once()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_process_bot_started_speaking_frame(processor):
    """Test processing BotStartedSpeakingFrame."""
    # Mock the tracker.bot_start method
    processor.tracker.bot_start = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a BotStartedSpeakingFrame
    frame = BotStartedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that tracker.bot_start was called
    processor.tracker.bot_start.assert_called_once()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_process_bot_stopped_speaking_frame(processor):
    """Test processing BotStoppedSpeakingFrame."""
    # Mock the tracker.bot_stop method
    processor.tracker.bot_stop = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a BotStoppedSpeakingFrame
    frame = BotStoppedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that tracker.bot_stop was called
    processor.tracker.bot_stop.assert_called_once()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_process_bot_interruption_frame(processor):
    """Test processing BotInterruptionFrame."""
    # Mock the tracker.explicit_interruption method
    processor.tracker.explicit_interruption = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a BotInterruptionFrame
    frame = BotInterruptionFrame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that tracker.explicit_interruption was called
    processor.tracker.explicit_interruption.assert_called_once()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_process_other_frame(processor):
    """Test processing an unrelated frame type."""
    # Mock the tracker methods
    processor.tracker.user_start = MagicMock()
    processor.tracker.user_stop = MagicMock()
    processor.tracker.bot_start = MagicMock()
    processor.tracker.bot_stop = MagicMock()
    processor.tracker.explicit_interruption = MagicMock()
    
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Process a generic Frame
    frame = Frame()
    await processor.process_frame(frame, FrameDirection.IN)
    
    # Check that no tracker methods were called
    processor.tracker.user_start.assert_not_called()
    processor.tracker.user_stop.assert_not_called()
    processor.tracker.bot_start.assert_not_called()
    processor.tracker.bot_stop.assert_not_called()
    processor.tracker.explicit_interruption.assert_not_called()
    
    # Check that push_frame was called with the frame
    processor.push_frame.assert_called_once_with(frame, FrameDirection.IN)


@pytest.mark.asyncio
async def test_start_metrics_export(processor):
    """Test starting the metrics export task."""
    # Mock asyncio.create_task
    with patch('asyncio.create_task') as mock_create_task:
        # Call start_metrics_export
        await processor.start_metrics_export()
        
        # Check that asyncio.create_task was called
        mock_create_task.assert_called_once()
        
        # Check that the metrics_task was set
        assert processor.metrics_task is not None


@pytest.mark.asyncio
async def test_stop_metrics_export_with_task(processor):
    """Test stopping the metrics export task when a task exists."""
    # Create a mock task
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()
    processor.metrics_task = mock_task
    
    # Call stop_metrics_export
    await processor.stop_metrics_export()
    
    # Check that the task was cancelled
    mock_task.cancel.assert_called_once()
    
    # Check that metrics_task was set to None
    assert processor.metrics_task is None


@pytest.mark.asyncio
async def test_stop_metrics_export_without_task(processor):
    """Test stopping the metrics export task when no task exists."""
    # Ensure metrics_task is None
    processor.metrics_task = None
    
    # Call stop_metrics_export
    await processor.stop_metrics_export()
    
    # Check that metrics_task is still None
    assert processor.metrics_task is None


@pytest.mark.asyncio
async def test_conversation_flow_integration(processor):
    """Test the complete conversation flow through the processor."""
    # Mock push_frame to avoid actual frame pushing
    processor.push_frame = AsyncMock()
    
    # Mock the tracker's methods to track calls
    processor.tracker.user_start = MagicMock()
    processor.tracker.user_stop = MagicMock()
    processor.tracker.bot_start = MagicMock()
    processor.tracker.bot_stop = MagicMock()
    
    # Process a sequence of frames representing a conversation
    frames = [
        UserStartedSpeakingFrame(),  # User starts speaking
        UserStoppedSpeakingFrame(),  # User stops speaking
        BotStartedSpeakingFrame(),   # Bot starts speaking
        BotStoppedSpeakingFrame(),   # Bot stops speaking
        UserStartedSpeakingFrame(),  # User starts speaking again
        BotInterruptionFrame(),      # Explicit interruption
        UserStoppedSpeakingFrame(),  # User stops speaking again
        BotStartedSpeakingFrame(),   # Bot starts speaking again
        BotStoppedSpeakingFrame(),   # Bot stops speaking again
    ]
    
    # Process each frame
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.IN)
    
    # Check the number of calls to each tracker method
    assert processor.tracker.user_start.call_count == 2
    assert processor.tracker.user_stop.call_count == 2
    assert processor.tracker.bot_start.call_count == 2
    assert processor.tracker.bot_stop.call_count == 2
    assert processor.push_frame.call_count == len(frames) 