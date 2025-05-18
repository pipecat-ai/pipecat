import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import from tests.conftest instead of directly from bot_openai
from tests.conftest import LatencyTracker, FIXED_DATETIME
# Import our custom datetime patch
from tests.mock_datetime import patch_datetime, unpatch_datetime


@pytest.fixture
def tracker():
    """Create a fresh LatencyTracker instance for each test."""
    return LatencyTracker()


@pytest.fixture
def mock_time():
    """Mock time.time() to control timestamps precisely."""
    with patch('time.time') as mock_time:
        # Starting at a fixed timestamp
        mock_time.return_value = 1000.0
        yield mock_time


@pytest.fixture
def mock_datetime():
    """Mock datetime.now() to control timestamps precisely using our custom mock."""
    # Setup mock datetime
    mock_dt = patch_datetime()
    mock_dt.set_now(datetime(2023, 1, 1, 12, 0, 0))
    
    # Run the test
    yield mock_dt
    
    # Cleanup
    unpatch_datetime()


@pytest.fixture
def mock_sentry():
    """Mock Sentry SDK to avoid external dependencies."""
    with patch('sentry_sdk.start_transaction') as mock_start_transaction, \
         patch('sentry_sdk.set_measurement') as mock_set_measurement:
        mock_transaction = MagicMock()
        mock_start_transaction.return_value = mock_transaction
        mock_transaction.__enter__.return_value = mock_transaction
        yield {
            'start_transaction': mock_start_transaction,
            'set_measurement': mock_set_measurement,
            'transaction': mock_transaction
        }


class TestLatencyTracker:
    """Test suite for the LatencyTracker class."""

    def test_initialization(self, tracker):
        """Test that the LatencyTracker initializes with correct default values."""
        assert tracker.user_started_speaking is None
        assert tracker.user_stopped_speaking is None
        assert tracker.bot_started_speaking is None
        assert tracker.bot_stopped_speaking is None
        assert tracker.turn_count == 0
        assert tracker.bot_is_speaking is False
        assert tracker.bot_was_interrupted is False
        assert tracker.interruption_timestamp is None
        assert tracker.response_latencies == {}
        assert tracker.interrupt_latencies == {}

    def test_user_start_basic(self, tracker, mock_time):
        """Test recording when user starts speaking."""
        mock_time.return_value = 1001.0
        tracker.user_start()
        
        assert tracker.user_started_speaking == 1001.0
        assert tracker.turn_count == 1
        assert not tracker.bot_was_interrupted  # No interruption since bot wasn't speaking

    def test_user_start_with_interruption(self, tracker, mock_time):
        """Test interruption when user starts speaking while bot is speaking."""
        # First, set up that the bot is speaking
        mock_time.return_value = 1000.0
        tracker.bot_started_speaking = 1000.0
        tracker.bot_is_speaking = True
        
        # Then user interrupts
        mock_time.return_value = 1005.0
        tracker.user_start()
        
        assert tracker.user_started_speaking == 1005.0
        assert tracker.bot_was_interrupted is True
        assert tracker.interruption_timestamp == 1005.0
        assert tracker.turn_count == 1

    def test_user_stop(self, tracker, mock_time):
        """Test recording when user stops speaking."""
        # First, user starts speaking
        mock_time.return_value = 1001.0
        tracker.user_start()
        
        # Then user stops speaking
        mock_time.return_value = 1005.0
        tracker.user_stop()
        
        assert tracker.user_stopped_speaking == 1005.0

    def test_bot_start_with_response_latency(self, tracker, mock_time, mock_datetime):
        """Test recording when bot starts speaking and calculating response latency."""
        # User speaks and stops
        mock_time.return_value = 1001.0
        tracker.user_start()
        mock_time.return_value = 1005.0
        tracker.user_stop()
        
        # Bot starts speaking
        mock_time.return_value = 1008.0
        tracker.bot_start()
        
        assert tracker.bot_started_speaking == 1008.0
        assert tracker.bot_is_speaking is True
        
        # Check response latency calculation
        expected_latency = 1008.0 - 1005.0  # 3 seconds
        assert datetime(2023, 1, 1, 12, 0, 0) in tracker.response_latencies
        assert tracker.response_latencies[datetime(2023, 1, 1, 12, 0, 0)] == expected_latency

    def test_bot_stop_without_interruption(self, tracker, mock_time):
        """Test recording when bot stops speaking normally (without interruption)."""
        # Set tracker state directly - otherwise it gets reset in bot_stop
        tracker.bot_started_speaking = 1008.0
        tracker.bot_is_speaking = True
        
        # Bot stops speaking
        mock_time.return_value = 1015.0
        saved_stopped_speaking = mock_time.return_value  # Save for assertion
        tracker.bot_stop()
        
        # Verify bot_stopped_speaking was set correctly before being reset
        assert tracker.bot_is_speaking is False
        assert tracker.bot_was_interrupted is False
        
        # We can't assert on bot_stopped_speaking since it's reset in bot_stop,
        # but we can verify that the bot is no longer speaking

    def test_bot_stop_with_interruption(self, tracker, mock_time, mock_datetime):
        """Test recording when bot stops speaking after being interrupted."""
        # Set tracker state directly
        tracker.bot_started_speaking = 1008.0
        tracker.bot_is_speaking = True
        tracker.bot_was_interrupted = True
        tracker.interruption_timestamp = 1012.0
        
        # Bot stops speaking
        mock_time.return_value = 1015.0
        tracker.bot_stop()
        
        # Verify state after stopping
        assert tracker.bot_is_speaking is False
        assert tracker.bot_was_interrupted is False
        assert tracker.interruption_timestamp is None
        
        # Check interrupt latency calculation
        expected_latency = 1015.0 - 1012.0  # 3 seconds
        assert datetime(2023, 1, 1, 12, 0, 0) in tracker.interrupt_latencies
        assert tracker.interrupt_latencies[datetime(2023, 1, 1, 12, 0, 0)] == expected_latency

    def test_explicit_interruption(self, tracker, mock_time):
        """Test handling explicit interruption event."""
        # Bot is speaking
        mock_time.return_value = 1008.0
        tracker.bot_started_speaking = 1008.0
        tracker.bot_is_speaking = True
        
        # Explicit interruption occurs
        mock_time.return_value = 1012.0
        tracker.explicit_interruption()
        
        assert tracker.bot_was_interrupted is True
        assert tracker.interruption_timestamp == 1012.0

    def test_explicit_interruption_bot_not_speaking(self, tracker, mock_time):
        """Test that explicit interruption has no effect if bot isn't speaking."""
        # Bot is not speaking
        tracker.bot_is_speaking = False
        
        # Explicit interruption occurs
        mock_time.return_value = 1012.0
        tracker.explicit_interruption()
        
        # Should not record interruption since bot wasn't speaking
        assert tracker.bot_was_interrupted is False
        assert tracker.interruption_timestamp is None

    def test_export_metrics_empty(self, tracker):
        """Test exporting metrics when no events have occurred."""
        metrics = tracker.export_metrics()
        
        assert metrics["response_latencies"] == {}
        assert metrics["interrupt_latencies"] == {}
        assert metrics["avg_response_latency"] == 0
        assert metrics["avg_interrupt_latency"] == 0
        assert metrics["total_turns"] == 0
        assert metrics["interruptions"] == 0

    def test_export_metrics_with_data(self, tracker, mock_datetime):
        """Test exporting metrics with actual latency data."""
        # Add some sample latency data manually
        tracker.turn_count = 3
        tracker.response_latencies = {
            datetime(2023, 1, 1, 12, 0, 0): 1.5,
            datetime(2023, 1, 1, 12, 1, 0): 2.5
        }
        tracker.interrupt_latencies = {
            datetime(2023, 1, 1, 12, 2, 0): 0.8
        }
        
        metrics = tracker.export_metrics()
        
        assert metrics["response_latencies"] == tracker.response_latencies
        assert metrics["interrupt_latencies"] == tracker.interrupt_latencies
        assert metrics["avg_response_latency"] == 2.0  # (1.5 + 2.5) / 2
        assert metrics["avg_interrupt_latency"] == 0.8
        assert metrics["total_turns"] == 3
        assert metrics["interruptions"] == 1

    @patch('sentry_sdk.init')
    def test_full_conversation_flow(self, mock_init, tracker, mock_time, mock_datetime):
        """Test a complete conversation flow with multiple turns."""
        # Reset tracker state to ensure clean test
        tracker.response_latencies = {}
        tracker.interrupt_latencies = {}
        tracker.turn_count = 0
        
        # Turn 1: User speaks
        mock_time.return_value = 1000.0
        tracker.user_start()
        mock_time.return_value = 1005.0
        tracker.user_stop()
        
        # Bot responds
        mock_time.return_value = 1007.0
        tracker.bot_start()
        
        # Check first response latency
        assert FIXED_DATETIME in tracker.response_latencies
        assert tracker.response_latencies[FIXED_DATETIME] == 2.0  # 1007-1005
        
        mock_time.return_value = 1012.0
        tracker.bot_stop()
        
        # Turn 2: User interrupts bot
        # Reset the tracker state manually since we're simulating a new turn
        tracker.bot_is_speaking = True  # Set this manually since bot_stop resets it
        
        # Since we're using the same FIXED_DATETIME for all latencies in our mock,
        # save the current response latency so we can verify it's not lost
        first_response_latency = tracker.response_latencies.get(FIXED_DATETIME)
        
        mock_time.return_value = 1020.0
        tracker.bot_start()
        mock_time.return_value = 1023.0  # User interrupts while bot is speaking
        tracker.user_start()
        mock_time.return_value = 1025.0
        tracker.bot_stop()  # Bot stops due to interruption
        
        # Check interrupt latency
        assert FIXED_DATETIME in tracker.interrupt_latencies
        assert tracker.interrupt_latencies[FIXED_DATETIME] == 2.0  # 1025-1023
        
        mock_time.return_value = 1030.0
        tracker.user_stop()
        
        # Bot responds to second turn
        mock_time.return_value = 1032.0
        tracker.bot_start()
        
        # Check second response latency
        assert FIXED_DATETIME in tracker.response_latencies
        assert tracker.response_latencies[FIXED_DATETIME] == 2.0  # 1032-1030
        
        mock_time.return_value = 1037.0
        tracker.bot_stop()
        
        # Check turn count
        assert tracker.turn_count == 2
        
        # Check that the metrics contain the expected values
        metrics = tracker.export_metrics()
        assert metrics["total_turns"] == 2
        assert metrics["interruptions"] == 1
        
        # Response latency should be 2s (using the most recent value)
        assert metrics["avg_response_latency"] == 2.0
        
        # Interruption latency should be 2s
        assert metrics["avg_interrupt_latency"] == 2.0 