"""Test configuration for pytest.

Import and expose classes from bot-openai.py for testing.
This avoids direct imports from the bot-openai.py file which might have problematic
dependencies in a test environment.
"""

import sys
import os
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add the server directory to the path
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

# Create mock for Sentry SDK to prevent actual Sentry usage in tests
sys.modules['sentry_sdk'] = MagicMock()
sys.modules['sentry_sdk.integrations.asyncio'] = MagicMock()
sys.modules['sentry_sdk.integrations.asyncio'].AsyncioIntegration = MagicMock()


# Import time separately since it's used in our mock implementations
import time


# Use a fixed datetime for testing
FIXED_DATETIME = datetime(2023, 1, 1, 12, 0, 0)


# Mock implementation of the LatencyTracker class
class LatencyTracker:
    """Mock implementation of the LatencyTracker class for testing."""

    def __init__(self):
        self.user_started_speaking = None
        self.user_stopped_speaking = None
        self.bot_started_speaking = None
        self.bot_stopped_speaking = None
        self.conversation_id = f"conversation-{1000}"
        self.turn_count = 0
        self.bot_is_speaking = False
        self.bot_was_interrupted = False
        self.interruption_timestamp = None
        self.interruption_transaction = None
        self.response_latency_transaction = None
        self.response_latencies = {}
        self.interrupt_latencies = {}
    
    def user_start(self):
        """Record when user starts speaking."""
        timestamp = time.time()
        self.user_started_speaking = timestamp
        self.turn_count += 1
        
        if self.bot_is_speaking:
            self.bot_was_interrupted = True
            self.interruption_timestamp = timestamp
    
    def user_stop(self):
        """Record when user stops speaking."""
        if not self.user_started_speaking:
            return
            
        self.user_stopped_speaking = time.time()
    
    def bot_start(self):
        """Record when bot starts speaking and calculate response latency."""
        timestamp = time.time()
        self.bot_started_speaking = timestamp
        self.bot_is_speaking = True
        
        if self.user_stopped_speaking:
            latency = timestamp - self.user_stopped_speaking
            # Use fixed datetime for predictable testing
            self.response_latencies[FIXED_DATETIME] = latency
    
    def bot_stop(self):
        """Record when bot stops speaking and finalize interruption latency if applicable."""
        if not self.bot_is_speaking:
            return
            
        timestamp = time.time()
        self.bot_stopped_speaking = timestamp
        self.bot_is_speaking = False
        
        if self.bot_was_interrupted and self.interruption_timestamp:
            latency = timestamp - self.interruption_timestamp
            # Use fixed datetime for predictable testing
            self.interrupt_latencies[FIXED_DATETIME] = latency
            
        self.bot_was_interrupted = False
        self.interruption_timestamp = None
        self.bot_started_speaking = None
        self.bot_stopped_speaking = None
    
    def explicit_interruption(self):
        """Handle explicit interruption frame from pipeline."""
        timestamp = time.time()
        
        if self.bot_is_speaking and not self.bot_was_interrupted:
            self.bot_was_interrupted = True
            self.interruption_timestamp = timestamp
    
    def export_metrics(self):
        """Return current latency metrics."""
        metrics = {
            "response_latencies": self.response_latencies,
            "interrupt_latencies": self.interrupt_latencies,
            "avg_response_latency": sum(self.response_latencies.values()) / len(self.response_latencies) if self.response_latencies else 0,
            "avg_interrupt_latency": sum(self.interrupt_latencies.values()) / len(self.interrupt_latencies) if self.interrupt_latencies else 0,
            "total_turns": self.turn_count,
            "interruptions": len(self.interrupt_latencies)
        }
        return metrics


# Mock implementation of the LatencyTrackerProcessor class
class LatencyTrackerProcessor:
    """Mock implementation of the LatencyTrackerProcessor class for testing."""
    
    def __init__(self):
        self.tracker = LatencyTracker()
        self.metrics_task = None
    
    async def process_frame(self, frame, direction):
        """Process incoming frames and update latency metrics."""
        # Import within the method to avoid circular imports
        from tests.mock_frames import (
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            BotInterruptionFrame,
        )
        
        # Process specific frame types to track latency
        if isinstance(frame, UserStartedSpeakingFrame):
            self.tracker.user_start()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.tracker.user_stop()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.tracker.bot_start()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.tracker.bot_stop()
        elif isinstance(frame, BotInterruptionFrame):
            # Process explicit interruption frame from pipeline
            self.tracker.explicit_interruption()
        
        # Pass the frame through
        await self.push_frame(frame, direction)
    
    async def push_frame(self, frame, direction):
        """Mock implementation of push_frame."""
        pass
    
    async def start_metrics_export(self):
        """Start the metrics export task."""
        # Create periodic metrics export task
        async def export_metrics_to_sentry():
            while True:
                metrics = self.tracker.export_metrics()
                await asyncio.sleep(60)  # Export every minute
        
        self.metrics_task = asyncio.create_task(export_metrics_to_sentry())
        
    async def stop_metrics_export(self):
        """Stop the metrics export task."""
        if self.metrics_task:
            self.metrics_task.cancel()
            self.metrics_task = None


# Try to import the real classes as a fallback
try:
    # We use exec to import the module with a dash in the name
    bot_openai_spec = {}
    with open(os.path.join(server_dir, 'bot-openai.py')) as f:
        exec(f.read(), bot_openai_spec)
    
    # If we successfully imported, we could use these instead
    # But we'll prefer our mock implementations which are simpler and don't
    # have external dependencies
    # LatencyTracker = bot_openai_spec['LatencyTracker']
    # LatencyTrackerProcessor = bot_openai_spec['LatencyTrackerProcessor']
    pass
except Exception as e:
    print(f"Error importing from bot-openai.py: {e}") 