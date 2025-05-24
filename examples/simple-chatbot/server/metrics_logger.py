"""Metrics logging utility for capturing Pipecat performance metrics.

This module provides a FrameProcessor that captures various performance metrics
from Pipecat frames and saves them to JSON files for later analysis.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from pipecat.frames.frames import Frame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MetricsLogger(FrameProcessor):
    """Frame processor that logs and saves various performance metrics.

    Captures metrics like TTFB, processing time, LLM usage, and TTS usage.
    Saves metrics to JSON files for later analysis.
    """

    def __init__(self, session_id=None):
        """Initialize metrics logger.

        Args:
            session_id: Optional identifier for the session being logged
        """
        super().__init__()
        self.session_id = session_id
        self.metrics_data: List[Dict[str, Any]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and capture any metrics it contains.

        Args:
            frame: The frame to process
            direction: Direction the frame is flowing
        """
        await super().process_frame(frame, direction)

        # Only process metrics if we have a session_id
        if isinstance(frame, MetricsFrame) and self.session_id:
            timestamp = datetime.now().isoformat()

            for data in frame.data:
                metric_entry = {
                    "timestamp": timestamp,
                    "session_id": self.session_id,
                    "frame_type": type(data).__name__,
                }

                if isinstance(data, TTFBMetricsData):
                    metric_entry.update(
                        {
                            "metric_type": "ttfb",
                            "processor": data.processor,
                            "value": data.value,
                            "unit": "seconds",
                            "model": data.model,
                        }
                    )
                    print(f"📊 CAPTURED TTFB: {data.processor} = {data.value}s")

                elif isinstance(data, ProcessingMetricsData):
                    metric_entry.update(
                        {
                            "metric_type": "processing_time",
                            "processor": data.processor,
                            "value": data.value,
                            "unit": "seconds",
                            "model": data.model,
                        }
                    )
                    print(f"📊 CAPTURED Processing: {data.processor} = {data.value}s")

                elif isinstance(data, LLMUsageMetricsData):
                    tokens = data.value
                    metric_entry.update(
                        {
                            "metric_type": "llm_usage",
                            "processor": data.processor,
                            "prompt_tokens": tokens.prompt_tokens,
                            "completion_tokens": tokens.completion_tokens,
                            "total_tokens": tokens.total_tokens,
                            "model": data.model,
                        }
                    )
                    print(
                        f"📊 CAPTURED LLM Usage: {data.processor} = {tokens.prompt_tokens}p + {tokens.completion_tokens}c tokens"
                    )

                elif isinstance(data, TTSUsageMetricsData):
                    metric_entry.update(
                        {
                            "metric_type": "tts_usage",
                            "processor": data.processor,
                            "characters": data.value,
                            "model": data.model,
                        }
                    )
                    print(f"📊 CAPTURED TTS Usage: {data.processor} = {data.value} characters")

                self.metrics_data.append(metric_entry)

        await self.push_frame(frame, direction)

    def save_session_metrics(self):
        """Save current session metrics to a file and return the filename."""
        if self.session_id and self.metrics_data:
            filename = f"session_data/{self.session_id}_metrics.json"
            os.makedirs("session_data", exist_ok=True)

            with open(filename, "w") as f:
                json.dump(self.metrics_data, f, indent=2)

            print(f"💾 Saved {len(self.metrics_data)} metrics to {filename}")

            return filename
        else:
            print(
                f"❌ No metrics to save. Session ID: {self.session_id}, Metrics count: {len(self.metrics_data)}"
            )
        return None
