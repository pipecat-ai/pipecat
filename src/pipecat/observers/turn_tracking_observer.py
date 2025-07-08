#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn tracking observer for conversation flow monitoring.

This module provides an observer that monitors conversation turns in a pipeline,
tracking when turns start and end based on user and bot speech patterns.
"""

import asyncio
from collections import deque

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    StartFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class TurnTrackingObserver(BaseObserver):
    """Observer that tracks conversation turns in a pipeline.

    This observer monitors the flow of conversation by tracking when turns
    start and end based on user and bot speaking patterns. It handles
    interruptions, timeouts, and maintains turn state throughout the pipeline.

    Turn tracking logic:

    - The first turn starts immediately when the pipeline starts (StartFrame)
    - Subsequent turns start when the user starts speaking
    - A turn ends when the bot stops speaking and either:

      - The user starts speaking again
      - A timeout period elapses with no more bot speech
    """

    def __init__(self, max_frames=100, turn_end_timeout_secs=2.5, **kwargs):
        """Initialize the turn tracking observer.

        Args:
            max_frames: Maximum number of frame IDs to keep in history for
                duplicate detection. Defaults to 100.
            turn_end_timeout_secs: Timeout in seconds after bot stops speaking
                before automatically ending the turn. Defaults to 2.5.
            **kwargs: Additional arguments passed to the parent observer.
        """
        super().__init__(**kwargs)
        self._turn_count = 0
        self._is_turn_active = False
        self._is_bot_speaking = False
        self._has_bot_spoken = False
        self._turn_start_time = 0
        self._turn_end_timeout_secs = turn_end_timeout_secs
        self._end_turn_timer = None

        # Track processed frames to avoid duplicates
        self._processed_frames = set()
        self._frame_history = deque(maxlen=max_frames)

        self._register_event_handler("on_turn_started")
        self._register_event_handler("on_turn_ended")

    async def on_push_frame(self, data: FramePushed):
        """Process frame events for turn tracking.

        Args:
            data: Frame push event data containing the frame and metadata.
        """
        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)
        self._frame_history.append(data.frame.id)

        # If we've exceeded our history size, remove the oldest frame ID
        # from the set of processed frames.
        if len(self._processed_frames) > len(self._frame_history):
            # Rebuild the set from the current deque contents
            self._processed_frames = set(self._frame_history)

        if isinstance(data.frame, StartFrame):
            # Start the first turn immediately when the pipeline starts
            if self._turn_count == 0:
                await self._start_turn(data)
        elif isinstance(data.frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(data)
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(data)
        # A BotStoppedSpeakingFrame can arrive after a UserStartedSpeakingFrame following an interruption
        # We only want to end the turn if the bot was previously speaking
        elif isinstance(data.frame, BotStoppedSpeakingFrame) and self._is_bot_speaking:
            await self._handle_bot_stopped_speaking(data)
        elif isinstance(data.frame, (EndFrame, CancelFrame)):
            await self._handle_pipeline_end(data)

    def _schedule_turn_end(self, data: FramePushed):
        """Schedule turn end with a timeout."""
        # Cancel any existing timer
        self._cancel_turn_end_timer()

        # Create a new timer
        loop = asyncio.get_event_loop()
        self._end_turn_timer = loop.call_later(
            self._turn_end_timeout_secs,
            lambda: asyncio.create_task(self._end_turn_after_timeout(data)),
        )

    def _cancel_turn_end_timer(self):
        """Cancel the turn end timer if it exists."""
        if self._end_turn_timer:
            self._end_turn_timer.cancel()
            self._end_turn_timer = None

    async def _end_turn_after_timeout(self, data: FramePushed):
        """End turn after timeout has expired."""
        if self._is_turn_active and not self._is_bot_speaking:
            logger.trace(f"Turn {self._turn_count} ending due to timeout")
            await self._end_turn(data, was_interrupted=False)
            self._end_turn_timer = None

    async def _handle_user_started_speaking(self, data: FramePushed):
        """Handle user speaking events, including interruptions."""
        if self._is_bot_speaking:
            # Handle interruption - end current turn and start a new one
            self._cancel_turn_end_timer()  # Cancel any pending end turn timer
            await self._end_turn(data, was_interrupted=True)
            self._is_bot_speaking = False  # Bot is considered interrupted
            await self._start_turn(data)
        elif self._is_turn_active and self._has_bot_spoken:
            # User started speaking during the turn_end_timeout_secs period after bot speech
            self._cancel_turn_end_timer()  # Cancel any pending end turn timer
            await self._end_turn(data, was_interrupted=False)
            await self._start_turn(data)
        elif not self._is_turn_active:
            # Start a new turn after previous one ended
            await self._start_turn(data)
        else:
            # User is speaking within the same turn (before bot has responded)
            logger.trace(f"User is already speaking in Turn {self._turn_count}")

    async def _handle_bot_started_speaking(self, data: FramePushed):
        """Handle bot speaking events."""
        self._is_bot_speaking = True
        self._has_bot_spoken = True
        # Cancel any pending turn end timer when bot starts speaking again
        self._cancel_turn_end_timer()

    async def _handle_bot_stopped_speaking(self, data: FramePushed):
        """Handle bot stopped speaking events."""
        self._is_bot_speaking = False
        # Schedule turn end with timeout
        # This is needed to handle cases where the bot's speech ends and then resumes
        # This can happen with HTTP TTS services or function calls
        self._schedule_turn_end(data)

    async def _handle_pipeline_end(self, data: FramePushed):
        """Handle pipeline end or cancellation by flushing any active turn."""
        if self._is_turn_active:
            # Cancel any pending turn end timer
            self._cancel_turn_end_timer()
            # End the current turn
            await self._end_turn(data, was_interrupted=True)

    async def _start_turn(self, data: FramePushed):
        """Start a new turn."""
        self._is_turn_active = True
        self._has_bot_spoken = False
        self._turn_count += 1
        self._turn_start_time = data.timestamp
        logger.trace(f"Turn {self._turn_count} started")
        await self._call_event_handler("on_turn_started", self._turn_count)

    async def _end_turn(self, data: FramePushed, was_interrupted: bool):
        """End the current turn."""
        if not self._is_turn_active:
            return

        duration = (data.timestamp - self._turn_start_time) / 1_000_000_000  # Convert to seconds
        self._is_turn_active = False

        status = "interrupted" if was_interrupted else "completed"
        logger.trace(f"Turn {self._turn_count} {status} after {duration:.2f}s")
        await self._call_event_handler("on_turn_ended", self._turn_count, duration, was_interrupted)
