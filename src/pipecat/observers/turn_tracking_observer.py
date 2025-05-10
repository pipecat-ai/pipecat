#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from collections import deque

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    StartFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class TurnTrackingObserver(BaseObserver):
    """Observer that tracks conversation turns in a pipeline.

    Turn tracking logic:
    - The first turn starts immediately when the pipeline starts (StartFrame)
    - Subsequent turns start when the user starts speaking
    - A turn ends when the bot stops speaking, unless it was already interrupted

    Events:
    - on_turn_started(turn_number)
    - on_turn_ended(turn_number, duration, was_interrupted)
    """

    def __init__(self, max_frames=100, **kwargs):
        super().__init__()
        self._turn_count = 0
        self._is_turn_active = False
        self._is_bot_speaking = False
        self._turn_start_time = 0

        # Track processed frames to avoid duplicates
        self._processed_frames = set()
        self._frame_history = deque(maxlen=max_frames)

        self._register_event_handler("on_turn_started")
        self._register_event_handler("on_turn_ended")

    async def on_push_frame(
        self,
        data: FramePushed,
    ):
        """Process frame events for turn tracking."""
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
                await self._start_turn(data.timestamp)
        elif isinstance(data.frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(data.timestamp)
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            self._is_bot_speaking = True
        elif isinstance(data.frame, BotStoppedSpeakingFrame) and self._is_bot_speaking:
            self._is_bot_speaking = False
            await self._maybe_end_turn(data.timestamp)

    async def _handle_user_started_speaking(self, timestamp: int):
        """Handle user speaking events, including interruptions."""
        if self._is_bot_speaking:
            # Handle interruption - end current turn and start a new one
            await self._end_turn(timestamp, was_interrupted=True)
            self._is_bot_speaking = False  # Bot is considered interrupted
            await self._start_turn(timestamp)
        elif not self._is_turn_active:
            # Start a new turn after previous one ended
            await self._start_turn(timestamp)

    async def _maybe_end_turn(self, timestamp: int):
        """End the current turn if one is active."""
        if self._is_turn_active:
            await self._end_turn(timestamp, was_interrupted=False)

    async def _start_turn(self, timestamp: int):
        """Start a new turn."""
        self._is_turn_active = True
        self._turn_count += 1
        self._turn_start_time = timestamp
        logger.debug(f"Turn {self._turn_count} started")
        await self._call_event_handler("on_turn_started", self._turn_count)

    async def _end_turn(self, timestamp: int, was_interrupted: bool):
        """End the current turn."""
        duration = (timestamp - self._turn_start_time) / 1_000_000_000  # Convert to seconds
        self._is_turn_active = False

        status = "interrupted" if was_interrupted else "completed"
        logger.debug(f"Turn {self._turn_count} {status} after {duration:.2f}s")
        await self._call_event_handler("on_turn_ended", self._turn_count, duration, was_interrupted)

    def _register_event_handler(self, event_name):
        """Register an event handler."""
        if not hasattr(self, "_event_handlers"):
            self._event_handlers = {}
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []

    async def _call_event_handler(self, event_name, *args, **kwargs):
        """Call registered event handlers."""
        if not hasattr(self, "_event_handlers"):
            return

        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                await handler(self, *args, **kwargs)

    def event_handler(self, event_name):
        """Decorator for registering event handlers."""

        def decorator(func):
            if not hasattr(self, "_event_handlers"):
                self._event_handlers = {}
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(func)
            return func

        return decorator
