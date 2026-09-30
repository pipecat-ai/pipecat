#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn tracking observer for conversation flow monitoring.

This module provides an observer that monitors conversation turns in a pipeline,
tracking when turns start and end based on user and bot speech patterns.
"""

import asyncio
import warnings
from collections import deque

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    StartFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
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

    User speaking frames observed while the user is muted are ignored: mute
    strategies suppress them at the user aggregator, so speech detected during
    a mute window (e.g. echo of the bot's own audio) must not advance turns or
    report user speech timing.

    Events:

    - on_turn_started: Triggered when a turn starts, providing the turn number
    - on_turn_ended: Triggered when a turn ends, providing the turn number, the turn's
      duration in seconds, and whether it was interrupted
    """

    def __init__(self, max_frames: int | None = None, turn_end_timeout_secs: float = 2.5, **kwargs):
        """Initialize the turn tracking observer.

        Args:
            max_frames: Unused.

                .. deprecated:: 1.12.0
                    No replacement. The observer receives each frame once.
                    Will be removed in 2.0.0.
            turn_end_timeout_secs: Timeout in seconds after bot stops speaking
                before automatically ending the turn. Defaults to 2.5.
            **kwargs: Additional arguments passed to the parent observer.
        """
        if max_frames is not None:
            warnings.warn(
                "`max_frames` parameter of `TurnTrackingObserver` is deprecated since 1.12.0 "
                "and will be removed in 2.0.0. No replacement.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(observe_every_push=False, **kwargs)
        self._turn_count = 0
        self._is_turn_active = False
        self._is_bot_speaking = False
        # The logical turn which owns the currently playing bot audio. During
        # an interruption the conversation advances to the next turn before
        # BaseOutputTransport emits BotStoppedSpeakingFrame for the interrupted
        # audio, so this must be tracked independently from ``_turn_count``.
        self._bot_speaking_turn: int | None = None
        self._user_speaking_turn: int | None = None
        self._is_user_muted = False
        self._has_bot_spoken = False
        self._turn_start_time = 0
        self._turn_end_timeout_secs = turn_end_timeout_secs
        self._end_turn_timer = None

        # WorkerObserver deduplicates pushes; broadcast siblings are distinct frames.
        self._broadcast_sibling_ids = deque(maxlen=100)

        # These handlers run inside this observer's worker task, so awaiting
        # them preserves the exact frame order without blocking the pipeline.
        self._register_event_handler("on_turn_started", sync=True)
        self._register_event_handler("on_turn_ended", sync=True)
        self._register_event_handler("on_user_speech_started_for_turn", sync=True)
        self._register_event_handler("on_user_speech_stopped_for_turn", sync=True)
        self._register_event_handler("on_bot_started_speaking", sync=True)
        self._register_event_handler("on_bot_stopped_speaking", sync=True)

    async def on_push_frame(self, data: FramePushed):
        """Process frame events for turn tracking.

        Args:
            data: Frame push event data containing the frame and metadata.
        """
        sibling_id = data.frame.broadcast_sibling_id
        if sibling_id is not None:
            if data.frame.id in self._broadcast_sibling_ids:
                return
            self._broadcast_sibling_ids.append(sibling_id)

        if isinstance(data.frame, StartFrame):
            # Start the first turn immediately when the pipeline starts
            if self._turn_count == 0:
                await self._start_turn(data)
        elif isinstance(data.frame, UserMuteStartedFrame):
            self._is_user_muted = True
        elif isinstance(data.frame, UserMuteStoppedFrame):
            self._is_user_muted = False
        elif isinstance(data.frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(data)
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(data)
        elif isinstance(data.frame, InterruptionFrame) and self._is_bot_speaking:
            # Word thresholds can interrupt after a non-interrupting speech onset.
            self._cancel_turn_end_timer()
            await self._end_turn(data, was_interrupted=True)
            self._is_bot_speaking = False
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(data)
        # A BotStoppedSpeakingFrame can arrive after UserStartedSpeakingFrame
        # has already advanced the logical conversation turn. The independent
        # owner retained in ``_bot_speaking_turn`` lets consumers still
        # correlate that late physical stop with the interrupted turn.
        elif isinstance(data.frame, BotStoppedSpeakingFrame) and (
            self._is_bot_speaking or self._bot_speaking_turn is not None
        ):
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
        if self._is_turn_active and not self._is_bot_speaking and self._user_speaking_turn is None:
            logger.trace(f"Turn {self._turn_count} ending due to timeout")
            await self._end_turn(data, was_interrupted=False)
            self._end_turn_timer = None

    async def _handle_user_started_speaking(self, data: FramePushed):
        """Handle user speaking events, including interruptions."""
        assert isinstance(data.frame, UserStartedSpeakingFrame)
        if self._is_user_muted:
            logger.trace(f"Ignoring muted user speech start in Turn {self._turn_count}")
            return

        if self._is_bot_speaking:
            self._cancel_turn_end_timer()
            if data.frame.enable_interruptions:
                await self._end_turn(data, was_interrupted=True)
                self._is_bot_speaking = False
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

        self._user_speaking_turn = self._turn_count
        await self._call_event_handler(
            "on_user_speech_started_for_turn", self._user_speaking_turn, data
        )

    async def _handle_user_stopped_speaking(self, data: FramePushed):
        """Associate the user speech stop with its owning logical turn."""
        # A stop whose start was suppressed by mute carries no useful timing.
        # A stop for speech that started before the mute engaged is still
        # processed so the owning turn's speech state closes correctly.
        if self._is_user_muted and self._user_speaking_turn is None:
            logger.trace(f"Ignoring muted user speech stop in Turn {self._turn_count}")
            return

        turn_number = self._user_speaking_turn or self._turn_count
        await self._call_event_handler("on_user_speech_stopped_for_turn", turn_number, data)
        self._user_speaking_turn = None
        if self._has_bot_spoken and not self._is_bot_speaking:
            self._schedule_turn_end(data)

    async def _handle_bot_started_speaking(self, data: FramePushed):
        """Handle bot speaking events."""
        if not self._is_turn_active:
            await self._start_turn(data)
        self._is_bot_speaking = True
        self._bot_speaking_turn = self._turn_count
        self._has_bot_spoken = True
        # Cancel any pending turn end timer when bot starts speaking again
        self._cancel_turn_end_timer()
        await self._call_event_handler("on_bot_started_speaking", self._bot_speaking_turn, data)

    async def _handle_bot_stopped_speaking(self, data: FramePushed):
        """Handle bot stopped speaking events."""
        turn_number = self._bot_speaking_turn or self._turn_count
        self._is_bot_speaking = False
        self._bot_speaking_turn = None
        await self._call_event_handler("on_bot_stopped_speaking", turn_number, data)

        # Only schedule the current logical turn to end. An interrupted bot's
        # physical stop can arrive after the next turn has already started and
        # must not schedule that new turn for completion.
        if (
            self._is_turn_active
            and turn_number == self._turn_count
            and self._user_speaking_turn is None
        ):
            # This delay handles cases where bot speech resumes within the same
            # turn, for example HTTP TTS services or function calls.
            self._schedule_turn_end(data)

    async def _handle_pipeline_end(self, data: FramePushed):
        """Handle pipeline end or cancellation by flushing any active turn."""
        if self._is_turn_active:
            # Cancel any pending turn end timer
            self._cancel_turn_end_timer()

            # Lets not end the current turn here, since the observers
            # get notified of the end frame first, and it will
            # prematurely set current context as None, leading to
            # floating spans in the observability tools
            # await self._end_turn(data, was_interrupted=True)

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
