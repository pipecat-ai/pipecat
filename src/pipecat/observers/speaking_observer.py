#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer reporting who was speaking, and when.

A conversation is a sequence of people taking the floor and occasionally
taking it from each other. This observer reports each of those moments as it
happens, leaving what counts as a turn to whoever reads them: a turn boundary
is a policy, and a policy that ships inside a record can never be revised.
"""

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class SpeechEventKind(StrEnum):
    """What happened, to whom, and at which layer.

    The user appears at two layers, and they answer different questions.
    ``user_speech_*`` is the speech itself, as the voice activity detector
    heard it: speech that never becomes a turn — a cough, a false start, a
    pause mid-sentence — appears only here. ``user_turn_*`` is the turn
    strategy's ruling on that speech, which is what the rest of the pipeline
    acts on, and follows the speech by however long the ruling took.
    """

    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_STOPPED = "user_speech_stopped"
    USER_TURN_STARTED = "user_turn_started"
    USER_TURN_STOPPED = "user_turn_stopped"
    BOT_SPEECH_STARTED = "bot_speech_started"
    BOT_SPEECH_STOPPED = "bot_speech_stopped"
    INTERRUPTION = "interruption"


class SpeechEvent(BaseModel):
    """One moment in the conversation's speaking lifecycle.

    Parameters:
        kind: What happened, to whom, and at which layer.
        timestamp: Unix timestamp of the moment itself. Speech is timed to
            when it began and ended, not to when the detector confirmed it, so
            an interval drawn from these matches what was said.
        started_at: When the matching stretch of speech began, on the moments
            that end one, so a stretch reads as an interval without pairing it
            with the moment that opened it.
    """

    kind: SpeechEventKind
    timestamp: float
    started_at: float | None = None


class SpeakingObserver(BaseObserver):
    """Reports the speaking lifecycle of a conversation.

    Every moment is reported as it happens, and the moments that close a
    stretch of speech name where it began, so an interval reads whole from one
    record: a stretch whose closing moment never arrives stays open rather than
    quietly joining itself to the next one. What a turn is stays with the
    reader: a turn built here would freeze one definition into every record,
    where the moments themselves can be grouped again later, differently, over
    the same history.

    Events:
        on_speech_event(observer, event): Emitted for each moment, as a
            :class:`SpeechEvent`.

    Example::

        observer = SpeakingObserver()

        @observer.event_handler("on_speech_event")
        async def on_speech_event(observer, event):
            logger.info(event.model_dump_json())
    """

    def __init__(self, *, time_source: Callable[[], float] = time.time, **kwargs):
        """Initialize the speaking observer.

        Args:
            time_source: Reads the current time in seconds. Supplying one lets
                a test place moments without waiting.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._now = time_source
        self._reported: set[int] = set()
        # When each open stretch of speech began, so the moment that closes one
        # can carry it.
        self._open: dict[SpeechEventKind, float] = {}

        self._register_event_handler("on_speech_event")

    async def on_push_frame(self, data: FramePushed):
        """Report the moment a frame represents, the first time it is seen.

        Args:
            data: Frame push event containing the frame and direction.
        """
        frame = data.frame

        # An interruption is broadcast, arriving as two frames with two IDs, so
        # an ID alone would not tell them apart. Read the downstream one.
        if frame.broadcast_sibling_id is not None and data.direction != FrameDirection.DOWNSTREAM:
            return
        if frame.id in self._reported:
            return

        event = self._as_event(frame)
        if not event:
            return

        self._reported.add(frame.id)
        await self._call_event_handler("on_speech_event", event)

    def _as_event(self, frame: Frame) -> SpeechEvent | None:
        """Build the moment a frame represents.

        Args:
            frame: The frame being pushed.

        Returns:
            The moment, or None if this frame is not part of the speaking
            lifecycle.
        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The detector's account of when speech began, which precedes its
            # confirmation by the time it needed to be sure.
            return self._opens(
                SpeechEventKind.USER_SPEECH_STARTED, frame.timestamp - frame.start_secs
            )
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            return self._closes(
                SpeechEventKind.USER_SPEECH_STOPPED,
                SpeechEventKind.USER_SPEECH_STARTED,
                frame.timestamp - frame.stop_secs,
            )
        elif isinstance(frame, UserStartedSpeakingFrame):
            return self._opens(SpeechEventKind.USER_TURN_STARTED, self._now())
        elif isinstance(frame, UserStoppedSpeakingFrame):
            return self._closes(
                SpeechEventKind.USER_TURN_STOPPED, SpeechEventKind.USER_TURN_STARTED, self._now()
            )
        elif isinstance(frame, BotStartedSpeakingFrame):
            return self._opens(SpeechEventKind.BOT_SPEECH_STARTED, self._now())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            return self._closes(
                SpeechEventKind.BOT_SPEECH_STOPPED, SpeechEventKind.BOT_SPEECH_STARTED, self._now()
            )
        elif isinstance(frame, InterruptionFrame):
            return SpeechEvent(kind=SpeechEventKind.INTERRUPTION, timestamp=self._now())
        return None

    def _opens(self, kind: SpeechEventKind, at: float) -> SpeechEvent:
        """Record the start of a stretch of speech."""
        self._open[kind] = at
        return SpeechEvent(kind=kind, timestamp=at)

    def _closes(self, kind: SpeechEventKind, opened_by: SpeechEventKind, at: float) -> SpeechEvent:
        """Record the end of a stretch, naming its start where one is known.

        A stretch that began before the observer was watching closes without a
        start rather than borrowing one from another stretch.
        """
        return SpeechEvent(kind=kind, timestamp=at, started_at=self._open.pop(opened_by, None))
