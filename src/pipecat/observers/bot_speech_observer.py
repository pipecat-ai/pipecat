#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer that reports the text the bot speaks."""

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.text.base_text_aggregator import TextType


class BotSpeechObserver(BaseObserver):
    """Reports the text the bot speaks, as the TTS service produces it.

    Each ``TTSTextFrame`` is reported once, when the TTS service pushes it,
    which is ahead of its playback. A TTS service that reports word timestamps
    pushes one word per frame, with a ``pts`` saying when its audio is due at
    the transport. One that doesn't pushes its text in the pieces it produces,
    often a sentence, with no ``pts``. Expression sounds and pauses are left
    out, since the bot doesn't say them as words.

    After an interruption, the text of the cut-off reply the TTS service still
    pushes is left out until it starts its next reply, since the transport
    drops its audio. Uninterruptible text plays on, so it is still reported.

    Event handlers available:

    - on_bot_text: Called with each ``TTSTextFrame`` the bot will speak.
    - on_bot_speech_ended: Called once after the text reported since it was
      last called, when the bot stops speaking or is interrupted, with whether
      it was interrupted. Text reported before an interrupted end may not have
      been heard.

    Both run in order on the observer's task, so they need to be fast.

    Example::

        observer = BotSpeechObserver()

        @observer.event_handler("on_bot_text")
        async def on_bot_text(observer, frame: TTSTextFrame):
            ...

        @observer.event_handler("on_bot_speech_ended")
        async def on_bot_speech_ended(observer, interrupted: bool):
            ...
    """

    def __init__(self, **kwargs):
        """Initialize the observer.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        # Every push: text the transport releases tells whether it played any audio.
        super().__init__(**kwargs)
        # Whether text was reported since speech last ended.
        self._speaking = False
        self._bot_speaking = False
        # From an interruption until the TTS service starts its next reply, the
        # text it pushes belongs to the reply that was cut off.
        self._interrupted = False

        self._register_event_handler("on_bot_text", sync=True)
        self._register_event_handler("on_bot_speech_ended", sync=True)

    async def on_push_frame(self, data: FramePushed):
        """Report the bot's text, and the end of its speech.

        Args:
            data: Frame push event data containing source, frame, direction, and timestamp.
        """
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        frame = data.frame
        if isinstance(frame, InterruptionFrame):
            if data.first_push:
                # Speech ends here rather than on a BotStoppedSpeakingFrame,
                # which an interruption doesn't always produce: the transport
                # pushes none while uninterruptible audio is still playing, nor
                # when the bot hadn't started speaking yet.
                self._interrupted = True
                await self._end_speech(interrupted=True)
        elif isinstance(frame, TTSStartedFrame):
            self._interrupted = False
        elif isinstance(frame, TTSTextFrame) and frame.text.strip() and self._is_speech(frame):
            if self._interrupted and frame.interruptible:
                return
            if data.first_push:
                self._speaking = True
                await self._call_event_handler("on_bot_text", frame)
            elif (
                not frame.pts
                and isinstance(data.source, BaseOutputTransport)
                and not self._bot_speaking
            ):
                # The transport released untimed text it played no audio for,
                # as when the TTS service failed, so no BotStoppedSpeakingFrame
                # will end it.
                await self._end_speech(interrupted=False)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            await self._end_speech(interrupted=False)

    @staticmethod
    def _is_speech(frame: TTSTextFrame) -> bool:
        """Whether the bot says this text out loud as words.

        An expression sound or pause is written into the reply by the LLM and
        carries the pts of the word beside it, but it is not the bot's words.
        """
        return frame.text_type not in (TextType.EXPRESSION_SOUND, TextType.EXPRESSION_PAUSE)

    async def _end_speech(self, *, interrupted: bool):
        if not self._speaking:
            return
        self._speaking = False
        await self._call_event_handler("on_bot_speech_ended", interrupted)
