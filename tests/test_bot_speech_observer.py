#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.bot_speech_observer import BotSpeechObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import TextType


def _observer():
    """An observer, and the events it reported: each frame's text, and each end."""
    observer = BotSpeechObserver()
    events = []

    @observer.event_handler("on_bot_text")
    async def on_bot_text(observer, frame):
        events.append(frame.text)

    @observer.event_handler("on_bot_speech_ended")
    async def on_bot_speech_ended(observer, interrupted):
        events.append(("ended", interrupted))

    return observer, events


def _pushed(frame, *, source=None, direction=FrameDirection.DOWNSTREAM, first_push=True):
    return FramePushed(
        # The TTS service, unless a test says otherwise.
        source=source or IdentityFilter(),
        destination=IdentityFilter(),
        frame=frame,
        direction=direction,
        timestamp=0,
        first_push=first_push,
    )


def _word(text, pts=1, text_type=TextType.WORD, interruptible=True):
    # A pts of 0 is how the rest of the framework says "untimed", so a timed
    # word in a test needs a real one.
    frame = TTSTextFrame(text, text_type=text_type)
    frame.pts = pts
    frame.interruptible = interruptible
    return frame


def _transport():
    return BaseOutputTransport(TransportParams())


@pytest.mark.asyncio
async def test_reports_the_bots_words_and_the_end_of_its_speech():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await observer.on_push_frame(_pushed(_word("there.")))
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(_word("Again")))

    assert events == ["Hello", "there.", ("ended", False), "Again"]


@pytest.mark.asyncio
async def test_reports_a_word_once_however_many_processors_pass_it_on():
    observer, events = _observer()

    word = _word("Hello")
    await observer.on_push_frame(_pushed(word))
    await observer.on_push_frame(_pushed(word, source=_transport(), first_push=False))

    assert events == ["Hello"]


@pytest.mark.asyncio
async def test_ignores_frames_going_upstream():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(_word("Hello"), direction=FrameDirection.UPSTREAM))
    await observer.on_push_frame(
        _pushed(BotStoppedSpeakingFrame(), direction=FrameDirection.UPSTREAM)
    )

    assert events == []


@pytest.mark.asyncio
async def test_leaves_out_what_the_bot_does_not_say_as_words():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await observer.on_push_frame(_pushed(_word("laugh", text_type=TextType.EXPRESSION_SOUND)))
    await observer.on_push_frame(_pushed(_word("400", text_type=TextType.EXPRESSION_PAUSE)))
    await observer.on_push_frame(_pushed(_word(" ")))
    await observer.on_push_frame(_pushed(_word("there.")))

    assert events == ["Hello", "there."]


@pytest.mark.asyncio
async def test_does_not_end_speech_that_reported_no_text():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(InterruptionFrame()))

    assert events == []


@pytest.mark.asyncio
async def test_ends_speech_once_on_an_interruption():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await observer.on_push_frame(_pushed(InterruptionFrame()))
    await observer.on_push_frame(_pushed(InterruptionFrame(), first_push=False))
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    assert events == ["Hello", ("ended", True)]


@pytest.mark.asyncio
async def test_leaves_out_words_of_the_cut_off_reply_pushed_after_the_interruption():
    observer, events = _observer()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await observer.on_push_frame(_pushed(InterruptionFrame()))
    # The TTS service had not dropped the reply yet.
    await observer.on_push_frame(_pushed(_word("there.")))
    await observer.on_push_frame(_pushed(_word("Still on.", interruptible=False)))
    await observer.on_push_frame(_pushed(TTSStartedFrame()))
    await observer.on_push_frame(_pushed(_word("Next.")))

    assert events == ["Hello", ("ended", True), "Still on.", "Next."]


@pytest.mark.asyncio
async def test_reports_untimed_text_when_the_tts_service_pushes_it():
    observer, events = _observer()

    sentence = _word("A whole sentence.", pts=None)
    await observer.on_push_frame(_pushed(sentence))
    await observer.on_push_frame(_pushed(BotStartedSpeakingFrame(), source=_transport()))
    await observer.on_push_frame(_pushed(sentence, source=_transport(), first_push=False))

    assert events == ["A whole sentence."]


@pytest.mark.asyncio
async def test_ends_speech_of_text_the_transport_played_no_audio_for():
    observer, events = _observer()

    # The TTS service failed, so its text reaches the transport with no audio
    # and the bot never starts speaking.
    sentence = _word("A whole sentence.", pts=None)
    await observer.on_push_frame(_pushed(sentence))
    await observer.on_push_frame(_pushed(sentence, source=_transport(), first_push=False))

    assert events == ["A whole sentence.", ("ended", False)]


@pytest.mark.asyncio
async def test_keeps_speech_going_when_a_timed_word_beats_the_bot_speaking():
    observer, events = _observer()

    # The transport's clock releases the first word before its audio task says
    # the bot started speaking.
    word = _word("Hello")
    await observer.on_push_frame(_pushed(word))
    await observer.on_push_frame(_pushed(word, source=_transport(), first_push=False))
    await observer.on_push_frame(_pushed(BotStartedSpeakingFrame(), source=_transport()))
    await observer.on_push_frame(_pushed(_word("there.")))

    assert events == ["Hello", "there."]
