#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for InterruptibleTTSService's reconnect-on-interruption tracking.

InterruptibleTTSService reconnects its websocket on interruption if the bot
was speaking (or about to start speaking) for the turn being interrupted. It
tracks this with two flags:

- ``_bot_speaking``: true only once BotStartedSpeakingFrame
  confirms the output transport actually received audio (see
  test_no_error_on_long_streaming_turn and
  test_no_deadlock_on_zero_audio_context_completion in test_tts_frame_ordering.py).
- ``_tts_started`` (InterruptibleTTSService only): true from the moment
  run_tts is invoked (TTSStartedFrame pushed) until consumed by an
  interruption, or cleared by BotStoppedSpeakingFrame (turn ended normally) or
  a following LLMFullResponseStartFrame (safety net for a turn that never got
  a BotStoppedSpeakingFrame at all), covering the narrow window before
  BotStartedSpeakingFrame confirmation arrives.

These are deliberately separate: folding _tts_started's early, unconfirmed
signal into _bot_speaking would let a turn that produces zero audio look
"confirmed".
"""

import asyncio
import unittest
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 16000


@dataclass
class MarkerFrame(DataFrame):
    """Marks how far downstream processing has got."""

    label: str = ""


class FakeInterruptibleTTSService(InterruptibleTTSService):
    """Minimal concrete InterruptibleTTSService for testing reconnect tracking.

    Never actually opens a websocket; _connect/_disconnect are patched or
    spied on in individual tests instead.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if False:
            yield


@pytest.mark.asyncio
async def test_reconnects_when_bot_confirmed_speaking():
    """BotStartedSpeakingFrame confirms speech; interrupting must reconnect."""
    tts = FakeInterruptibleTTSService()

    reconnected = {"disconnect": False, "connect": False}

    async def fake_disconnect():
        reconnected["disconnect"] = True

    async def fake_connect():
        reconnected["connect"] = True

    tts._disconnect = fake_disconnect
    tts._connect = fake_connect

    await tts.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert tts._bot_speaking is True
    assert tts._tts_started is False

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert reconnected == {"disconnect": True, "connect": True}


@pytest.mark.asyncio
async def test_reconnects_when_started_but_not_yet_confirmed():
    """run_tts invoked (TTSStartedFrame) but BotStartedSpeakingFrame hasn't
    arrived yet — the narrow race window _tts_started exists for. Interrupting
    here must still reconnect, even though _bot_speaking is still False.
    """
    tts = FakeInterruptibleTTSService()

    reconnected = {"disconnect": False, "connect": False}

    async def fake_disconnect():
        reconnected["disconnect"] = True

    async def fake_connect():
        reconnected["connect"] = True

    tts._disconnect = fake_disconnect
    tts._connect = fake_connect

    await tts.push_frame(TTSStartedFrame())
    assert tts._tts_started is True
    assert tts._bot_speaking is False

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert reconnected == {"disconnect": True, "connect": True}
    # Consumed by the interruption it was needed for.
    assert tts._tts_started is False


@pytest.mark.asyncio
async def test_no_reconnect_when_bot_never_spoke():
    """No TTSStartedFrame or BotStartedSpeakingFrame this turn — interrupting
    (e.g. the user talking over silence) must not reconnect.
    """
    tts = FakeInterruptibleTTSService()

    reconnected = {"disconnect": False, "connect": False}

    async def fake_disconnect():
        reconnected["disconnect"] = True

    async def fake_connect():
        reconnected["connect"] = True

    tts._disconnect = fake_disconnect
    tts._connect = fake_connect

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert reconnected == {"disconnect": False, "connect": False}


@pytest.mark.asyncio
async def test_tts_started_cleared_on_new_turn():
    """_tts_started must not leak into a new turn.

    Models a turn that invoked run_tts (TTSStartedFrame) but never got a
    BotStartedSpeakingFrame or BotStoppedSpeakingFrame — a turn that completed
    with no audio, say — so nothing ever clears _tts_started via the normal
    BotStoppedSpeakingFrame path.
    Without the LLMFullResponseStartFrame reset, an interruption during the
    *next* turn (before it has invoked run_tts itself) would incorrectly
    reconnect because of the stale flag.
    """
    tts = FakeInterruptibleTTSService()

    await tts.push_frame(TTSStartedFrame())
    assert tts._tts_started is True

    # New turn begins without the previous one ever resolving _tts_started.
    await tts.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    assert tts._tts_started is False

    reconnected = {"disconnect": False, "connect": False}

    async def fake_disconnect():
        reconnected["disconnect"] = True

    async def fake_connect():
        reconnected["connect"] = True

    tts._disconnect = fake_disconnect
    tts._connect = fake_connect

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert reconnected == {"disconnect": False, "connect": False}


@pytest.mark.asyncio
async def test_silent_turn_resumes_frame_processing():
    """A turn that plays nothing must not leave frame processing paused for an
    InterruptibleTTSService subclass that combines it with
    pause_frame_processing=True (e.g. the deprecated RimeNonJsonTTSService).

    A context completes (TTSStoppedFrame) with zero TTSAudioRawFrames, and no
    BotStartedSpeakingFrame/BotStoppedSpeakingFrame ever arrives — as in
    production, where the output transport never receives audio to react to.
    Whether the pause is taken at all depends on how long the provider holds
    the context open, so this asserts only that the pipeline keeps moving.
    """

    class FakeInterruptiblePauseTTSService(FakeInterruptibleTTSService):
        def __init__(self, **kwargs):
            super().__init__(
                pause_frame_processing=True,
                **kwargs,
            )

        async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
            async def _deliver_zero_audio_completion():
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)

            self.create_task(_deliver_zero_audio_completion(), name=f"fake_zero_audio_{context_id}")
            if False:
                yield

    tts = FakeInterruptiblePauseTTSService()

    frames_to_send = [
        LLMFullResponseStartFrame(),
        TextFrame(text="Hello."),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.4),  # let the silent context complete
        MarkerFrame(label="after_silence"),
    ]

    down, _ = await run_test(tts, frames_to_send=frames_to_send)

    markers = [f for f in down if isinstance(f, MarkerFrame)]
    assert any(f.label == "after_silence" for f in markers), (
        "Frame processing stayed paused after a zero-audio completion, meaning "
        "TTSStartedFrame's early marker masked both recoveries"
    )


@pytest.mark.asyncio
async def test_reconnect_flags_track_full_turn_via_process_frame_and_push_frame():
    """End-to-end sanity check through the public frame-processing API (not
    direct attribute pokes): a normal turn with confirmed playback reconnects
    on interruption, and after BotStoppedSpeakingFrame ends the turn cleanly,
    a later interruption with no new speech does not reconnect.
    """
    tts = FakeInterruptibleTTSService()

    calls = []

    async def fake_disconnect():
        calls.append("disconnect")

    async def fake_connect():
        calls.append("connect")

    tts._disconnect = fake_disconnect
    tts._connect = fake_connect

    await tts.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await tts.push_frame(TTSStartedFrame())
    await tts.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await tts.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert tts._bot_speaking is False
    assert tts._tts_started is False

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert calls == [], f"Should not reconnect after a clean, already-finished turn: {calls}"


class FakeStreamingInterruptibleTTSService(FakeInterruptibleTTSService):
    """Streams requests in order on one connection and labels audio with the active context.

    Reconnecting drops the audio still to come (e.g. FishAudioTTSService).
    """

    def __init__(self):
        super().__init__(
            push_stop_frames=True, stop_frame_timeout_s=0.2, pause_frame_processing=True
        )
        self._requests: asyncio.Queue[str] = asyncio.Queue()
        self._stream_task: asyncio.Task | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def _connect(self):
        await super()._connect()
        self._requests = asyncio.Queue()
        self._stream_task = self.create_task(self._stream())

    async def _disconnect(self):
        await super()._disconnect()
        if self._stream_task:
            await self.cancel_task(self._stream_task)
            self._stream_task = None

    async def _stream(self):
        while True:
            await self._requests.get()
            for _ in range(4):
                await asyncio.sleep(0.05)
                context_id = self.get_active_audio_context_id()
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(b"\x00" * 320, _SAMPLE_RATE, 1, context_id=context_id),
                )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        await self._requests.put(text)
        yield None


@pytest.mark.asyncio
async def test_interrupted_reply_stays_interruptible_behind_uninterruptible_speech():
    """Uninterruptible speech sent mid-reply doesn't keep the interrupted reply playing."""
    speech = TTSSpeakFrame("One moment.", append_to_context=False)
    speech.interruptible = False
    down, _ = await asyncio.wait_for(
        run_test(
            FakeStreamingInterruptibleTTSService(),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                TextFrame("Hello there. "),
                TextFrame("How"),
                speech,
                SleepFrame(sleep=0.1),
                BotStartedSpeakingFrame(),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.4),
            ],
        ),
        timeout=5,
    )
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert audio
    assert all(f.interruptible for f in audio)


if __name__ == "__main__":
    unittest.main()
