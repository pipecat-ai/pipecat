#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how websocket TTS services handle a new provider connection.

Providers keep context state per connection, so a new connection ends every
open audio context, and a turn in progress continues under a new context ID.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services import websocket_service
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 16000


@pytest.fixture(autouse=True)
def _fake_websocket_connect(monkeypatch):
    async def fake_connect(*args, **kwargs):
        return object()

    monkeypatch.setattr(websocket_service, "websocket_connect", fake_connect)


class MockWebsocketTTSService(WebsocketTTSService):
    """Websocket-style service whose audio arrives after run_tts returns.

    Each sentence's audio is tagged with the sentence's position. After the
    audio for ``reconnect_after`` arrives, the service opens a new connection.

    Args:
        reconnect_after: The sentence after which to reconnect.
        first_audio: Audio for the first sentence, if not the default.
    """

    def __init__(self, reconnect_after: int = 1, first_audio: bytes | None = None, **kwargs):
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=0.3,
            **kwargs,
        )
        self._reconnect_after = reconnect_after
        self._first_audio = first_audio
        self._sentences = 0

    def can_generate_metrics(self) -> bool:
        return False

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self._sentences += 1
        sentence = self._sentences
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)

        async def deliver():
            await asyncio.sleep(0.02)
            audio = bytes([sentence, 0]) * 160
            if sentence == 1 and self._first_audio is not None:
                audio = self._first_audio
            await self.append_to_audio_context(
                context_id, TTSAudioRawFrame(audio, _SAMPLE_RATE, 1, context_id=context_id)
            )
            if sentence == self._reconnect_after:
                await asyncio.sleep(0.05)
                await self._websocket_connect("ws://test")

        self.create_task(deliver(), name=f"deliver_{sentence}")
        yield None


async def _speak_two_sentences(service: MockWebsocketTTSService) -> list[Frame]:
    # "One." is released when "Two" arrives, and "Two." when the turn ends, so
    # the reconnect after sentence 1 happens before sentence 2 is synthesized.
    down_frames, _ = await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            TextFrame("One. "),
            TextFrame("Two"),
            SleepFrame(sleep=0.3),
            TextFrame("."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.6),
        ],
    )
    return down_frames


def _audio_by_sentence(frames: list[Frame]) -> dict[int, list[TTSAudioRawFrame]]:
    audio: dict[int, list[TTSAudioRawFrame]] = {}
    for frame in frames:
        if isinstance(frame, TTSAudioRawFrame):
            audio.setdefault(frame.audio[0], []).append(frame)
    return audio


@pytest.mark.asyncio
async def test_reconnect_continues_turn_under_new_context():
    service = MockWebsocketTTSService()

    down_frames = await _speak_two_sentences(service)

    audio = _audio_by_sentence(down_frames)
    assert set(audio) == {1, 2}
    first_context = audio[1][0].context_id
    second_context = audio[2][0].context_id
    assert first_context != second_context
    # The old context is stopped before the new one starts.
    events = [
        (type(f).__name__, f.context_id)
        for f in down_frames
        if isinstance(f, (TTSStartedFrame, TTSStoppedFrame))
    ]
    assert events == [
        ("TTSStartedFrame", first_context),
        ("TTSStoppedFrame", first_context),
        ("TTSStartedFrame", second_context),
        ("TTSStoppedFrame", second_context),
    ]


@pytest.mark.asyncio
async def test_reconnect_flushes_held_partial_sample():
    first_audio = bytes([1, 0]) * 160 + b"\x01"
    service = MockWebsocketTTSService(first_audio=first_audio)

    down_frames = await _speak_two_sentences(service)

    audio = _audio_by_sentence(down_frames)
    assert b"".join(f.audio for f in audio[1]) == first_audio + b"\x00"
    assert all(len(f.audio) % 2 == 0 for f in audio[2])


@pytest.mark.asyncio
async def test_first_connection_leaves_state_untouched():
    service = MockWebsocketTTSService()

    await service._websocket_connect("ws://test")

    assert service.get_audio_contexts() == []
    assert service._turn_context_id is None
