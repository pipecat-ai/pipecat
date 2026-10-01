#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how websocket TTS services handle a replaced provider connection.

Providers keep context state per connection, so replacing a connection ends
every open audio context, and a turn in progress continues under a new context
ID. Reopening a closed connection to send the text at hand leaves audio contexts
alone.
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
    TTSUpdateSettingsFrame,
)
from pipecat.services import websocket_service
from pipecat.services.settings import TTSSettings
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
    audio for ``reconnect_after`` arrives, the service replaces its connection.
    Like several real services, it also replaces its connection when a setting
    changes.

    Args:
        reconnect_after: The sentence after which to replace the connection, or
            None.
        reconnect_by_disconnecting: Replace the connection with ``_disconnect()``
            and ``_connect()``, as settings changes and failed sends do, rather
            than through ``_reconnect_websocket``.
        first_audio: Audio for the first sentence, if not the default.
        connect_lazily: Open the connection in the first run_tts call, the way
            services reconnect after the provider closed an idle connection.
    """

    def __init__(
        self,
        reconnect_after: int | None = 1,
        reconnect_by_disconnecting: bool = False,
        first_audio: bytes | None = None,
        connect_lazily: bool = False,
        **kwargs,
    ):
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=0.3,
            **kwargs,
        )
        self._reconnect_after = reconnect_after
        self._reconnect_by_disconnecting = reconnect_by_disconnecting
        self._first_audio = first_audio
        self._needs_connection = connect_lazily
        self._sentences = 0

    def can_generate_metrics(self) -> bool:
        return False

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        pass

    async def _verify_connection(self) -> bool:
        return True

    async def _update_settings(self, delta):
        changed = await super()._update_settings(delta)
        if changed:
            await self._disconnect()
            await self._connect()
        return changed

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self._sentences += 1
        sentence = self._sentences
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)
        if self._needs_connection:
            self._needs_connection = False
            await self._websocket_connect("ws://test")

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
                if self._reconnect_by_disconnecting:
                    await self._disconnect()
                    await self._connect()
                else:
                    await self._reconnect_websocket(1)

        self.create_task(deliver(), name=f"deliver_{sentence}")
        yield None


async def _speak_two_sentences(
    service: MockWebsocketTTSService, between: list[Frame] | None = None
) -> list[Frame]:
    # "One." is released when "Two" arrives, and "Two." when the turn ends, so
    # anything that happens after sentence 1 comes before sentence 2.
    down_frames, _ = await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            TextFrame("One. "),
            TextFrame("Two"),
            SleepFrame(sleep=0.15),
            *(between or []),
            SleepFrame(sleep=0.15),
            TextFrame("."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.6),
        ],
    )
    return down_frames


def _assert_turn_continued_under_new_context(down_frames: list[Frame]):
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

    _assert_turn_continued_under_new_context(down_frames)


@pytest.mark.asyncio
async def test_disconnect_and_connect_continues_turn_under_new_context():
    service = MockWebsocketTTSService(reconnect_by_disconnecting=True)

    down_frames = await _speak_two_sentences(service)

    _assert_turn_continued_under_new_context(down_frames)


@pytest.mark.asyncio
async def test_settings_change_mid_turn_continues_turn_under_new_context():
    service = MockWebsocketTTSService(reconnect_after=None)

    down_frames = await _speak_two_sentences(
        service, between=[TTSUpdateSettingsFrame(delta=TTSSettings(voice="other-voice"))]
    )

    _assert_turn_continued_under_new_context(down_frames)


@pytest.mark.asyncio
async def test_reconnect_flushes_held_partial_sample():
    first_audio = bytes([1, 0]) * 160 + b"\x01"
    service = MockWebsocketTTSService(first_audio=first_audio)

    down_frames = await _speak_two_sentences(service)

    audio = _audio_by_sentence(down_frames)
    assert b"".join(f.audio for f in audio[1]) == first_audio + b"\x00"
    assert all(len(f.audio) % 2 == 0 for f in audio[2])


@pytest.mark.asyncio
async def test_lazy_connection_keeps_the_context_it_sends_to():
    service = MockWebsocketTTSService(reconnect_after=None, connect_lazily=True)

    down_frames = await _speak_two_sentences(service)

    audio = _audio_by_sentence(down_frames)
    assert set(audio) == {1, 2}
    assert audio[1][0].context_id == audio[2][0].context_id
