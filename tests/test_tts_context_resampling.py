#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for resampling TTS audio to the service's sample rate as it plays.

Each audio context has its own resampler, so a context's audio comes out as if
its whole clip had been resampled in one go, however the provider spaced its
chunks, and an interruption drops whatever the resampler still holds.
"""

import asyncio
import io
import wave
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
import pytest

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SERVICE_RATE = 24000


def _tone(rate: int, samples: int) -> bytes:
    t = np.arange(samples) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()


def _split(data: bytes, parts: int) -> list[bytes]:
    size = len(data) // parts // 2 * 2
    return [data[i : i + size] for i in range(0, len(data), size)]


async def _resampled(clip: bytes, in_rate: int) -> bytes:
    """Resample a whole clip in one go, the audio a context should come out as."""
    resampler = create_stream_resampler(clear_after_secs=None)
    return await resampler.resample(clip, in_rate, _SERVICE_RATE) + await resampler.flush()


class MockTTSService(TTSService):
    """HTTP-style TTS service that yields each utterance as the given chunks."""

    def __init__(self, chunks: list[tuple[bytes, int]], delay_s: float = 0):
        super().__init__(push_text_frames=False, push_stop_frames=True, sample_rate=_SERVICE_RATE)
        self._chunks = chunks
        self._delay_s = delay_s

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        for audio, rate in self._chunks:
            yield TTSAudioRawFrame(audio, rate, 1, context_id=context_id)
            if self._delay_s:
                await asyncio.sleep(self._delay_s)


class IteratorTTSService(TTSService):
    """TTS service that streams raw PCM through the iterator helper."""

    def __init__(self, chunks: list[bytes], in_sample_rate: int, delay_s: float = 0):
        super().__init__(push_text_frames=False, push_stop_frames=True, sample_rate=_SERVICE_RATE)
        self._chunks = chunks
        self._in_sample_rate = in_sample_rate
        self._delay_s = delay_s

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
            if self._delay_s:
                await asyncio.sleep(self._delay_s)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async for frame in self._stream_audio_frames_from_iterator(
            self._iterate(), in_sample_rate=self._in_sample_rate, context_id=context_id
        ):
            yield frame


async def _run(service: TTSService, *frames: Frame) -> list[Frame]:
    down_frames, _ = await run_test(service, frames_to_send=[TTSSpeakFrame(text="Hello."), *frames])
    return down_frames


def _audio(frames: list[Frame]) -> list[TTSAudioRawFrame]:
    return [f for f in frames if isinstance(f, TTSAudioRawFrame)]


@pytest.mark.asyncio
async def test_audio_at_another_rate_comes_out_resampled_in_one_go():
    clip = _tone(16000, 4000)
    service = MockTTSService([(chunk, 16000) for chunk in _split(clip, 4)])

    audio = _audio(await _run(service))

    assert all(f.sample_rate == _SERVICE_RATE for f in audio)
    assert b"".join(f.audio for f in audio) == await _resampled(clip, 16000)


@pytest.mark.asyncio
async def test_a_pause_in_the_middle_of_a_reply_leaves_no_gap():
    clip = _tone(16000, 4000)
    service = IteratorTTSService(_split(clip, 2), in_sample_rate=16000, delay_s=0.3)

    audio = _audio(await _run(service))

    assert b"".join(f.audio for f in audio) == await _resampled(clip, 16000)


@pytest.mark.asyncio
async def test_the_held_audio_comes_out_before_the_stopped_frame():
    service = MockTTSService([(_tone(16000, 4000), 16000)])

    frames = await _run(service)

    stopped = next(i for i, f in enumerate(frames) if isinstance(f, TTSStoppedFrame))
    last_audio = max(i for i, f in enumerate(frames) if isinstance(f, TTSAudioRawFrame))
    assert last_audio < stopped
    assert len(_audio(frames)) == 2


@pytest.mark.asyncio
async def test_an_interruption_drops_the_audio_the_resampler_holds():
    clip = _tone(16000, 4000)
    service = MockTTSService([(chunk, 16000) for chunk in _split(clip, 4)], delay_s=0.05)

    frames = await _run(
        service,
        SleepFrame(sleep=0.07),
        InterruptionFrame(),
        SleepFrame(sleep=0.2),
        TTSSpeakFrame(text="Again."),
    )

    audio = _audio(frames)
    last_context = audio[-1].context_id
    assert audio[0].context_id != last_context
    reply = b"".join(f.audio for f in audio if f.context_id == last_context)
    assert reply == await _resampled(clip, 16000)


@pytest.mark.asyncio
async def test_a_rate_change_flushes_the_audio_at_the_previous_rate():
    first, second = _tone(16000, 2000), _tone(22050, 2000)
    service = MockTTSService([(first, 16000), (second, 22050)])

    audio = _audio(await _run(service))

    expected = await _resampled(first, 16000) + await _resampled(second, 22050)
    assert b"".join(f.audio for f in audio) == expected


@pytest.mark.asyncio
async def test_audio_at_the_service_rate_passes_through_untouched():
    clip = _tone(_SERVICE_RATE, 4000)
    service = MockTTSService([(chunk, _SERVICE_RATE) for chunk in _split(clip, 4)])

    audio = _audio(await _run(service))

    assert b"".join(f.audio for f in audio) == clip


@pytest.mark.asyncio
async def test_the_iterator_helper_streams_audio_at_the_source_rate():
    clip = _tone(22050, 4410)
    service = IteratorTTSService(_split(clip, 3), in_sample_rate=22050)

    audio = _audio(await _run(service))

    assert b"".join(f.audio for f in audio) == await _resampled(clip, 22050)


@pytest.mark.asyncio
async def test_the_iterator_helper_takes_the_rate_from_a_wav_header():
    clip = _tone(16000, 1600)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(clip)
    service = MockTTSService([])

    async def chunks() -> AsyncIterator[bytes]:
        yield buffer.getvalue()

    frames = [
        frame
        async for frame in service._stream_audio_frames_from_iterator(
            chunks(), strip_wav_header=True, context_id="wav"
        )
    ]

    assert [f.sample_rate for f in frames] == [16000]
    assert b"".join(f.audio for f in frames) == clip


@pytest.mark.asyncio
async def test_chunks_cut_mid_sample_play_whole_at_another_rate():
    clip = _tone(16000, 4000)
    chunks = [clip[:1001], clip[1001:4000], clip[4000:6999], clip[6999:]]
    service = MockTTSService([(chunk, 16000) for chunk in chunks])

    audio = _audio(await _run(service))

    assert b"".join(f.audio for f in audio) == await _resampled(clip, 16000)


@pytest.mark.asyncio
async def test_audio_back_at_the_service_rate_follows_the_held_audio():
    first, second = _tone(16000, 2000), _tone(_SERVICE_RATE, 3000)
    service = MockTTSService([(first, 16000), (second, _SERVICE_RATE)])

    audio = _audio(await _run(service))

    assert b"".join(f.audio for f in audio) == await _resampled(first, 16000) + second
