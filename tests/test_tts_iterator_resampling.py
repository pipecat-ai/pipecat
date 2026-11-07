#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for resampling complete TTS audio streams without losing samples."""

import asyncio
import io
import wave
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
import pytest

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test

_OUTPUT_RATE = 24000


def _pcm(rate: int, samples: int) -> bytes:
    return (np.sin(2 * np.pi * 220 * np.arange(samples) / rate) * 8000).astype(np.int16).tobytes()


async def _chunks(data: bytes, size: int, gap: float = 0) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        if start and gap:
            await asyncio.sleep(gap)
        yield data[start : start + size]


class IteratorTTSService(TTSService):
    def __init__(self, data: bytes, rate: int, *, wav: bool = False, gap: float = 0):
        super().__init__(
            sample_rate=_OUTPUT_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            reuse_context_id_within_turn=False,
            settings=TTSSettings(model=None, voice=None, language=None),
        )
        self._data = data
        self._rate = rate
        self._wav = wav
        self._gap = gap

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async for frame in self._stream_audio_frames_from_iterator(
            _chunks(self._data, 11025, self._gap),
            strip_wav_header=self._wav,
            in_sample_rate=None if self._wav else self._rate,
            context_id=context_id,
        ):
            yield frame


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rate,wav,gap,samples",
    [
        (22050, False, 0, 22050),
        (22050, True, 0, 22050),
        (48000, False, 0, 48000),
        (24000, False, 0, 24000),
        (22050, False, 0.25, 22050),
        (22050, False, 0, 441),
        (22050, False, 0, 0),
    ],
)
async def test_tts_iterator_preserves_complete_audio(rate, wav, gap, samples):
    data = _pcm(rate, samples)
    if wav:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(data)
        data = buffer.getvalue()
    service = IteratorTTSService(data, rate, wav=wav, gap=gap)

    down, _ = await run_test(service, frames_to_send=[TTSSpeakFrame("Hello.")])

    audio = [frame for frame in down if isinstance(frame, TTSAudioRawFrame)]
    assert sum(len(frame.audio) for frame in audio) // 2 == round(samples * _OUTPUT_RATE / rate)
    assert all(frame.sample_rate == _OUTPUT_RATE and frame.num_channels == 1 for frame in audio)
    assert all(len(frame.audio) % 2 == 0 for frame in audio)
    if audio:
        stopped = next(i for i, frame in enumerate(down) if isinstance(frame, TTSStoppedFrame))
        assert all(
            i < stopped for i, frame in enumerate(down) if isinstance(frame, TTSAudioRawFrame)
        )
        assert all(frame.context_id == down[stopped].context_id for frame in audio)
        if rate == _OUTPUT_RATE:
            assert b"".join(frame.audio for frame in audio) == data


@pytest.mark.asyncio
async def test_tts_iterators_complete_each_utterance_independently():
    service = IteratorTTSService(_pcm(22050, 22050), 22050)

    down, _ = await run_test(
        service, frames_to_send=[TTSSpeakFrame("First."), TTSSpeakFrame("Second.")]
    )

    samples_by_context = Counter()
    for frame in down:
        if isinstance(frame, TTSAudioRawFrame):
            samples_by_context[frame.context_id] += len(frame.audio) // 2
    assert len(samples_by_context) == 2
    assert list(samples_by_context.values()) == [_OUTPUT_RATE, _OUTPUT_RATE]


@pytest.mark.asyncio
async def test_abandoned_tts_iterator_does_not_leak_audio_into_the_next_stream():
    service = IteratorTTSService(b"", 22050)
    service._sample_rate = _OUTPUT_RATE
    abandoned = service._stream_audio_frames_from_iterator(
        _chunks(_pcm(22050, 22050), 11025), in_sample_rate=22050, context_id="abandoned"
    )
    await anext(abandoned)
    await abandoned.aclose()

    # The next stream has no input, so it must not emit the abandoned tail.
    frames = [
        frame
        async for frame in service._stream_audio_frames_from_iterator(
            _chunks(b"", 11025), in_sample_rate=22050, context_id="next"
        )
    ]
    assert frames == []


@pytest.mark.asyncio
async def test_interleaved_tts_iterators_keep_separate_resampling_state():
    service = IteratorTTSService(b"", 22050)
    service._sample_rate = _OUTPUT_RATE
    first = service._stream_audio_frames_from_iterator(
        _chunks(_pcm(22050, 22050), 11025), in_sample_rate=22050, context_id="first"
    )
    second = service._stream_audio_frames_from_iterator(
        _chunks(_pcm(48000, 48000), 11025), in_sample_rate=48000, context_id="second"
    )

    first_frames = [await anext(first)]
    second_frames = [await anext(second)]
    first_frames.extend([frame async for frame in first])
    second_frames.extend([frame async for frame in second])

    for frames, context_id in [(first_frames, "first"), (second_frames, "second")]:
        assert sum(len(frame.audio) for frame in frames) // 2 == _OUTPUT_RATE
        assert all(frame.context_id == context_id for frame in frames)
