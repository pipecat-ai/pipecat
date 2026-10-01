#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for keeping TTS audio frames aligned to whole 16-bit samples.

Providers may cut their PCM stream at any byte, so a chunk can end mid-sample.
TTSService holds the partial sample back and prepends it to the context's next
frame, so every audio frame it emits holds whole samples.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.audio.utils import detect_speech_onset
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 16000
_PCM = bytes(i % 251 for i in range(4096))


def _split(data: bytes, sizes: list[int]) -> list[bytes]:
    chunks, pos = [], 0
    for size in sizes:
        chunks.append(data[pos : pos + size])
        pos += size
    return chunks


class MockTTSService(TTSService):
    """HTTP-style TTS service that yields each utterance as the given chunks."""

    def __init__(self, chunks: list[bytes], num_channels: int = 1, delay_s: float = 0, **kwargs):
        super().__init__(
            push_text_frames=False, push_stop_frames=True, sample_rate=_SAMPLE_RATE, **kwargs
        )
        self._chunks = chunks
        self._num_channels = num_channels
        self._delay_s = delay_s
        self.yielded_frames: list[TTSAudioRawFrame] = []

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        for chunk in self._chunks:
            frame = TTSAudioRawFrame(chunk, _SAMPLE_RATE, self._num_channels, context_id=context_id)
            self.yielded_frames.append(frame)
            yield frame
            if self._delay_s:
                await asyncio.sleep(self._delay_s)


async def _run(service: TTSService, *frames: Frame) -> list[Frame]:
    down_frames, _ = await run_test(
        service,
        frames_to_send=[TTSSpeakFrame(text="Hello."), *frames],
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    return down_frames


async def _speak(service: TTSService, *frames: Frame) -> list[TTSAudioRawFrame]:
    return [f for f in await _run(service, *frames) if isinstance(f, TTSAudioRawFrame)]


@pytest.mark.asyncio
async def test_split_samples_are_realigned():
    # Odd chunk sizes, including a single byte, totaling an even length.
    service = MockTTSService(_split(_PCM, [1023, 1, 2047, 1, 1024]))

    audio_frames = await _speak(service)

    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert all(f.num_frames == len(f.audio) // 2 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == _PCM


@pytest.mark.asyncio
async def test_aligned_frames_pass_through_untouched():
    service = MockTTSService(_split(_PCM, [1024, 2048, 1024]))

    audio_frames = await _speak(service)

    assert [f.id for f in audio_frames] == [f.id for f in service.yielded_frames]
    assert [f.audio for f in audio_frames] == [f.audio for f in service.yielded_frames]


@pytest.mark.asyncio
async def test_stereo_frames_align_to_whole_sample_pairs():
    service = MockTTSService(_split(_PCM, [1022, 3, 2049, 1022]), num_channels=2)

    audio_frames = await _speak(service)

    assert all(len(f.audio) % 4 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == _PCM


@pytest.mark.asyncio
async def test_trailing_partial_sample_is_padded_at_end_of_context():
    service = MockTTSService(_split(_PCM[:2047], [1023, 1024]))

    down_frames = await _run(service)

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == _PCM[:2047] + b"\x00"
    # The padded sample is part of the utterance, so it plays before the stop frame.
    last_audio = max(i for i, f in enumerate(down_frames) if isinstance(f, TTSAudioRawFrame))
    stopped = next(i for i, f in enumerate(down_frames) if isinstance(f, TTSStoppedFrame))
    assert last_audio < stopped
    assert service._audio_remainders == {}


@pytest.mark.asyncio
async def test_context_timeout_drops_partial_sample():
    # The response stalls mid-sample until its audio context times out.
    service = MockTTSService([_PCM[:1023]], delay_s=0.3, stop_frame_timeout_s=0.1)

    await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            TextFrame("Hello."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
    )

    assert service._audio_remainders == {}


@pytest.mark.asyncio
async def test_interruption_discards_partial_sample():
    service = MockTTSService(_split(_PCM, [1023, 1024, 2049]), delay_s=0.05)

    await _speak(service, SleepFrame(sleep=0.02), InterruptionFrame(), SleepFrame(sleep=0.2))

    assert service._audio_remainders == {}


def test_speech_onset_tolerates_partial_trailing_sample():
    silence = b"\x00\x00" * 1600
    speech = (b"\x00\x40" + b"\x00\xc0") * 1600

    onset = detect_speech_onset(silence + speech + b"\x01", _SAMPLE_RATE)

    assert onset is not None
