#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for KokoroTTSService chunked streaming.

The service module is imported with ``pytest.importorskip`` so the suite is
skipped when the kokoro extra is not installed. The Kokoro model is replaced
with a stand-in that records what it is asked to synthesize, so the chunking
is checked without the real model; the splitter itself is kokoro-onnx's own,
exercised for real.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pipecat.services.kokoro.tts", exc_type=ImportError)

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame  # noqa: E402
from pipecat.services.kokoro.tts import KokoroTTSService  # noqa: E402

SAMPLE_RATE = 24000
CHUNK_SAMPLES = 1000


class _FakeKokoro:
    """Stand-in for ``kokoro_onnx.Kokoro`` that records requests, never synthesizes."""

    def __init__(self, phonemes: str):
        self.voices = {"af_heart": object()}
        self._phonemes = " ".join(phonemes.split())
        self.phonemized: list[str] = []
        self.stream_calls: list[dict] = []
        self.created: list[dict] = []
        tokenizer = MagicMock()
        tokenizer.phonemize.side_effect = self._phonemize
        tokenizer.tokenize.side_effect = lambda piece: piece.strip()
        self.tokenizer = tokenizer

    def _phonemize(self, text, lang):
        self.phonemized.append(text)
        return self._phonemes

    def create(self, text, *, voice, speed, lang):
        self.created.append({"text": text, "voice": voice, "speed": speed, "lang": lang})

    def create_stream(self, text, *, voice, lang, speed, is_phonemes=False):
        self.stream_calls.append({"text": text, "is_phonemes": is_phonemes})

        async def _stream():
            yield np.zeros(CHUNK_SAMPLES, dtype=np.float32), SAMPLE_RATE

        return _stream()


class _FakeResampler:
    """Pass-through resampler that records the rates it was asked to convert."""

    def __init__(self):
        self.rates: list[tuple[int, int]] = []

    async def resample(self, audio, in_rate, out_rate):
        self.rates.append((in_rate, out_rate))
        return audio

    async def flush(self):
        return b""


def _build(fake: _FakeKokoro, stop_frame_timeout_s: float | None = None, **settings):
    with (
        patch("pipecat.services.kokoro.tts._ensure_model_files"),
        patch("pipecat.services.kokoro.tts.Kokoro", return_value=fake),
        patch("pipecat.services.kokoro.tts.create_stream_resampler", return_value=_FakeResampler()),
    ):
        kwargs = (
            {} if stop_frame_timeout_s is None else {"stop_frame_timeout_s": stop_frame_timeout_s}
        )
        service = KokoroTTSService(
            settings=KokoroTTSService.Settings(voice="af_heart", **settings), **kwargs
        )
    return service


async def _run(service, text):
    frames = []
    async for frame in service.run_tts(text, "ctx"):
        frames.append(frame)
    return frames


def test_construction_defaults_chunk_and_timeout():
    """The defaults exist to keep the first audio one short inference away."""
    service = _build(_FakeKokoro("x"), warmup=False)
    assert service._settings.stream_chunk_phonemes == 60
    assert service._stop_frame_timeout_s == 10.0


def test_stop_frame_timeout_is_overridable():
    service = _build(_FakeKokoro("x"), warmup=False, stop_frame_timeout_s=5.0)
    assert service._stop_frame_timeout_s == 5.0


@pytest.mark.asyncio
async def test_long_text_is_phonemized_once_and_streamed_as_chunks():
    """One phonemization, then one phoneme stream per chunk, in text order."""
    # 3 atoms under a 4-phoneme cap: "aaa." (sentence), "bbb," (clause), "ccc".
    fake = _FakeKokoro("aaa.  bbb,  ccc")
    service = _build(fake, warmup=False, stream_chunk_phonemes=4)

    frames = await _run(service, "irrelevant, the fake phonemizes to a constant")

    assert fake.phonemized == ["irrelevant, the fake phonemizes to a constant"]
    assert [call["text"] for call in fake.stream_calls] == ["aaa.", "bbb,", "ccc"]
    assert all(call["is_phonemes"] for call in fake.stream_calls)
    assert len(frames) == 3
    assert all(isinstance(f, TTSAudioRawFrame) for f in frames)


@pytest.mark.asyncio
async def test_pauses_are_spliced_between_chunks_but_not_after_the_last():
    """Each chunk is the last batch of its own stream, so the pause its
    trailing punctuation calls for has to be spliced in — except on the final
    chunk, whose trailing pause belongs to the next utterance.
    """
    fake = _FakeKokoro("aaa.  bbb,  ccc")
    service = _build(fake, warmup=False, stream_chunk_phonemes=4)

    frames = await _run(service, "text")

    # 1000 samples of audio per chunk + pause samples at 24000 Hz:
    # "aaa." -> 0.25s = 6000, "bbb," -> 0.1s = 2400, "ccc" -> none (last).
    sizes = [len(f.audio) // 2 for f in frames]
    assert sizes == [CHUNK_SAMPLES + 6000, CHUNK_SAMPLES + 2400, CHUNK_SAMPLES]


@pytest.mark.asyncio
async def test_short_text_stays_a_single_stream():
    """A short text is one chunk; nothing changes for it."""
    fake = _FakeKokoro("hi there")
    service = _build(fake, warmup=False)

    frames = await _run(service, "hi there")

    assert [call["text"] for call in fake.stream_calls] == ["hi there"]
    assert len(frames) == 1
    assert len(frames[0].audio) == CHUNK_SAMPLES * 2  # no pause spliced


@pytest.mark.asyncio
async def test_no_phonemes_surfaces_as_error_frame():
    service = _build(_FakeKokoro(""), warmup=False)

    frames = await _run(service, "   ")

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "Nothing to synthesize" in frames[0].error


@pytest.mark.asyncio
async def test_warmup_synthesizes_once_at_construction():
    """The warm-up primes espeak-ng and the first ONNX inference off the
    critical path, so the first reply doesn't pay for them.
    """
    fake = _FakeKokoro("x")
    _build(fake)  # warmup defaults to True

    deadline = time.monotonic() + 5
    while not fake.created and time.monotonic() < deadline:
        await asyncio.sleep(0.05)

    assert len(fake.created) == 1
    assert fake.created[0]["text"] == "Ready."


def test_warmup_can_be_disabled():
    with (
        patch("pipecat.services.kokoro.tts._ensure_model_files"),
        patch("pipecat.services.kokoro.tts.Kokoro", return_value=_FakeKokoro("x")),
        patch("pipecat.services.kokoro.tts.create_stream_resampler"),
        patch("pipecat.services.kokoro.tts.threading.Thread") as thread,
    ):
        KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart", warmup=False))
    thread.assert_not_called()
