#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWSPollyTTSService word-timestamp handling (issue #1015).

All Polly calls are mocked, so no live AWS credentials are needed.
"""

import json
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from pipecat.frames.frames import TTSStoppedFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.tts_service import TTSContext, TTSService

CTX = "ctx-1"

# A live SpeechMarks response (newline-delimited JSON) captured from Polly for
# "Hello world from Pipecat". `time` is the word start offset in milliseconds;
# `start`/`end` are byte offsets into the input text (unused).
MARKS_SAMPLE = (
    b'{"time":6,"type":"word","start":0,"end":5,"value":"Hello"}\n'
    b'{"time":273,"type":"word","start":6,"end":11,"value":"world"}\n'
    b'{"time":612,"type":"word","start":12,"end":16,"value":"from"}\n'
    b'{"time":794,"type":"word","start":17,"end":24,"value":"Pipecat"}\n'
)


# --------------------------------------------------------------------------- #
# Test doubles for the aiobotocore Polly client
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Stands in for a Polly StreamingBody with an async read()."""

    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakePolly:
    """Fake Polly client: returns marks for json output, audio otherwise."""

    def __init__(self, pcm: bytes = b"", marks: bytes = b""):
        self._pcm = pcm
        self._marks = marks
        self.calls: list[dict] = []

    async def synthesize_speech(self, **params):
        self.calls.append(params)
        if params.get("OutputFormat") == "json":
            return {"AudioStream": _FakeStream(self._marks)}
        return {"AudioStream": _FakeStream(self._pcm)}


class _FakeClientCtx:
    """Async context manager returned by session.create_client()."""

    def __init__(self, polly):
        self._polly = polly

    async def __aenter__(self):
        return self._polly

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Fake aiobotocore session whose create_client yields a fake Polly."""

    def __init__(self, polly):
        self._polly = polly

    def create_client(self, *args, **kwargs):
        return _FakeClientCtx(self._polly)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_service(**kwargs) -> AWSPollyTTSService:
    # Explicit credentials short-circuit the botocore chain (no network).
    return AWSPollyTTSService(
        api_key="test-secret",
        aws_access_key_id="test-key",
        region="us-east-1",
        **kwargs,
    )


def _pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    """Return `seconds` of 16-bit mono silence at the given sample rate."""
    return b"\x00" * int(seconds * sample_rate * 2)


def _ndjson(words: list[tuple[str, int]]) -> bytes:
    """Build newline-delimited Polly word marks from (value, time_ms) pairs."""
    lines = [json.dumps({"time": t, "type": "word", "value": w}).encode() for w, t in words]
    return b"\n".join(lines) + b"\n"


def _stub_for_run_tts(service: AWSPollyTTSService):
    """Stub sample rate, resampler and metrics so run_tts can run offline."""
    service._sample_rate = 16000

    async def _aresample(data, in_rate, out_rate):
        return data

    async def _anoop(*args, **kwargs):
        pass

    service._resampler.resample = _aresample
    service.start_tts_usage_metrics = _anoop
    service.stop_ttfb_metrics = _anoop


def _capture_word_timestamps(service: AWSPollyTTSService) -> list:
    """Replace add_word_timestamps with a capturing stub; return the capture."""
    captured: list = []

    async def _fake(word_times, context_id=None, **kwargs):
        captured.extend(word_times)

    service.add_word_timestamps = _fake
    return captured


async def _drain(service: AWSPollyTTSService, text: str):
    return [frame async for frame in service.run_tts(text, CTX)]


# --------------------------------------------------------------------------- #
# Constructor wiring
# --------------------------------------------------------------------------- #


def test_word_timestamps_enabled_by_default():
    """Word timestamps are on by default and activate the per-word path."""
    service = _make_service()
    assert service._word_timestamps is True
    # Word events produce the TTSTextFrames, so the base must not push whole text.
    assert service._push_text_frames is False
    assert service._cumulative_time == 0.0


def test_word_timestamps_disabled_pushes_whole_text():
    """Disabling word timestamps flips the service back to whole-text frames."""
    service = _make_service(word_timestamps=False)
    assert service._word_timestamps is False
    assert service._push_text_frames is True


# --------------------------------------------------------------------------- #
# Speech-marks fetch + parse
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fetch_word_marks_parses_ndjson():
    """Newline-delimited Polly marks parse into (word, start_seconds) pairs."""
    service = _make_service()
    polly = _FakePolly(marks=MARKS_SAMPLE)

    result = await service._fetch_word_marks(
        polly, {"Text": "x", "VoiceId": "Joanna", "SampleRate": "16000"}
    )

    assert result == [
        ("Hello", pytest.approx(0.006)),
        ("world", pytest.approx(0.273)),
        ("from", pytest.approx(0.612)),
        ("Pipecat", pytest.approx(0.794)),
    ]
    # The marks call asks for json word marks and drops SampleRate.
    sent = polly.calls[0]
    assert sent["OutputFormat"] == "json"
    assert sent["SpeechMarkTypes"] == ["word"]
    assert "SampleRate" not in sent


@pytest.mark.asyncio
async def test_fetch_word_marks_returns_empty_on_failure():
    """A Polly error or a missing AudioStream yields [] and never raises."""
    service = _make_service()

    class _RaisingPolly:
        async def synthesize_speech(self, **params):
            raise ClientError({"Error": {"Code": "Boom", "Message": "boom"}}, "SynthesizeSpeech")

    class _NoStreamPolly:
        async def synthesize_speech(self, **params):
            return {}

    assert await service._fetch_word_marks(_RaisingPolly(), {}) == []
    assert await service._fetch_word_marks(_NoStreamPolly(), {}) == []


# --------------------------------------------------------------------------- #
# run_tts behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cumulative_offset_across_sentences():
    """A later sentence in the same turn is offset by the prior audio duration."""
    service = _make_service()
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)

    # Sentence 1: 1.0s of audio (32000 bytes), word "Hello" at 0 ms.
    service._aws_session = _FakeSession(_FakePolly(pcm=_pcm(1.0), marks=_ndjson([("Hello", 0)])))
    await _drain(service, "Hello")

    # Sentence 2: word "World" at 100 ms -> 0.1 + 1.0 (prior duration) = 1.1s.
    service._aws_session = _FakeSession(_FakePolly(pcm=_pcm(0.5), marks=_ndjson([("World", 100)])))
    await _drain(service, "World")

    assert captured == [("Hello", pytest.approx(0.0)), ("World", pytest.approx(1.1))]


@pytest.mark.asyncio
async def test_no_marks_pushes_fallback_text():
    """Empty marks emit a single whole-text frame and add no word timestamps."""
    service = _make_service()
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)
    service._tts_contexts[CTX] = TTSContext(append_to_context=True)

    pushed: list = []

    async def _fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    service.push_frame = _fake_push
    service._aws_session = _FakeSession(_FakePolly(pcm=_pcm(0.5), marks=b""))

    await _drain(service, "Hello world")

    assert captured == []
    text_frames = [f for f in pushed if isinstance(f, TTSTextFrame)]
    assert len(text_frames) == 1
    assert text_frames[0].text == "Hello world"
    assert text_frames[0].append_to_context is True


@pytest.mark.asyncio
async def test_word_timestamps_disabled_skips_marks_call():
    """With word timestamps off, run_tts never makes the json marks call."""
    service = _make_service(word_timestamps=False)
    _stub_for_run_tts(service)
    captured = _capture_word_timestamps(service)
    polly = _FakePolly(pcm=_pcm(0.5), marks=MARKS_SAMPLE)
    service._aws_session = _FakeSession(polly)

    await _drain(service, "Hello world")

    assert captured == []
    assert all(call.get("OutputFormat") != "json" for call in polly.calls)


# --------------------------------------------------------------------------- #
# Offset reset
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cumulative_time_resets_on_tts_stopped():
    """A TTSStoppedFrame (end of turn) clears the accumulated offset."""
    service = _make_service()

    async def _noop_super(self, frame, direction=FrameDirection.DOWNSTREAM):
        pass

    service._cumulative_time = 5.0
    # Stub the parent push so the override runs without a wired pipeline.
    with patch.object(TTSService, "push_frame", _noop_super):
        await service.push_frame(TTSStoppedFrame(context_id=CTX))

    assert service._cumulative_time == 0.0
