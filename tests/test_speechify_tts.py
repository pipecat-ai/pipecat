#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Speechify
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SpeechifyTTSService."""

import base64
import inspect
import unittest
from types import SimpleNamespace
from typing import Any

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.speechify import tts as speechify_tts
from pipecat.services.speechify.tts import (
    SpeechifyTTSService,
    SpeechifyTTSSettings,
    language_to_speechify_language,
)
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language


def _fake_response(pcm: bytes, words: list[tuple[str | None, int | None, int | None]]):
    """Build a stand-in for the Speechify speech response."""
    chunks = [SimpleNamespace(value=v, start_time=s, end_time=e) for (v, s, e) in words]
    return SimpleNamespace(
        audio_data=base64.b64encode(pcm).decode(),
        speech_marks=SimpleNamespace(chunks=chunks),
    )


class _FakeAudio:
    """Minimal Speechify audio client stand-in that records speech calls."""

    def __init__(self, response: Any = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def speech(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


class _FakeClient:
    """Minimal Speechify client stand-in."""

    def __init__(self, response: Any = None, error: Exception | None = None):
        self.audio = _FakeAudio(response, error)


def _service(response: Any, **kwargs: Any) -> SpeechifyTTSService:
    # Pin sample_rate to Speechify's native rate so the harness performs no
    # resampling and audio bytes pass through unchanged.
    return SpeechifyTTSService(client=_FakeClient(response), sample_rate=24000, **kwargs)


def test_word_times_converts_ms_to_seconds():
    svc = _service(_fake_response(b"\x00\x00", []))
    marks = SimpleNamespace(
        chunks=[
            SimpleNamespace(value="Hello", start_time=0, end_time=500),
            SimpleNamespace(value="world", start_time=500, end_time=1000),
        ]
    )

    word_times, utterance_end = svc._word_times(marks)

    assert word_times == [("Hello", 0.0), ("world", 0.5)]
    assert utterance_end == 1.0


def test_word_times_applies_cumulative_offset():
    svc = _service(_fake_response(b"\x00\x00", []))
    svc._cumulative_time = 2.0
    marks = SimpleNamespace(chunks=[SimpleNamespace(value="again", start_time=250, end_time=750)])

    word_times, utterance_end = svc._word_times(marks)

    assert word_times == [("again", 2.25)]
    # utterance_end is relative to this utterance, not cumulative.
    assert utterance_end == 0.75


def test_word_times_skips_empty_values_and_missing_starts():
    svc = _service(_fake_response(b"\x00\x00", []))
    marks = SimpleNamespace(
        chunks=[
            SimpleNamespace(value=None, start_time=0, end_time=100),
            SimpleNamespace(value="", start_time=100, end_time=200),
            SimpleNamespace(value="missing", start_time=None, end_time=300),
            SimpleNamespace(value="ok", start_time=200, end_time=300),
        ]
    )

    word_times, _ = svc._word_times(marks)

    assert word_times == [("ok", 0.2)]


def test_request_kwargs_defaults_omit_options():
    svc = _service(_fake_response(b"\x00\x00", []))

    kwargs = svc._speech_request_kwargs("hi")

    assert kwargs == {
        "audio_format": "pcm",
        "input": "hi",
        "voice_id": "dominic_32",
        "model": "simba-3.2",
    }


def test_request_kwargs_includes_options_only_when_set():
    svc = _service(
        _fake_response(b"\x00\x00", []),
        language="en-US",
        settings=SpeechifyTTSSettings(loudness_normalization=True, text_normalization=False),
    )

    kwargs = svc._speech_request_kwargs("hi")

    assert kwargs["language"] == "en-US"
    assert kwargs["options"].loudness_normalization is True
    assert kwargs["options"].text_normalization is False


def test_language_helper_maps_base_and_regional_languages():
    # Base language -> Speechify's default locale for that language.
    assert language_to_speechify_language(Language.EN) == "en-US"
    assert language_to_speechify_language(Language.PT) == "pt-BR"
    assert language_to_speechify_language(Language.ES) == "es-MX"
    # A supported regional variant resolves to its own BCP-47 value.
    assert language_to_speechify_language(Language.EN_GB) == "en-GB"


def test_language_enum_is_converted_at_construction():
    # The base class normalizes a Language enum to a string via
    # language_to_service_language at construction time.
    svc = _service(_fake_response(b"\x00\x00", []), language=Language.EN)

    kwargs = svc._speech_request_kwargs("hi")

    assert kwargs["language"] == "en-US"


@pytest.mark.asyncio
async def test_run_tts_emits_audio_and_word_timestamps():
    pcm = b"\x01\x02" * 2400  # 0.1s of 24 kHz mono 16-bit PCM
    response = _fake_response(pcm, [("Hello", 0, 400), ("world", 400, 800)])
    svc = _service(response)

    down, _up = await run_test(svc, frames_to_send=[TTSSpeakFrame("Hello world.")])

    assert any(isinstance(f, TTSStartedFrame) for f in down)
    assert any(isinstance(f, TTSStoppedFrame) for f in down)

    spoken = [f.text for f in down if isinstance(f, TTSTextFrame)]
    assert spoken == ["Hello", "world"]

    audio = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    assert audio == pcm  # 24 kHz in, 24 kHz out: bytes pass through unchanged.

    call = svc._client.audio.calls[0]
    assert call["input"] == "Hello world."
    assert call["audio_format"] == "pcm"


def test_service_constructs_real_client_from_api_key():
    # Exercises the real AsyncSpeechify constructor (construction makes no
    # network call). Guards against SDK constructor-signature drift.
    from speechify.client import AsyncSpeechify

    svc = SpeechifyTTSService(api_key="sk_test_dummy", sample_rate=24000)

    assert isinstance(svc._client, AsyncSpeechify)


def test_service_requires_api_key(monkeypatch):
    monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)

    with pytest.raises(ValueError):
        SpeechifyTTSService(sample_rate=24000)


@pytest.mark.asyncio
async def test_run_tts_yields_error_frame_on_api_error(monkeypatch):
    class FakeApiError(Exception):
        """Test double for speechify.core.api_error.ApiError."""

        status_code = 401
        body = {"message": "unauthorized"}

    monkeypatch.setattr(speechify_tts, "ApiError", FakeApiError)
    svc = SpeechifyTTSService(
        client=_FakeClient(error=FakeApiError("unauthorized")),
        sample_rate=24000,
    )

    _down, up = await run_test(svc, frames_to_send=[TTSSpeakFrame("Hello.")])

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "Speechify API error" in errors[0].error
    assert "unauthorized" in errors[0].error


def test_run_tts_is_async_generator():
    svc = _service(_fake_response(b"\x00\x00", []))

    generator = svc.run_tts("Hello.", "ctx-1")

    assert inspect.isasyncgen(generator)


if __name__ == "__main__":
    unittest.main()
