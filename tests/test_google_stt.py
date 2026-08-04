#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Google STT streaming responses and adaptation handling."""

import time
from types import SimpleNamespace

import pytest
from google.cloud.speech_v2.types import cloud_speech

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.google.stt import (
    GoogleSTTService,
    google_stt_model_supports_adaptation,
    normalize_google_speech_adaptation,
)


class AsyncResponses:
    """Minimal async iterator for Google streaming responses."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def result(*, transcript: str, is_final: bool):
    return SimpleNamespace(
        alternatives=[SimpleNamespace(transcript=transcript)],
        is_final=is_final,
    )


@pytest.mark.asyncio
async def test_google_final_result_emits_finalized_transcription_frame():
    service = object.__new__(GoogleSTTService)
    service._stream_start_time = int(time.time() * 1000)
    service._user_id = "user"
    service._last_transcript_was_final = False
    service._get_language_codes = lambda: ["en-US"]
    service._stt_usage_pending_seconds = 0.0
    service._enable_usage_metrics = False

    frames = []
    transcriptions = []

    async def push_frame(frame):
        frames.append(frame)

    async def stop_processing_metrics():
        pass

    async def handle_transcription(transcript, is_final, language=None):
        transcriptions.append((transcript, is_final, language))

    service.push_frame = push_frame
    service.stop_processing_metrics = stop_processing_metrics
    service._handle_transcription = handle_transcription

    responses = AsyncResponses(
        [
            SimpleNamespace(results=[result(transcript="hel", is_final=False)]),
            SimpleNamespace(results=[result(transcript="hello", is_final=True)]),
        ]
    )

    await service._process_responses(responses)

    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].finalized is True
    assert transcriptions == [("hello", True, "en-US")]


def test_google_stt_model_supports_adaptation():
    assert google_stt_model_supports_adaptation("latest_long") is True
    assert google_stt_model_supports_adaptation("telephony") is False
    assert google_stt_model_supports_adaptation("TELEPHONY") is False
    assert google_stt_model_supports_adaptation(None) is True


def test_normalize_google_speech_adaptation_accepts_native_message():
    adaptation = cloud_speech.SpeechAdaptation()

    normalized = normalize_google_speech_adaptation(adaptation)

    assert normalized is adaptation


def test_normalize_google_speech_adaptation_converts_phrase_set_references():
    normalized = normalize_google_speech_adaptation(
        {
            "phrase_set_references": [
                "projects/test/locations/global/phraseSets/support-terms",
            ]
        }
    )

    assert len(normalized.phrase_sets) == 1
    assert normalized.phrase_sets[0].phrase_set == (
        "projects/test/locations/global/phraseSets/support-terms"
    )


@pytest.mark.parametrize("field", ["phrase_set_references", "phrase_sets"])
def test_normalize_google_speech_adaptation_accepts_single_phrase_set_string(field):
    phrase_set = "projects/test/locations/global/phraseSets/support-terms"

    normalized = normalize_google_speech_adaptation({field: phrase_set})

    assert len(normalized.phrase_sets) == 1
    assert normalized.phrase_sets[0].phrase_set == phrase_set


def test_normalize_google_speech_adaptation_converts_string_and_inline_phrase_sets():
    normalized = normalize_google_speech_adaptation(
        {
            "phrase_sets": [
                "projects/test/locations/global/phraseSets/catalog",
                {
                    "phrases": [
                        {"value": "pipecat", "boost": 15.0},
                        {"value": "voice pipeline"},
                    ]
                },
            ]
        }
    )

    assert normalized.phrase_sets[0].phrase_set == (
        "projects/test/locations/global/phraseSets/catalog"
    )
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[0].value == "pipecat"
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[0].boost == 15.0
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[1].value == "voice pipeline"


def test_normalize_google_speech_adaptation_rejects_invalid_phrase_set_entries():
    with pytest.raises(ValueError, match="Invalid Google SpeechAdaptation phrase_set entry"):
        normalize_google_speech_adaptation({"phrase_sets": [123]})


def test_google_stt_rejects_invalid_adaptation_during_initialization():
    settings = GoogleSTTService.Settings(adaptation={"phrase_sets": [{"phrases": ["hello"]}]})

    with pytest.raises(TypeError, match="expected.*Phrase.*got.*str"):
        GoogleSTTService(settings=settings)


@pytest.mark.asyncio
async def test_google_stt_rejects_invalid_runtime_adaptation_before_commit():
    service = object.__new__(GoogleSTTService)
    service._settings = GoogleSTTService.Settings(adaptation=None)
    delta = GoogleSTTService.Settings(adaptation={"phrase_sets": [{"phrases": ["hello"]}]})

    with pytest.raises(TypeError, match="expected.*Phrase.*got.*str"):
        await service._update_settings(delta)

    assert service._settings.adaptation is None
