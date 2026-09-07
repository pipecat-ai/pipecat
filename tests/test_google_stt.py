#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Google STT streaming responses and adaptation handling."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from google.cloud.speech_v2.types import cloud_speech

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.google.stt import (
    GoogleSTTService,
    _normalize_speech_adaptation,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


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
    service._setup = frame_processor_setup(TaskManager(), enable_usage_metrics=False)

    frames = []
    transcriptions = []

    async def push_frame(frame):
        frames.append(frame)

    async def handle_transcription(transcript, is_final, language=None):
        transcriptions.append((transcript, is_final, language))

    service.push_frame = push_frame
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


def test_normalize_speech_adaptation_accepts_native_message():
    adaptation = cloud_speech.SpeechAdaptation()

    normalized = _normalize_speech_adaptation(adaptation)

    assert normalized is adaptation


def test_normalize_speech_adaptation_accepts_single_phrase_set_string():
    phrase_set = "projects/test/locations/global/phraseSets/support-terms"

    normalized = _normalize_speech_adaptation({"phrase_sets": phrase_set})

    assert len(normalized.phrase_sets) == 1
    assert normalized.phrase_sets[0].phrase_set == phrase_set


def test_normalize_speech_adaptation_accepts_single_inline_phrase_set():
    normalized = _normalize_speech_adaptation({"phrase_sets": {"phrases": [{"value": "pipecat"}]}})

    assert len(normalized.phrase_sets) == 1
    assert normalized.phrase_sets[0].inline_phrase_set.phrases[0].value == "pipecat"


def test_normalize_speech_adaptation_converts_string_and_inline_phrase_sets():
    normalized = _normalize_speech_adaptation(
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


def test_normalize_speech_adaptation_rejects_invalid_phrase_set_entries():
    with pytest.raises(ValueError, match="Invalid Google SpeechAdaptation phrase_set entry"):
        _normalize_speech_adaptation({"phrase_sets": [123]})


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


async def connected_recognition_config(adaptation, model="latest_long"):
    """Run _connect() on a bare service and return the config it built."""
    service = object.__new__(GoogleSTTService)
    service._settings = GoogleSTTService.Settings(
        model=model,
        enable_automatic_punctuation=True,
        enable_spoken_punctuation=False,
        enable_spoken_emojis=False,
        profanity_filter=False,
        enable_word_time_offsets=False,
        enable_word_confidence=False,
        enable_interim_results=True,
        enable_voice_activity_events=False,
        adaptation=adaptation,
    )
    service._sample_rate = 16000
    service._get_language_codes = lambda: ["en-US"]
    service._call_event_handler = lambda *args, **kwargs: asyncio.sleep(0)

    def create_task(coro):
        coro.close()

    service.create_task = create_task

    await service._connect()

    return service._config.config


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["latest_long", "short", "chirp_2", "telephony_short"])
async def test_google_connect_sends_adaptation_for_supporting_models(model):
    phrase_set = "projects/test/locations/global/phraseSets/catalog"

    config = await connected_recognition_config({"phrase_sets": [phrase_set]}, model=model)

    assert config.model == model
    assert config.adaptation.phrase_sets[0].phrase_set == phrase_set


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["telephony", "TELEPHONY"])
async def test_google_connect_omits_adaptation_for_the_telephony_model(model):
    phrase_set = "projects/test/locations/global/phraseSets/catalog"

    config = await connected_recognition_config({"phrase_sets": [phrase_set]}, model=model)

    assert "adaptation" not in config


@pytest.mark.asyncio
async def test_google_connect_leaves_adaptation_unset_when_not_configured():
    config = await connected_recognition_config(None)

    assert "adaptation" not in config
