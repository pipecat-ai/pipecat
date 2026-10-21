#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestSTTService's Pulse 2.0 model support."""

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest

from pipecat.frames.frames import TranscriptionFrame, TranscriptionSpeaker
from pipecat.services.smallest.stt import SmallestSTTModel, SmallestSTTService
from pipecat.services.stt_service import STTSettings
from pipecat.transcriptions.language import Language


def test_default_model_is_pulse():
    service = SmallestSTTService(api_key="test-key")
    assert service._settings.model == SmallestSTTModel.PULSE.value


def test_pulse_2_defaults_to_english():
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(model=SmallestSTTModel.PULSE_2.value),
    )
    assert service._settings.language == "en"


def test_pulse_2_accepts_regional_english_string():
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(
            model=SmallestSTTModel.PULSE_2.value, language="en-US"
        ),
    )
    assert service._settings.language == "en"


def test_pulse_2_rejects_non_english_language_at_construction():
    with pytest.raises(ValueError, match="English-only"):
        SmallestSTTService(
            api_key="test-key",
            settings=SmallestSTTService.Settings(
                model=SmallestSTTModel.PULSE_2.value, language=Language.HI
            ),
        )


def test_pulse_rejects_english_only_restriction_does_not_apply():
    """Non-English languages are fine on the default `pulse` model."""
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(language=Language.HI),
    )
    assert service._settings.language == "hi"


@pytest.mark.asyncio
async def test_switching_to_pulse_2_with_non_english_language_raises():
    service = SmallestSTTService(api_key="test-key")

    with pytest.raises(ValueError, match="English-only"):
        await service._update_settings(
            STTSettings(model=SmallestSTTModel.PULSE_2.value, language=Language.HI)
        )


@pytest.mark.asyncio
async def test_switching_to_pulse_2_with_regional_english_string_succeeds():
    service = SmallestSTTService(api_key="test-key")
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._update_settings(
        STTSettings(model=SmallestSTTModel.PULSE_2.value, language="en-US")
    )
    assert service._settings.model == SmallestSTTModel.PULSE_2.value


def test_emotion_and_gender_detection_default_on():
    service = SmallestSTTService(api_key="test-key")
    assert service._settings.emotion_detection is True
    assert service._settings.gender_detection is True


async def _connect_query(service: SmallestSTTService) -> dict[str, list[str]]:
    """Return the query the service connects with, without opening a socket."""
    urls = []

    async def fake_websocket_connect(url, **kwargs):
        urls.append(url)
        raise ConnectionError("not connecting in tests")

    service._websocket_connect = fake_websocket_connect
    service.push_error = AsyncMock()
    service._call_event_handler = AsyncMock()
    service._sample_rate = 16000
    await service._connect_websocket()
    return parse_qs(urlparse(urls[0]).query)


@pytest.mark.asyncio
async def test_diarize_requests_word_timestamps():
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(diarize=True, speaker_format="{speaker}: {text}"),
    )

    query = await _connect_query(service)

    assert query["diarize"] == ["true"]
    assert query["word_timestamps"] == ["true"]
    assert "speaker_format" not in query


@pytest.mark.asyncio
async def test_diarized_final_emits_one_frame_per_speaker_run():
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(diarize=True, speaker_format="{speaker}: {text}"),
    )
    service._user_id = "session-user"
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame

    await service._process_response(
        {
            "transcript": "Hello, there. Hi.",
            "is_final": True,
            "words": [
                {"word": "Hello,", "speaker": 0},
                {"word": "there.", "speaker": 0},
                {"word": "Hi.", "speaker": 1},
            ],
        }
    )

    frames = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("0", 1), "0: Hello, there."),
        (TranscriptionSpeaker("1", 2), "1: Hi."),
    ]
    assert all(f.user_id == "session-user" for f in frames)


@pytest.mark.asyncio
async def test_speaker_format_update_does_not_reconnect():
    service = SmallestSTTService(api_key="test-key")
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    changed = await service._update_settings(
        SmallestSTTService.Settings(speaker_format="{speaker}: {text}")
    )

    assert "speaker_format" in changed
    service._disconnect.assert_not_awaited()
