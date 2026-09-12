#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Azure Voice Live service configuration."""

import pytest

from pipecat.services.azure.voicelive import events
from pipecat.services.azure.voicelive.llm import AzureVoiceLiveLLMService


def _service(**kwargs) -> AzureVoiceLiveLLMService:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("endpoint", "https://my-resource.services.ai.azure.com")
    return AzureVoiceLiveLLMService(**kwargs)


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        (
            "https://my-resource.services.ai.azure.com",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "https://my-resource.services.ai.azure.com/",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "my-resource.services.ai.azure.com",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "https://my-resource.cognitiveservices.azure.com",
            "wss://my-resource.cognitiveservices.azure.com/voice-live/realtime",
        ),
    ],
)
def test_endpoint_is_normalized_to_a_websocket_url(endpoint, expected):
    """The portal shows an https resource endpoint, not the realtime URL."""
    assert _service(endpoint=endpoint).base_url == expected


def test_credentials_are_required():
    with pytest.raises(ValueError):
        AzureVoiceLiveLLMService(endpoint="https://my-resource.services.ai.azure.com")


def test_token_provider_is_accepted_without_an_api_key():
    async def provider() -> str:
        return "token"

    service = AzureVoiceLiveLLMService(
        endpoint="https://my-resource.services.ai.azure.com",
        token_provider=provider,
    )

    assert service.api_key is None


def test_voice_shorthand_populates_session_properties():
    service = _service(voice="en-US-Andrew:DragonHDLatestNeural")

    voice = service._settings.session_properties.voice
    assert isinstance(voice, events.AzureStandardVoice)
    assert voice.name == "en-US-Andrew:DragonHDLatestNeural"


def test_server_turn_detection_is_on_by_default():
    assert _service()._is_manual_turn_detection() is False


def test_turn_detection_none_selects_manual_mode():
    """Callers driving turns from transport VAD disable the server's own."""
    service = _service(
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(turn_detection=None)
        )
    )

    assert service._is_manual_turn_detection() is True


@pytest.mark.parametrize(
    "output_rate, expected_format",
    [(8000, "pcm16_8000hz"), (16000, "pcm16_16000hz"), (24000, "pcm16")],
)
def test_output_sample_rate_selects_the_matching_format(output_rate, expected_format):
    """Voice Live picks the output rate through the format, not a rate field."""
    service = _service()

    service._ensure_audio_config(16000, output_rate)

    assert service._settings.session_properties.output_audio_format == expected_format
    assert service._get_output_sample_rate() == output_rate


def test_unsupported_output_sample_rate_falls_back_to_24khz():
    service = _service()

    service._ensure_audio_config(16000, 44100)

    assert service._settings.session_properties.output_audio_format == "pcm16"
    assert service._get_output_sample_rate() == 24000


def test_input_sample_rate_is_sent_as_configured():
    service = _service()

    service._ensure_audio_config(8000, 24000)

    assert service._settings.session_properties.input_audio_sampling_rate == 8000


def test_settings_from_mapping_routes_session_keys():
    """Plain dicts of session keys land in session_properties, not extra."""
    settings = AzureVoiceLiveLLMService.Settings.from_mapping(
        {"model": "gpt-realtime", "temperature": 0.7, "modalities": ["audio"]}
    )

    assert settings.model == "gpt-realtime"
    assert settings.session_properties.modalities == ["audio"]
    assert not settings.extra


def test_turn_detection_false_also_selects_manual_mode():
    service = _service(
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(turn_detection=False)
        )
    )

    assert service._is_manual_turn_detection() is True
