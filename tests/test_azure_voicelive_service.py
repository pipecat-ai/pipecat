#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Azure Voice Live service configuration."""

import io

import pytest
from loguru import logger
from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

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
        (
            "ws://my-resource.services.ai.azure.com",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "HTTPS://my-resource.services.ai.azure.com",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
        ),
        (
            "WSS://my-resource.services.ai.azure.com/voice-live/realtime",
            "wss://my-resource.services.ai.azure.com/voice-live/realtime",
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


@pytest.mark.parametrize("model_in_settings", [False, True], ids=["model-argument", "settings"])
def test_azure_realtime_is_left_to_pick_its_own_voice(model_in_settings):
    """`azure-realtime` rejects Azure standard voices and picks a native one when none is sent."""
    if model_in_settings:
        service = _service(settings=AzureVoiceLiveLLMService.Settings(model="azure-realtime"))
    else:
        service = _service(model="azure-realtime")

    assert service._settings.session_properties.voice is None


@pytest.mark.parametrize(
    "session_kwargs, manual",
    [
        ({}, False),
        ({"turn_detection": None}, True),
        ({"turn_detection": False}, True),
        ({"turn_detection": events.TurnDetection(type="azure_semantic_vad")}, False),
    ],
    ids=["unset", "none", "false", "configured"],
)
def test_manual_mode_matches_what_the_session_update_sends(session_kwargs, manual):
    """The service and the wire have to agree on who is detecting turns.

    An unset ``turn_detection`` is omitted from the session update, leaving
    Voice Live's own VAD running, so it is not manual mode.
    """
    service = _service(
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(**session_kwargs)
        )
    )
    session = service._settings.session_properties
    dump = events.SessionUpdateEvent(session=session.model_copy()).model_dump(exclude_none=True)
    wire_disables_detection = (
        "turn_detection" in dump["session"] and not (dump["session"]["turn_detection"])
    )

    assert service._is_manual_turn_detection() is manual
    assert wire_disables_detection is manual


@pytest.mark.asyncio
async def test_a_model_update_is_reported_as_unsupported():
    """The model is fixed by the connection URL, so an update can't take effect."""
    service = _service(model="gpt-4o-mini")
    sent = []

    async def _record(event):
        sent.append(event)

    service.send_client_event = _record

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._update_settings(
            AzureVoiceLiveLLMService.Settings.from_mapping({"model": "gpt-realtime"})
        )
    finally:
        logger.remove(handler_id)

    assert "model" in sink.getvalue()
    assert sent == []


@pytest.mark.parametrize(
    "modalities, sends_voice",
    [(["text", "audio"], True), (["text"], False)],
    ids=["audio", "text-only"],
)
@pytest.mark.asyncio
async def test_a_text_only_session_update_leaves_out_the_voice(modalities, sends_voice):
    """Voice Live rejects a repeated update naming a voice when audio is off."""
    service = _service(voice="en-US-Ava:DragonHDLatestNeural")
    service._settings.session_properties.modalities = modalities
    sent = []

    async def _record(event):
        sent.append(event)

    service.send_client_event = _record

    await service._send_session_update()

    assert ("voice" in sent[0].model_dump(exclude_none=True)["session"]) is sends_voice
    assert service._settings.session_properties.voice is not None


@pytest.mark.asyncio
async def test_a_model_given_in_settings_selects_the_connection_model(monkeypatch):
    """The connection URL picks the model, so a model set only in settings must reach it."""
    import pipecat.services.azure.voicelive.llm as llm_module

    uris = []

    async def fake_connect(uri, **kwargs):
        uris.append(uri)
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(llm_module, "websocket_connect", fake_connect)
    service = _service(settings=AzureVoiceLiveLLMService.Settings(model="gpt-realtime"))
    service.push_error = _noop

    await service._connect()

    assert "model=gpt-realtime" in uris[0]


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


def test_a_g711_output_format_is_replaced_with_a_warning():
    """Output frames carry PCM, so the transport's rate decides the format."""
    service = _service(
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(output_audio_format="g711_ulaw")
        )
    )

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service._ensure_audio_config(8000, 8000)
    finally:
        logger.remove(handler_id)

    assert "g711_ulaw" in sink.getvalue()
    assert service._settings.session_properties.output_audio_format == "pcm16_8000hz"


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


@pytest.mark.asyncio
async def test_a_failed_disconnect_still_clears_the_disconnecting_flag():
    """Sends are dropped while the flag is set, so a stuck flag mutes the service."""
    service = _service()

    class _FailingWebSocket:
        async def close(self):
            raise RuntimeError("close failed")

    service._websocket = _FailingWebSocket()
    await service._disconnect()

    assert service._disconnecting is False


class _EndedWebSocket:
    """A connection whose event stream has ended, in the given state."""

    def __init__(self, state: State, error: Exception | None = None):
        self.state = state
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._error:
            raise self._error
        raise StopAsyncIteration

    async def send(self, message):
        raise ConnectionClosedError(None, None)


def _record_errors(service) -> list[bool]:
    reported: list[bool] = []

    async def record(error_msg, **kwargs):
        reported.append(kwargs.get("force_treat_as_permanent", False))

    service.push_error = record
    return reported


@pytest.mark.parametrize(
    "error", [None, ConnectionClosedError(None, None)], ids=["clean-close", "error-close"]
)
@pytest.mark.asyncio
async def test_a_lost_connection_is_reported_once_as_permanent(error):
    """Without its connection the service can't respond; the unusable policy decides next."""
    service = _service()
    reported = _record_errors(service)
    service._websocket = _EndedWebSocket(State.CLOSED, error)

    await service._run_receive_loop()
    await service.send_client_event(events.InputAudioBufferClearEvent())

    assert reported == [True]
    assert service._websocket is None


@pytest.mark.parametrize(
    "disconnecting, state",
    [(True, State.CLOSED), (False, State.OPEN)],
    ids=["own-disconnect", "stopped-after-error-event"],
)
@pytest.mark.asyncio
async def test_an_expected_end_of_the_receive_loop_reports_nothing(disconnecting, state):
    """Closing the connection itself, or stopping after an error event already reported."""
    service = _service()
    reported = _record_errors(service)
    service._websocket = _EndedWebSocket(state)
    service._disconnecting = disconnecting

    await service._run_receive_loop()

    assert reported == []


@pytest.mark.asyncio
async def test_sending_on_a_closed_connection_reports_nothing():
    """The receive loop reports the lost connection; each send would report it again."""
    service = _service()
    reported = _record_errors(service)
    service._websocket = _EndedWebSocket(State.CLOSED)

    await service.send_client_event(events.InputAudioBufferClearEvent())

    assert reported == []


@pytest.mark.asyncio
async def test_reset_conversation_before_any_context_is_a_no_op():
    service = _service()
    reconnected = []
    service._connect = lambda: reconnected.append(True) or _noop()
    service._disconnect = _noop

    await service.reset_conversation()

    assert reconnected == []


@pytest.mark.asyncio
async def test_reset_conversation_sends_the_history_to_the_new_session():
    """Conversation setup only runs when _handle_context has no context yet."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    service = _service()
    service._connect = _noop
    service._disconnect = _noop
    service._api_session_ready = True

    created = []
    service.send_client_event = lambda e: created.append(type(e).__name__) or _noop()

    context = LLMContext([{"role": "user", "content": "remember this"}])
    await service._handle_context(context)
    service._llm_needs_conversation_setup = False
    # The first response finishes before the reset.
    service._response_in_flight = False
    created.clear()

    await service.reset_conversation()

    assert service._llm_needs_conversation_setup is False
    assert "ConversationItemCreateEvent" in created


async def _noop(*args, **kwargs):
    return None
