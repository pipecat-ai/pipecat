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

from pipecat.services.azure.voice_live import events
from pipecat.services.azure.voice_live.llm import AzureVoiceLiveLLMService


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


def test_the_default_session_speaks_with_an_azure_voice():
    service = _service()

    assert service._settings.model == "gpt-4o-mini"
    voice = service._settings.session_properties.voice
    assert isinstance(voice, events.AzureStandardVoice)
    assert voice.name == "en-US-Ava:DragonHDLatestNeural"


def test_azure_realtime_is_left_to_pick_its_own_voice():
    """`azure-realtime` rejects Azure standard voices and picks a native one when none is sent."""
    service = _service(settings=AzureVoiceLiveLLMService.Settings(model="azure-realtime"))

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
    service = _service()
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
    service = _service()
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
    import pipecat.services.azure.voice_live.llm as llm_module

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

    assert service._settings.session_properties.input_audio_format == "pcm16"
    assert service._settings.session_properties.input_audio_sampling_rate == 8000


def test_a_g711_input_format_is_replaced_with_a_warning():
    """Input frames carry PCM, whatever format the session properties name."""
    service = _service(
        settings=AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(
                input_audio_format="g711_ulaw", input_audio_transcription=None
            )
        )
    )

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service._ensure_audio_config(8000, 8000)
    finally:
        logger.remove(handler_id)

    assert "g711_ulaw" in sink.getvalue()
    assert service._settings.session_properties.input_audio_format == "pcm16"


@pytest.mark.parametrize(
    "session_properties, warns",
    [
        (None, False),
        (events.SessionProperties(voice=events.AzureStandardVoice(name="en-US-Andrew")), True),
        (events.SessionProperties(input_audio_transcription=None), False),
        (
            events.SessionProperties(
                input_audio_transcription=events.InputAudioTranscription(model="azure-speech")
            ),
            False,
        ),
    ],
    ids=["defaults", "left-out", "explicitly-disabled", "configured"],
)
def test_session_properties_without_transcription_warn(session_properties, warns):
    """The caller's turns reach the context only as transcripts."""
    settings = (
        AzureVoiceLiveLLMService.Settings(session_properties=session_properties)
        if session_properties is not None
        else None
    )

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        _service(settings=settings)
    finally:
        logger.remove(handler_id)

    assert ("input_audio_transcription" in sink.getvalue()) is warns


@pytest.mark.asyncio
async def test_a_session_update_without_transcription_warns():
    service = _service()

    async def _record(event):
        pass

    service.send_client_event = _record

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._update_settings(
            AzureVoiceLiveLLMService.Settings(
                session_properties=events.SessionProperties(modalities=["text", "audio"])
            )
        )
    finally:
        logger.remove(handler_id)

    assert "input_audio_transcription" in sink.getvalue()


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
    """A connection whose event stream has ended, optionally with an error."""

    def __init__(self, error: Exception | None = None):
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
    service._websocket = _EndedWebSocket(error)

    await service._run_receive_loop()
    await service.send_client_event(events.InputAudioBufferClearEvent())

    assert reported == [True]
    assert service._websocket is None


@pytest.mark.asyncio
async def test_a_failed_connect_is_reported_as_permanent(monkeypatch):
    """Without a connection the service can't respond; the unusable policy decides next."""
    import pipecat.services.azure.voice_live.llm as llm_module

    async def fail_connect(uri, additional_headers):
        raise TimeoutError("timed out during opening handshake")

    monkeypatch.setattr(llm_module, "websocket_connect", fail_connect)
    service = _service()
    reported = _record_errors(service)

    await service._connect()

    assert reported == [True]
    assert service._websocket is None


@pytest.mark.asyncio
async def test_closing_the_connection_itself_reports_nothing():
    service = _service()
    reported = _record_errors(service)
    service._websocket = _EndedWebSocket()
    service._disconnecting = True

    await service._run_receive_loop()

    assert reported == []


@pytest.mark.asyncio
async def test_sending_on_a_closed_connection_reports_nothing():
    """The receive loop reports the lost connection; each send would report it again."""
    service = _service()
    reported = _record_errors(service)
    service._websocket = _EndedWebSocket()

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


def test_session_properties_in_an_update_set_the_top_level_fields():
    """``session_properties`` replaces the stored one, and its values reach the top level."""
    service = _service()
    settings = service._settings

    changed = settings.apply_update(
        AzureVoiceLiveLLMService.Settings(
            session_properties=events.SessionProperties(
                model="gpt-realtime", instructions="Be brief."
            )
        )
    )

    assert settings.model == "gpt-realtime"
    assert settings.system_instruction == "Be brief."
    assert {"model", "system_instruction"} <= changed.keys()


def test_top_level_fields_win_over_session_properties_in_the_same_update():
    service = _service()
    settings = service._settings

    settings.apply_update(
        AzureVoiceLiveLLMService.Settings(
            system_instruction="Top level.",
            session_properties=events.SessionProperties(instructions="Session."),
        )
    )

    assert settings.system_instruction == "Top level."
    assert settings.session_properties.instructions == "Top level."


@pytest.mark.asyncio
async def test_a_response_asked_for_before_the_session_is_ready_waits_for_it():
    """Voice Live applies the first session.update before it can answer."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    service = _service()
    service.start_processing_metrics = _noop
    service.start_ttfb_metrics = _noop
    sent = []

    async def _record(event):
        sent.append(type(event).__name__)

    service.send_client_event = _record

    await service._handle_context(LLMContext([{"role": "user", "content": "Hello"}]))
    assert "ResponseCreateEvent" not in sent

    await service._handle_evt_session_updated(None)

    assert sent.count("ResponseCreateEvent") == 1


@pytest.mark.parametrize("use_token_provider", [False, True], ids=["api-key", "token-provider"])
@pytest.mark.asyncio
async def test_connect_authenticates_with_the_configured_credential(
    monkeypatch, use_token_provider
):
    import pipecat.services.azure.voice_live.llm as llm_module

    headers = []

    async def record_connect(uri, additional_headers):
        headers.append(additional_headers)
        raise ConnectionError("no network in tests")

    async def token_provider():
        return "entra-token"

    monkeypatch.setattr(llm_module, "websocket_connect", record_connect)
    if use_token_provider:
        service = AzureVoiceLiveLLMService(
            endpoint="https://my-resource.services.ai.azure.com", token_provider=token_provider
        )
    else:
        service = _service()
    service.push_error = _noop

    await service._connect()

    if use_token_provider:
        assert headers == [{"Authorization": "Bearer entra-token"}]
    else:
        assert headers == [{"api-key": "test-key"}]


@pytest.mark.asyncio
async def test_frames_reach_voice_live_and_pass_through():
    """Audio is streamed and a tools update is sent as a session update; every frame
    continues downstream."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.frames.frames import (
        InputAudioRawFrame,
        LLMServiceMetadataFrame,
        LLMSetToolsFrame,
    )
    from pipecat.tests.utils import run_test

    service = _service()
    service._connect = _noop
    service._disconnect = _noop
    service._llm_needs_conversation_setup = False
    sent = []

    async def _record(event):
        sent.append(type(event).__name__)

    service.send_client_event = _record

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_current_weather",
                description="Get the weather.",
                properties={},
                required=[],
            )
        ]
    )
    audio = InputAudioRawFrame(audio=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1)

    await run_test(
        service,
        frames_to_send=[audio, LLMSetToolsFrame(tools=tools)],
        expected_down_frames=[LLMServiceMetadataFrame, InputAudioRawFrame, LLMSetToolsFrame],
    )

    assert "InputAudioBufferAppendEvent" in sent
    assert "SessionUpdateEvent" in sent
