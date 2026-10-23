#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Fish Audio TTS."""

from unittest.mock import AsyncMock

import ormsgpack
import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import FishAudioS2ExpressionAdapter
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.utils.types import NOT_GIVEN


@pytest.mark.asyncio
async def test_one_silent_context_writes_off_the_service():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=1)

    assert service._max_consecutive_zero_audio_contexts == 1


@pytest.mark.asyncio
async def test_the_silent_context_limit_can_be_raised():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=4)

    assert service._max_consecutive_zero_audio_contexts == 4


async def _start_request(settings=None):
    """Connect with a mocked socket and return the start message's request."""
    service = FishAudioTTSService(api_key="key", settings=settings)
    websocket = AsyncMock()
    service._websocket_connect = AsyncMock(return_value=websocket)

    await service._connect_websocket()

    message = ormsgpack.unpackb(websocket.send.await_args.args[0])
    assert message["event"] == "start"
    return message["request"]


async def _connect_headers(settings=None):
    """Connect with a mocked socket and return the handshake headers."""
    service = FishAudioTTSService(api_key="key", settings=settings)
    service._websocket_connect = AsyncMock(return_value=AsyncMock())

    await service._connect_websocket()

    return service._websocket_connect.await_args.kwargs["additional_headers"]


@pytest.mark.asyncio
async def test_default_model_is_s2_1_pro():
    headers = await _connect_headers()

    assert headers["model"] == "s2.1-pro"


@pytest.mark.asyncio
async def test_drama_3_model_is_sent_in_handshake():
    headers = await _connect_headers(FishAudioTTSService.Settings(model="drama-3"))

    assert headers["model"] == "drama-3"


@pytest.mark.asyncio
async def test_start_message_omits_unset_request_settings():
    request = await _start_request()

    for name in ("chunk_length", "min_chunk_length", "condition_on_previous_chunks"):
        assert name not in request
    assert request["prosody"] == {"speed": 1.0, "volume": 0}


@pytest.mark.asyncio
async def test_start_message_carries_set_request_settings():
    request = await _start_request(
        FishAudioTTSService.Settings(
            chunk_length=150,
            min_chunk_length=20,
            condition_on_previous_chunks=False,
            prosody_normalize_loudness=False,
        )
    )

    assert request["chunk_length"] == 150
    assert request["min_chunk_length"] == 20
    assert request["condition_on_previous_chunks"] is False
    assert request["prosody"]["normalize_loudness"] is False
    assert "normalize_loudness" not in request


def test_nested_prosody_mapping_leaves_missing_keys_unset():
    delta = FishAudioTTSService.Settings.from_mapping({"prosody": {"speed": 1.2}})

    assert delta.prosody_speed == 1.2
    assert delta.prosody_volume is NOT_GIVEN
    assert delta.prosody_normalize_loudness is NOT_GIVEN


def test_nested_prosody_mapping_carries_normalize_loudness():
    delta = FishAudioTTSService.Settings.from_mapping({"prosody": {"normalize_loudness": False}})

    assert delta.prosody_normalize_loudness is False


class _Messages:
    """A websocket stand-in that yields the given messages, then ends."""

    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.mark.asyncio
async def test_short_audio_chunks_are_kept():
    # A decoded chunk is PCM, so a short one is real audio (an utterance's tail).
    service = FishAudioTTSService(api_key="key")
    chunks = [b"\x01\x00" * 160, b"\x02\x00" * 1024]
    service._websocket = _Messages(
        ormsgpack.packb({"event": "audio", "audio": chunk}) for chunk in chunks
    )
    service.get_active_audio_context_id = lambda: "ctx"
    service.stop_ttfb_metrics = AsyncMock()
    appended = []

    async def append_to_audio_context(context_id, frame):
        appended.append(frame.audio)

    service.append_to_audio_context = append_to_audio_context

    await service._receive_messages()

    assert appended == chunks


class TestS2ExpressionAdapter:
    def test_capabilities(self):
        capabilities = FishAudioS2ExpressionAdapter().capabilities()
        assert capabilities.free_text_modes and capabilities.free_text_events
        assert capabilities.emphasis
        assert "laughing" in [term.name for term in capabilities.events]
        # A mode tag separates modes by spaces.
        assert all(len(term.name.split()) == 1 for term in capabilities.modes)

    def test_tags_where_modes_change(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.render_text("Hi.", ["excited"], None) == "[excited] Hi."
        assert adapter.render_text("Hi.", ["excited"], ["excited"]) == "Hi."
        assert adapter.render_text("Hi.", ["sad", "whispering"], None) == "[sad][whispering] Hi."

    def test_return_to_baseline_writes_no_tag(self):
        # Fish reads a [neutral] tag as a delivery of its own.
        assert FishAudioS2ExpressionAdapter().render_text("Hi.", [], ["excited"]) == "Hi."

    def test_emphasis(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.render_text("now", ["emphasis"], None) == "[emphasis] now"
        assert adapter.render_text("now", ["excited", "emphasis"], ["excited"]) == "[emphasis] now"

    def test_events_and_pauses(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.render_event("clear throat") == "[clear throat]"
        assert adapter.render_pause(600) == "[break]"
        assert adapter.render_pause(1200) == "[long-break]"


@pytest.mark.parametrize(
    "model, adapter",
    [
        ("s2.1-pro", FishAudioS2ExpressionAdapter),
        ("s2-pro", FishAudioS2ExpressionAdapter),
        ("s1", BaseExpressionAdapter),
        ("drama-3", BaseExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model(model, adapter):
    service = FishAudioTTSService(api_key="key", settings=FishAudioTTSService.Settings(model=model))
    assert type(service.get_expression_adapter()) is adapter
