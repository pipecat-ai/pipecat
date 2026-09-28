#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Fish Audio TTS."""

from unittest.mock import AsyncMock

import ormsgpack
import pytest

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
