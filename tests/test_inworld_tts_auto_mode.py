#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Inworld WebSocket TTS auto-mode strategy."""

import json
from unittest.mock import AsyncMock

import pytest

from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.tts_service import TextAggregationMode


async def _create_config(**kwargs) -> dict:
    model = kwargs.pop("model", "inworld-tts-2")
    service = InworldTTSService(
        api_key="test", settings=InworldTTSService.Settings(model=model), **kwargs
    )
    service._websocket = AsyncMock()
    await service._send_context("ctx-1")
    return json.loads(service._websocket.send.await_args.args[0])["create"]


@pytest.mark.asyncio
async def test_token_mode_uses_sentence_boundary():
    config = await _create_config(text_aggregation_mode=TextAggregationMode.TOKEN)
    assert config["autoMode"] is True
    assert config["autoModeStrategy"] == "SENTENCE_BOUNDARY"
    assert "bufferCharThreshold" not in config


@pytest.mark.asyncio
async def test_sentence_boundary_sends_a_set_buffer_char_threshold():
    config = await _create_config(
        text_aggregation_mode=TextAggregationMode.TOKEN, buffer_char_threshold=400
    )
    assert config["autoModeStrategy"] == "SENTENCE_BOUNDARY"
    assert config["bufferCharThreshold"] == 400


@pytest.mark.asyncio
async def test_buffer_args_override_params():
    with pytest.deprecated_call():
        params = InworldTTSService.InputParams(max_buffer_delay_ms=1000, buffer_char_threshold=400)
        config = await _create_config(
            params=params, max_buffer_delay_ms=500, buffer_char_threshold=100
        )
    assert config["maxBufferDelayMs"] == 500
    assert config["bufferCharThreshold"] == 100


@pytest.mark.asyncio
async def test_buffer_params_apply_without_args():
    with pytest.deprecated_call():
        params = InworldTTSService.InputParams(max_buffer_delay_ms=1000, buffer_char_threshold=400)
        config = await _create_config(params=params)
    assert config["maxBufferDelayMs"] == 1000
    assert config["bufferCharThreshold"] == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"text_aggregation_mode": TextAggregationMode.SENTENCE},
        {"text_aggregation_mode": TextAggregationMode.TOKEN, "model": "inworld-tts-1.5-max"},
        {"text_aggregation_mode": TextAggregationMode.TOKEN, "auto_mode": False},
    ],
)
async def test_sentence_boundary_not_requested(kwargs):
    config = await _create_config(**kwargs)
    assert "autoModeStrategy" not in config
    assert config["bufferCharThreshold"] == 250
