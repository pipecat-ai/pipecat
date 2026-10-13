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
    service.send_with_retry = AsyncMock()
    await service._send_context("ctx-1")
    return json.loads(service.send_with_retry.await_args.args[0])["create"]


@pytest.mark.asyncio
async def test_token_mode_uses_sentence_boundary():
    config = await _create_config(text_aggregation_mode=TextAggregationMode.TOKEN)
    assert config["autoMode"] is True
    assert config["autoModeStrategy"] == "SENTENCE_BOUNDARY"


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
