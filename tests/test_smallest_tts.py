#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestTTSService.

Unit tests run without any API key. Integration tests require SMALLEST_API_KEY
and hit the live Waves v4.0.0 API.

Run all:
    uv run pytest tests/test_smallest_tts.py -v -s

Run only unit tests (no API key needed):
    uv run pytest tests/test_smallest_tts.py -v -s -k "not integration"
"""

import base64
import json
import os

import pytest
from websockets.asyncio.client import connect as websocket_connect

from pipecat.services.smallest.tts import (
    SmallestTTSModel,
    SmallestTTSService,
    _MODEL_DEFAULT_VOICES,
)
from pipecat.transcriptions.language import Language

API_KEY = os.environ.get("SMALLEST_API_KEY", "")


# ---------------------------------------------------------------------------
# Unit tests — no network, no API key required
# ---------------------------------------------------------------------------


def test_default_voice_lightning_v3_1():
    """lightning_v3.1 should default to sophia."""
    tts = SmallestTTSService(
        api_key="dummy",
        settings=SmallestTTSService.Settings(model=SmallestTTSModel.LIGHTNING_V3_1.value),
    )
    assert tts._settings.voice == _MODEL_DEFAULT_VOICES[SmallestTTSModel.LIGHTNING_V3_1]
    assert tts._settings.voice == "sophia"


def test_default_voice_lightning_v3_1_pro():
    """lightning_v3.1_pro should default to meher."""
    tts = SmallestTTSService(
        api_key="dummy",
        settings=SmallestTTSService.Settings(model=SmallestTTSModel.LIGHTNING_V3_1_PRO.value),
    )
    assert tts._settings.voice == _MODEL_DEFAULT_VOICES[SmallestTTSModel.LIGHTNING_V3_1_PRO]
    assert tts._settings.voice == "meher"


def test_explicit_voice_overrides_default():
    """An explicitly provided voice should always win over the model default."""
    tts = SmallestTTSService(
        api_key="dummy",
        settings=SmallestTTSService.Settings(
            model=SmallestTTSModel.LIGHTNING_V3_1_PRO.value,
            voice="magnus",
        ),
    )
    assert tts._settings.voice == "magnus"


def test_no_settings_defaults_to_lightning_v3_1_pro_and_meher():
    """No settings at all should give lightning_v3.1_pro + meher."""
    tts = SmallestTTSService(api_key="dummy")
    assert tts._settings.model == SmallestTTSModel.LIGHTNING_V3_1_PRO.value
    assert tts._settings.voice == "meher"


def test_websocket_url():
    """WebSocket URL should be the fixed v4 endpoint regardless of model."""
    for model in SmallestTTSModel:
        tts = SmallestTTSService(
            api_key="dummy",
            settings=SmallestTTSService.Settings(model=model.value),
        )
        assert tts._build_websocket_url() == "wss://api.smallest.ai/waves/v1/tts/live"


def test_build_msg_includes_model():
    """Message payload should include model as a field."""
    tts = SmallestTTSService(
        api_key="dummy",
        settings=SmallestTTSService.Settings(model=SmallestTTSModel.LIGHTNING_V3_1_PRO.value),
    )
    tts._sample_rate = 16000
    msg = tts._build_msg("hello")
    assert msg["model"] == SmallestTTSModel.LIGHTNING_V3_1_PRO.value
    assert "text" in msg
    assert "voice_id" in msg


def test_build_msg_output_format():
    """output_format defaults to pcm."""
    tts = SmallestTTSService(api_key="dummy")
    tts._sample_rate = 16000
    assert tts._build_msg("hello")["output_format"] == "pcm"


# ---------------------------------------------------------------------------
# Integration tests — require SMALLEST_API_KEY, hit the live API directly
# ---------------------------------------------------------------------------


async def _synthesize(model: str, voice: str, text: str) -> list[bytes]:
    """Connect to the live WebSocket and collect audio chunks."""
    url = "wss://api.smallest.ai/waves/v1/tts/live"
    chunks = []
    async with websocket_connect(
        url, additional_headers={"Authorization": f"Bearer {API_KEY}"}
    ) as ws:
        msg = {
            "text": text,
            "voice_id": voice,
            "model": model,
            "language": "en",
            "sample_rate": 16000,
        }
        await ws.send(json.dumps(msg))
        async for message in ws:
            data = json.loads(message)
            if data["status"] == "chunk":
                chunks.append(base64.b64decode(data["data"]["audio"]))
            elif data["status"] == "complete":
                break
            elif data["status"] == "error":
                raise AssertionError(f"API error: {data}")
    return chunks


@pytest.mark.asyncio
@pytest.mark.skipif(not API_KEY, reason="SMALLEST_API_KEY not set")
async def test_integration_lightning_v3_1_returns_audio():
    """lightning_v3.1 with default voice (sophia) should return audio."""
    chunks = await _synthesize(
        model=SmallestTTSModel.LIGHTNING_V3_1.value,
        voice=_MODEL_DEFAULT_VOICES[SmallestTTSModel.LIGHTNING_V3_1],
        text="Hello from Smallest AI.",
    )
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


@pytest.mark.asyncio
@pytest.mark.skipif(not API_KEY, reason="SMALLEST_API_KEY not set")
async def test_integration_lightning_v3_1_pro_returns_audio():
    """lightning_v3.1_pro with default voice (meher) should return audio."""
    chunks = await _synthesize(
        model=SmallestTTSModel.LIGHTNING_V3_1_PRO.value,
        voice=_MODEL_DEFAULT_VOICES[SmallestTTSModel.LIGHTNING_V3_1_PRO],
        text="Hello from Smallest AI Pro.",
    )
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)
