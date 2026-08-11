#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI streaming STT service connection parameters."""

import asyncio
from urllib.parse import parse_qs, urlparse

from pipecat.frames.frames import StartFrame
from pipecat.services.xai.stt import XAISTTService


def _query(service: XAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def test_sample_rate_inherits_start_frame_when_omitted(monkeypatch):
    service = XAISTTService(api_key="test-key")

    async def fake_connect():
        pass

    monkeypatch.setattr(service, "_connect", fake_connect)

    asyncio.run(service.start(StartFrame(audio_in_sample_rate=8000)))

    assert service.sample_rate == 8000
    assert _query(service)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_overrides_start_frame(monkeypatch):
    service = XAISTTService(api_key="test-key", sample_rate=16000)

    async def fake_connect():
        pass

    monkeypatch.setattr(service, "_connect", fake_connect)

    asyncio.run(service.start(StartFrame(audio_in_sample_rate=8000)))

    assert service.sample_rate == 16000
    assert _query(service)["sample_rate"] == ["16000"]
