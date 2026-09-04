#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI streaming STT service connection parameters."""

import asyncio
from urllib.parse import parse_qs, urlparse

from pipecat.services.xai.stt import XAISTTService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _query(service: XAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def _setup_service(service: XAISTTService, monkeypatch, sample_rate: int) -> None:
    """Set the service up with the given input sample rate, without connecting."""

    async def fake_connect():
        pass

    monkeypatch.setattr(service, "_connect", fake_connect)

    async def run():
        await service.setup(frame_processor_setup(TaskManager(), audio_in_sample_rate=sample_rate))

    asyncio.run(run())


def test_sample_rate_inherits_setup_when_omitted(monkeypatch):
    service = XAISTTService(api_key="test-key")

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 8000
    assert _query(service)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_overrides_setup(monkeypatch):
    service = XAISTTService(api_key="test-key", sample_rate=16000)

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 16000
    assert _query(service)["sample_rate"] == ["16000"]
