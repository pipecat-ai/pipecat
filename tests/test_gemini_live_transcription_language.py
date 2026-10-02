#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``input_transcription_language_codes`` setting.

The codes must land on the ``LiveConnectConfig`` sent to ``live.connect`` as
``input_audio_transcription.language_codes``; unset, the transcription config
stays empty so the model auto-detects as before.
"""

import pytest

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


async def _connect_config(**settings):
    """Run ``_connect`` and return the config it hands to the connection task."""
    service = GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(**settings)
    )
    captured = {}

    def _capture(config):
        captured["config"] = config

    service._connection_task_handler = _capture  # type: ignore[method-assign]
    service.create_task = lambda *args, **kwargs: None  # type: ignore[method-assign]
    await service._connect()
    return captured["config"]


@pytest.mark.asyncio
async def test_language_codes_reach_the_connect_config():
    config = await _connect_config(input_transcription_language_codes=["te-IN", "en-IN"])

    assert config.input_audio_transcription.language_codes == ["te-IN", "en-IN"]


@pytest.mark.asyncio
async def test_unset_language_codes_leave_transcription_auto_detected():
    config = await _connect_config()

    assert config.input_audio_transcription is not None
    assert config.input_audio_transcription.language_codes is None
    assert config.output_audio_transcription.language_codes is None
