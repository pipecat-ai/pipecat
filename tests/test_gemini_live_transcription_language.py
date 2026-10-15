#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ``input_transcription_languages`` setting."""

from unittest.mock import Mock

import pytest

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transcriptions.language import Language


async def _connect_config(**settings):
    """Run ``_connect`` and return the config it hands to the connection task."""
    service = GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(**settings)
    )
    service._connection_task_handler = Mock()  # type: ignore[method-assign]
    service.create_task = Mock()  # type: ignore[method-assign]
    await service._connect()
    return service._connection_task_handler.call_args.kwargs["config"]


@pytest.mark.asyncio
async def test_languages_reach_the_connect_config():
    config = await _connect_config(input_transcription_languages=[Language.TE_IN, "en-IN"])

    assert config.input_audio_transcription.language_codes == ["te-IN", "en-IN"]


@pytest.mark.asyncio
async def test_unset_languages_leave_transcription_auto_detected():
    config = await _connect_config()

    assert config.input_audio_transcription.language_codes is None
