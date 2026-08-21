#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Fish Audio TTS."""

import pytest

from pipecat.services.fish.tts import FishAudioTTSService


@pytest.mark.asyncio
async def test_one_silent_context_writes_off_the_service():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=1)

    assert service._max_consecutive_zero_audio_contexts == 1


@pytest.mark.asyncio
async def test_the_silent_context_limit_can_be_raised():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=4)

    assert service._max_consecutive_zero_audio_contexts == 4
