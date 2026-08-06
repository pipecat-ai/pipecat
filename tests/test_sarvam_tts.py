#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SarvamHttpTTSService settings updates."""

import aiohttp
import pytest

from pipecat.services.sarvam.tts import SarvamHttpTTSService


@pytest.mark.asyncio
async def test_update_settings_recomputes_config_on_model_change():
    """Switching to a v3 model at runtime should recompute derived config."""
    async with aiohttp.ClientSession() as session:
        tts = SarvamHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(
                model="bulbul:v2",
                pitch=0.5,
                loudness=2.0,
                pace=2.5,
            ),
        )

        assert tts._config.supports_pitch is True
        assert tts._config.supports_temperature is False

        await tts._update_settings(SarvamHttpTTSService.Settings(model="bulbul:v3-beta", pace=2.5))

        # Config should now reflect the v3 model's capabilities.
        assert tts._config.supports_pitch is False
        assert tts._config.supports_loudness is False
        assert tts._config.supports_temperature is True

        # pitch/loudness are unsupported on v3 and should be nulled out.
        assert tts._settings.pitch is None
        assert tts._settings.loudness is None

        # Preprocessing is always enabled for v3 models.
        assert tts._settings.enable_preprocessing is True

        # pace 2.5 is outside the v3 range (0.5-2.0) and should be clamped.
        assert tts._settings.pace == 2.0


@pytest.mark.asyncio
async def test_update_settings_clamps_pace_back_to_v2_range():
    """Switching back to v2 should re-clamp pace to the v2 range."""
    async with aiohttp.ClientSession() as session:
        tts = SarvamHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(model="bulbul:v3-beta", pace=2.0),
        )

        await tts._update_settings(SarvamHttpTTSService.Settings(model="bulbul:v2", pace=3.5))

        assert tts._config.pace_range == (0.3, 3.0)
        assert tts._settings.pace == 3.0


@pytest.mark.asyncio
async def test_update_settings_without_model_change_keeps_config():
    """Updating an unrelated setting should not recompute the model config."""
    async with aiohttp.ClientSession() as session:
        tts = SarvamHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(model="bulbul:v2"),
        )

        original_config = tts._config

        await tts._update_settings(SarvamHttpTTSService.Settings(voice="abhilash"))

        assert tts._config is original_config
        assert tts._settings.voice == "abhilash"
