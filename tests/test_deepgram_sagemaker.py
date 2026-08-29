#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.services.deepgram.sagemaker.stt import DeepgramSageMakerSTTService
from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService

# ---------------------------------------------------------------------------
# TTS — issue #4739
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_update_settings_triggers_reconnect():
    """Regression test for issue #4739.

    _update_settings() previously stored new settings but never reconnected,
    so voice/model changes were silently ignored. All TTS settings are baked
    into the SageMaker query string at connect time, so a fresh connection is
    required for any change to take effect.
    """
    service = DeepgramSageMakerTTSService.__new__(DeepgramSageMakerTTSService)
    service._name = "DeepgramSageMakerTTSService"
    service._settings = DeepgramSageMakerTTSService.Settings(voice="aura-2-helena-en")
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()
    service._sync_model_name_to_metrics = MagicMock()

    with patch(
        "pipecat.services.tts_service.TTSService._update_settings",
        new_callable=AsyncMock,
        return_value={"voice": "aura-2-helena-en"},
    ):
        await service._update_settings(
            DeepgramSageMakerTTSService.Settings(voice="aura-2-arcas-en")
        )

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_update_settings_no_reconnect_when_nothing_changed():
    """No reconnect when the settings delta produces no change."""
    service = DeepgramSageMakerTTSService.__new__(DeepgramSageMakerTTSService)
    service._name = "DeepgramSageMakerTTSService"
    service._settings = DeepgramSageMakerTTSService.Settings(voice="aura-2-helena-en")
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    with patch(
        "pipecat.services.tts_service.TTSService._update_settings",
        new_callable=AsyncMock,
        return_value={},
    ):
        await service._update_settings(
            DeepgramSageMakerTTSService.Settings(voice="aura-2-helena-en")
        )

    service._disconnect.assert_not_awaited()
    service._connect.assert_not_awaited()


# ---------------------------------------------------------------------------
# STT — issue #4737
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_do_reconnect_calls_disconnect_then_connect():
    """_do_reconnect() must disconnect then reconnect so STTService._reconnect()
    can use it as the connection-reset hook."""
    service = DeepgramSageMakerSTTService.__new__(DeepgramSageMakerSTTService)
    service._name = "DeepgramSageMakerSTTService"
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._do_reconnect()

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stt_update_settings_triggers_reconnect():
    """Regression test for issue #4737.

    _update_settings() previously stored new settings but never reconnected.
    After the fix, it calls _request_reconnect() which defers the reconnect
    until after the current user turn when the user is speaking.
    """
    service = DeepgramSageMakerSTTService.__new__(DeepgramSageMakerSTTService)
    service._name = "DeepgramSageMakerSTTService"
    service._settings = DeepgramSageMakerSTTService.Settings(model="nova-3")
    service._request_reconnect = AsyncMock()

    with patch(
        "pipecat.services.stt_service.STTService._update_settings",
        new_callable=AsyncMock,
        return_value={"model": "nova-3"},
    ):
        await service._update_settings(DeepgramSageMakerSTTService.Settings(model="nova-2"))

    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stt_update_settings_no_reconnect_when_nothing_changed():
    """No reconnect when the settings delta produces no change."""
    service = DeepgramSageMakerSTTService.__new__(DeepgramSageMakerSTTService)
    service._name = "DeepgramSageMakerSTTService"
    service._settings = DeepgramSageMakerSTTService.Settings(model="nova-3")
    service._request_reconnect = AsyncMock()

    with patch(
        "pipecat.services.stt_service.STTService._update_settings",
        new_callable=AsyncMock,
        return_value={},
    ):
        await service._update_settings(DeepgramSageMakerSTTService.Settings(model="nova-3"))

    service._request_reconnect.assert_not_awaited()
