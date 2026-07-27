#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.services.gladia.stt import GladiaSTTService


@pytest.mark.asyncio
async def test_update_settings_triggers_reconnect():
    """Regression test for issue #4719.

    Runtime settings updates were silently ignored because _update_settings()
    stored the new settings but never reconnected. Gladia bakes all config into
    the session at POST /v2/live time, so a new session must be created for any
    settings change to take effect.

    After the fix, _update_settings() clears the session URL/ID and calls
    _request_reconnect(), following the same pattern as DeepgramSTTService.
    """
    service = GladiaSTTService.__new__(GladiaSTTService)
    service._name = "GladiaSTTService"
    service._settings = GladiaSTTService.Settings(model="solaria-1")
    service._session_url = "wss://fake-session.gladia.io/live/abc123"
    service._session_id = "abc123"
    service._request_reconnect = AsyncMock()

    with patch(
        "pipecat.services.stt_service.STTService._update_settings",
        new_callable=AsyncMock,
        return_value={"model": "old-model"},
    ):
        await service._update_settings(GladiaSTTService.Settings(model="accurate"))

    assert service._session_url is None, "session_url should be cleared on settings change"
    assert service._session_id is None, "session_id should be cleared on settings change"
    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_settings_no_reconnect_when_nothing_changed():
    """When settings delta produces no change, _request_reconnect() must not be called."""
    service = GladiaSTTService.__new__(GladiaSTTService)
    service._name = "GladiaSTTService"
    service._settings = GladiaSTTService.Settings(model="solaria-1")
    service._session_url = "wss://fake-session.gladia.io/live/abc123"
    service._session_id = "abc123"
    service._request_reconnect = AsyncMock()

    with patch(
        "pipecat.services.stt_service.STTService._update_settings",
        new_callable=AsyncMock,
        return_value={},  # empty dict → nothing changed
    ):
        await service._update_settings(GladiaSTTService.Settings(model="solaria-1"))

    assert service._session_url == "wss://fake-session.gladia.io/live/abc123"
    assert service._session_id == "abc123"
    service._request_reconnect.assert_not_awaited()
