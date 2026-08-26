#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GladiaSTTService runtime settings updates."""

import asyncio
import io
from collections.abc import Awaitable, Callable

from loguru import logger

from pipecat.frames.frames import VADUserStartedSpeakingFrame
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _stub_session(service: GladiaSTTService) -> list[dict]:
    """Stub out the network and record every session-init payload.

    Gladia configures a session over HTTP (``POST /v2/live``) and then opens a
    websocket to the URL that call returns, so the recorded payloads are exactly
    the settings each live session runs with.
    """
    payloads = []

    async def fake_setup_gladia(settings):
        payloads.append(settings)
        return {
            "url": f"wss://example.invalid/session-{len(payloads)}",
            "id": f"session-{len(payloads)}",
        }

    async def fake_connect_websocket():
        pass

    async def fake_disconnect_websocket():
        pass

    service._setup_gladia = fake_setup_gladia
    service._connect_websocket = fake_connect_websocket
    service._disconnect_websocket = fake_disconnect_websocket
    return payloads


def _run(service: GladiaSTTService, body: Callable[[], Awaitable[None]]) -> None:
    """Set the service up, run ``body``, and always tear the service down."""

    async def run():
        await service.setup(frame_processor_setup(TaskManager()))
        try:
            await body()
        finally:
            await service.cleanup()

    asyncio.run(run())


def test_settings_update_starts_a_new_session_carrying_the_change():
    # Gladia only reads configuration at session init, so an update that doesn't
    # start a new session never reaches the transcriber.
    service = GladiaSTTService(api_key="test-key")
    payloads = _stub_session(service)

    async def body():
        await service._update_settings(GladiaSTTService.Settings(endpointing=0.75))

    _run(service, body)

    assert len(payloads) == 2, "a changed setting must open a new Gladia session"
    assert "endpointing" not in payloads[0]
    assert payloads[1]["endpointing"] == 0.75
    assert service._session_id == "session-2"


def test_settings_update_with_no_change_keeps_the_session():
    # Re-sending the settings the session already runs with must not churn it.
    service = GladiaSTTService(
        api_key="test-key",
        settings=GladiaSTTService.Settings(endpointing=0.75),
    )
    payloads = _stub_session(service)

    async def body():
        await service._update_settings(GladiaSTTService.Settings(endpointing=0.75))

    _run(service, body)

    assert len(payloads) == 1
    assert service._session_id == "session-1"


def test_settings_update_mid_turn_defers_until_the_user_stops_speaking():
    # Reconnecting mid-utterance would drop the audio the caller is still speaking.
    service = GladiaSTTService(api_key="test-key")
    payloads = _stub_session(service)

    async def body():
        await service._handle_vad_user_started_speaking(VADUserStartedSpeakingFrame())

        await service._update_settings(GladiaSTTService.Settings(endpointing=0.75))
        assert len(payloads) == 1, "must not reconnect while the user is speaking"

        await service._maybe_reconnect_on_user_stopped_speaking()

    _run(service, body)

    assert len(payloads) == 2
    assert payloads[1]["endpointing"] == 0.75


def test_language_update_is_still_reported_as_unhandled():
    # Gladia picks languages through language_config, so a bare `language` never
    # reaches the session payload and must keep warning rather than look applied.
    service = GladiaSTTService(api_key="test-key")
    payloads = _stub_session(service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")

    async def body():
        await service._update_settings(GladiaSTTService.Settings(language=Language.ES))

    try:
        _run(service, body)
    finally:
        logger.remove(handler_id)

    assert "language" in sink.getvalue()
    assert all("language" not in payload for payload in payloads)
