#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily transport."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import BotConnectedFrame, STTMetadataFrame
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.transports.daily.transport import DailyParams, DailyTransport


def _make_transport(**params_kwargs) -> DailyTransport:
    with (
        patch("pipecat.transports.daily.transport.Daily"),
        patch("pipecat.transports.daily.transport.CallClient"),
    ):
        return DailyTransport(
            "https://mock.daily.co/mock", None, "bot", params=DailyParams(**params_kwargs)
        )


def _make_client_session(
    *, status: int = 200, body: str = "", error: Exception | None = None
) -> MagicMock:
    response = MagicMock(status=status)
    response.text = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    if error:
        session.post.side_effect = error
    else:
        session.post.return_value = response

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    return context


def _make_dialin_transport() -> DailyTransport:
    return _make_transport(
        api_key="test-api-key",
        dialin_settings={"call_id": "test-call", "call_domain": "test-domain"},
    )


@pytest.mark.asyncio
async def test_on_joined_pushes_stt_metadata_when_transcription_starts():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value=None)
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_awaited_once()
    # BotConnectedFrame is pushed before the STT metadata frame.
    call_names = [name for (name, _, _) in transport._input.mock_calls]
    assert call_names.index("push_frame") < call_names.index("push_stt_metadata_frame")
    (frame,) = transport._input.push_frame.await_args.args
    assert isinstance(frame, BotConnectedFrame)


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_fails():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value="some error")
    transport._on_error = AsyncMock()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._on_error.assert_awaited_once()
    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_disabled():
    transport = _make_transport()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_push_stt_metadata_frame_contents():
    transport = _make_transport(transcription_enabled=True)
    input_transport = transport.input()
    input_transport.broadcast_frame = AsyncMock()

    await input_transport.push_stt_metadata_frame()

    input_transport.broadcast_frame.assert_awaited_once_with(
        STTMetadataFrame,
        service_name=input_transport.name,
        ttfs_p99_latency=DEEPGRAM_TTFS_P99,
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_http_error_and_ready_event():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    transport._call_event_handler = AsyncMock()
    session = _make_client_session(status=503, body="service unavailable")

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._on_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Unable to handle dialin-ready event (status: 503, error: service unavailable)"
    )
    transport._call_event_handler.assert_awaited_once_with(
        "on_dialin_ready", "sip:test@example.com"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_timeout():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=TimeoutError())

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Timeout handling dialin-ready event (https://api.daily.co/v1/dialin/pinlessCallUpdate)"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_unexpected_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=RuntimeError("connection failed"))

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Error handling dialin-ready event "
        "(https://api.daily.co/v1/dialin/pinlessCallUpdate): connection failed"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_does_not_report_success_as_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session()

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_not_awaited()
