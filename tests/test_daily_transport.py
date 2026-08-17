#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily transport."""

import asyncio
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


def _prepare_client_for_cleanup(transport: DailyTransport):
    """Give the shared DailyTransportClient what cleanup() needs in a test.

    Simulates the state after both the input and output transports have called
    setup(): a task manager whose loop is the running loop, and a cleanup
    counter of 2.
    """
    client = transport._client
    task_manager = MagicMock()
    task_manager.get_event_loop.return_value = asyncio.get_running_loop()
    client._task_manager = task_manager
    client._cleanup_counter = 2
    return client


@pytest.mark.asyncio
async def test_final_cleanup_releases_client_and_shuts_down_executor():
    transport = _make_transport()
    client = _prepare_client_for_cleanup(transport)
    call_client = client._client

    # First call (input transport): refcounted, nothing released yet.
    await client.cleanup()
    call_client.release.assert_not_called()
    assert not client._executor._shutdown

    # Final call (output transport): releases the client and shuts down the
    # executor so its worker thread doesn't outlive the session.
    await client.cleanup()
    call_client.release.assert_called_once()
    assert client._client is None
    assert client._executor._shutdown


@pytest.mark.asyncio
async def test_extra_cleanup_calls_are_a_no_op():
    # Some wrappers (e.g. the LemonSlice transport) call the shared client's
    # cleanup() from both input and output while setup() ran only once, so
    # cleanup() can be reached again after the executor is shut down.
    transport = _make_transport()
    client = _prepare_client_for_cleanup(transport)
    call_client = client._client

    await client.cleanup()
    await client.cleanup()
    await client.cleanup()  # One more than setup() calls: must not raise.

    call_client.release.assert_called_once()


@pytest.mark.asyncio
async def test_executor_shuts_down_even_when_release_raises():
    transport = _make_transport()
    client = _prepare_client_for_cleanup(transport)
    client._cleanup_counter = 1
    client._client.release.side_effect = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        await client.cleanup()

    assert client._executor._shutdown


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
