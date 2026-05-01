#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService


def _make_ws_service(**settings_kwargs) -> DeepgramTTSService:
    settings = DeepgramTTSService.Settings(**settings_kwargs) if settings_kwargs else None
    service = DeepgramTTSService(api_key="test-key", settings=settings)
    # Bypass start() lifecycle: sample_rate is the only field _connect_websocket reads.
    service._sample_rate = 16000
    return service


@pytest.mark.asyncio
async def test_ws_mip_opt_out_true_in_url():
    service = _make_ws_service(mip_opt_out=True)

    fake_ws = MagicMock()
    fake_ws.response.headers = {}

    with patch(
        "pipecat.services.deepgram.tts.websocket_connect",
        new=AsyncMock(return_value=fake_ws),
    ) as mock_connect:
        await service._connect_websocket()

    url = mock_connect.call_args.args[0]
    assert "mip_opt_out=true" in url


@pytest.mark.asyncio
async def test_ws_mip_opt_out_false_in_url():
    service = _make_ws_service(mip_opt_out=False)

    fake_ws = MagicMock()
    fake_ws.response.headers = {}

    with patch(
        "pipecat.services.deepgram.tts.websocket_connect",
        new=AsyncMock(return_value=fake_ws),
    ) as mock_connect:
        await service._connect_websocket()

    url = mock_connect.call_args.args[0]
    assert "mip_opt_out=false" in url


@pytest.mark.asyncio
async def test_ws_mip_opt_out_default_absent():
    service = _make_ws_service()

    fake_ws = MagicMock()
    fake_ws.response.headers = {}

    with patch(
        "pipecat.services.deepgram.tts.websocket_connect",
        new=AsyncMock(return_value=fake_ws),
    ) as mock_connect:
        await service._connect_websocket()

    url = mock_connect.call_args.args[0]
    assert "mip_opt_out" not in url


@pytest.mark.asyncio
async def test_ws_explicit_empty_settings_omits_mip_opt_out():
    """Explicit Settings() with no kwargs must not leak the NOT_GIVEN sentinel."""
    service = DeepgramTTSService(api_key="test-key", settings=DeepgramTTSService.Settings())
    # Bypass start() lifecycle: sample_rate is the only field _connect_websocket reads.
    service._sample_rate = 16000

    fake_ws = MagicMock()
    fake_ws.response.headers = {}

    with patch(
        "pipecat.services.deepgram.tts.websocket_connect",
        new=AsyncMock(return_value=fake_ws),
    ) as mock_connect:
        await service._connect_websocket()

    url = mock_connect.call_args.args[0]
    assert "mip_opt_out" not in url


class _FakeResponse:
    def __init__(self):
        self.status = 200
        self.content = MagicMock()

        async def _empty_iter(_chunk_size):
            return
            yield  # unreachable; makes this an async generator

        self.content.iter_chunked = _empty_iter

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_http_service(**settings_kwargs) -> DeepgramHttpTTSService:
    settings = DeepgramHttpTTSService.Settings(**settings_kwargs) if settings_kwargs else None
    session = MagicMock(spec=aiohttp.ClientSession)
    service = DeepgramHttpTTSService(api_key="test-key", aiohttp_session=session, settings=settings)
    # Bypass start() lifecycle: sample_rate is the only field run_tts reads.
    service._sample_rate = 16000
    service._session.post = MagicMock(return_value=_FakeResponse())
    return service


async def _drain(gen):
    async for _ in gen:
        pass


@pytest.mark.asyncio
async def test_http_mip_opt_out_true_in_params():
    service = _make_http_service(mip_opt_out=True)

    await _drain(service.run_tts("hello", "ctx"))

    params = service._session.post.call_args.kwargs["params"]
    assert params["mip_opt_out"] == "true"


@pytest.mark.asyncio
async def test_http_mip_opt_out_false_in_params():
    service = _make_http_service(mip_opt_out=False)

    await _drain(service.run_tts("hello", "ctx"))

    params = service._session.post.call_args.kwargs["params"]
    assert params["mip_opt_out"] == "false"


@pytest.mark.asyncio
async def test_http_mip_opt_out_default_absent():
    service = _make_http_service()

    await _drain(service.run_tts("hello", "ctx"))

    params = service._session.post.call_args.kwargs["params"]
    assert "mip_opt_out" not in params


@pytest.mark.asyncio
async def test_http_explicit_empty_settings_omits_mip_opt_out():
    """Explicit Settings() with no kwargs must not leak the NOT_GIVEN sentinel."""
    session = MagicMock(spec=aiohttp.ClientSession)
    service = DeepgramHttpTTSService(
        api_key="test-key",
        aiohttp_session=session,
        settings=DeepgramHttpTTSService.Settings(),
    )
    # Bypass start() lifecycle: sample_rate is the only field run_tts reads.
    service._sample_rate = 16000
    service._session.post = MagicMock(return_value=_FakeResponse())

    await _drain(service.run_tts("hello", "ctx"))

    params = service._session.post.call_args.kwargs["params"]
    assert "mip_opt_out" not in params
