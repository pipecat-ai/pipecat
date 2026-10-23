#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import websockets
from aiohttp import web
from aiohttp.test_utils import TestServer

from pipecat.audio.filters import krisp_viva_api_filter
from pipecat.audio.filters.krisp_viva_api_filter import KrispVivaApiFilter
from pipecat.frames.frames import FilterEnableFrame
from pipecat.services.krisp import common as krisp_common


@dataclass
class FakeKrisp:
    """A fake token endpoint and VIVA API WebSocket server."""

    token_requests: list = field(default_factory=list)
    auth_headers: list = field(default_factory=list)
    hellos: list = field(default_factory=list)
    frame_sizes: list = field(default_factory=list)
    connections: int = 0
    token_status: int = 200
    hello_error: dict | None = None
    # Close the first connection after this many frames.
    drop_after: int | None = None


def _clean(frame: bytes) -> bytes:
    """What the fake server returns for a frame, so cleaned audio is recognizable."""
    return bytes(255 - b for b in frame)


@pytest.fixture(autouse=True)
def _clear_token_cache():
    """Session tokens are cached per process; start every test without one."""
    krisp_common._token_cache.clear()
    yield
    krisp_common._token_cache.clear()


@pytest_asyncio.fixture
async def krisp():
    fake = FakeKrisp()

    async def token(request):
        fake.token_requests.append(await request.json())
        if fake.token_status != 200:
            return web.json_response(
                {"error_code": "AUTH_API_KEY_INVALID", "message": "bad key"},
                status=fake.token_status,
            )
        return web.json_response({"data": {"session_key": "sess-viva"}})

    app = web.Application()
    app.router.add_post("/v2/sdk/tokens/session", token)
    rest = TestServer(app)
    await rest.start_server()

    async def voice_isolation(ws):
        fake.auth_headers.append(ws.request.headers.get("Authorization"))
        fake.hellos.append(json.loads(await ws.recv()))
        fake.connections += 1
        if fake.hello_error:
            await ws.send(json.dumps({"type": "error", "error": fake.hello_error}))
            return
        await ws.send(json.dumps({"type": "ready", "request_id": "r1", "tracks_ms": [30]}))
        drop_after = fake.drop_after if fake.connections == 1 else None
        frames = 0
        async for message in ws:
            fake.frame_sizes.append(len(message))
            await ws.send(_clean(message))
            frames += 1
            if drop_after is not None and frames == drop_after:
                return

    async with websockets.serve(voice_isolation, "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield fake, f"ws://127.0.0.1:{port}", str(rest.make_url("")).rstrip("/")
    await rest.close()


def _filter(urls, **kwargs) -> KrispVivaApiFilter:
    _, ws_url, rest_url = urls
    return KrispVivaApiFilter(api_key="master", url=ws_url, rest_base_url=rest_url, **kwargs)


async def _until(predicate, timeout: float = 3.0):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout)


async def _filter_for(filt, chunk: bytes, chunks: int) -> bytes:
    """Feed audio at roughly real time and collect what the filter returns."""
    out = b""
    for _ in range(chunks):
        out += await filt.filter(chunk)
        await asyncio.sleep(0.01)
    for _ in range(20):  # let in-flight frames come back
        await asyncio.sleep(0.01)
        out += await filt.filter(b"")
    return out


@pytest.mark.asyncio
async def test_connects_with_a_viva_session_token_and_hello(krisp):
    fake, *_ = krisp
    filt = _filter(krisp, suppression_level=80)
    await filt.start(16000)
    await _until(lambda: filt._websocket is not None)
    await filt.stop()

    assert fake.token_requests == [{"usage_type": "viva", "expiration_ttl": 60}]
    assert fake.auth_headers == ["api-key sess-viva"]
    assert fake.hellos == [{"model": "VIVA-tel", "sample_rate": 16000, "suppression_level": 80}]


@pytest.mark.asyncio
async def test_sends_30ms_frames_and_returns_cleaned_audio(krisp):
    fake, *_ = krisp
    filt = _filter(krisp)
    await filt.start(16000)
    await _until(lambda: filt._websocket is not None)

    chunk = bytes(range(160)) * 2  # 10 ms at 16 kHz
    out = await _filter_for(filt, chunk, 9)  # 90 ms -> three 30 ms frames
    await filt.stop()

    assert fake.frame_sizes == [960, 960, 960]
    assert out == _clean(chunk * 9)


@pytest.mark.asyncio
async def test_resamples_other_transport_rates(krisp):
    fake, *_ = krisp
    filt = _filter(krisp)
    await filt.start(48000)
    await _until(lambda: filt._websocket is not None)

    out = await _filter_for(filt, b"\x00\x00" * 480, 30)  # 300 ms at 48 kHz
    await filt.stop()

    assert fake.hellos[0]["sample_rate"] == 16000
    assert set(fake.frame_sizes) == {960}
    # Cleaned audio comes back at 48 kHz, minus what the two resamplers hold back.
    assert 30 * 960 * 0.6 < len(out) <= 30 * 960 and len(out) % 2 == 0


@pytest.mark.asyncio
async def test_8khz_transport_uses_8khz_frames(krisp):
    fake, *_ = krisp
    filt = _filter(krisp)
    await filt.start(8000)
    await _until(lambda: filt._websocket is not None)

    out = await _filter_for(filt, b"\x01\x00" * 80, 6)  # 60 ms at 8 kHz
    await filt.stop()

    assert fake.hellos[0]["sample_rate"] == 8000
    assert fake.frame_sizes == [480, 480]
    assert out == _clean(b"\x01\x00" * 80 * 6)


@pytest.mark.asyncio
async def test_passes_audio_through_until_connected_and_when_disabled(krisp):
    fake, *_ = krisp
    filt = _filter(krisp)
    chunk = b"\x02\x00" * 160

    # Not started: no connection yet.
    assert await filt.filter(chunk) == chunk

    await filt.start(16000)
    await _until(lambda: filt._websocket is not None)
    await filt.process_frame(FilterEnableFrame(enable=False))
    assert await filt.filter(chunk) == chunk
    await filt.stop()

    assert fake.frame_sizes == []


@pytest.mark.asyncio
async def test_rejected_credentials_stop_retrying_and_pass_through(krisp):
    fake, *_ = krisp
    fake.token_status = 401
    filt = _filter(krisp)
    await filt.start(16000)
    await _until(lambda: filt._connection_task.done())

    chunk = b"\x03\x00" * 160
    assert await filt.filter(chunk) == chunk
    await filt.stop()

    assert len(fake.token_requests) == 1
    assert fake.connections == 0


@pytest.mark.asyncio
async def test_invalid_hello_is_not_retried(krisp):
    fake, *_ = krisp
    fake.hello_error = {"code": "invalid_client_hello", "status_code": 400, "message": "bad"}
    filt = _filter(krisp, suppression_level=150)
    await filt.start(16000)
    await _until(lambda: filt._connection_task.done())
    await filt.stop()

    assert fake.connections == 1


@pytest.mark.asyncio
async def test_reconnects_after_the_connection_drops(krisp, monkeypatch):
    fake, *_ = krisp
    fake.drop_after = 1
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep(asyncio.sleep))
    filt = _filter(krisp)
    await filt.start(16000)
    await _until(lambda: filt._websocket is not None)

    chunk = b"\x04\x00" * 480  # 30 ms
    await filt.filter(chunk)
    await _until(lambda: fake.connections == 2 and filt._websocket is not None)
    await filt.filter(chunk)
    await _until(lambda: len(fake.frame_sizes) == 2)
    await filt.stop()

    # The reconnect reuses the cached session token.
    assert [r["usage_type"] for r in fake.token_requests] == ["viva"]


@pytest.mark.asyncio
async def test_permanent_error_mid_session_closes_the_socket_and_passes_audio_through(krisp):
    fake, *_ = krisp
    filt = _filter(krisp)
    await filt.start(16000)
    await _until(lambda: filt._websocket is not None)
    ws = filt._websocket

    # The server reports a permanent error but leaves the socket open.
    async def receive_error():
        from pipecat.services.krisp.common import KrispServiceError

        raise KrispServiceError("unauthorized: expired", code="unauthorized", status_code=403)

    filt._receive_audio = receive_error
    filt._connection_task.cancel()
    filt._connection_task = asyncio.create_task(filt._run_connection())
    await _until(lambda: filt._connection_task.done())

    chunk = b"\x05\x00" * 160
    assert filt._websocket is None or filt._websocket is not ws
    assert await filt.filter(chunk) == chunk
    await filt.stop()


async def _backoff_delays(filt, monkeypatch, sessions: int) -> list[float]:
    """Run the connection loop over sessions that close at once; return its sleeps."""
    delays = []

    async def record_sleep(delay, *args, **kwargs):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    filt._connect = AsyncMock()
    filt._close_websocket = AsyncMock()
    filt._receive_audio = AsyncMock(side_effect=[None] * sessions + [asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await filt._run_connection()
    return delays


@pytest.mark.asyncio
async def test_quick_disconnects_keep_the_backoff_growing(krisp, monkeypatch):
    filt = _filter(krisp)

    assert await _backoff_delays(filt, monkeypatch, 4) == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_stable_connections_reset_the_backoff(krisp, monkeypatch):
    monkeypatch.setattr(krisp_viva_api_filter, "_STABLE_CONNECTION_SECS", 0.0)
    filt = _filter(krisp)

    assert await _backoff_delays(filt, monkeypatch, 3) == [1.0, 1.0, 1.0]


def _fast_sleep(real_sleep):
    async def sleep(delay, *args, **kwargs):
        return await real_sleep(min(delay, 0.01), *args, **kwargs)

    return sleep
