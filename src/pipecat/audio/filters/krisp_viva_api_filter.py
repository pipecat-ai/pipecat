#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp VIVA API audio filter for Pipecat.

Cleans input audio with Krisp VIVA over its WebSocket API (documented by Krisp
as the Realtime Voice Isolation API). It is the API counterpart of
:class:`~pipecat.audio.filters.krisp_viva_filter.KrispVivaFilter`, which runs
the Krisp VIVA SDK locally with a ``.kef`` model.
"""

import asyncio
import json
import os
from typing import Any

import aiohttp
from loguru import logger
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

KRISP_VIVA_API_URL = "wss://streaming.krisp.ai/voice-isolation"
KRISP_REST_BASE_URL = "https://api.developers.krisp.ai"
_SESSION_TOKEN_PATH = "/v2/sdk/tokens/session"

# The API processes 16-bit mono PCM at 8 or 16 kHz in frames of exactly 30 ms.
_FRAME_MS = 30

_HANDSHAKE_TIMEOUT = 10.0
_MAX_RECONNECT_DELAY = 10.0


class KrispVivaApiError(Exception):
    """An error reported by the Krisp VIVA API or its token endpoint."""

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        """Initialize the error.

        Args:
            message: Human-readable error message.
            code: Krisp error code (e.g. ``invalid_client_hello``).
            status_code: HTTP-style status code reported by the server.
        """
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    @property
    def is_permanent(self) -> bool:
        """Whether retrying can't help (rejected credentials or configuration)."""
        return self.status_code in (400, 401, 403)


class KrispVivaApiFilter(BaseAudioFilter):
    """Audio filter that removes noise and background voices with the Krisp VIVA API.

    Use it instead of :class:`~pipecat.audio.filters.krisp_viva_filter.KrispVivaFilter`
    to run VIVA in Krisp's cloud with just an API key, without the local SDK
    and ``.kef`` model.

    Input audio is resampled to 16 kHz (8 kHz for 8 kHz transports), sent in
    30 ms frames, and the cleaned audio is resampled back to the transport
    rate. Cleaned audio is returned as it arrives, so the filter's output lags
    its input by the network round trip; the first calls return no audio,
    which the input transport skips.

    If the service can't be reached or the connection drops, audio passes
    through unfiltered while the filter reconnects in the background.
    Rejected credentials or configuration stop the reconnect attempts.

    Example::

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_filter=KrispVivaApiFilter(),
            ),
        )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        rest_base_url: str | None = None,
        suppression_level: float = 100.0,
        model: str = "VIVA-tel",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Krisp VIVA API filter.

        Args:
            api_key: Krisp master API key. Defaults to ``KRISP_API_KEY``. It is
                only used to create session tokens and is never sent on the
                WebSocket.
            url: VIVA API WebSocket endpoint. Defaults to
                ``KRISP_VIVA_API_URL`` or production.
            rest_base_url: Krisp REST API base URL used to create session
                tokens. Defaults to ``KRISP_REST_BASE_URL`` or production.
            suppression_level: Noise suppression level, 0-100.
            model: VIVA model.
            metadata: Optional object echoed in the server's ``ready`` message
                (at most 4 KB JSON-encoded).
        """
        self._api_key = api_key or os.getenv("KRISP_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Krisp API key is required. Pass api_key or set the KRISP_API_KEY "
                "environment variable."
            )
        self._url = url or os.getenv("KRISP_VIVA_API_URL") or KRISP_VIVA_API_URL
        self._rest_base_url = (
            rest_base_url or os.getenv("KRISP_REST_BASE_URL") or KRISP_REST_BASE_URL
        )
        self._suppression_level = suppression_level
        self._model = model
        self._metadata = metadata

        self._filtering = True
        self._sample_rate = 0
        self._api_sample_rate = 16000
        self._frame_bytes = 0
        # HQ instead of the default VHQ: the filter resamples twice, and VHQ
        # holds back ~70 ms per direction versus ~30 ms for HQ.
        self._in_resampler = create_stream_resampler(quality="HQ")
        self._out_resampler = create_stream_resampler(quality="HQ")

        self._websocket: ClientConnection | None = None
        self._connection_task: asyncio.Task | None = None
        self._send_buffer = bytearray()
        self._received = bytearray()

    async def start(self, sample_rate: int):
        """Connect to the VIVA API for the transport's sample rate.

        Connecting happens in the background; audio passes through unfiltered
        until the connection is ready.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        self._api_sample_rate = 8000 if sample_rate <= 8000 else 16000
        self._frame_bytes = self._api_sample_rate * _FRAME_MS // 1000 * 2
        self._connection_task = asyncio.create_task(self._run_connection())

    async def stop(self):
        """Close the connection and release buffered audio."""
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None
        await self._close_websocket()
        self._send_buffer.clear()
        self._received.clear()

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Send audio to the VIVA API and return the cleaned audio received so far.

        Args:
            audio: Raw 16-bit mono PCM at the transport sample rate.

        Returns:
            Cleaned audio at the transport sample rate (possibly empty while
            the first frames are in flight), or the input unmodified when
            filtering is disabled or the service is unavailable.
        """
        websocket = self._websocket
        if not self._filtering or websocket is None or websocket.state is not State.OPEN:
            return self._take_received() + audio

        api_audio = audio
        if self._sample_rate != self._api_sample_rate:
            api_audio = await self._in_resampler.resample(
                audio, self._sample_rate, self._api_sample_rate
            )
        self._send_buffer.extend(api_audio)
        try:
            while len(self._send_buffer) >= self._frame_bytes:
                await websocket.send(bytes(self._send_buffer[: self._frame_bytes]))
                del self._send_buffer[: self._frame_bytes]
        except Exception as e:
            logger.warning(f"{self}: send failed, passing audio through: {e}")
            self._send_buffer.clear()
            return self._take_received() + audio

        return self._take_received()

    def _take_received(self) -> bytes:
        if not self._received:
            return b""
        cleaned = bytes(self._received)
        self._received.clear()
        return cleaned

    async def _run_connection(self):
        """Keep a session open, reconnecting with backoff until stopped."""
        delay = 1.0
        while True:
            try:
                await self._connect()
                delay = 1.0
                await self._receive_audio()
                logger.warning(f"{self}: connection closed, reconnecting")
            except asyncio.CancelledError:
                raise
            except KrispVivaApiError as e:
                if e.is_permanent:
                    logger.error(f"{self}: {e}; passing audio through unfiltered")
                    return
                logger.warning(f"{self}: {e}; retrying in {delay:.0f}s")
            except Exception as e:
                logger.warning(f"{self}: connection failed ({e}); retrying in {delay:.0f}s")
            await self._close_websocket()
            self._send_buffer.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_RECONNECT_DELAY)

    async def _connect(self):
        token = await self._create_session_token()
        websocket = await websocket_connect(
            self._url, additional_headers={"Authorization": f"api-key {token}"}
        )
        try:
            hello: dict[str, Any] = {
                "model": self._model,
                "sample_rate": self._api_sample_rate,
                "suppression_level": self._suppression_level,
            }
            if self._metadata is not None:
                hello["metadata"] = self._metadata
            await websocket.send(json.dumps(hello))
            ready = await self._wait_ready(websocket)
        except BaseException:
            await websocket.close()
            raise
        # Exposed only once ready: no audio may be sent before that.
        self._websocket = websocket
        logger.debug(
            f"{self}: connected to Krisp VIVA API (request_id={ready.get('request_id')}, "
            f"{self._api_sample_rate} Hz)"
        )

    async def _wait_ready(self, websocket: ClientConnection) -> dict[str, Any]:
        while True:
            message = await asyncio.wait_for(websocket.recv(), _HANDSHAKE_TIMEOUT)
            if isinstance(message, bytes):
                continue
            msg = json.loads(message)
            if msg.get("type") == "ready":
                return msg
            if msg.get("type") == "error":
                raise _error_from_message(msg)

    async def _receive_audio(self):
        websocket = self._websocket
        assert websocket is not None
        async for message in websocket:
            if isinstance(message, bytes):
                if self._sample_rate != self._api_sample_rate:
                    message = await self._out_resampler.resample(
                        message, self._api_sample_rate, self._sample_rate
                    )
                self._received.extend(message)
                continue
            msg = json.loads(message)
            if msg.get("type") == "error":
                raise _error_from_message(msg)

    async def _close_websocket(self):
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"{self}: error closing websocket: {e}")

    async def _create_session_token(self) -> str:
        url = f"{self._rest_base_url}{_SESSION_TOKEN_PATH}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={"Authorization": f"api-key {self._api_key}"},
                json={"usage_type": "viva"},
            ) as resp:
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = {}
                body = body if isinstance(body, dict) else {}
                if resp.status >= 400:
                    code = body.get("error_code")
                    raise KrispVivaApiError(
                        f"session token request failed: {code or resp.status}: "
                        f"{body.get('message') or resp.reason} (req_id={body.get('req_id')})",
                        code=code,
                        status_code=resp.status,
                    )
        token = (body.get("data") or {}).get("session_key")
        if not token:
            raise KrispVivaApiError(f"Unexpected session token response: {body!r}")
        return token


def _error_from_message(msg: dict[str, Any]) -> KrispVivaApiError:
    error = msg.get("error") or {}
    return KrispVivaApiError(
        f"{error.get('code', 'error')}: {error.get('message') or 'Unknown error'}",
        code=error.get("code"),
        status_code=error.get("status_code"),
    )
