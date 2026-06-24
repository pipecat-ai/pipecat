#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh Flux STT service: Deepgram Flux turn detection proxied via the Dograh MPS."""

import json
import time
from collections.abc import AsyncGenerator
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)
from pipecat.services.dograh.mps_billing import (
    MPS_BILLING_VERSION_KEY,
    MPS_BILLING_VERSION_V2,
    get_correlation_id,
    uses_mps_billing_v2,
)
from pipecat.services.websocket_service import WebsocketService

__all__ = [
    "DograhFluxSTTService",
    "DeepgramFluxSTTSettings",
]


class DograhFluxSTTService(DeepgramFluxSTTBase, WebsocketService):
    """Dograh Flux speech-to-text service.

    Provides Deepgram Flux turn detection through the Dograh managed model
    services (MPS) proxy. All Flux protocol handling (turn detection, metrics,
    settings) is inherited from ``DeepgramFluxSTTBase``; this class only
    implements the transport: a WebSocket to the Dograh MPS Flux endpoint with
    Bearer auth and MPS billing/correlation carried in the query string. The
    proxy forwards native Flux messages, so the inherited message handling
    applies unchanged.

    Event handlers available (in addition to base events):

    - on_start_of_turn(service, transcript): start of speech detected
    - on_end_of_turn(service, transcript): end of turn (EOT) detected
    - on_eager_end_of_turn(service, transcript): end of turn predicted (EagerEOT)
    - on_turn_resumed(service): user resumed speaking after EagerEOT

    Example::

        @stt.event_handler("on_end_of_turn")
        async def on_end_of_turn(service, transcript):
            ...
    """

    Settings = DeepgramFluxSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://services.dograh.com",
        ws_path: str = "/api/v1/stt/flux",
        correlation_id: str | None = None,
        sample_rate: int | None = None,
        flux_encoding: str = "linear16",
        mip_opt_out: bool | None = None,
        tag: list | None = None,
        should_interrupt: bool = False,
        watchdog_min_timeout: float = 0.5,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Dograh Flux STT service.

        Args:
            api_key: Dograh API key for authentication (sent as a Bearer token).
            base_url: WebSocket base URL for the Dograh MPS. Defaults to
                "wss://services.dograh.com".
            ws_path: WebSocket path for the Flux STT endpoint. Defaults to
                "/api/v1/stt/flux".
            correlation_id: Optional server-generated correlation ID for MPS
                billing v2. Falls back to the StartFrame metadata when omitted.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            flux_encoding: Audio encoding format required by Flux. Must be
                "linear16" (signed little-endian 16-bit PCM).
            mip_opt_out: Opt out of the Deepgram Model Improvement Program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether the service interrupts the bot when Flux
                detects user speech. Defaults to False so the user-turn
                aggregator owns interruption (via the external turn strategies).
            watchdog_min_timeout: Minimum idle timeout before sending silence to
                prevent dangling turns. Defaults to 0.5.
            settings: Runtime-updatable Flux settings.
            **kwargs: Additional arguments passed to the parent classes.
        """
        default_settings = self.Settings(
            model="flux-general-multi",
            language=None,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
            language_hints=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        DeepgramFluxSTTBase.__init__(
            self,
            encoding=flux_encoding,
            mip_opt_out=mip_opt_out,
            tag=tag,
            should_interrupt=should_interrupt,
            watchdog_min_timeout=watchdog_min_timeout,
            settings=default_settings,
            sample_rate=sample_rate,
            **kwargs,
        )
        # reconnect_on_error stays False: like Deepgram Flux, the connection is
        # only considered ready once the server confirms it, and send_with_retry
        # handles reconnection on send.
        WebsocketService.__init__(self, reconnect_on_error=False)

        self._api_key = api_key
        self._base_url = base_url
        self._ws_path = ws_path
        self._correlation_id = correlation_id
        self._websocket_url: str | None = None
        self._receive_task = None
        self._start_metadata = None

    @property
    def supports_ttfs(self) -> bool:
        """TTFS doesn't apply: Flux defines turn boundaries directly."""
        return False

    # ------------------------------------------------------------------
    # MPS billing
    # ------------------------------------------------------------------

    def _get_correlation_id(self) -> str | None:
        return get_correlation_id(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    def _uses_mps_billing_v2(self) -> bool:
        return uses_mps_billing_v2(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    def _build_dograh_query_string(self) -> str:
        """Append MPS billing/correlation params to the inherited Flux query."""
        query = self._build_query_string()
        correlation_id = self._get_correlation_id()
        if correlation_id:
            query += "&" + urlencode({"correlation_id": correlation_id})
            if self._uses_mps_billing_v2():
                query += "&" + urlencode({MPS_BILLING_VERSION_KEY: MPS_BILLING_VERSION_V2})
        return query

    # ------------------------------------------------------------------
    # Transport interface implementation
    # ------------------------------------------------------------------

    async def _transport_send_audio(self, audio: bytes):
        if self._websocket is None:  # caller gates on _transport_is_active()
            return
        await self._websocket.send(audio)

    async def _transport_send_json(self, message: dict):
        if self._websocket is None:  # caller gates on _transport_is_active()
            return
        await self._websocket.send(json.dumps(message))

    def _transport_is_active(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        """Capture StartFrame metadata (for billing) before connecting."""
        # Must run before super().start(), which triggers _connect() and the
        # query-string build that resolves the correlation id from metadata.
        self._start_metadata = frame.metadata
        await super().start(frame)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Build the MPS Flux URL and open the WebSocket connection."""
        await super()._connect()
        self._websocket_url = f"{self._base_url}{self._ws_path}?{self._build_dograh_query_string()}"
        await self._connect_websocket()

    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks."""
        await super()._disconnect()

        try:
            await self._disconnect_websocket()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the Dograh MPS Flux endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._connection_established_event.clear()
            self._user_is_speaking = False
            assert self._websocket_url is not None
            websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            self._websocket = websocket

            if not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )

            if not self._watchdog_task:
                self._watchdog_task = self.create_task(self._watchdog_task_handler())

            logger.debug("WebSocket connected, waiting for server confirmation...")
            await self._connection_established_event.wait()
            logger.debug("Connected to Dograh Flux WebSocket")
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clean up state."""
        try:
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None
            if self._watchdog_task:
                await self.cancel_task(self._watchdog_task, timeout=2.0)
                self._watchdog_task = None
                self._last_stt_time = None

            self._connection_established_event.clear()
            await self.stop_all_metrics()

            if self._websocket:
                await self._send_close_stream()
                logger.debug("Disconnecting from Dograh Flux WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Audio sending and receiving
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to the Dograh MPS for Flux transcription.

        Args:
            audio: Raw audio bytes in linear16 format.

        Yields:
            Frame: None (results are delivered via WebSocket callbacks).
        """
        if not self._websocket:
            return

        try:
            self._last_stt_time = time.monotonic()
            self._last_audio_chunk_duration = len(audio) / (self.sample_rate * 2)
            await self.send_with_retry(audio, self._report_error)
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
            return

        yield None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive messages from the proxy and dispatch them.

        Native Flux messages (capitalized ``type`` values) are handled by the
        base. The Dograh proxy injects its own control messages with a
        lowercase ``"error"`` type for billing/quota conditions, handled here.
        """
        async for message in self._get_websocket():
            if not isinstance(message, str):
                logger.warning(f"Received non-string message: {type(message)}")
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {e}")
                continue

            if data.get("type") == "error":
                await self._handle_proxy_error(data)
                continue

            try:
                await self._handle_message(data)
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
                # Surfaced + stopped by _receive_task_handler (reconnect disabled).
                raise

    async def _handle_proxy_error(self, data: dict):
        """Handle a Dograh proxy control error (e.g. quota exceeded)."""
        error_msg = data.get("error") or data.get("message", "Unknown error")
        is_quota_error = "quota" in error_msg.lower()

        if is_quota_error:
            logger.info(f"STT quota exceeded: {error_msg}")
            await self.push_frame(
                ErrorFrame(error=f"STT service quota exceeded: {error_msg}", fatal=True),
                direction=FrameDirection.UPSTREAM,
            )
            try:
                if self._websocket:
                    await self._websocket.close(
                        code=1000, reason="Quota exceeded - closing gracefully"
                    )
                    self._websocket = None
            except Exception as close_error:
                logger.debug(f"Error while closing websocket: {close_error}")
            return

        await self.push_error(error_msg=f"STT error: {error_msg}")

    async def _report_error(self, error):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
