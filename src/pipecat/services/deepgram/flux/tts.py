#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux text-to-speech service implementation (WebSocket transport).

This module provides integration with Deepgram's Flux TTS WebSocket API
(``/v2/speak``), a streaming-first speech synthesis service designed for
voice agents.
"""

import json

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.deepgram.flux.tts_base import (
    DeepgramFluxTTSBase,
    DeepgramFluxTTSSettings,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.websocket_service import WebsocketService

# `DeepgramFluxTTSSettings` is defined with the base but belongs to this
# service's public API.
__all__ = [
    "DeepgramFluxTTSService",
    "DeepgramFluxTTSSettings",
]


class DeepgramFluxTTSService(DeepgramFluxTTSBase, WebsocketService):
    """Deepgram Flux WebSocket text-to-speech service.

    Provides real-time speech synthesis using Deepgram's Flux TTS API at
    ``wss://api.deepgram.com/v2/speak``. Flux keeps acoustic state across
    turns on a single connection, so prosody and pacing stay consistent
    throughout a conversation.

    By default, LLM tokens are streamed to Flux as they arrive
    (``TextAggregationMode.TOKEN``) — Flux is built to take raw LLM output
    and places synthesis boundaries internally, so buffering for sentence
    punctuation only adds latency. Pass
    ``text_aggregation_mode=TextAggregationMode.SENTENCE`` to aggregate
    text into sentences before synthesis instead.

    Event handlers:

    - on_connected: Called when the websocket connection is established.
    - on_disconnected: Called when the websocket connection is closed.
    - on_connection_error: Called when a websocket connection error occurs.

    Example::

        tts = DeepgramFluxTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            settings=DeepgramFluxTTSService.Settings(
                voice="flux-alexis-en",
                speed=1.05,
                expressivity=1,
            ),
        )
    """

    Settings = DeepgramFluxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.deepgram.com/v2/speak",
        sample_rate: int | None = None,
        mip_opt_out: bool | None = None,
        tag: list[str] | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux WebSocket TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            url: WebSocket URL for the Flux TTS API. Defaults to
                "wss://api.deepgram.com/v2/speak".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                default. Must be one of :attr:`SUPPORTED_SAMPLE_RATES`.
            mip_opt_out: Opt out of the Deepgram Model Improvement Program. See
                https://dpgr.am/deepgram-mip for pricing impacts before setting to True.
            tag: Tags to label requests for identification during usage reporting.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Flux for the lowest latency.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent classes.
        """
        default_settings = self.Settings(
            model=None,
            voice="flux-heather-en",
            language=None,
            speed=None,
            expressivity=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        DeepgramFluxTTSBase.__init__(
            self,
            sample_rate=sample_rate,
            mip_opt_out=mip_opt_out,
            tag=tag,
            text_aggregation_mode=text_aggregation_mode,
            settings=default_settings,
            **kwargs,
        )
        WebsocketService.__init__(self, **kwargs)

        self._api_key = api_key
        self._url = url

        self._receive_task = None

    # ------------------------------------------------------------------
    # Transport interface implementation
    # ------------------------------------------------------------------

    async def _transport_send_json(self, message: dict):
        if (
            self._websocket is None
        ):  # should never happen — caller should gate on _transport_is_active()
            return
        await self._websocket.send(json.dumps(message))

    def _transport_is_active(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the Flux websocket and start the receive task."""
        # Reaching WebsocketService takes an explicit call: it comes after the
        # Flux base in the MRO, so `super()` resolves to the abstract `_connect`
        # this method overrides.
        await WebsocketService._connect(self)

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Flux websocket and clean up tasks."""
        await WebsocketService._disconnect(self)

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to the Deepgram Flux WebSocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Deepgram Flux WebSocket")

            self._validate_sample_rate()

            url = f"{self._url}?{self._build_query_string()}"

            headers = {"Authorization": f"Token {self._api_key}"}

            websocket = await self._websocket_connect(url, additional_headers=headers)
            self._websocket = websocket

            # `response` is populated after the handshake completes (which it
            # has, since the connect call already returned).
            response_headers = websocket.response.headers if websocket.response else {}
            headers = {k: v for k, v in response_headers.items() if k.startswith("dg-")}
            logger.debug(f'{self}: Websocket connection initialized: {{"headers": {headers}}}')

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and reset state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Deepgram Flux WebSocket")
                # No `Close` message here: in Flux, `Close` asks the server to
                # drain the active turn, generating all of its remaining audio,
                # which a teardown has no use for. Closing the socket ends the
                # session outright.
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from the Flux WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                await self._handle_audio(message)
            elif isinstance(message, str):
                try:
                    await self._handle_message(json.loads(message))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    async def _report_error(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)
