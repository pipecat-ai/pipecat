#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux text-to-speech service implementations (WebSocket and HTTP).

This module provides integration with Deepgram's Flux TTS API (``/v2/speak``):
the streaming WebSocket transport designed for voice agents, and the batch
HTTP transport that returns the audio for a complete text in one response.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.deepgram._errors import describe_refusal
from pipecat.services.deepgram.flux.tts_base import (
    DeepgramFluxTTSBase,
    DeepgramFluxTTSSettings,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.utils.tracing.service_decorators import traced_tts

# `DeepgramFluxTTSSettings` is defined with the base but belongs to this
# service's public API.
__all__ = [
    "DeepgramFluxHttpTTSService",
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
            normalize_volume=None,
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
            refusal = describe_refusal(e)
            await self.push_error_frame(
                ErrorFrame(error=refusal or f"{self} error: {e}", exception=e)
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", refusal or f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and reset state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Deepgram Flux WebSocket")
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


class DeepgramFluxHttpTTSService(TTSService):
    """Deepgram Flux HTTP text-to-speech service.

    Synthesizes speech with Deepgram's Flux TTS batch API at
    ``https://api.deepgram.com/v2/speak``. Each request is independent, so
    there is no connection to manage and no acoustic state carried across
    turns. Use :class:`DeepgramFluxTTSService` for live conversations.

    Example::

        tts = DeepgramFluxHttpTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            aiohttp_session=session,
            settings=DeepgramFluxHttpTTSService.Settings(
                voice="flux-anjali-hi",
            ),
        )
    """

    Settings = DeepgramFluxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.deepgram.com",
        sample_rate: int | None = None,
        mip_opt_out: bool | None = None,
        tag: list[str] | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux HTTP TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            aiohttp_session: Shared aiohttp session for HTTP requests with connection pooling.
            base_url: Custom base URL for Deepgram API. Defaults to "https://api.deepgram.com".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                default. Flux synthesizes 8000, 16000, 24000, 32000, 44100
                and 48000 Hz.
            mip_opt_out: Opt out of the Deepgram Model Improvement Program. See
                https://dpgr.am/deepgram-mip for pricing impacts before setting to True.
            tag: Tags to label requests for identification during usage reporting.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        default_settings = self.Settings(
            model=None,
            voice="flux-heather-en",
            language=None,
            speed=None,
            expressivity=None,
            normalize_volume=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        # Deepgram passes the voice identifier as its `model` query parameter,
        # so keep `model` in sync with `voice` for metrics.
        default_settings.model = default_settings.voice

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url
        self._mip_opt_out = mip_opt_out
        self._tag = tag or []

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram Flux TTS supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, which takes effect on the next request.

        Args:
            delta: A :class:`TTSSettings` (or ``Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        # Deepgram uses voice as the model, so keep them in sync for metrics
        if "voice" in changed:
            self._settings.model = self._settings.voice
            self._sync_model_name_to_metrics()

        return changed

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Deepgram's Flux TTS batch API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        url = f"{self._base_url}/v2/speak"

        headers = {"Authorization": f"Token {self._api_key}", "Content-Type": "application/json"}

        # Audio is always requested as raw linear16 PCM, the format Pipecat
        # pipelines use internally.
        params: list[tuple[str, str]] = [
            ("model", str(self._settings.voice)),
            ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)),
            ("container", "none"),
        ]

        if self._settings.speed is not None:
            params.append(("speed", str(self._settings.speed)))

        if self._settings.expressivity is not None:
            params.append(("expressivity", str(self._settings.expressivity)))

        if self._settings.normalize_volume is not None:
            params.append(("normalize_volume", str(self._settings.normalize_volume).lower()))

        if self._mip_opt_out is not None:
            params.append(("mip_opt_out", str(self._mip_opt_out).lower()))

        for tag_value in self._tag:
            params.append(("tag", tag_value))

        payload = {
            "text": text,
        }

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                url, headers=headers, json=payload, params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                await self.start_tts_usage_metrics(text)

                # HTTP chunk boundaries can fall mid-sample, so let the iterator
                # helper keep emitted frames sample-aligned.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except Exception as e:
            yield ErrorFrame(f"Error getting audio: {str(e)}")
