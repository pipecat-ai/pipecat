#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux text-to-speech service implementation.

This module provides integration with Deepgram's Flux TTS WebSocket API
(``/v2/speak``), a streaming-first speech synthesis service designed for
voice agents.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TextAggregationMode
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class DeepgramFluxTTSSettings(TTSSettings):
    """Settings for DeepgramFluxTTSService.

    The Flux voice is a single ``flux-{voice}-{language}`` identifier (e.g.
    ``flux-alexis-en``), carried by ``voice``. Deepgram's API passes it as
    its ``model`` query parameter, so ``model`` is kept in sync with
    ``voice`` and is not directly settable.
    """

    pass


class DeepgramFluxTTSService(InterruptibleTTSService):
    """Deepgram Flux WebSocket text-to-speech service (early access).

    Provides real-time speech synthesis using Deepgram's Flux TTS API at
    ``wss://api.deepgram.com/v2/speak``. LLM tokens are streamed to the
    server as ``Speak`` messages and each agent response is synthesized as
    a discrete turn ended by a ``Flush``. Flux keeps acoustic state across
    turns on a single connection, so prosody and pacing stay consistent
    throughout a conversation.

    By default, LLM tokens are streamed to Flux as they arrive
    (``TextAggregationMode.TOKEN``) — Flux is built to take raw LLM output
    and places synthesis boundaries internally, so buffering for sentence
    punctuation only adds latency. Pass
    ``text_aggregation_mode=TextAggregationMode.SENTENCE`` to aggregate
    text into sentences before synthesis instead.

    Flux does not yet provide a message to cancel the active turn, so
    interruptions are handled by the :class:`InterruptibleTTSService`
    behavior of reconnecting the websocket while the bot is speaking. A
    reconnect (interruption or the server's one-hour session cap) resets
    Flux's cross-turn prosody state, which is expected. Once Deepgram ships
    the planned ``Interrupt`` message, the base class can change to
    :class:`WebsocketTTSService` and send it instead of reconnecting.

    Flux TTS is early access: the voice catalog and parts of the protocol
    may change before general availability.

    Event handlers:

    - on_connected: Called when the websocket connection is established.
    - on_disconnected: Called when the websocket connection is closed.
    - on_connection_error: Called when a websocket connection error occurs.

    Example::

        tts = DeepgramFluxTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            settings=DeepgramFluxTTSService.Settings(voice="flux-alexis-en"),
        )
    """

    Settings = DeepgramFluxTTSSettings
    _settings: Settings

    # Audio is always requested as linear16 (raw PCM), the format Pipecat
    # pipelines use internally.
    SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 32000, 44100, 48000)

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
                default. Must be one of SUPPORTED_SAMPLE_RATES.
            mip_opt_out: Opt out of the Deepgram Model Improvement Program. See
                https://dpgr.am/deepgram-mip for pricing impacts before setting to True.
            tag: Tags to label requests for identification during usage reporting.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Flux for the lowest latency.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService class.

        Raises:
            ValueError: If an explicit sample_rate is not supported.
        """
        if sample_rate is not None and sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. "
                f"Must be one of {self.SUPPORTED_SAMPLE_RATES}."
            )

        default_settings = self.Settings(
            model=None,
            voice="flux-alexis-en",
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        # Deepgram passes the voice identifier as its `model` query parameter,
        # so keep `model` in sync with `voice` for metrics.
        default_settings.model = default_settings.voice

        super().__init__(
            sample_rate=sample_rate,
            pause_frame_processing=True,
            push_stop_frames=False,
            push_start_frame=True,
            text_aggregation_mode=text_aggregation_mode,
            # Flux never inserts or strips whitespace between Speak messages,
            # so consecutive sentences would otherwise glue together. Applies
            # in sentence mode only; when streaming tokens, the LLM's own
            # whitespace is used as-is.
            append_trailing_space=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._mip_opt_out = mip_opt_out
        self._tag = tag or []

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram Flux TTS supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Deepgram Flux TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def _connect(self):
        """Connect to the Flux websocket and start the receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Flux websocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Args:
            delta: A :class:`TTSSettings` (or ``DeepgramFluxTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        # Deepgram uses voice as the model, so keep them in sync for metrics
        if "voice" in changed:
            self._settings.model = self._settings.voice
            self._sync_model_name_to_metrics()

        # Settings are query parameters, so a change requires a new connection.
        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_query_string(self) -> str:
        """Build query string from current settings and init-only connection config."""
        params = [
            f"model={self._settings.voice}",
            "encoding=linear16",
            f"sample_rate={self.sample_rate}",
        ]

        if self._mip_opt_out is not None:
            params.append(f"mip_opt_out={str(self._mip_opt_out).lower()}")

        # Add tag parameters (can have multiple)
        for tag_value in self._tag:
            params.append(urlencode({"tag": tag_value}))

        return "&".join(params)

    async def _connect_websocket(self):
        """Connect to the Deepgram Flux WebSocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Deepgram Flux WebSocket")

            if self.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
                logger.warning(
                    f"{self}: sample rate {self.sample_rate} is not supported. "
                    f"Supported rates: {self.SUPPORTED_SAMPLE_RATES}."
                )

            url = f"{self._url}?{self._build_query_string()}"

            headers = {"Authorization": f"Token {self._api_key}"}

            websocket = await websocket_connect(url, additional_headers=headers)
            self._websocket = websocket

            # `response` is populated after the handshake completes (which it
            # has, since `websocket_connect` already returned).
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
                # drain the active turn (generating all remaining audio),
                # which is the opposite of what an interruption-driven
                # disconnect needs. Closing the socket ends the session.
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
                # Binary audio frames carry no speech_id, so audio is
                # attributed to the active audio context. This is safe because
                # `pause_frame_processing=True` serializes turns and
                # interruptions tear the connection down.
                ctx_id = self.get_active_audio_context_id()
                frame = TTSAudioRawFrame(message, self.sample_rate, 1, context_id=ctx_id)
                await self.append_to_audio_context(ctx_id, frame)
            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "Connected":
                        logger.debug(
                            f"{self}: connected (request_id: {msg.get('request_id')}, "
                            f"model: {msg.get('model_name')})"
                        )
                    elif msg_type == "SpeechStarted":
                        logger.trace(f"Received SpeechStarted: {msg}")
                    elif msg_type == "Flushed":
                        # Not end-of-turn: Flux acknowledges the flush and may
                        # still send audio afterwards. SpeechMetadata is the
                        # definitive end-of-turn signal.
                        logger.trace(f"Received Flushed: {msg}")
                    elif msg_type == "SpeechMetadata":
                        # Sent once per turn after all of its audio.
                        logger.debug(
                            f"{self}: speech complete (speech_id: {msg.get('speech_id')}, "
                            f"duration: {msg.get('audio_duration_ms')}ms, "
                            f"billable characters: {msg.get('billable_character_count')})"
                        )
                        ctx_id = self.get_active_audio_context_id()
                        await self.append_to_audio_context(
                            ctx_id, TTSStoppedFrame(context_id=ctx_id)
                        )
                        await self.remove_audio_context(ctx_id)
                    elif msg_type == "SessionMetadata":
                        logger.debug(f"{self}: session totals: {msg}")
                    elif msg_type == "Warning":
                        logger.warning(
                            f"{self} warning {msg.get('code')}: "
                            f"{msg.get('description', 'Unknown warning')}"
                        )
                    elif msg_type == "Error":
                        error_msg = (
                            f"{self} error {msg.get('code')}: "
                            f"{msg.get('description', 'Unknown error')}"
                        )
                        logger.error(error_msg)
                        await self.push_error(error_msg=error_msg)
                    else:
                        logger.debug(f"Received unknown message type: {msg}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis by sending a Flush message.

        This ends the active turn: the server generates any remaining audio
        and reports the turn's ``SpeechMetadata``.
        """
        if self._websocket:
            try:
                flush_msg = {"type": "Flush"}
                await self._websocket.send(json.dumps(flush_msg))
            except Exception as e:
                logger.error(f"{self} error sending Flush message: {e}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Deepgram's Flux WebSocket TTS API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        # When streaming tokens, the base class logs the accumulated turn text
        # at flush time, so per-token logs are kept at trace level.
        if self._is_streaming_tokens:
            logger.trace(f"{self}: Generating TTS [{text}]")
        else:
            logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                # Full disconnect/connect cycle: after a server-initiated close
                # (e.g. Flux's one-hour session cap) the receive task has
                # completed but is still set, so a plain _connect() would not
                # restart the receive loop.
                await self._disconnect()
                await self._connect()

            speak_msg = {"type": "Speak", "text": text}
            await self._get_websocket().send(json.dumps(speak_msg))

            await self.start_tts_usage_metrics(text)

            # The audio frames will be handled in _receive_messages
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
