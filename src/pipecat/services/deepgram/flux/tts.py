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
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class DeepgramFluxTTSSettings(TTSSettings):
    """Settings for DeepgramFluxTTSService.

    The Flux voice is a single ``flux-{voice}-{language}`` identifier (e.g.
    ``flux-alexis-en``), carried by ``voice``. Deepgram's API passes it as
    its ``model`` query parameter, so ``model`` is kept in sync with
    ``voice`` and is not directly settable.

    Parameters:
        speed: Speech-rate multiplier, from 0.85 to 1.15 in steps of 0.05.
            ``None`` leaves Flux at its default rate. Applied to the open
            connection, so a speed change keeps the cross-turn acoustic state.
        expressivity: Expressive range on a calm-to-animated axis. ``None``
            leaves Flux at its default range. Flux fixes expressivity when the
            connection opens, so a change reconnects.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    expressivity: Literal[-2, -1, 0, 1, 2] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class DeepgramFluxTTSService(WebsocketTTSService):
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

    # Audio is always requested as linear16 (raw PCM), the format Pipecat
    # pipelines use internally.
    SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 32000, 44100, 48000)

    # Settings Flux can change on an open connection, via `Configure`. Every
    # other setting is a query parameter, fixed for the life of the connection.
    _CONFIGURE_FIELDS = frozenset({"speed"})

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
            **kwargs: Additional arguments passed to parent WebsocketTTSService class.
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

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
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

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the turn on the Flux connection when the user barges in.

        Flux's ``Interrupt`` message ends the active turn without closing the
        connection, so the cross-turn acoustic state survives a barge-in.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "Interrupt"}))
            except Exception as e:
                logger.error(f"{self} error sending Interrupt message: {e}")

        await super().on_audio_context_interrupted(context_id)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        A speed change is sent to Flux with a ``Configure`` message. Every other
        setting is a query parameter, so a change reconnects.

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

        if changed.keys() - self._CONFIGURE_FIELDS:
            await self._disconnect()
            await self._connect()
        elif changed:
            await self._send_configure()

        return changed

    async def _send_configure(self):
        """Apply the current speed to the open connection with a Configure message."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        # Flux's own default is 1.0, so clearing the setting restores that rate
        # explicitly — there is no way to ask Flux to forget a configured speed.
        speed = self._settings.speed if self._settings.speed is not None else 1.0

        try:
            await self._websocket.send(json.dumps({"type": "Configure", "speed": speed}))
        except Exception as e:
            logger.error(f"{self} error sending Configure message: {e}")

    def _build_query_string(self) -> str:
        """Build query string from current settings and init-only connection config."""
        params = [
            f"model={self._settings.voice}",
            "encoding=linear16",
            f"sample_rate={self.sample_rate}",
        ]

        if self._settings.speed is not None:
            params.append(f"speed={self._settings.speed}")

        if self._settings.expressivity is not None:
            params.append(f"expressivity={self._settings.expressivity}")

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
                # Binary audio frames carry no speech_id, so audio is
                # attributed to the active audio context. This is safe because
                # `pause_frame_processing=True` serializes turns, and an
                # interruption leaves no active context, so audio the server
                # had already generated for the abandoned turn is discarded.
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
                    elif msg_type == "SpeechInterrupted":
                        # Acknowledges an Interrupt. The reported total covers
                        # audio the server generated, not audio the user heard,
                        # since the request carries no playback offset.
                        logger.trace(
                            f"{self}: speech interrupted (speech_id: {msg.get('speech_id')}, "
                            f"audio played: {msg.get('audio_played_ms')}ms)"
                        )
                    elif msg_type == "SessionMetadata":
                        logger.debug(f"{self}: session totals: {msg}")
                    elif msg_type == "ConfigureSuccess":
                        logger.debug(f"{self}: configuration applied: {msg.get('applied')}")
                    elif msg_type == "ConfigureFailure":
                        # Non-fatal: synthesis continues with the previous
                        # configuration, so application code decides what a
                        # rejected settings update means for the conversation.
                        error_msg = (
                            f"{self} configuration rejected {msg.get('code')} "
                            f"({msg.get('field')}={msg.get('value')}): "
                            f"{msg.get('description', 'Unknown failure')}"
                        )
                        await self.push_error(error_msg=error_msg)
                    elif msg_type == "Warning":
                        code = msg.get("code")
                        if code == "NO_ACTIVE_SPEECH":
                            # Routine: an audio context outlives the turn that
                            # filled it, so a barge-in late in playback sends an
                            # Interrupt the server has nothing left to cancel.
                            logger.trace(f"{self}: no active turn to interrupt")
                        else:
                            logger.warning(
                                f"{self} warning {code}: "
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
