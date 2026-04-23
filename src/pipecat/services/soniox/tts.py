#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox text-to-speech service implementation.

This module provides integration with Soniox's text-to-speech API
for generating speech from text using various voice models.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.soniox.utils import language_to_soniox_language
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SonioxTTSService, you need to `pip install pipecat-ai[soniox]`.")
    raise Exception(f"Missing module: {e}")

SONIOX_TTS_WEBSOCKET_URL = "wss://tts-rt.soniox.com/tts-websocket"

DEFAULT_MODEL = "tts-rt-v1-preview"
DEFAULT_LANGUAGE = Language.EN
DEFAULT_VOICE = "Maya"


@dataclass
class SonioxTTSSettings(TTSSettings):
    """Settings for SonioxTTSService.

    Parameters:
        bitrate: Codec bitrate in bps (for compressed formats).
    """

    bitrate: int | None = None


class SonioxTTSService(WebsocketTTSService):
    """Soniox WebSocket-based text-to-speech service.

    Provides real-time text-to-speech synthesis using Soniox's WebSocket TTS API.
    Supports streaming text input and streaming audio generation with interruption
    handling for conversational AI use cases.

    Event handlers available:

    - on_connected: Called when the WebSocket connection is established.
    - on_connection_error: Called when a connection error occurs. Receives the error message.
    - on_disconnected: Called when the WebSocket connection is closed.
    - on_tts_request: Called before a TTS request is made. Receives context_id and text.

    Example::

        tts = SonioxTTSService(
            api_key="your-api-key",
            settings=SonioxTTSService.Settings(
                voice="Maya",
                language=Language.EN,
            ),
        )
    """

    Settings = SonioxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = SONIOX_TTS_WEBSOCKET_URL,
        sample_rate: int | None = None,
        encoding: str = "pcm_s16le",
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the Soniox TTS service.

        Args:
            api_key: Soniox API key.
            url: WebSocket URL for Soniox API. Defaults to "wss://tts-rt.soniox.com/tts-websocket".
            sample_rate: Audio sample rate in Hz. If None, uses 24kHz.
            encoding: Audio encoding format. Defaults to "pcm_s16le". Available formats are listed in
                https://soniox.com/docs/tts/concepts/audio-formats
            settings: Runtime-updatable settings (includes bitrate for compressed formats).
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            **kwargs: Additional arguments passed to parent WebsocketTTSService class.
        """
        default_settings = self.Settings(
            model=DEFAULT_MODEL,
            voice=DEFAULT_VOICE,
            language=DEFAULT_LANGUAGE,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            pause_frame_processing=True,
            push_stop_frames=False,
            push_start_frame=True,
            text_aggregation_mode=text_aggregation_mode,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._encoding = encoding

        self._receive_task = None
        self._config_sent = {}

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Soniox TTS service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Soniox language format.

        Args:
            language: The language to convert.

        Returns:
            The Soniox-specific language code.
        """
        if isinstance(language, Language):
            return language_to_soniox_language(language)
        return None

    async def start(self, frame: StartFrame):
        """Start the Soniox TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Soniox TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Soniox TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to Soniox TTS WebSocket and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Soniox TTS WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

        self._config_sent.clear()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Args:
            delta: A :class:`TTSSettings` (or ``SonioxTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def _connect_websocket(self):
        """Connect to Soniox TTS WebSocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Soniox TTS WebSocket")

            self._websocket = await websocket_connect(self._url)
            logger.debug(f"{self}: Websocket connection initialized.")

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(f"{self} error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and reset state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Soniox TTS WebSocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(f"{self} error: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Send a cancel message to Soniox when an audio context is interrupted.

        This will stop sending audio for the current stream.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        if self._websocket:
            try:
                await self._websocket.send(
                    json.dumps(
                        {
                            "cancel": True,
                            "stream_id": context_id,
                        }
                    )
                )
                # Clean up config tracking for cancelled stream
                self._config_sent.pop(context_id, None)
            except Exception as e:
                logger.error(f"{self} error sending cancel message: {e}")

    async def _receive_messages(self):
        """Receive and process messages from Soniox TTS WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                logger.warning(f"{self} received unsupported binary message: {message}")
            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                    stream_id = msg.get("stream_id")

                    if not stream_id:
                        logger.warning(
                            f"{self} received unsupported message, stream_id is missing: {msg}"
                        )
                        continue

                    error_code = msg.get("error_code")
                    error_message = msg.get("error_message")

                    if error_code or error_message:
                        logger.error(f"{self} received error: {error_code} - {error_message}")
                        await self.push_error(f"Soniox TTS error: {error_code} - {error_message}")
                        continue

                    context_id = self.get_active_audio_context_id()
                    if context_id != stream_id:
                        # Message for a stream that is not active (interrupted or completed)
                        logger.debug(f"{self} received message for inactive stream_id: {stream_id}")
                        continue

                    audio_base64 = msg.get("audio")
                    audio_end = msg.get("audio_end")
                    terminated = msg.get("terminated")

                    if audio_base64:
                        audio = base64.b64decode(audio_base64)
                        frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=stream_id)
                        await self.append_to_audio_context(stream_id, frame)
                    if audio_end:
                        logger.trace(f"Received audio end for {stream_id}")
                    if terminated:
                        logger.trace(f"Received stream terminated for {stream_id}")

                        await self.append_to_audio_context(
                            stream_id, TTSStoppedFrame(context_id=stream_id)
                        )
                        await self.remove_audio_context(stream_id)
                        self._config_sent.pop(stream_id, None)

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    async def flush_audio(self, context_id: str | None = None):
        """Finish the current audio synthesis by sending a text_end message.

        This should be called when the LLM finishes a complete response.
        """
        if not context_id:
            context_id = self.get_active_audio_context_id()
            if not context_id:
                logger.warning("No active audio context, cannot flush audio.")
                return

        if self._websocket:
            try:
                await self._websocket.send(
                    json.dumps(
                        {
                            "text": "",
                            "text_end": True,
                            "stream_id": context_id,
                        }
                    )
                )
            except Exception as e:
                logger.error(f"{self} error sending text_end message: {e}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Soniox's WebSocket TTS API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Reconnect if the websocket is closed
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Send config message if not already sent for this stream
            if context_id not in self._config_sent:
                language_setting = self._settings.language
                if isinstance(language_setting, Language):
                    language_code = self.language_to_service_language(language_setting)
                elif isinstance(language_setting, str):
                    language_code = language_setting
                else:
                    language_code = language_to_soniox_language(DEFAULT_LANGUAGE)
                config = {
                    "api_key": self._api_key,
                    "stream_id": context_id,
                    "model": self._settings.model or DEFAULT_MODEL,
                    "language": language_code,
                    "voice": self._settings.voice or DEFAULT_VOICE,
                    "audio_format": self._encoding,
                }
                if self.sample_rate:
                    config["sample_rate"] = self.sample_rate
                if self._settings.bitrate:
                    config["bitrate"] = self._settings.bitrate
                try:
                    await self._get_websocket().send(json.dumps(config))
                    self._config_sent[context_id] = True
                except Exception as e:
                    yield ErrorFrame(error=f"Failed to send config: {e}")
                    return

            # Send text message to Soniox. Partial text is supported.
            # When the LLM finishes a complete response, text_end is sent in flush_audio().
            await self._get_websocket().send(
                json.dumps(
                    {
                        "text": text,
                        "text_end": False,
                        "stream_id": context_id,
                    }
                )
            )

            # The audio frames will be handled in _receive_messages
            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
