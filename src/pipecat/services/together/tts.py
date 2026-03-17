#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together AI text-to-speech service implementation."""

import base64
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Together, you need to `pip install pipecat-ai[together]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class TogetherTTSSettings(TTSSettings):
    """Settings for the Together AI TTS service.

    ``model``, ``voice``, and ``language`` are inherited from ``TTSSettings`` /
    ``ServiceSettings``.

    Parameters:
        model: Together AI TTS model to use.
        voice: Voice to use for synthesis.
        language: Language of the text input.
        max_partial_length: Maximum partial text length for streaming.
    """

    max_partial_length: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TogetherTTSService(WebsocketTTSService):
    """Together AI TTS service with WebSocket streaming.

    Provides text-to-speech using Together AI's realtime WebSocket API.
    Supports streaming synthesis with configurable voice and model options.
    """

    Settings = TogetherTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        output_format: str = "raw",
        encoding: str = "pcm_s16le",
        url: str = "wss://api.together.ai/v1/audio/speech/websocket",
        sample_rate: Optional[int] = 24000,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Together AI TTS service.

        Args:
            api_key: Together AI API key for authentication.
            output_format: Audio output container format. Supported values:
                ``"raw"``, ``"mp3"``, ``"wav"``, ``"opus"``, ``"aac"``,
                ``"flac"``, ``"pcm"``. Defaults to ``"raw"``.
            encoding: PCM encoding when ``output_format`` is ``"raw"``.
                Supported values: ``"pcm_s16le"``, ``"pcm_f32le"``.
                Defaults to ``"pcm_s16le"``.
            url: WebSocket URL for Together AI TTS API.
            sample_rate: Audio sample rate (default: 24000).
            settings: Runtime-updatable settings. Allows overriding model, voice,
                language, and max_partial_length.
            **kwargs: Additional arguments passed to the parent service.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="canopylabs/orpheus-3b-0.1-ft",
            voice="tara",
            language=Language.EN,
            max_partial_length=None,
        )

        # 2. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=False,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._output_format = output_format
        self._encoding = encoding
        self._session_id = None
        self._receive_task = None
        self._context_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Together TTS service supports metrics generation.
        """
        return True

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL with query parameters."""
        url = (
            f"{self._url}?model={self._settings.model}"
            f"&voice={self._settings.voice}"
            f"&response_format={self._output_format}"
            f"&response_encoding={self._encoding}"
        )
        if self._settings.max_partial_length is not None:
            url += f"&max_partial_length={self._settings.max_partial_length}"
        return url

    async def start(self, frame: StartFrame):
        """Start the Together AI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Together AI TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Together AI TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the transcription endpoint and start receiving."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect and clean up background tasks."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the Together AI TTS endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_url = self._build_websocket_url()
            logger.debug(f"Connecting to Together AI TTS: {ws_url}")

            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")

            # Ensure voice is set on the server side
            try:
                voice_update_msg = {
                    "type": "tts_session.updated",
                    "session": {"voice": self._settings.voice},
                }
                await self._websocket.send(json.dumps(voice_update_msg))
                logger.debug(f"Sent initial voice setting to WebSocket: {self._settings.voice}")
            except Exception as e:
                logger.error(f"Error sending initial voice setting: {e}")

            logger.debug("Connected to Together AI TTS")

        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Together AI TTS: {e}",
                exception=e,
            )
            self._websocket = None

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Error disconnecting: {e}",
                exception=e,
            )
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._session_id = None
            await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Client events
    # ------------------------------------------------------------------

    async def _ws_send(self, message: dict):
        """Send a JSON message over the WebSocket.

        Args:
            message: The message dict to serialize and send.
        """
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error sending message: {e}",
                exception=e,
            )

    async def flush_audio(self, context_id: Optional[str] = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        await self._ws_send({"type": "input_text_buffer.commit"})

    # ------------------------------------------------------------------
    # Server event handling
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Receive and dispatch server events from the TTS session.

        Called by ``WebsocketService._receive_task_handler`` which wraps
        this method with automatic reconnection on connection errors.
        """
        async for message in self._websocket:
            # Together sends audio as binary WebSocket frames (raw PCM)
            # and control/status messages as JSON text frames.
            if not isinstance(message, str):
                await self._handle_audio_binary(message)
                continue

            try:
                evt = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} failed to parse WebSocket message")
                continue

            evt_type = evt.get("type", "")

            if evt_type == "session.created":
                await self._handle_session_created(evt)
            elif evt_type == "session.updated":
                await self._handle_session_updated(evt)
            elif evt_type == "conversation.item.input_text.received":
                logger.trace(f"{self} text received")
            elif evt_type == "conversation.item.audio_output.delta":
                await self._handle_audio_delta(evt)
            elif evt_type == "conversation.item.audio_output.done":
                await self._handle_audio_done(evt)
            elif evt_type == "conversation.item.tts.failed":
                await self._handle_tts_failed(evt)
            elif evt_type == "error":
                await self._handle_error(evt)
            else:
                logger.trace(f"{self} unhandled event: {evt_type}")

    async def _handle_session_created(self, evt: dict):
        """Handle ``session.created`` event.

        Args:
            evt: The session created event from the server.
        """
        session = evt.get("session", {})
        self._session_id = session.get("id")
        logger.debug(f"{self} session created: {self._session_id}")

    async def _handle_session_updated(self, evt: dict):
        """Handle ``session.updated`` event.

        Args:
            evt: The session updated event from the server.
        """
        session = evt.get("session", {})
        if "voice" in session:
            updated_voice = session.get("voice")
            logger.debug(f"{self} voice updated to: {updated_voice}")

    async def _handle_audio_binary(self, data: bytes):
        """Handle a binary WebSocket frame containing raw PCM audio.

        Together sends audio as binary frames and control messages as JSON text.

        Args:
            data: Raw PCM audio bytes.
        """
        context_id = self._context_id
        if not context_id or not self.audio_context_available(context_id):
            return

        await self.stop_ttfb_metrics()
        frame = TTSAudioRawFrame(
            audio=data,
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
        await self.append_to_audio_context(context_id, frame)

    async def _handle_audio_delta(self, evt: dict):
        """Handle a JSON audio output delta containing base64-encoded audio.

        Args:
            evt: The delta event from the server.
        """
        context_id = self._context_id
        if not context_id or not self.audio_context_available(context_id):
            return

        delta = evt.get("delta")
        if delta:
            await self.stop_ttfb_metrics()
            audio_chunk = base64.b64decode(delta)
            frame = TTSAudioRawFrame(
                audio=audio_chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
            await self.append_to_audio_context(context_id, frame)

    async def _handle_audio_done(self, evt: dict):
        """Handle audio output completion for a speech segment.

        Args:
            evt: The done event from the server.
        """
        context_id = self._context_id
        item_id = evt.get("item_id")
        logger.debug(f"{self} audio generation complete for: {item_id}")
        if context_id and self.audio_context_available(context_id):
            await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)], context_id)
            await self.remove_audio_context(context_id)

    async def _handle_tts_failed(self, evt: dict):
        """Handle a TTS failure.

        Args:
            evt: The failed event containing error details.
        """
        context_id = self._context_id
        error = evt.get("error", {})
        await self.push_error(error_msg=f"TTS error: {error}")
        await self.push_frame(TTSStoppedFrame(context_id=context_id))
        await self.stop_all_metrics()
        self.reset_active_audio_context()

    async def _handle_error(self, evt: dict):
        """Handle a fatal error from the TTS session.

        Raises an exception so that ``WebsocketService`` can decide
        whether to attempt reconnection.

        Args:
            evt: The error event.
        """
        error = evt.get("error", {})
        error_msg = error.get("message", "Unknown error")
        error_code = error.get("code", "")
        msg = f"Together AI TTS error [{error_code}]: {error_msg}"
        await self.push_error(error_msg=msg)

    # ------------------------------------------------------------------
    # Audio context lifecycle
    # ------------------------------------------------------------------

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active Together context when the bot is interrupted.

        Args:
            context_id: The context that was interrupted.
        """
        await self.stop_all_metrics()
        await self._ws_send({"type": "input_text_buffer.clear"})

    async def on_audio_context_completed(self, context_id: str):
        """Handle context completion after all audio has been played.

        No server-side cleanup is needed — the Together AI server considers
        the context done once it has sent its ``audio_output.done`` message.

        Args:
            context_id: The context that completed playback.
        """
        pass

    # ------------------------------------------------------------------
    # TTS generation
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Together AI's streaming API.

        Audio context creation and ``TTSStartedFrame`` are managed by the base
        class (``push_start_frame=True``).  This method only sends the text to
        the WebSocket; audio frames arrive via ``_receive_messages``.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        self._context_id = context_id

        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()

            try:
                await self._ws_send({"type": "input_text_buffer.append", "text": text})
                await self._ws_send({"type": "input_text_buffer.commit"})
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Error sending TTS text: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}")
            yield TTSStoppedFrame(context_id=context_id)
