#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together AI text-to-speech service implementation."""

import base64
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.services.settings import TTSSettings

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
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class TogetherTTSSettings(TTSSettings):
    """Settings for the Together AI TTS service.

    Parameters:
        model: Together AI TTS model to use.
        voice: Voice to use for synthesis.
        language: Language of the text input.
        max_partial_length: Maximum partial text length for streaming.
    """

    model: str = "canopylabs/orpheus-3b-0.1-ft"
    language: Optional[Language] = Language.EN
    voice: Optional[str] = "tara"
    max_partial_length: Optional[int] = None


class TogetherTTSService(WebsocketTTSService):
    """Together AI TTS service with WebSocket streaming.

    Provides text-to-speech using Together AI's realtime WebSocket API.
    Supports streaming synthesis with configurable voice and model options.
    """

    _settings: TogetherTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "canopylabs/orpheus-3b-0.1-ft",
        voice: str = "tara",
        language: Optional[Language] = Language.EN,
        max_partial_length: Optional[int] = None,
        url: str = "wss://api.together.ai/v1/audio/speech/websocket",
        sample_rate: Optional[int] = 24000,
        **kwargs,
    ):
        """Initialize the Together AI TTS service.

        Args:
            api_key: Together AI API key for authentication.
            model: Together AI TTS model. Defaults to "canopylabs/orpheus-3b-0.1-ft".
            voice: Voice to use for synthesis. Defaults to "tara".
            language: Language of the text input. Defaults to English.
            max_partial_length: Maximum partial text length for streaming.
            url: WebSocket URL for Together AI TTS API.
            sample_rate: Audio sample rate (default: 24000).
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            sample_rate=sample_rate,
            settings=TogetherTTSSettings(
                model=model,
                voice=voice,
                language=language,
                max_partial_length=max_partial_length,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
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
        url = f"{self._url}?model={self._settings.model}&voice={self._settings.voice}"
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

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
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
            if not isinstance(message, str):
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
                text = evt.get("text", "")
                logger.debug(f"{self} text received: {text[:50]}{'...' if len(text) > 50 else ''}")
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

    async def _handle_audio_delta(self, evt: dict):
        """Handle an audio output delta containing a chunk of synthesized audio.

        Args:
            evt: The delta event from the server.
        """
        delta = evt.get("delta")
        if delta:
            try:
                await self.stop_ttfb_metrics()
                audio_chunk = base64.b64decode(delta)
                frame = TTSAudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=self._context_id,
                )
                await self.push_frame(frame)
            except Exception as e:
                logger.error(f"{self} error processing audio delta: {e}")

    async def _handle_audio_done(self, evt: dict):
        """Handle audio output completion for a speech segment.

        Args:
            evt: The done event from the server.
        """
        item_id = evt.get("item_id")
        logger.debug(f"{self} audio generation complete for: {item_id}")
        await self.push_frame(TTSStoppedFrame(context_id=self._context_id))

    async def _handle_tts_failed(self, evt: dict):
        """Handle a TTS failure.

        Args:
            evt: The failed event containing error details.
        """
        error = evt.get("error", {})
        await self.push_error(error_msg=f"TTS error: {error}")
        await self.push_frame(TTSStoppedFrame(context_id=self._context_id))

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
        raise Exception(msg)

    # ------------------------------------------------------------------
    # Interruption handling
    # ------------------------------------------------------------------

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by canceling current generation.

        Args:
            frame: The interruption frame.
            direction: Frame processing direction.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        await self._ws_send({"type": "input_text_buffer.clear"})

    # ------------------------------------------------------------------
    # TTS generation
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Together AI's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()
                if not self._websocket or self._websocket.state is not State.OPEN:
                    logger.error(f"{self} failed to connect to WebSocket")
                    yield TTSStoppedFrame(context_id=context_id)
                    return

            self._context_id = context_id

            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            try:
                await self._ws_send({"type": "input_text_buffer.append", "text": text})
                await self._ws_send({"type": "input_text_buffer.commit"})
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return

            yield None

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(error_msg=f"Error generating TTS: {e}", exception=e)
            yield TTSStoppedFrame(context_id=context_id)
