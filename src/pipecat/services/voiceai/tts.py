#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice.ai text-to-speech service implementation."""

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Voice.ai, you need to `pip install pipecat-ai[voiceai]`.")
    raise Exception(f"Missing module: {e}")


def language_to_voiceai_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Voice.ai language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Voice.ai language code (ISO 639-1 format), or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.CA: "ca",  # Catalan
        Language.DE: "de",  # German
        Language.EN: "en",  # English
        Language.ES: "es",  # Spanish
        Language.FR: "fr",  # French
        Language.IT: "it",  # Italian
        Language.NL: "nl",  # Dutch
        Language.PL: "pl",  # Polish
        Language.PT: "pt",  # Portuguese
        Language.RU: "ru",  # Russian
        Language.SV: "sv",  # Swedish
    }

    return LANGUAGE_MAP.get(language)


class VoiceAiTTSService(AudioContextTTSService):
    """Text-to-speech service using Voice.ai's Multi-Context WebSocket API.

    By Voice.ai.

    Converts text to speech using Voice.ai's TTS models with support for multiple
    languages. Maintains a persistent WebSocket connection that handles multiple
    concurrent TTS streams (contexts), with automatic context management and
    interruption handling.

    Supported features:

    - Multi-context WebSocket for concurrent TTS streams
    - Multiple language support (en, ca, sv, es, fr, de, it, pt, pl, ru, nl)
    - Configurable voice selection per context
    - Temperature and top_p control for generation variety
    - Raw PCM audio output at 32kHz mono
    - Multiple TTS model options
    - Automatic interruption and context cleanup
    - Connection keepalive for idle timeout prevention
    - Context-based audio ordering
    - Flow control: Limits concurrent in-flight requests (max 2) to reduce
      wasted compute on user interruptions

    Example::

        tts = VoiceAiTTSService(
            api_key="vk_your-api-key",
            voice_id="your-voice-id",
            params=VoiceAiTTSService.InputParams(
                language=Language.EN,
                temperature=1.0,
                top_p=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Voice.ai TTS.

        Parameters:
            language: Target language for synthesis. Supported languages include English (en),
                Catalan (ca), Swedish (sv), Spanish (es), French (fr), German (de),
                Italian (it), Portuguese (pt), Polish (pl), Russian (ru), Dutch (nl).
                Defaults to English (en).
            model: TTS model to use. Supported models include voiceai-tts-v1-latest,
                voiceai-tts-v1-2025-12-19 (English only), voiceai-tts-multilingual-v1-latest,
                voiceai-tts-multilingual-v1-2025-01-14 (multilingual). If not provided,
                automatically selected based on language. English uses non-multilingual models;
                other languages use multilingual models.
            audio_format: Audio format for output. Supported formats include "pcm" for raw PCM
                audio. Defaults to "pcm".
            temperature: Temperature for generation (0.0-2.0). Higher values produce more
                random output. Defaults to 1.0.
            top_p: Top-p sampling parameter (0.0-1.0). Controls diversity of output.
                Defaults to 0.8.
        """

        language: Language = Language.EN
        model: Optional[str] = None
        audio_format: str = "pcm"
        temperature: float = Field(default=1.0, ge=0.0, le=2.0)
        top_p: float = Field(default=0.8, ge=0.0, le=1.0)

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "wss://dev.voice.ai/api/v1/tts/multi-stream",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        aggregate_sentences: bool = True,
        **kwargs,
    ):
        """Initialize the Voice.ai TTS service with multi-context WebSocket connection.

        Args:
            api_key: Voice.ai API key for authentication (format: vk_*).
            voice_id: Voice identifier for synthesis. If not provided, uses default built-in voice.
            url: WebSocket URL for Voice.ai multi-context TTS API.
            sample_rate: Output audio sample rate. Defaults to 32000 Hz (Voice.ai native rate).
            params: Optional input parameters to configure voice synthesis settings.
            aggregate_sentences: Whether to aggregate text by sentences before TTS. When True
                (default), each sentence is sent separately which provides lower latency but may
                cause minor audio artifacts between sentences. When False, larger text chunks
                are batched together for more natural speech flow at the cost of higher latency.
            **kwargs: Additional keyword arguments passed to AudioContextTTSService base class.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            # Pause frame processing while TTS is generating to prevent
            # pile-up of LLM text (reduces wasted TTS if interrupted)
            pause_frame_processing=True,
            # Auto-push stop frames after idle timeout
            push_stop_frames=True,
            sample_rate=sample_rate or 32000,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._url = url
        self._settings = {}

        # Register voice for service tracking
        if voice_id:
            self.set_voice(voice_id)

        # WebSocket state
        self._websocket = None
        self._receive_task = None
        self._keepalive_task = None
        self._started = False
        self._disconnecting = False

        # Context management (like ElevenLabs - persist across sentences)
        self._context_id = None

        # Flow control: limit concurrent in-flight requests to prevent waste on interruption
        self._max_in_flight = 2  # Allow max 2 sentences being processed at once
        self._in_flight_semaphore = asyncio.Semaphore(self._max_in_flight)

        # Set up parameters
        if params:
            self._settings["language"] = language_to_voiceai_language(params.language) or "en"
            self._settings["model"] = params.model
            self._settings["audio_format"] = params.audio_format
            self._settings["temperature"] = params.temperature
            self._settings["top_p"] = params.top_p
        else:
            # Defaults
            self._settings["language"] = "en"
            self._settings["model"] = None
            self._settings["audio_format"] = "pcm"
            self._settings["temperature"] = 1.0
            self._settings["top_p"] = 0.8

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Voice.ai service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Voice.ai language format.

        Args:
            language: The language to convert.

        Returns:
            The Voice.ai-specific language code (ISO 639-1), or None if not supported.
        """
        return language_to_voiceai_language(language)

    async def start(self, frame: StartFrame):
        """Start the Voice.ai TTS service and establish WebSocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Voice.ai TTS service and close WebSocket connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Voice.ai TTS service and close WebSocket connection.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with end-of-turn flush handling.

        Calls flush_audio() when LLMFullResponseEndFrame or EndFrame is received,
        triggering Voice.ai to generate audio for all buffered text.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # Flush audio at end of LLM response (like WordTTSService does)
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False
            # Clean up context on TTS stop (end of turn)
            if isinstance(frame, TTSStoppedFrame) and self._context_id:
                if self.audio_context_available(self._context_id):
                    await self.remove_audio_context(self._context_id)
                self._context_id = None

                # Release any held semaphore slots on TTS stop
                while self._in_flight_semaphore._value < self._max_in_flight:
                    self._in_flight_semaphore.release()

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect with new configuration.

        Args:
            settings: Dictionary of settings to update.
        """
        await super()._update_settings(settings)
        # Reconnect to apply new settings
        logger.info(f"Reconnecting Voice.ai TTS with updated settings")
        await self._disconnect()
        await self._connect()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by closing context and resetting state.

        Args:
            frame: The interruption frame.
            direction: The direction of the frame.
        """
        await super()._handle_interruption(frame, direction)

        # Close the current context when interrupted (like ElevenLabs)
        if self._context_id and self._websocket:
            try:
                await self._websocket.send(
                    json.dumps({"context_id": self._context_id, "close_context": True})
                )
            except Exception as e:
                logger.warning(f"Error closing context on interruption: {e}")

        # Release any held semaphore slots (drain to max value)
        # This ensures we don't deadlock if interruption happens mid-processing
        while self._in_flight_semaphore._value < self._max_in_flight:
            self._in_flight_semaphore.release()

        # Reset state
        self._context_id = None
        self._started = False

    async def _connect(self):
        """Connect to Voice.ai WebSocket and start background tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Voice.ai WebSocket and clean up tasks."""
        await super()._disconnect()

        try:
            # Set flag to prevent new operations
            self._disconnecting = True

            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            if self._keepalive_task:
                await self.cancel_task(self._keepalive_task, timeout=2.0)
                self._keepalive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            await self.push_error(error_msg=f"Error during disconnect: {e}", exception=e)
        finally:
            # Reset state
            self._started = False
            self._websocket = None
            self._disconnecting = False

    async def _connect_websocket(self):
        """Establish WebSocket connection for multi-context streaming."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            # Connect with authentication
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)

            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Voice.ai WebSocket: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._started = False
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current websocket connection.

        Returns:
            The active websocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    async def flush_audio(self):
        """Flush any buffered text and finalize audio generation.

        Called by Pipecat when LLMFullResponseEndFrame or EndFrame is received,
        signaling the end of a conversational turn. This triggers Voice.ai to
        generate audio for any buffered text.
        """
        if not self._context_id or not self._websocket:
            return

        msg = {"context_id": self._context_id, "flush": True}
        await self._websocket.send(json.dumps(msg))

    async def _receive_messages(self):
        """Receive and process messages from Voice.ai multi-context WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)

                # Get context_id from the response
                received_ctx_id = msg.get("context_id")

                # Check if this message has the completion signal
                is_final = msg.get("is_last", False)

                # Skip messages for unavailable contexts (old/closed contexts)
                if received_ctx_id and not self.audio_context_available(received_ctx_id):
                    if "audio" in msg:
                        audio_size = len(msg["audio"]) if msg["audio"] else 0
                        logger.error(
                            f"Dropping audio for unavailable context {received_ctx_id}: "
                            f"{audio_size} base64 chars (~{audio_size * 3 // 4} bytes)"
                        )
                    continue

                # Handle audio chunk
                if "audio" in msg:
                    await self.stop_ttfb_metrics()

                    # Decode base64 to get raw PCM audio bytes
                    try:
                        audio_data = base64.b64decode(msg["audio"])
                    except Exception as e:
                        logger.error(f"Failed to decode base64 audio for {received_ctx_id}: {e}")
                        continue

                    # Voice.ai returns PCM audio (16-bit samples, mono)
                    frame = TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

                    # Append audio to the appropriate context
                    if received_ctx_id:
                        if self.audio_context_available(received_ctx_id):
                            await self.append_to_audio_context(received_ctx_id, frame)
                        else:
                            logger.error(
                                f"Dropping audio at append: Context {received_ctx_id} not available, "
                                f"audio size: {len(audio_data)} bytes"
                            )

                # Handle completion signal from Voice.ai
                # With per-sentence flush, is_last fires after each sentence
                # Release the in-flight semaphore to allow next sentence to be sent
                if is_final and received_ctx_id:
                    # Release semaphore to allow next text to be sent
                    self._in_flight_semaphore.release()

                # Handle error
                if "error" in msg:
                    error_msg = msg["error"]
                    await self.push_error(error_msg=f"TTS Error: {error_msg}")

                    # Release semaphore on error to prevent deadlock
                    if self._in_flight_semaphore._value < self._max_in_flight:
                        self._in_flight_semaphore.release()

                    # Clean up context on error
                    if received_ctx_id and self.audio_context_available(received_ctx_id):
                        await self.remove_audio_context(received_ctx_id)

    async def _receive_task_handler(self, report_error):
        """Background task to receive messages from WebSocket.

        Args:
            report_error: Callback to report errors.
        """
        try:
            await self._receive_messages()
        except Exception as e:
            if not self._disconnecting:
                logger.error(f"Error in receive task: {e}")
                await report_error(ErrorFrame(error=f"Voice.ai receive error: {e}", exception=e))

    async def _keepalive_task_handler(self):
        """Background task to send keepalive messages."""
        KEEPALIVE_SLEEP = 30  # Send keepalive every 30 seconds
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive ping to maintain connection."""
        if self._disconnecting:
            return

        if self._websocket and self._websocket.state == State.OPEN:
            try:
                await self._websocket.ping()
            except Exception as e:
                logger.warning(f"Keepalive ping failed: {e}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Voice.ai's multi-context WebSocket API.

        Uses a persistent context across sentences within a turn. Each sentence is
        flushed immediately to trigger audio streaming. Context is reused for
        potential prosodic continuity.

        Flow control: Uses a semaphore to limit concurrent in-flight requests.
        This prevents wasting compute if the user interrupts - only 1-2 sentences
        will be in-flight at any time.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: TTSStartedFrame. Audio frames are pushed via the receive task
                to the context queue.
        """
        try:
            # Ensure we're connected
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Flow control: Wait if we already have too many requests in flight
            # This prevents sending all sentences at once and wasting compute on interruption
            await self._in_flight_semaphore.acquire()

            try:
                # First sentence in turn: create context and yield started frame
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True

                    # Create a new context for this turn (reused across sentences)
                    if not self._context_id:
                        self._context_id = str(uuid.uuid4())
                    if not self.audio_context_available(self._context_id):
                        await self.create_audio_context(self._context_id)

                    # Send first message with settings and flush
                    init_message = {
                        "context_id": self._context_id,
                        "audio_format": self._settings["audio_format"],
                        "temperature": self._settings["temperature"],
                        "top_p": self._settings["top_p"],
                        "language": self._settings["language"],
                        "text": text,
                        "flush": True,  # Flush to trigger audio streaming
                    }

                    # Add optional fields
                    if self._voice_id:
                        init_message["voice_id"] = self._voice_id
                    if self._settings["model"]:
                        init_message["model"] = self._settings["model"]

                    await self._websocket.send(json.dumps(init_message))
                else:
                    # Subsequent sentences: reuse context, send text with flush
                    if self._websocket and self._context_id:
                        msg = {"context_id": self._context_id, "text": text, "flush": True}
                        await self._websocket.send(json.dumps(msg))

                await self.start_tts_usage_metrics(text)

            except Exception as e:
                logger.error(f"Error in Voice.ai TTS: {e}")
                # Release semaphore on error
                self._in_flight_semaphore.release()
                yield ErrorFrame(error=f"Voice.ai TTS error: {e}", exception=e)
                yield TTSStoppedFrame()
                self._started = False
                self._context_id = None
                return

            # Yield None - audio streams immediately after flush
            yield None

        except Exception as e:
            logger.error(f"Error in Voice.ai TTS: {e}")
            # Release semaphore on outer exception
            self._in_flight_semaphore.release()
            yield ErrorFrame(error=f"Voice.ai TTS error: {e}", exception=e)

    async def _report_error(self, error: ErrorFrame):
        """Report errors from background tasks.

        Args:
            error: The error frame to report.
        """
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
