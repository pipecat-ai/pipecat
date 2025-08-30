#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Murf AI text-to-speech service implementation."""

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, Dict, Optional, Mapping, Any

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for Murf configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class MurfTTSService(AudioContextWordTTSService):
    """Murf AI WebSocket-based text-to-speech service.

    Provides real-time text-to-speech synthesis using Murf's WebSocket API.
    Supports various voice customization options including style, rate, pitch,
    and pronunciation dictionaries.
    """

    class InputParams(BaseModel):
        """Input parameters for Murf TTS configuration.

        Parameters:
            voice_id: Voice ID to use for TTS. Defaults to "en-UK-ruby".
            style: The style of speech. Defaults to "Conversational".
            rate: Speech rate (optional).
            pitch: Speech pitch (optional).
            pronunciation_dictionary: A map of words to their pronunciation details.
            variation: Higher values add more variation in Pause, Pitch, and Speed.
            multi_native_locale: Language for generated audio in Gen2 model.
            sample_rate: The sample rate for audio output. Defaults to 44100.
            channel_type: The channel type for audio output. Defaults to "MONO".
            format: The audio format for output. Defaults to "PCM".
        """

        voice_id: Optional[str] = "en-UK-ruby"
        style: Optional[str] = "Conversational"
        rate: Optional[int] = 0
        pitch: Optional[int] = 0
        pronunciation_dictionary: Optional[Dict[str, Dict[str, str]]] = None
        variation: Optional[int] = 1
        multi_native_locale: Optional[str] = None
        sample_rate: Optional[int] = 44100
        channel_type: Optional[str] = "MONO"
        format: Optional[str] = "PCM"

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.murf.ai/v1/speech/stream-input",
        params: Optional[InputParams] = None,
        aggregate_sentences: bool = True,
        **kwargs,
    ):
        """Initialize the Murf TTS service.

        Args:
            api_key: Murf API key for authentication.
            url: WebSocket URL for Murf TTS API.
            sample_rate: Audio sample rate (overrides params.sample_rate if provided).
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate sentences before synthesis.
            **kwargs: Additional arguments passed to parent AudioContextWordTTSService.
        """
        params = params or MurfTTSService.InputParams()

        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=params.sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._settings = {
            "voice_id": params.voice_id,
            "style": params.style,
            "rate": params.rate,
            "pitch": params.pitch,
            "pronunciation_dictionary": params.pronunciation_dictionary or {},
            "variation": params.variation,
            "multi_native_locale": params.multi_native_locale,
            "sample_rate": params.sample_rate,  
            "channel_type": params.channel_type,
            "format": params.format,
        }

        # Context management
        self._context_id: Optional[str] = None
        self._receive_task: Optional[asyncio.Task] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Murf service supports metrics generation.
        """
        return True

    def set_voice(self, voice_id: str):
        logger.info(f"Setting Murf TTS voice to: [{voice_id}]")
        self._settings["voice_id"] = voice_id

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if URL parameters changed.

        Args:
            settings: Dictionary of settings to update.
        """
        await super()._update_settings(settings)

        url_params = {"sample_rate", "format", "channel_type"}
        needs_reconnect = any(key in url_params for key in settings.keys())

        if needs_reconnect:
            await self._disconnect()
            await self._connect()
            logger.info(f"Reconnected Murf TTS due to URL parameter changes")

    async def _verify_connection(self) -> bool:
        """Verify the websocket connection is active and responsive.

        Returns:
            True if connection is verified working, False otherwise.
        """
        try:
            if not self._websocket or self._websocket.closed:
                return False
            await self._websocket.ping()
            return True
        except Exception as e:
            logger.error(f"{self} connection verification failed: {e}")
            return False

    async def start(self, frame: StartFrame):
        """Start the Murf TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Murf TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Murf TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to Murf WebSocket and start receive task."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Murf WebSocket and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Murf websocket."""
        try:
            if self._websocket and self._websocket.open:
                return

            url = (
                f"{self._url}"
                f"?api-key={self._api_key}"
                f"&sample_rate={self._settings['sample_rate']}"
                f"&format={self._settings['format']}"
                f"&channel_type={self._settings['channel_type']}"
            )

            self._websocket = await websockets.connect(url)
            logger.debug(f"Connected to Murf")

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Disconnect from Murf websocket."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Murf")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            if self._context_id:
                if self.audio_context_available(self._context_id):
                    await self.remove_audio_context(self._context_id)
            self._context_id = None
            self._websocket = None

    def _get_websocket(self):
        """Get the WebSocket connection if available."""
        if self._websocket and not self._websocket.closed:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        """Handle interruption by clearing the current context.

        Uses Murf's context clearing API instead of disconnecting/reconnecting.
        This is more efficient and follows Murf's design for handling interruptions.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()

        # Clear the current context if we have one using Murf's clear context API
        if self._context_id and self._websocket and not self._websocket.closed:
            try:
                clear_context_msg = {"clear": True, "context_id": self._context_id}
                await self._websocket.send(json.dumps(clear_context_msg))
                logger.debug(f"{self} cleared context {self._context_id}")
            except Exception as e:
                logger.error(f"{self} error clearing context: {e}")

        self._context_id = None

    async def _receive_messages(self):
        logger.debug(f"{self} receiving messages")
        """Receive and process messages from Murf WebSocket."""
        async for message in WatchdogAsyncIterator(self._websocket, manager=self.task_manager):
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._process_json_message(data)
                else:
                    logger.warning(
                        f"{self} received unexpected non-string message: {type(message)}"
                    )
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _process_json_message(self, data: Dict):
        """Process JSON messages from Murf.

        Handles two message types:
        1. audioOutput: Contains base64-encoded audio data
        2. finalOutput: Indicates end of synthesis (final=true)
        """
        received_ctx_id = data.get("context_id", self._context_id)
        if not self.audio_context_available(received_ctx_id):
            logger.warning(f"Ignoring message from unavailable context: {received_ctx_id}")
            return

        if "error" in data:
            logger.error(f"{self} error: {data['error']}")
            await self.push_frame(TTSStoppedFrame())
            await self.stop_all_metrics()
            await self.push_error(ErrorFrame(f"{self} error: {data['error']}"))
            await self.remove_audio_context(received_ctx_id)
            self._context_id = None
            return

        if "audio" in data:
            try:
                audio_b64 = data["audio"]
                audio_data = base64.b64decode(audio_b64)
                await self._process_audio_data_to_context(received_ctx_id, audio_data)
            except Exception as e:
                logger.error(f"{self} error decoding audio data: {e}")
            return

        if data.get("final") is True:
            logger.debug(f"{self} received final output for context {received_ctx_id}")
            await self.push_frame(TTSStoppedFrame())
            await self.stop_all_metrics()
            await self.remove_audio_context(received_ctx_id)
            self._context_id = None
            return

        logger.debug(f"{self} received unknown message: {data}")

    async def _process_audio_data_to_context(self, context_id: str, audio_data: bytes):
        """Process decoded audio data from Murf and append to context."""
        await self.stop_ttfb_metrics()
        frame = TTSAudioRawFrame(
            audio=audio_data,
            sample_rate=self.sample_rate,
            num_channels=1,
        )
        await self.append_to_audio_context(context_id, frame)

    def _build_voice_config_message(self, text: str, is_last: bool = False) -> Dict:
        """Build voice configuration message for Murf API.

        Args:
            text: The text to synthesize.
            is_last: Whether this is the last message in the sequence.

        Returns:
            The voice configuration message according to Murf API format.
        """
        message = {
            "voice_config": {
                "voice_id": self._settings["voice_id"],
                "style": self._settings["style"],
                "rate": self._settings["rate"],
                "pitch": self._settings["pitch"],
                "pronunciation_dictionary": self._settings["pronunciation_dictionary"],
                "variation": self._settings["variation"],
            },
            "context_id": self._context_id,
            "text": text,
            "end": is_last,
        }
        logger.debug(f"{self} voice config message: {message}")

        if self._settings["multi_native_locale"]:
            message["voice_config"]["multi_native_locale"] = self._settings["multi_native_locale"]

    
        return message

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Murf's streaming WebSocket API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._context_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._context_id = str(uuid.uuid4())
                await self.create_audio_context(self._context_id)

            voice_config_msg = self._build_voice_config_message(text, is_last=True)

            try:
                await self._get_websocket().send(json.dumps(voice_config_msg))
                await self.start_tts_usage_metrics(text)
                logger.debug(f"{self} sent voice config message for context {self._context_id}")
                logger.debug(f"{self} voice config message: {voice_config_msg}")
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self.stop_all_metrics()
                self._context_id = None
                return

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield TTSStoppedFrame()
            await self.stop_all_metrics()


__all__ = ["MurfTTSService"]
