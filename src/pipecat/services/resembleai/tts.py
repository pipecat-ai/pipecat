#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resemble AI text-to-speech service implementations."""

import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Resemble AI, you need to `pip install pipecat-ai[resembleai]`.")
    raise Exception(f"Missing module: {e}")


class ResembleAITTSService(AudioContextWordTTSService):
    """Resemble AI TTS service with WebSocket streaming and word timestamps.

    Provides text-to-speech using Resemble AI's streaming WebSocket API.
    Supports word-level timestamps and audio context management for handling
    multiple simultaneous synthesis requests with proper interruption support.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "wss://websocket.cluster.resemble.ai/stream",
        precision: Optional[str] = "PCM_16",
        output_format: Optional[str] = "wav",
        sample_rate: Optional[int] = 22050,
        **kwargs,
    ):
        """Initialize the Resemble AI TTS service.

        Args:
            api_key: Resemble AI API key for authentication.
            voice_id: Voice UUID to use for synthesis.
            url: WebSocket URL for Resemble AI TTS API.
            precision: PCM bit depth (PCM_32, PCM_24, PCM_16, or MULAW).
            output_format: Audio format (wav or mp3).
            sample_rate: Audio sample rate (8000, 16000, 22050, 32000, or 44100). Defaults to 22050.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._url = url
        self._settings = {
            "precision": precision,
            "output_format": output_format,
            "sample_rate": sample_rate,
        }

        self._websocket = None
        self._request_id_counter = 0
        self._current_request_id = None
        self._receive_task = None

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Resemble AI service supports metrics generation.
        """
        return True

    def _build_msg(self, text: str = "") -> str:
        """Build a JSON message for the Resemble AI WebSocket API.

        Args:
            text: The text or SSML to synthesize.

        Returns:
            JSON string containing the request payload.
        """
        msg = {
            "voice_uuid": self._voice_id,
            "data": text,
            "binary_response": False,  # Use JSON frames to get timestamps
            "request_id": self._request_id_counter,
            "output_format": self._settings["output_format"],
            "sample_rate": self._settings["sample_rate"],
            "precision": self._settings["precision"],
            "no_audio_header": True,
        }

        self._request_id_counter += 1
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the Resemble AI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Resemble AI TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Resemble AI TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to the Resemble AI WebSocket."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Resemble AI WebSocket."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Resemble AI."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Resemble AI TTS")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection to Resemble AI."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Resemble AI")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._current_request_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The active WebSocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by stopping current synthesis.

        Args:
            frame: The interruption frame.
            direction: The direction of frame processing.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        # Note: Resemble AI doesn't have an explicit cancel mechanism,
        # but we can stop processing by resetting our current request_id
        self._current_request_id = None

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if not self._current_request_id:
            return
        logger.trace(f"{self}: flushing audio")
        # For Resemble AI, we just wait for the audio_end message
        # which is handled in _process_messages
        self._current_request_id = None

    async def _process_messages(self):
        """Process incoming WebSocket messages from Resemble AI."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"{self} received invalid JSON: {message}")
                continue

            if not msg:
                continue

            msg_type = msg.get("type")
            request_id = msg.get("request_id")

            # Convert request_id to string for audio context tracking
            request_id_str = str(request_id)

            # Check if this message belongs to a valid audio context
            if not self.audio_context_available(request_id_str):
                continue

            if msg_type == "audio":
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()

                # Decode base64 audio content
                audio_content = msg.get("audio_content", "")
                if audio_content:
                    audio_data = base64.b64decode(audio_content)
                    frame = TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    await self.append_to_audio_context(request_id_str, frame)

                # Process timestamps if available
                timestamps = msg.get("audio_timestamps", {})
                if timestamps:
                    graph_chars = timestamps.get("graph_chars", [])
                    graph_times = timestamps.get("graph_times", [])

                    # Convert graph_times (start, end pairs) to word timestamps
                    word_times = []
                    for char, times in zip(graph_chars, graph_times):
                        if times and len(times) >= 2:
                            start_time = times[0]
                            word_times.append((char, start_time))

                    if word_times:
                        await self.add_word_timestamps(word_times)

            elif msg_type == "audio_end":
                await self.stop_ttfb_metrics()
                await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)])
                await self.remove_audio_context(request_id_str)
                # Clear current request if this was it
                if self._current_request_id == request_id:
                    self._current_request_id = None

            elif msg_type == "error":
                error_name = msg.get("error_name", "Unknown")
                error_msg = msg.get("message", "Unknown error")
                status_code = msg.get("status_code", 0)
                logger.error(f"{self} error: {error_name} (status {status_code}): {error_msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(error=f"{self} error: {error_name} - {error_msg}"))

                # Clear current request if this was it
                if self._current_request_id == request_id:
                    self._current_request_id = None

                # Check if this is an unrecoverable error (connection-level failure)
                if status_code in [401, 403]:
                    # Close and reconnect for auth errors
                    await self._disconnect_websocket()
                    await self._connect_websocket()
            else:
                logger.warning(f"{self} unknown message type: {msg_type}")

    async def _receive_messages(self):
        """Main loop for receiving messages from Resemble AI."""
        while True:
            try:
                await self._process_messages()
            except Exception as e:
                logger.error(f"{self} error in receive loop: {e}")
                # Try to reconnect
                logger.debug(f"{self} Resemble AI connection lost, reconnecting")
                await self._connect_websocket()

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Resemble AI's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._current_request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                # Track the current request_id we're processing
                self._current_request_id = self._request_id_counter

            # Create audio context using request_id (converted to string)
            request_id_str = str(self._request_id_counter)
            await self.create_audio_context(request_id_str)

            msg = self._build_msg(text=text)

            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} exception: {e}")
                yield ErrorFrame(error=f"{self} error: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
