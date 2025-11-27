#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram text-to-speech service implementation.

This module provides integration with Deepgram's text-to-speech API
for generating speech from text using various voice models.
"""

import json
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

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
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use DeepgramWebsocketTTSService, you need to `pip install pipecat-ai[deepgram]`."
    )
    raise Exception(f"Missing module: {e}")


class DeepgramTTSService(WebsocketTTSService):
    """Deepgram WebSocket-based text-to-speech service.

    Provides real-time text-to-speech synthesis using Deepgram's WebSocket API.
    Supports streaming audio generation with interruption handling via the Clear
    message for conversational AI use cases.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-2-helena-en",
        base_url: str = "wss://api.deepgram.com",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        """Initialize the Deepgram WebSocket TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            voice: Voice model to use for synthesis. Defaults to "aura-2-helena-en".
            base_url: WebSocket base URL for Deepgram API. Defaults to "wss://api.deepgram.com".
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            encoding: Audio encoding format. Defaults to "linear16".
            **kwargs: Additional arguments passed to parent InterruptibleTTSService class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram WebSocket TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Deepgram WebSocket TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram WebSocket TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram WebSocket TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for LLM response end.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # When the LLM finishes responding, flush any remaining text in Deepgram's buffer
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _connect(self):
        """Connect to Deepgram WebSocket and start receive task."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Deepgram WebSocket and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Deepgram WebSocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Deepgram WebSocket")

            # Build WebSocket URL with query parameters
            params = []
            params.append(f"model={self._voice_id}")
            params.append(f"encoding={self._settings['encoding']}")
            params.append(f"sample_rate={self.sample_rate}")

            url = f"{self._base_url}/v1/speak?{'&'.join(params)}"

            headers = {"Authorization": f"Token {self._api_key}"}

            self._websocket = await websocket_connect(url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and reset state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Deepgram WebSocket")
                # Send Close message to gracefully close the connection
                await self._websocket.send(json.dumps({"type": "Close"}))
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by sending Clear message to Deepgram.

        The Clear message will clear Deepgram's internal text buffer and stop
        sending audio, allowing for a new response to be generated.
        """
        await super()._handle_interruption(frame, direction)

        # Send Clear message to stop current audio generation
        if self._websocket:
            try:
                clear_msg = {"type": "Clear"}
                await self._websocket.send(json.dumps(clear_msg))
            except Exception as e:
                logger.error(f"{self} error sending Clear message: {e}")

    async def _receive_messages(self):
        """Receive and process messages from Deepgram WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Binary message contains audio data
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(message, self.sample_rate, 1)
                await self.push_frame(frame)
            elif isinstance(message, str):
                # Text message contains metadata or control messages
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "Metadata":
                        logger.trace(f"Received metadata: {msg}")
                    elif msg_type == "Flushed":
                        logger.trace(f"Received Flushed: {msg}")
                        # Flushed indicates the end of audio generation for the current buffer
                        # This happens after flush_audio() is called
                        await self.push_frame(TTSStoppedFrame())
                    elif msg_type == "Cleared":
                        logger.trace(f"Received Cleared: {msg}")
                        # Buffer has been cleared after interruption
                        # TTSStoppedFrame will be sent by the interruption handler
                    elif msg_type == "Warning":
                        logger.warning(
                            f"{self} warning: {msg.get('description', 'Unknown warning')}"
                        )
                    else:
                        logger.debug(f"Received unknown message type: {msg}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending Flush command.

        This should be called when the LLM finishes a complete response to force
        generation of audio from Deepgram's internal text buffer.
        """
        if self._websocket:
            try:
                flush_msg = {"type": "Flush"}
                await self._websocket.send(json.dumps(flush_msg))
            except Exception as e:
                logger.error(f"{self} error sending Flush message: {e}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepgram's WebSocket TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Reconnect if the websocket is closed
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Send text message to Deepgram
            # Note: We don't send Flush here - that should only be sent when the
            # LLM finishes a complete response via flush_audio()
            speak_msg = {"type": "Speak", "text": text}
            await self._get_websocket().send(json.dumps(speak_msg))

            # The actual audio frames will be handled in _receive_messages
            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class DeepgramHttpTTSService(TTSService):
    """Deepgram HTTP text-to-speech service.

    Provides text-to-speech synthesis using Deepgram's HTTP TTS API.
    Supports various voice models and audio encoding formats with
    configurable sample rates and quality settings.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-2-helena-en",
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.deepgram.com",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        """Initialize the Deepgram TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            voice: Voice model to use for synthesis. Defaults to "aura-2-helena-en".
            aiohttp_session: Shared aiohttp session for HTTP requests with connection pooling.
            base_url: Custom base URL for Deepgram API. Defaults to "https://api.deepgram.com".
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            encoding: Audio encoding format. Defaults to "linear16".
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url
        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram TTS service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepgram's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Build URL with parameters
        url = f"{self._base_url}/v1/speak"

        headers = {"Authorization": f"Token {self._api_key}", "Content-Type": "application/json"}

        params = {
            "model": self._voice_id,
            "encoding": self._settings["encoding"],
            "sample_rate": self.sample_rate,
            "container": "none",
        }

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
                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                first_chunk = True
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    if chunk:
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )

            yield TTSStoppedFrame()

        except Exception as e:
            yield ErrorFrame(f"Error getting audio: {str(e)}")
