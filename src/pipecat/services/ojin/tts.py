#
# Copyright (c) 2024–2025, Journee Technologies GmbH
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger
from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_tts_client import OjinTTSClient
from ojin.ojin_tts_messages import (
    IOjinTTSClient,
    OjinTTSCancelInteractionMessage,
    OjinTTSEndInteractionMessage,
    OjinTTSInteractionResponseMessage,
    OjinTTSSessionReadyMessage,
    OjinTTSTextInputMessage,
)
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


@dataclass
class OjinTTSServiceSettings:
    """Settings for Ojin TTS Service."""

    api_key: str = field(default="")
    ws_url: str = field(default="wss://models.ojin.ai/realtime")
    config_id: str = field(default="")
    sample_rate: int = field(default=24000)
    client_connect_max_retries: int = field(default=3)
    client_reconnect_delay: float = field(default=1.0)


class OjinTTSService(TTSService):
    """Ojin TTS Service for Pipecat.

    This service connects to the Ojin TTS inference server via WebSocket,
    sends text input, and receives streaming audio output.

    Usage:
        settings = OjinTTSServiceSettings(
            api_key="your-api-key",
            config_id="your-config-id",
        )
        tts = OjinTTSService(settings)

        # Use in a pipeline or directly
        async for frame in tts.run_tts("Hello, world!"):
            # Process audio frames
            pass
    """

    def __init__(
        self,
        settings: OjinTTSServiceSettings,
        client: IOjinTTSClient | None = None,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=settings.sample_rate, **kwargs)
        logger.debug(f"OjinTTSService initialized with settings {settings}")

        self._settings = settings
        if client is None:
            self._client = OjinTTSClient(
                ws_url=settings.ws_url,
                api_key=settings.api_key,
                config_id=settings.config_id,
                reconnect_attempts=settings.client_connect_max_retries,
                reconnect_delay=settings.client_reconnect_delay,
                mode=os.getenv("OJIN_MODE", ""),
            )
        else:
            self._client = client

        self._session_data: Optional[Dict[str, Any]] = None
        self._receive_msg_task: Optional[asyncio.Task] = None
        self._audio_queue: asyncio.Queue[OjinTTSInteractionResponseMessage | None] = asyncio.Queue()

    def can_generate_metrics(self) -> bool:
        return True

    async def connect_with_retry(self) -> bool:
        """Attempt to connect with configurable retry mechanism."""
        last_error: Optional[Exception] = None
        assert self._client is not None

        for attempt in range(self._settings.client_connect_max_retries):
            try:
                logger.info(
                    f"Connection attempt {attempt + 1}/{self._settings.client_connect_max_retries}"
                )
                await self._client.connect()
                logger.info("Successfully connected to TTS server!")
                return True

            except ConnectionError as e:
                last_error = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < self._settings.client_connect_max_retries - 1:
                    logger.info(f"Retrying in {self._settings.client_reconnect_delay} seconds...")
                    await asyncio.sleep(self._settings.client_reconnect_delay)

        logger.error(
            f"Failed to connect to TTS after {self._settings.client_connect_max_retries} attempts. Last error: {last_error}"
        )
        return False

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the TTS server."""
        if isinstance(message, OjinTTSSessionReadyMessage):
            if message.parameters is not None:
                self._session_data = message.parameters

            logger.info(f"Received Session Ready session data: {message}")
            if self._session_data and self._session_data.get("server_id"):
                logger.info(f"Connected to server: {self._session_data.get('server_id')}")

        elif isinstance(message, OjinTTSInteractionResponseMessage):
            # Queue audio response for consumption
            await self._audio_queue.put(message)
            logger.debug(
                f"Received TTS audio chunk {message.index}, "
                f"is_final={message.is_final_response}, "
                f"size={len(message.audio_bytes)}"
            )

        elif isinstance(message, ErrorResponseMessage):
            logger.error(f"TTS Error: {message.payload.code}")
            # Signal error by putting None in queue
            await self._audio_queue.put(None)

    async def _receive_messages_loop(self):
        """Background task to receive messages from the TTS server."""
        try:
            while self._client.is_connected():
                try:
                    message = await asyncio.wait_for(
                        self._client.receive_message(), timeout=1.0
                    )
                    if message:
                        await self._handle_ojin_message(message)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving TTS message: {e}")
                    break
        except asyncio.CancelledError:
            pass

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio for the given text.

        Args:
            text: The text to synthesize.

        Yields:
            TTSAudioRawFrame frames containing the synthesized audio.
        """
        await self.start(None)

        if not self._client.is_connected():
            logger.error("Not connected to TTS server")
            yield ErrorFrame(error="Not connected to TTS server")
            return

        # Clear queue before starting new interaction
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Start interaction
        await self._client.start_interaction()

        # Send text input
        try:
            await self._client.send_message(
                OjinTTSTextInputMessage(text=text, params=None)
            )
            logger.debug(f"Sent TTS text input: {text[:50]}...")

            # Send end interaction to signal we're done with input
            await self._client.send_message(OjinTTSEndInteractionMessage())
            logger.debug("Sent TTS end interaction")

        except Exception as e:
            logger.error(f"Failed to send TTS text: {e}")
            yield ErrorFrame(error=f"Failed to send text: {e}")
            return

        # Yield TTSStartedFrame
        yield TTSStartedFrame()

        # Receive audio chunks
        while True:
            try:
                response = await asyncio.wait_for(self._audio_queue.get(), timeout=30.0)

                if response is None:
                    # Error or end of stream
                    break

                # Yield audio frame only if it has data
                if response.audio_bytes:
                    yield TTSAudioRawFrame(
                        audio=response.audio_bytes,
                        sample_rate=self._settings.sample_rate,
                        num_channels=1,
                    )

                if response.is_final_response:
                    logger.debug("Received final TTS audio chunk")
                    break

            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for TTS audio chunk")
                break

        # Yield TTSStoppedFrame
        yield TTSStoppedFrame()

    async def _start(self) -> None:
        """Internal start method to connect and begin receiving messages."""
        if not await self.connect_with_retry():
            logger.error("Failed to start TTS service - connection failed")
            return

        # Start background task to receive messages
        self._receive_msg_task = asyncio.create_task(self._receive_messages_loop())

        # Wait for session ready
        try:
            message = await asyncio.wait_for(self._client.receive_message(), timeout=10.0)
            if message:
                await self._handle_ojin_message(message)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for TTS session ready")

    async def _stop(self) -> None:
        """Internal stop method to disconnect and cleanup."""
        if self._receive_msg_task:
            self._receive_msg_task.cancel()
            try:
                await self._receive_msg_task
            except asyncio.CancelledError:
                pass
            self._receive_msg_task = None

        if self._client:
            await self._client.close()

    async def start(self, frame: StartFrame) -> None:
        """Start the TTS service.
        
        Args:
            frame: The StartFrame from the pipeline (can be None for standalone use).
        """
        if frame:
            await super().start(frame)
        if not self._client.is_connected():
            await self._start()

    async def stop(self, frame: EndFrame) -> None:
        """Stop the TTS service.
        
        Args:
            frame: The EndFrame from the pipeline (can be None for standalone use).
        """
        await self._stop()
        if frame:
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame = None) -> None:
        """Cancel the current TTS interaction.
        
        Args:
            frame: The CancelFrame from the pipeline (can be None for standalone use).
        """
        if self._client.is_connected():
            try:
                await self._client.send_message(OjinTTSCancelInteractionMessage())
                logger.debug("Sent TTS cancel interaction")
            except Exception as e:
                logger.warning(f"Failed to send TTS cancel: {e}")
        if frame:
            await super().cancel(frame)


class OjinTTSServiceInitializedFrame(Frame):
    """Frame indicating that the TTS service has been initialized."""

    def __init__(self, session_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.session_data = session_data
