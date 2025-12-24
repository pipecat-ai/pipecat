# Copyright (c) 2024–2025, Daily & Voxium Tech
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements STT transcription by connecting to a remote Voxium server."""

import asyncio
import base64
import json
import urllib.parse
from typing import Optional

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class VoxiumSTTService(FrameProcessor):
    """Transcribes audio using Voxium server via WebSocket.

    Args:
        url (str): The WebSocket URL of the Voxium server (e.g., "wss://voxium.tech/asr/ws").
        api_key (str): The API key for authenticating with the Voxium server.
        language (Optional[Language]): The language hint to pass to the server. Defaults to None (auto-detect).
        sample_rate (int): The sample rate of the incoming audio. Defaults to 16000.
        channels (int): The number of channels in the incoming audio. Defaults to 1.
        input_format (str): The expected format identifier for the server ('pcm', 'mulaw', 'base64'). Defaults to 'base64'.
        vad_threshold (float): VAD threshold parameter for the server. Defaults to 0.5.
        silence_threshold_s (float): Silence duration threshold for the server (in seconds). This controls the minimum duration of silence that will trigger a new transcription. Defaults to 0.5.
        speech_pad_ms (int): Speech padding parameter for the server (in milliseconds). This pads the audio to ensure all of the audio is transcribed. Defaults to 100.
        beam_size (int): Beam size parameter. Defaults to 3
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        language: Optional[Language] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        input_format: str = "base64",
        vad_threshold: float = 0.5,
        silence_threshold_s: float = 0.5,
        speech_pad_ms: int = 100,
        beam_size: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = url
        self._api_key = api_key
        self._language = language
        self._sample_rate = sample_rate
        self._channels = channels
        self._input_format = input_format
        self._vad_threshold = vad_threshold
        self._silence_threshold_s = silence_threshold_s
        self._speech_pad_ms = speech_pad_ms
        self._beam_size = beam_size

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connection_id: Optional[str] = None
        self._is_connected: bool = False
        self._connection_lock = asyncio.Lock()
        self._current_language_str: Optional[str] = None

    def _build_connection_url(self) -> str:
        """Builds the WebSocket URL with query parameters.

        Returns:
            str: The complete WebSocket URL with all query parameters.
        """
        params = {
            "apiKey": self._api_key,
            "input_format": self._input_format,
            "sample_rate": str(self._sample_rate),
            "vad_threshold": str(self._vad_threshold),
            "silence_threshold": str(self._silence_threshold_s),
            "speech_pad_ms": str(self._speech_pad_ms),
            "beam_size": str(self._beam_size),
        }
        if self._current_language_str:
            params["language"] = self._current_language_str

        query_string = urllib.parse.urlencode(params)
        return f"{self._url}?{query_string}"

    async def _connect(self):
        """Establishes the WebSocket connection and starts the receiver task."""
        async with self._connection_lock:
            if self._is_connected:
                return

            connection_url = self._build_connection_url()
            logger.info(f"Connecting to Voxium STT server: {self._url} with specified params...")
            logger.debug(f"Full connection URL: {connection_url}")

            try:
                self._websocket = await websockets.connect(
                    connection_url,
                    ping_interval=20,  # Send pings to keep connection alive
                    ping_timeout=20,
                    # max_size= desired_max_message_size_bytes, # Default is 1MB
                    # write_limit=desired_write_limit_bytes_per_second # Throttling
                )
                self._is_connected = True
                logger.info("WebSocket connection established.")
                self._receive_task = asyncio.create_task(self._receive_loop())
                logger.info("Started background task to receive transcriptions.")

            except websockets.exceptions.InvalidURI as e:
                logger.error(f"Invalid WebSocket URI: {e}")
                await self.push_frame(ErrorFrame(f"Invalid WebSocket URI: {connection_url}"))
                self._is_connected = False
            except websockets.exceptions.InvalidHandshake as e:
                logger.error(
                    f"WebSocket handshake failed: {e}. Check URL, API key, and server status."
                )
                await self.push_frame(ErrorFrame(f"WebSocket handshake failed: {e}"))
                self._is_connected = False
            except ConnectionRefusedError:
                logger.error(f"Connection refused by server at {self._url}. Is the server running?")
                await self.push_frame(ErrorFrame(f"Connection refused by server at {self._url}"))
                self._is_connected = False
            except Exception as e:
                logger.error(f"Failed to connect to WebSocket: {e}", exc_info=True)
                await self.push_frame(ErrorFrame(f"WebSocket connection error: {e}"))
                self._is_connected = False  # Ensure state reflects failure

    async def _receive_loop(self):
        """Listens for messages from the server and pushes frames.

        This method runs in a loop while the WebSocket connection is active, processing
        incoming messages and pushing appropriate frames to the output queue.
        """
        logger.info("Receive loop started.")
        while self._websocket and self._is_connected:
            try:
                message_str = await self._websocket.recv()
                message = json.loads(message_str)
                logger.debug(f"Received message from server: {message}")

                status = message.get("status")

                if status == "connected":
                    self._connection_id = message.get("connection_id")
                    logger.info(f"Server confirmed connection (ID: {self._connection_id})")

                elif status in ["complete"]:
                    transcription = message.get("transcription", "").strip()
                    if transcription:  # Only push if there's text
                        lang = message.get("language")
                        # Convert language code back to pipecat enum if possible, otherwise keep as string
                        # Note: This requires reversing the mapping or a new map. For simplicity,
                        # we'll just use the code provided by the server for now.
                        detected_language = lang  # Use the string code directly. could map 'lang' back to transcriptions.language.Language if needed

                        await self.push_frame(
                            TranscriptionFrame(
                                text=transcription,
                                user_id="",
                                timestamp=time_now_iso8601(),
                                language=detected_language,
                            )
                        )
                    else:
                        logger.debug("Received transcription message with empty text, skipping.")

                elif status == "error":
                    error_message = message.get("message", "Unknown server error")
                    logger.error(f"Received error from server: {error_message}")
                    await self.push_frame(ErrorFrame(f"Voxium server error: {error_message}"))
                    if (
                        "Usage limit exceeded" in error_message
                        or "Invalid or inactive API Key" in error_message
                    ):
                        logger.warning("Closing connection due to server-reported error.")
                        await self._close_connection()
                        break

                else:
                    logger.warning(f"Received unknown message status: {status}")

            except ConnectionClosedOK:
                logger.info("WebSocket connection closed normally by server.")
                self._is_connected = False
                break
            except ConnectionClosedError as e:
                logger.warning(f"WebSocket connection closed unexpectedly: {e.code} {e.reason}")
                self._is_connected = False
                await self.push_frame(
                    ErrorFrame(f"WebSocket closed unexpectedly: {e.code} {e.reason}")
                )
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {e}. Message: '{message_str}'")
            except Exception as e:
                logger.error(f"Error in receive loop: {e}", exc_info=True)
                self._is_connected = False
                await self.push_frame(ErrorFrame(f"Receive loop error: {e}"))
                break

        logger.info("Receive loop finished.")
        async with self._connection_lock:
            self._is_connected = False
            self._websocket = None

    async def _close_connection(self):
        """Closes the WebSocket connection and stops the receiver task."""
        logger.info("Attempting to close WebSocket connection...")
        async with self._connection_lock:
            if self._receive_task and not self._receive_task.done():
                logger.debug("Cancelling receive task...")
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=2.0)
                    logger.debug("Receive task cancelled.")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for receive task to cancel.")
                except asyncio.CancelledError:
                    logger.debug("Receive task already cancelled.")  # Expected
                except Exception as e:
                    logger.error(f"Error waiting for receive task cancellation: {e}")
            self._receive_task = None

            # Close the websocket connection
            if self._websocket:
                logger.debug("Closing WebSocket...")
                try:
                    await self._websocket.close()
                    logger.info("WebSocket connection closed.")
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
            self._websocket = None
            self._is_connected = False
            self._connection_id = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames, sending audio to the server."""
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (StartFrame)):
            logger.info("Start frame received, ensuring connection...")
            await self._connect()
            await self.push_frame(frame)
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            logger.info(f"{type(frame).__name__} received, closing connection.")
            await self._close_connection()
            await self.push_frame(frame)
            return

        if not self._is_connected:
            if isinstance(frame, AudioRawFrame):
                logger.warning("Dropping audio frame as WebSocket is not connected.")
                return
            else:
                await self.push_frame(frame)
                return

        # Process audio frames
        if isinstance(frame, AudioRawFrame):
            if not frame.audio:
                logger.debug("Empty audio frame received, skipping.")
                return

            encoded_audio = base64.b64encode(frame.audio).decode("utf-8")
            message = {"audio_data": encoded_audio}

            if self._websocket and self._is_connected:
                try:
                    # Use create_task to avoid blocking frame processing if send takes time
                    # logger.debug(f"Sending audio chunk ({len(frame.audio)} bytes)...")
                    asyncio.create_task(self._websocket.send(json.dumps(message)))
                except ConnectionClosedError:
                    logger.warning(
                        "WebSocket closed while trying to send audio. Attempting cleanup."
                    )
                    self._is_connected = False  # Update state immediately
                    await self._close_connection()  # Clean up resources
                    await self.push_frame(ErrorFrame("WebSocket closed during audio send"))
                except Exception as e:
                    logger.error(f"Error sending audio frame: {e}", exc_info=True)
                    # Should we close connection on send error? Maybe.
                    await self.push_frame(ErrorFrame(f"Error sending audio: {e}"))
            else:
                logger.warning("WebSocket is None or not connected, cannot send audio.")

        else:
            await self.push_frame(frame)

    async def stop(self):
        """Overrides FrameProcessor stop method for cleanup."""
        logger.info("Stopping VoxiumSTTService...")
        await self._close_connection()
        await super().stop()
        logger.info("VoxiumSTTService stopped.")
