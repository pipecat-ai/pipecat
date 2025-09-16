#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pinch implementation for Pipecat.

This module provides integration with the Pinch platform for real-time translation
with audio streaming. It manages translation sessions and provides real-time
audio streaming capabilities through Pinch API.
"""

import asyncio
import json
from typing import Awaitable, Callable, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import StartFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.pinch.api import (
    PinchApi,
    PinchConfigurationError,
    PinchConnectionError,
    PinchSession,
    PinchSessionRequest,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from livekit import rtc
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Pinch, you need to `pip install pipecat-ai[livekit]`.")
    raise Exception(f"Missing module: {e}")

PINCH_INPUT_SAMPLE_RATE = 16000  # Pinch expects 16kHz input
PINCH_OUTPUT_SAMPLE_RATE = 48000  # Pinch outputs 48kHz audio (received via WebRTC)


class PinchCallbacks(BaseModel):
    """Callback handlers for Pinch events.

    Parameters:
        on_original_transcript: Called when original transcript is received
        on_translated_transcript: Called when translated transcript is received
        on_audio_data: Called when translated audio is received
        on_session_started: Called when session starts
        on_session_ended: Called when session ends
    """

    on_original_transcript: Optional[Callable[[str, bool], Awaitable[None]]] = None
    on_translated_transcript: Optional[Callable[[str, bool], Awaitable[None]]] = None
    on_audio_data: Optional[Callable[[bytes], Awaitable[None]]] = None
    on_session_started: Optional[Callable[[], Awaitable[None]]] = None
    on_session_ended: Optional[Callable[[], Awaitable[None]]] = None


class PinchClient:
    """A client for interacting with Pinch's Translation API.

    This client manages audio connections for real-time translation streaming,
    handling bi-directional audio communication and transcript messages.

    The client manages the following:
    1. Connection for audio input/output
    2. Data messages for transcript exchange
    3. Audio resampling for optimal quality

    Attributes:
        PINCH_INPUT_SAMPLE_RATE (int): The required sample rate for Pinch input (16000 Hz)
        PINCH_OUTPUT_SAMPLE_RATE (int): The sample rate for Pinch output (48000 Hz)
    """

    def __init__(
        self,
        *,
        api_token: str,
        session: aiohttp.ClientSession,
        params: TransportParams,
        session_request: PinchSessionRequest = PinchSessionRequest(),
        callbacks: PinchCallbacks = PinchCallbacks(),
    ) -> None:
        """Initialize the Pinch client.

        Args:
            api_token: Pinch API token for authentication
            session: HTTP client session for API requests
            params: Transport configuration parameters
            session_request: Configuration for the Pinch session
            callbacks: Callback handlers for Pinch events
        """
        self._api = PinchApi(api_token, session=session)
        self._pinch_session: Optional[PinchSession] = None
        self._livekit_room = None
        self._task_manager: Optional[BaseTaskManager] = None
        self._params = params
        self._in_sample_rate = 0
        self._out_sample_rate = 0
        self._connected = False
        self._session_request = session_request
        self._callbacks = callbacks
        self._event_task = None
        # Audio processing
        self._audio_task = None
        # Audio resampling
        self._resampler = create_stream_resampler()
        self._session_active = False
        # Audio frame counter for limiting log spam
        self._audio_frame_counter = 0
        # Audio source and track for LiveKit audio streaming
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None

    async def _initialize(self):
        self._pinch_session = await self._api.new_session(self._session_request)
        logger.info(f"Pinch session started (ID: {self._pinch_session.session_id})")

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Setup the client and initialize the translation session.

        Establishes a new session with Pinch's API if one doesn't exist.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._pinch_session is not None:
            return
        self._task_manager = setup.task_manager
        try:
            await self._initialize()
        except Exception as e:
            logger.error(f"Failed to setup PinchClient: {e}")
            await self.cleanup()
            raise PinchConfigurationError(f"Failed to setup PinchClient: {str(e)}") from e

    async def cleanup(self) -> None:
        """Cleanup client resources.

        Closes the active Pinch session and disconnects from LiveKit room.
        """
        try:
            # End the session via API if we have a session
            if self._pinch_session is not None:
                try:
                    await self._api.end_session(self._pinch_session.session_id)
                except Exception as e:
                    logger.warning(f"Failed to end Pinch session: {e}")

            # Disconnect from LiveKit room
            await self._disconnect_audio_stream()

            self._pinch_session = None
            self._session_active = False
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def start(self, frame: StartFrame) -> None:
        """Start the client and establish all necessary connections.

        Initializes connection using the provided configuration.
        Sets up audio processing with the specified sample rates.

        Args:
            frame: Initial configuration frame containing audio parameters
        """
        if self._livekit_room:
            logger.info("Pinch client already connected")
            return

        logger.debug("Starting Pinch client")
        self._in_sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        # Reset audio frame counter for new session
        self._audio_frame_counter = 0
        await self._connect_audio_stream()

    async def stop(self) -> None:
        """Stop the client and terminate all connections.

        Disconnects from audio stream and performs cleanup.
        """
        await self._disconnect_audio_stream()
        await self.cleanup()

    # Audio streaming methods
    async def _connect_audio_stream(self):
        """Connect to Pinch audio streaming service."""
        try:
            # Ensure we have a Pinch session (create via API if needed)
            if self._pinch_session is None:
                await self._initialize()
            self._livekit_room = rtc.Room()

            @self._livekit_room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                # Check if this is the Pinch service participant (use identity as primary, fallback to metadata)
                if participant.identity == "pinch-translation-agent":
                    self._session_active = True
                    logger.info("Pinch translation service participant connected")
                    self._call_event_callback(self._callbacks.on_session_started)

            @self._livekit_room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                if participant.identity == "pinch-translation-agent":
                    self._session_active = False
                    logger.info("Pinch translation service participant disconnected")
                    # Attempt simple reconnect after a brief delay
                    asyncio.create_task(self._attempt_reconnect())

            @self._livekit_room.on("disconnected")
            def on_room_disconnected():
                logger.warning("Pinch disconnected, attempting reconnect")
                self._session_active = False
                self._connected = False
                asyncio.create_task(self._attempt_reconnect())

            @self._livekit_room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    audio_stream = rtc.AudioStream(track)
                    self._audio_task = self._task_manager.create_task(
                        self._process_audio_frames(audio_stream), name="PinchClient_Receive_Audio"
                    )

            @self._livekit_room.on("data_received")
            def on_data_received(data_packet: rtc.DataPacket):
                try:
                    message_str = data_packet.data.decode("utf-8")
                    message = json.loads(message_str)
                    asyncio.create_task(self._handle_data_message(message))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON data message: {e}")
                except UnicodeDecodeError as e:
                    logger.error(f"Failed to decode data message as UTF-8: {e}")
                except Exception as e:
                    logger.error(f"Error processing data message: {e}")

            # Connect to LiveKit room
            try:
                await self._livekit_room.connect(
                    self._pinch_session.livekit_url,
                    self._pinch_session.access_token,
                    rtc.RoomOptions(
                        auto_subscribe=True,
                        dynacast=True,
                    ),
                )

                # Set up audio source and track for sending user audio to Pinch
                self._audio_source = rtc.AudioSource(
                    PINCH_INPUT_SAMPLE_RATE,
                    1,  # 16kHz mono for Pinch input
                )
                self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                    "pinch-user-audio", self._audio_source
                )
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_MICROPHONE
                await self._livekit_room.local_participant.publish_track(self._audio_track, options)

                logger.info("Connected to Pinch audio streaming service")
                self._connected = True
            except Exception as e:
                logger.error(f"Failed to connect to Pinch audio streaming: {e}")
                raise

        except Exception as e:
            logger.error(f"Audio streaming connection error: {e}")
            self._livekit_room = None
            raise PinchConnectionError(f"Failed to connect to Pinch audio streaming: {str(e)}", e)

    async def _handle_data_message(self, message: dict) -> None:
        """Handle data messages received."""
        msg_type = message.get("type")

        if msg_type == "original_transcript":
            text = message.get("text", "")
            is_final = message.get("is_final", True)

            if self._callbacks.on_original_transcript:
                self._call_event_callback(self._callbacks.on_original_transcript, text, is_final)

        elif msg_type == "translated_transcript":
            text = message.get("translated_text", "")
            is_final = message.get("is_final", True)

            if self._callbacks.on_translated_transcript:
                self._call_event_callback(self._callbacks.on_translated_transcript, text, is_final)

        elif msg_type == "session_started":
            self._session_active = True
            logger.info("Pinch session started via data message")
            if self._callbacks.on_session_started:
                self._call_event_callback(self._callbacks.on_session_started)

        elif msg_type == "session_ended":
            self._session_active = False
            logger.info("Pinch session ended via data message")
            if self._callbacks.on_session_ended:
                self._call_event_callback(self._callbacks.on_session_ended)

        else:
            logger.debug(f"Received unknown message type: {msg_type}")

    @property
    def out_sample_rate(self) -> int:
        """Get the output sample rate.

        Returns:
            The output sample rate in Hz.
        """
        return self._out_sample_rate

    @property
    def in_sample_rate(self) -> int:
        """Get the input sample rate.

        Returns:
            The input sample rate in Hz.
        """
        return self._in_sample_rate

    async def send_audio(self, audio: bytes, sample_rate: int) -> None:
        """Send audio data to Pinch for translation via LiveKit audio track.

        Args:
            audio: Raw PCM audio bytes (assumed 16-bit signed integer format)
            sample_rate: Sample rate of the input audio in Hz

        Note:
            Audio is assumed to be 16-bit PCM mono. Stereo audio will be downmixed
            to mono before processing. The audio is resampled to 16kHz if needed.
        """
        if not self._connected:
            logger.warning("Not connected to Pinch, cannot send audio")
            return

        try:
            # Increment frame counter for logging throttling
            self._audio_frame_counter += 1

            # Validate and prepare audio data
            if not audio:
                return  # Skip empty audio frames

            # Resample audio to 16kHz for Pinch if needed
            if sample_rate != PINCH_INPUT_SAMPLE_RATE:
                if self._audio_frame_counter % 100 == 0:
                    logger.debug(
                        f"Resampling audio from {sample_rate}Hz to {PINCH_INPUT_SAMPLE_RATE}Hz"
                    )
                resampled_audio = await self._resampler.resample(
                    audio, sample_rate, PINCH_INPUT_SAMPLE_RATE
                )
            else:
                resampled_audio = audio

            # Send via LiveKit audio track instead of DataChannel
            if self._audio_source:
                try:
                    # Convert audio frame
                    bytes_per_sample = 2  # 16-bit audio
                    total_samples = len(resampled_audio) // bytes_per_sample
                    samples_per_channel = total_samples // 1  # Mono

                    audio_frame = rtc.AudioFrame(
                        data=resampled_audio,
                        sample_rate=PINCH_INPUT_SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=samples_per_channel,
                    )

                    # Publish audio frame via audio track
                    await self._audio_source.capture_frame(audio_frame)

                    # Only log audio sends every 100 frames to reduce spam
                    if self._audio_frame_counter % 100 == 0:
                        logger.debug(
                            f"Sent audio via Pinch audio track (frame #{self._audio_frame_counter})"
                        )

                except Exception as e:
                    # Only log audio errors every 100 frames to reduce spam
                    if self._audio_frame_counter % 100 == 0:
                        logger.warning(f"Failed to send audio: {e}")
                    raise
            else:
                logger.error("Audio source not available, cannot send audio")

        except Exception as e:
            logger.error(f"Error sending audio to Pinch: {e}")
            raise

    # Audio processing to receive translated audio
    async def _process_audio_frames(self, stream: rtc.AudioStream):
        """Process translated audio frames from Pinch audio stream."""
        try:
            async for frame_event in stream:
                try:
                    audio_frame = frame_event.frame

                    # Skip empty audio frames to reduce unnecessary processing
                    if not audio_frame.data or len(audio_frame.data) == 0:
                        continue

                    # Use tobytes() following the established LiveKit pattern for efficiency
                    audio_bytes = audio_frame.data.tobytes()

                    # Only process if callback is available
                    if self._callbacks.on_audio_data:
                        self._call_event_callback(self._callbacks.on_audio_data, audio_bytes)

                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}")
                    continue  # Continue processing other frames even if one fails
        except Exception as e:
            logger.error(f"Audio processing stream error: {e}")

    async def _disconnect_audio_stream(self):
        """Disconnect from Pinch audio streaming service."""
        try:
            if self._audio_task:
                await self._task_manager.cancel_task(self._audio_task)
                self._audio_task = None

            if self._livekit_room:
                await self._livekit_room.disconnect()
                self._livekit_room = None
                self._connected = False
                self._audio_source = None
                self._audio_track = None
        except Exception as e:
            logger.error(f"Error disconnecting from audio stream: {e}")

    async def _attempt_reconnect(self):
        """Attempt to reconnect to the Pinch service after disconnection."""
        try:
            # Wait a bit before attempting reconnect
            await asyncio.sleep(2.0)

            if self._connected or not self._pinch_session:
                return  # Already connected or no session to reconnect to

            logger.info("Attempting to reconnect to Pinch service...")
            await self._connect_audio_stream()

        except Exception as e:
            logger.error(f"Failed to reconnect to Pinch service: {e}")
            # Could implement exponential backoff here for more sophisticated retry logic

    def _call_event_callback(self, callback, *args):
        """Call event callback asynchronously."""
        if callback:
            try:
                # Create a task to call the callback asynchronously
                asyncio.create_task(callback(*args))
            except Exception as e:
                logger.error(f"Error calling event callback: {e}")
