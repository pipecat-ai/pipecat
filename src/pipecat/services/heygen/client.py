#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HeyGen implementation for Pipecat.

This module provides integration with the HeyGen platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the HeyGen API.
"""

import asyncio
import base64
import json
import time
import uuid
from typing import Awaitable, Callable, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    ImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.heygen.api import HeyGenApi, HeyGenSession, NewSessionRequest
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from livekit import rtc
    from livekit.rtc._proto.video_frame_pb2 import VideoBufferType
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.exceptions import ConnectionClosedOK
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use HeyGen, you need to `pip install pipecat-ai[heygen]`.")
    raise Exception(f"Missing module: {e}")

HEY_GEN_SAMPLE_RATE = 24000


class HeyGenCallbacks(BaseModel):
    """Callback handlers for HeyGen events.

    Parameters:
        on_participant_connected: Called when a participant connects
        on_participant_disconnected: Called when a participant disconnects
    """

    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]


class HeyGenClient:
    """A client for interacting with HeyGen's Interactive Avatar Realtime API.

    This client manages both WebSocket and LiveKit connections for real-time avatar streaming,
    handling bi-directional audio/video communication and avatar control. It implements the API defined in
    https://docs.heygen.com/docs/interactive-avatar-realtime-api

    The client manages the following connections:
    1. WebSocket connection for avatar control and audio streaming
    2. LiveKit connection for receiving avatar video and audio

    Attributes:
        HEY_GEN_SAMPLE_RATE (int): The required sample rate for HeyGen's audio processing (24000 Hz)
    """

    def __init__(
        self,
        *,
        api_key: str,
        session: aiohttp.ClientSession,
        params: TransportParams,
        session_request: NewSessionRequest = NewSessionRequest(
            avatarName="Shawn_Therapist_public",
            version="v2",
        ),
        callbacks: HeyGenCallbacks,
    ) -> None:
        """Initialize the HeyGen client.

        Args:
            api_key: HeyGen API key for authentication
            session: HTTP client session for API requests
            params: Transport configuration parameters
            session_request: Configuration for the HeyGen session (default: uses Shawn_Therapist_public avatar)
            callbacks: Callback handlers for HeyGen events
        """
        self._api = HeyGenApi(api_key, session=session)
        self._heyGen_session: Optional[HeyGenSession] = None
        self._websocket = None
        self._task_manager: Optional[BaseTaskManager] = None
        self._params = params
        self._in_sample_rate = 0
        self._out_sample_rate = 0
        self._connected = False
        self._session_request = session_request
        self._callbacks = callbacks
        self._event_queue: Optional[asyncio.Queue] = None
        self._event_task = None
        # Currently supporting to capture the audio and video from a single participant
        self._video_task = None
        self._audio_task = None
        self._video_frame_callback = None
        self._audio_frame_callback = None
        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0
        self._audio_seconds_sent = 0.0
        self._transport_ready = False

    async def _initialize(self):
        self._heyGen_session = await self._api.new_session(self._session_request)
        logger.debug(f"HeyGen sessionId: {self._heyGen_session.session_id}")
        logger.debug(f"HeyGen realtime_endpoint: {self._heyGen_session.realtime_endpoint}")
        logger.debug(f"HeyGen livekit URL: {self._heyGen_session.url}")
        logger.debug(f"HeyGen livekit toke: {self._heyGen_session.access_token}")
        logger.info(
            f"Full Link: https://meet.livekit.io/custom?liveKitUrl={self._heyGen_session.url}&token={self._heyGen_session.access_token}"
        )

        await self._api.start_session(self._heyGen_session.session_id)
        logger.info("HeyGen session started")

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Setup the client and initialize the conversation.

        Establishes a new session with HeyGen's API if one doesn't exist.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._heyGen_session is not None:
            logger.debug("heygen_session already initialized")
            return
        self._task_manager = setup.task_manager
        try:
            await self._initialize()

            self._event_queue = asyncio.Queue()
            self._event_task = self._task_manager.create_task(
                self._callback_task_handler(self._event_queue),
                f"{self}::event_callback_task",
            )
        except Exception as e:
            logger.error(f"Failed to setup HeyGenClient: {e}")
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup client resources.

        Closes the active HeyGen session and resets internal state.
        """
        try:
            if self._heyGen_session is not None:
                await self._api.close_session(self._heyGen_session.session_id)
                self._heyGen_session = None
                self._connected = False

            if self._event_task and self._task_manager:
                await self._task_manager.cancel_task(self._event_task)
                self._event_task = None
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def start(self, frame: StartFrame, audio_chunk_size: int) -> None:
        """Start the client and establish all necessary connections.

        Initializes WebSocket and LiveKit connections using the provided configuration.
        Sets up audio processing with the specified sample rates.

        Args:
            frame: Initial configuration frame containing audio parameters
            audio_chunk_size: Audio chunk size for output processing
        """
        if self._websocket:
            logger.debug("heygen client already started")
            return

        logger.debug(f"HeyGenClient starting")
        self._in_sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._send_interval = (audio_chunk_size / self._out_sample_rate) / 2
        logger.debug(f"HeyGenClient send_interval: {self._send_interval}")
        await self._ws_connect()
        await self._livekit_connect()

    async def stop(self) -> None:
        """Stop the client and terminate all connections.

        Disconnects from WebSocket and LiveKit endpoints, and performs cleanup.
        """
        logger.debug(f"HeyGenVideoService stopping")
        await self._ws_disconnect()
        await self._livekit_disconnect()
        await self.cleanup()

    # websocket connection methods
    async def _ws_connect(self):
        """Connect to HeyGen websocket endpoint."""
        try:
            if self._websocket:
                logger.debug(f"HeyGenClient ws already connected!")
                return
            logger.debug(f"HeyGenClient ws connecting")
            self._websocket = await websocket_connect(
                uri=self._heyGen_session.realtime_endpoint,
            )
            self._connected = True
            self._receive_task = self._task_manager.create_task(
                self._ws_receive_task_handler(), name="HeyGenClient_Websocket"
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _ws_receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        while self._connected:
            try:
                message = await self._websocket.recv()
                parsed_message = json.loads(message)
                await self._handle_ws_server_event(parsed_message)
            except ConnectionClosedOK:
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                break

    async def _handle_ws_server_event(self, event: dict) -> None:
        """Handle an event from HeyGen websocket."""
        event_type = event.get("type")
        if event_type == "agent.state":
            logger.debug(f"HeyGenClient ws received agent status: {event}")
        else:
            logger.trace(f"HeyGenClient ws received unknown event: {event_type}")

    async def _ws_disconnect(self) -> None:
        """Disconnect from HeyGen websocket endpoint."""
        try:
            self._connected = False
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} disconnect error: {e}")
        finally:
            self._websocket = None

    async def _ws_send(self, message: dict) -> None:
        """Send a message to HeyGen websocket."""
        if not self._connected:
            logger.debug(f"{self} websocket is not connected anymore.")
            return
        try:
            if self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to HeyGen websocket: {e}")
            raise e

    async def interrupt(self, event_id: str) -> None:
        """Interrupt the avatar's current action.

        Stops the current animation/speech and returns the avatar to idle state.
        Useful for handling user interruptions during avatar speech.
        """
        logger.debug("HeyGenClient interrupt")
        self._reset_audio_timing()
        await self._ws_send(
            {
                "type": "agent.interrupt",
                "event_id": event_id,
            }
        )

    async def start_agent_listening(self) -> None:
        """Start the avatar's listening animation.

        Triggers visual cues indicating the avatar is listening to user input.
        """
        logger.debug("HeyGenClient start_agent_listening")
        await self._ws_send(
            {
                "type": "agent.start_listening",
                "event_id": str(uuid.uuid4()),
            }
        )

    async def stop_agent_listening(self) -> None:
        """Stop the avatar's listening animation.

        Returns the avatar to idle state from listening state.
        """
        await self._ws_send(
            {
                "type": "agent.stop_listening",
                "event_id": str(uuid.uuid4()),
            }
        )

    def transport_ready(self) -> None:
        """Indicates that the output transport is ready and able to receive frames."""
        self._transport_ready = True

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

    async def agent_speak(self, audio: bytes, event_id: str) -> None:
        """Send audio data to the agent speak.

        Args:
            audio: Audio data in base64 encoded format
            event_id: Unique identifier for the event
        """
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._ws_send(
            {
                "type": "agent.speak",
                "audio": audio_base64,
                "event_id": event_id,
            }
        )
        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    def _reset_audio_timing(self):
        """Reset audio timing control variables."""
        self._audio_seconds_sent = 0.0
        self._next_send_time = 0

    async def _write_audio_sleep(self):
        """Simulate audio playback timing with appropriate delays."""
        # Only sleep after we've sent the first second of audio
        # This appears to reduce the latency to receive the answer from HeyGen
        if self._audio_seconds_sent < 3.0:
            self._audio_seconds_sent += self._send_interval
            self._next_send_time = time.monotonic() + self._send_interval
            return

        # After first second, use normal timing
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
            self._next_send_time += self._send_interval
        else:
            self._next_send_time = time.monotonic() + self._send_interval

    async def agent_speak_end(self, event_id: str) -> None:
        """Send signaling that the agent has finished speaking.

        Args:
            event_id: Unique identifier for the event
        """
        self._reset_audio_timing()
        await self._ws_send(
            {
                "type": "agent.speak_end",
                "event_id": event_id,
            }
        )

    async def capture_participant_audio(self, participant_id: str, callback) -> None:
        """Capture audio frames from the HeyGen avatar.

        Args:
            participant_id: Identifier of the participant to capture audio from
            callback: Async function to handle received audio frames
        """
        logger.debug(f"capture_participant_audio: {participant_id}")
        self._audio_frame_callback = callback
        if self._audio_task is not None:
            logger.warning(
                "Trying to capture more than one audio stream. It is currently not supported."
            )
            return

        # Check if we already have audio tracks and participant is connected
        if self._livekit_room and participant_id in self._livekit_room.remote_participants:
            participant = self._livekit_room.remote_participants[participant_id]
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and track_pub.track is not None:
                    logger.debug(f"Starting audio capture for existing track: {track_pub.sid}")
                    audio_stream = rtc.AudioStream(track_pub.track)
                    self._audio_task = self._task_manager.create_task(
                        self._process_audio_frames(audio_stream), name="HeyGenClient_Receive_Audio"
                    )
                    break

    async def capture_participant_video(self, participant_id: str, callback) -> None:
        """Capture video frames from the HeyGen avatar.

        Args:
            participant_id: Identifier of the participant to capture video from
            callback: Async function to handle received video frames
        """
        logger.debug(f"capture_participant_video: {participant_id}")
        self._video_frame_callback = callback
        if self._video_task is not None:
            logger.warning(
                "Trying to capture more than one audio stream. It is currently not supported."
            )
            return

        # Check if we already have video tracks and participant is connected
        if self._livekit_room and participant_id in self._livekit_room.remote_participants:
            participant = self._livekit_room.remote_participants[participant_id]
            for track_pub in participant.track_publications.values():
                if track_pub.kind == rtc.TrackKind.KIND_VIDEO and track_pub.track is not None:
                    logger.debug(f"Starting video capture for existing track: {track_pub.sid}")
                    video_stream = rtc.VideoStream(track_pub.track)
                    self._video_task = self._task_manager.create_task(
                        self._process_video_frames(video_stream), name="HeyGenClient_Receive_Video"
                    )
                    break

    # Livekit integration to receive audio and video
    async def _process_audio_frames(self, stream: rtc.AudioStream):
        """Process audio frames from LiveKit stream."""
        try:
            logger.debug("Starting audio frame processing...")
            async for frame_event in stream:
                try:
                    audio_frame = frame_event.frame
                    # Convert audio to raw bytes
                    audio_data = bytes(audio_frame.data)

                    audio_frame = AudioRawFrame(
                        audio=audio_data,
                        sample_rate=audio_frame.sample_rate,
                        num_channels=1,  # HeyGen uses mono audio
                    )
                    if self._transport_ready and self._audio_frame_callback:
                        await self._audio_frame_callback(audio_frame)

                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}")
        except Exception as e:
            logger.error(f"Error processing audio frames: {e}")
        finally:
            logger.debug(f"Audio frame processing ended.")

    async def _process_video_frames(self, stream: rtc.VideoStream):
        """Process video frames from LiveKit stream."""
        try:
            logger.debug("Starting video frame processing...")
            async for frame_event in stream:
                try:
                    video_frame = frame_event.frame

                    # Convert to RGB24 if not already
                    if video_frame.type != VideoBufferType.RGB24:
                        video_frame = video_frame.convert(VideoBufferType.RGB24)

                    # Create frame with original dimensions
                    image_frame = ImageRawFrame(
                        image=bytes(video_frame.data),
                        size=(video_frame.width, video_frame.height),
                        format="RGB",
                    )
                    image_frame.pts = frame_event.timestamp_us // 1000  # Convert to milliseconds

                    if self._transport_ready and self._video_frame_callback:
                        await self._video_frame_callback(image_frame)
                except Exception as e:
                    logger.error(f"Error processing individual video frame: {e}")
        except Exception as e:
            logger.error(f"Error processing video frames: {e}")
        finally:
            logger.debug(f"Video frame processing ended.")

    async def _livekit_connect(self):
        """Connect to LiveKit room."""
        try:
            logger.debug(f"HeyGenClient livekit connecting to room URL: {self._heyGen_session.url}")
            self._livekit_room = rtc.Room()

            @self._livekit_room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.debug(
                    f"Participant connected - SID: {participant.sid}, Identity: {participant.identity}"
                )
                for track_pub in participant.track_publications.values():
                    logger.debug(
                        f"Available track - SID: {track_pub.sid}, Kind: {track_pub.kind}, Name: {track_pub.name}"
                    )
                self._call_event_callback(
                    self._callbacks.on_participant_connected, participant.identity
                )

            @self._livekit_room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                if (
                    track.kind == rtc.TrackKind.KIND_VIDEO
                    and self._video_frame_callback is not None
                    and self._video_task is None
                ):
                    logger.debug(f"Creating video stream processor for track: {publication.sid}")
                    video_stream = rtc.VideoStream(track)
                    self._video_task = self._task_manager.create_task(
                        self._process_video_frames(video_stream), name="HeyGenClient_Receive_Video"
                    )
                elif (
                    track.kind == rtc.TrackKind.KIND_AUDIO
                    and self._audio_frame_callback is not None
                    and self._audio_task is None
                ):
                    logger.debug(f"Creating audio stream processor for track: {publication.sid}")
                    audio_stream = rtc.AudioStream(track)
                    self._audio_task = self._task_manager.create_task(
                        self._process_audio_frames(audio_stream), name="HeyGenClient_Receive_Audio"
                    )

            @self._livekit_room.on("track_unsubscribed")
            def on_track_unsubscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.debug(f"Track unsubscribed - SID: {publication.sid}, Kind: {track.kind}")

            @self._livekit_room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.debug(
                    f"Participant disconnected - SID: {participant.sid}, Identity: {participant.identity}"
                )
                self._call_event_callback(
                    self._callbacks.on_participant_disconnected, participant.identity
                )

            await self._livekit_room.connect(
                self._heyGen_session.url, self._heyGen_session.access_token
            )
            logger.debug(f"Successfully connected to LiveKit room: {self._livekit_room.name}")
            logger.debug(f"Local participant SID: {self._livekit_room.local_participant.sid}")
            logger.debug(
                f"Number of remote participants: {len(self._livekit_room.remote_participants)}"
            )

            # Log existing participants and their tracks
            for participant in self._livekit_room.remote_participants.values():
                logger.debug(
                    f"Existing participant - SID: {participant.sid}, Identity: {participant.identity}"
                )
                self._call_event_callback(
                    self._callbacks.on_participant_connected, participant.identity
                )
                for track_pub in participant.track_publications.values():
                    logger.debug(
                        f"Existing track - SID: {track_pub.sid}, Kind: {track_pub.kind}, Name: {track_pub.name}"
                    )

        except Exception as e:
            logger.error(f"LiveKit initialization error: {e}")
            self._livekit_room = None

    async def _livekit_disconnect(self):
        """Disconnect from LiveKit room."""
        try:
            logger.debug("Starting LiveKit disconnect...")
            if self._video_task:
                await self._task_manager.cancel_task(self._video_task)
                self._video_task = None

            if self._audio_task:
                await self._task_manager.cancel_task(self._audio_task)
                self._audio_task = None

            if self._livekit_room:
                logger.debug("Disconnecting from LiveKit room")
                await self._livekit_room.disconnect()
                self._livekit_room = None
                logger.debug("Successfully disconnected from LiveKit room")
        except Exception as e:
            logger.error(f"LiveKit disconnect error: {e}")

    #
    # Queue callback handling
    #

    def _call_event_callback(self, callback, *args):
        """Queue an event callback for async execution."""
        self._event_queue.put_nowait((callback, *args))

    async def _callback_task_handler(self, queue: asyncio.Queue):
        """Handle queued callbacks from the specified queue."""
        while True:
            (callback, *args) = await queue.get()
            await callback(*args)
            queue.task_done()
