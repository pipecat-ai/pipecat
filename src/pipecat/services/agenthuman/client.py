#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AgentHuman implementation for Pipecat.

This module provides integration with the AgentHuman platform for creating conversational
AI applications with avatars. It manages conversation sessions and provides real-time
audio/video streaming capabilities through the AgentHuman API.
"""

import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    ImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.agenthuman.api import AgentHumanApi, AgentHumanSession, NewSessionRequest
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from livekit import rtc
    from livekit.agents.voice.avatar import DataStreamAudioOutput
    from livekit.rtc._proto.video_frame_pb2 import VideoBufferType
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AgentHuman, you need to `pip install pipecat-ai[agenthuman]`.")
    raise Exception(f"Missing module: {e}")

AGENTHUMAN_SAMPLE_RATE = 16000
_AVATAR_AGENT_IDENTITY = "agenthuman-avatar-agent"
# Same RPC as livekit.agents DataStreamAudioOutput / agenthuman-webrtc LivekitConductor
_LK_CLEAR_BUFFER_RPC = "lk.clear_buffer"

class AgentHumanCallbacks(BaseModel):
    """Callback handlers for AgentHuman events.

    Parameters:
        on_participant_connected: Called when a participant connects
        on_participant_disconnected: Called when a participant disconnects
    """

    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]


class AgentHumanClient:
    """A client for interacting with AgentHuman's Interactive Avatar Realtime API.

    This client manages LiveKit connections for real-time avatar streaming,
    handling bi-directional audio/video communication and avatar control. It implements the API defined in
    https://docs.heygen.com/docs/interactive-avatar-realtime-api

    The client manages the following connections:
    1. LiveKit connection for receiving avatar video and audio

    Attributes:
        AGENTHUMAN_SAMPLE_RATE (int): The required sample rate for AgentHuman's audio processing (16000 Hz)
    """

    def __init__(
        self,
        *,
        api_key: str,
        params: TransportParams,
        session_request: NewSessionRequest = NewSessionRequest(),
        callbacks: AgentHumanCallbacks,
    ) -> None:
        """Initialize the AgentHuman client.

        Args:
            api_key: AgentHuman API key for authentication
            params: Transport configuration parameters
            session_request: Configuration for the AgentHuman session (default: uses avat_01KMZHXFPBVCXA5ATK85HCP8G1 avatar with 4:3 aspect ratio)
            callbacks: Callback handlers for AgentHuman events
        """
        self._api = AgentHumanApi(api_key)
        self._agentHuman_session: Optional[AgentHumanSession] = None
        self._task_manager: Optional[BaseTaskManager] = None
        self._params = params
        self._session_request = session_request
        self._callbacks = callbacks
        self._event_queue: Optional[asyncio.Queue] = None
        self._event_task = None
        # Currently supporting to capture the audio and video from a single participant
        self._video_task = None
        self._audio_task = None
        self._video_frame_callback = None
        self._audio_frame_callback = None
        # TTS is chunked for pacing in AgentHumanVideoService (_audio_chunk_size in start());
        # this client only forwards PCM via agent_speak to the LiveKit data stream.
        self._transport_ready = False
        self._data_stream_audio: Optional[DataStreamAudioOutput] = None

    async def _initialize(self):
        self._agentHuman_session = await self._api.new_session(self._session_request)
        logger.debug(f"AgentHuman session_id: {self._agentHuman_session.session_id}")
        logger.debug(f"AgentHuman livekit URL: {self._agentHuman_session.roomURL}")
        logger.debug(f"AgentHuman livekit token: {self._agentHuman_session.participantToken}")
        logger.info(
            f"Full Link: https://meet.livekit.io/custom?liveKitUrl={self._agentHuman_session.roomURL}&token={self._agentHuman_session.participantToken}"
        )

        logger.info("AgentHuman session started")

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Setup the client and initialize the conversation.

        Establishes a new session with AgentHuman's API if one doesn't exist.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._agentHuman_session is not None:
            logger.debug("AgentHuman session already initialized")
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
            logger.error(f"Failed to setup AgentHumanClient: {e}")
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup client resources.

        Closes the active AgentHuman session and resets internal state.
        """
        try:
            if self._agentHuman_session is not None:
                await self._api.end_session(self._agentHuman_session.session_id)
                self._agentHuman_session = None

            if self._event_task and self._task_manager:
                await self._task_manager.cancel_task(self._event_task)
                self._event_task = None
        except Exception as e:
            logger.exception(f"Exception during cleanup: {e}")

    async def start(self, frame: StartFrame) -> None:
        """Start the client and establish all necessary connections.

        Initializes LiveKit connections using the provided configuration.

        Args:
            frame: Initial configuration frame containing audio parameters
        """

        logger.debug(f"AgentHumanClient starting")
        await self._livekit_connect()

    async def stop(self) -> None:
        """Stop the client and terminate all connections.

        Disconnects from LiveKit endpoints, and performs cleanup.
        """
        logger.debug(f"AgentHumanVideoService stopping")
        await self._livekit_disconnect()
        await self.cleanup()

    async def interrupt(self) -> None:
        """Interrupt the avatar's current speech, clearing its audio buffer.

        Sends ``lk.clear_buffer`` to the avatar worker (same path as LiveKit
        ``DataStreamAudioOutput.clear_buffer``). That RPC is required so the
        server clears its pipeline and notifies ComfyUI; calling only
        ``DataStreamAudioOutput.clear_buffer()`` is a no-op until the stream
        has finished starting, so user barge-in could miss the server.
        """
        logger.debug("AgentHumanClient interrupt")

        if self._livekit_room and self._livekit_room.isconnected():
            try:
                await self._livekit_room.local_participant.perform_rpc(
                    destination_identity=_AVATAR_AGENT_IDENTITY,
                    method=_LK_CLEAR_BUFFER_RPC,
                    payload="",
                )
            except Exception as e:
                logger.warning(
                    "AgentHumanClient interrupt: lk.clear_buffer RPC failed "
                    f"(avatar may not be connected yet): {e}"
                )
                if self._data_stream_audio:
                    self._data_stream_audio.clear_buffer()
        elif self._data_stream_audio:
            self._data_stream_audio.clear_buffer()

    def transport_ready(self) -> None:
        """Indicates that the output transport is ready and able to receive frames."""
        self._transport_ready = True

    async def agent_speak(self, audio: bytes) -> None:
        """Send audio data to the avatar via LiveKit data stream.

        Args:
            audio: Raw PCM audio bytes (16-bit, mono)
        """

        if self._data_stream_audio:
            audio_frame = rtc.AudioFrame(
                data=audio,
                sample_rate=AGENTHUMAN_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(audio) // 2,
            )
            await self._data_stream_audio.capture_frame(audio_frame)

    async def capture_participant_audio(self, participant_id: str, callback) -> None:
        """Capture audio frames from the AgentHuman avatar.

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
                        self._process_audio_frames(audio_stream), name="AgentHumanClient_Receive_Audio"
                    )
                    break

    async def capture_participant_video(self, participant_id: str, callback) -> None:
        """Capture video frames from the AgentHuman avatar.

        Args:
            participant_id: Identifier of the participant to capture video from
            callback: Async function to handle received video frames
        """
        logger.debug(f"capture_participant_video: {participant_id}")
        self._video_frame_callback = callback
        if self._video_task is not None:
            logger.warning(
                "Trying to capture more than one video stream. It is currently not supported."
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
                        self._process_video_frames(video_stream), name="AgentHumanClient_Receive_Video"
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
                        num_channels=1,  # AgentHuman uses mono audio
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
            logger.debug(f"AgentHumanClient livekit connecting to room URL: {self._agentHuman_session.roomURL}")
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
                        self._process_video_frames(video_stream), name="AgentHumanClient_Receive_Video"
                    )
                elif (
                    track.kind == rtc.TrackKind.KIND_AUDIO
                    and self._audio_frame_callback is not None
                    and self._audio_task is None
                ):
                    logger.debug(f"Creating audio stream processor for track: {publication.sid}")
                    audio_stream = rtc.AudioStream(track)
                    self._audio_task = self._task_manager.create_task(
                        self._process_audio_frames(audio_stream), name="AgentHumanClient_Receive_Audio"
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
                self._agentHuman_session.roomURL, self._agentHuman_session.participantToken
            )

            self._data_stream_audio = DataStreamAudioOutput(
                room=self._livekit_room,
                destination_identity=_AVATAR_AGENT_IDENTITY,
                sample_rate=AGENTHUMAN_SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
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

            self._data_stream_audio = None

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