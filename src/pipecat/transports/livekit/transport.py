#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit transport implementation for Pipecat.

This module provides comprehensive LiveKit real-time communication integration
including audio streaming, data messaging, participant management, and room
event handling for conversational AI applications.
"""

import asyncio
import functools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler, mix_audio
from pipecat.frames.frames import (
    AudioRawFrame,
    BotConnectedFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    ImageRawFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.video_in_sampler import _capture_framerate, _VideoInSamplers
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from livekit import rtc
    from livekit.rtc._proto import video_frame_pb2 as proto_video_frame
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use LiveKit, you need to `uv add "pipecat-ai[livekit]"`.')
    raise ImportError(f"Missing module: {e}") from e

# DTMF mapping according to RFC 4733
DTMF_CODE_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "*": 10,
    "#": 11,
}

# Maps Pipecat's PIL-style color format strings (``OutputImageRawFrame.format``,
# configured via ``TransportParams.video_out_color_format``) to LiveKit's
# ``VideoBufferType`` enum used by ``rtc.VideoFrame`` and its bytes per pixel.
LIVEKIT_VIDEO_BUFFER_TYPES = {
    "RGB": (proto_video_frame.VideoBufferType.RGB24, 3),
    "RGBA": (proto_video_frame.VideoBufferType.RGBA, 4),
    "BGRA": (proto_video_frame.VideoBufferType.BGRA, 4),
    "ARGB": (proto_video_frame.VideoBufferType.ARGB, 4),
}

CAM_VIDEO_SOURCE = "camera"
SCREEN_VIDEO_SOURCE = "screenVideo"


def _video_source(publication: rtc.RemoteTrackPublication) -> str:
    """The Pipecat video source of a published video track.

    A screen share is ``"screenVideo"``. Any other video, including a track
    published without a source, is ``"camera"``.
    """
    if publication.source == rtc.TrackSource.SOURCE_SCREENSHARE:
        return SCREEN_VIDEO_SOURCE
    return CAM_VIDEO_SOURCE


@dataclass
class LiveKitInputTransportMessageFrame(InputTransportMessageFrame):
    """Frame for incoming transport messages from LiveKit rooms.

    Parameters:
        participant_id: ID of the participant this message is from.
    """

    participant_id: str | None = None


@dataclass
class LiveKitOutputTransportMessageFrame(OutputTransportMessageFrame):
    """Frame for transport messages in LiveKit rooms.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: str | None = None


@dataclass
class LiveKitOutputTransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in LiveKit rooms.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: str | None = None


class LiveKitParams(TransportParams):
    """Configuration parameters for LiveKit transport.

    Video output publishes a single ``"pipecat-video"`` camera track (mirroring how
    audio output always publishes one ``"pipecat-audio"`` microphone track) when
    ``video_out_enabled`` is set. The track is sized using
    ``video_out_width``/``video_out_height`` and encodes frames according to
    ``video_out_color_format`` (default ``"RGB"``); ``video_out_framerate``
    governs how often ``BaseOutputTransport`` draws frames. Per-destination
    video routing (multiple named output tracks, as supported by Daily's
    ``camera_out_enabled``/``register_video_destination``) is not yet
    implemented for LiveKit.

    ``video_out_codec`` selects the published video codec (``"VP8"``, ``"H264"``,
    ``"VP9"``, ``"AV1"`` or ``"H265"``); LiveKit picks one when it is unset.

    With ``video_in_enabled``, a user's camera (``"camera"``) and screen share
    (``"screenVideo"``) are received separately. Without ``video_in_sources``,
    every frame of every video track is passed on. With it, only the listed
    sources are, each at its own framerate, plus any source captured with
    ``LiveKitTransport.capture_participant_video()``.

    Parameters:
        audio_in_user_tracks: Receive each participant's audio as its own stream of
            ``UserAudioRawFrame``s tagged with their identity (the default). All of
            these streams enter the pipeline through the one input transport, so a
            pipeline that runs a single STT and VAD should only receive one of them.
            When False, the participants' audio is mixed into a single stream of
            ``InputAudioRawFrame``s.
        audio_out_queue_size_ms: Buffer size of the outgoing audio source, in milliseconds
            (LiveKit's default is 1000).
        video_out_max_bitrate: Maximum bitrate of the published video track, in bits
            per second, capped at ``video_out_framerate``. LiveKit chooses the encoding
            from the track resolution when unset.
    """

    audio_in_user_tracks: bool = True
    audio_out_queue_size_ms: int = 1000
    video_out_max_bitrate: int | None = None


class LiveKitCallbacks(BaseModel):
    """Callback handlers for LiveKit events.

    Parameters:
        on_connected: Called when connected to the LiveKit room.
        on_disconnected: Called when disconnected from the LiveKit room.
        on_participant_connected: Called when a participant joins the room.
        on_participant_disconnected: Called when a participant leaves the room.
        on_audio_track_subscribed: Called when an audio track is subscribed.
        on_audio_track_unsubscribed: Called when an audio track is unsubscribed.
        on_data_received: Called when data is received. The sender is None for
            packets sent by a server SDK, which LiveKit delivers unattributed.
        on_first_participant_joined: Called when the first participant joins.
        on_dtmf_event: Called when a SIP DTMF tone is received.
        on_active_speaker_changed: Called with the identity of the room's loudest speaker
            when it changes. The bot can be the speaker while it talks.
        on_video_track_muted: Called with the participant and video source when a
            participant mutes a video track, e.g. turns their camera off.
    """

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_before_disconnect: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_video_track_subscribed: Callable[[str], Awaitable[None]]
    on_video_track_unsubscribed: Callable[[str, str], Awaitable[None]]
    on_data_received: Callable[[bytes, str | None], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]
    on_dtmf_event: Callable[[Any], Awaitable[None]]
    on_active_speaker_changed: Callable[[str], Awaitable[None]]
    on_video_track_muted: Callable[[str, str], Awaitable[None]]


class LiveKitTransportClient:
    """Core client for interacting with LiveKit rooms.

    Manages the connection to LiveKit rooms and handles all low-level API interactions
    including room management, audio streaming, data messaging, and event handling.
    """

    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: LiveKitParams,
        callbacks: LiveKitCallbacks,
        transport_name: str,
    ):
        """Initialize the LiveKit transport client.

        Args:
            url: LiveKit server URL to connect to.
            token: Authentication token for the room.
            room_name: Name of the LiveKit room to join.
            params: Configuration parameters for the transport.
            callbacks: Event callback handlers.
            transport_name: Name identifier for the transport.
        """
        self._url = url
        self._token = token
        self._room_name = room_name
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name
        self._room: rtc.Room | None = None
        self._participant_id: str = ""
        self._connected = False
        self._disconnect_counter = 0
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._audio_tracks = {}
        self._audio_queue = asyncio.Queue()
        # Per-participant ``(AudioStream, Task)`` so unsubscribe can close
        # the owned native stream and cancel its producer task instead of
        # leaking both on every track republish.
        self._audio_streams: dict[str, tuple[rtc.AudioStream, asyncio.Task]] = {}
        self._video_source: rtc.VideoSource | None = None
        self._video_track: rtc.LocalVideoTrack | None = None
        # Video tracks and streams by participant and video source, so a user's
        # camera and screen share are received side by side.
        self._video_tracks: dict[tuple[str, str], rtc.Track] = {}
        self._video_publications: dict[tuple[str, str], rtc.RemoteTrackPublication] = {}
        self._video_queue = asyncio.Queue()
        self._video_streams: dict[tuple[str, str], tuple[rtc.VideoStream, asyncio.Task]] = {}
        self._other_participant_has_joined = False
        self._task_manager: BaseTaskManager | None = None
        self._active_speaker_id: str | None = None
        self._async_lock = asyncio.Lock()

    @property
    def participant_id(self) -> str:
        """Get the participant ID for this client.

        Returns:
            The participant ID assigned by LiveKit.
        """
        return self._participant_id

    @property
    def room(self) -> rtc.Room:
        """Get the LiveKit room instance.

        Returns:
            The LiveKit room object.

        Raises:
            Exception: If room object is not available.
        """
        if not self._room:
            raise Exception(f"{self}: missing room object (pipeline not started?)")
        return self._room

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client with task manager and room initialization.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._task_manager:
            return

        self._task_manager = setup.task_manager
        self._room = rtc.Room(loop=self._task_manager.get_event_loop())

        self._out_sample_rate = self._params.audio_out_sample_rate or setup.audio_out_sample_rate

        # Set up room event handlers
        self.room.on("participant_connected")(self._on_participant_connected_wrapper)
        self.room.on("participant_disconnected")(self._on_participant_disconnected_wrapper)
        self.room.on("track_subscribed")(self._on_track_subscribed_wrapper)
        self.room.on("track_unsubscribed")(self._on_track_unsubscribed_wrapper)
        self.room.on("track_muted")(self._on_track_muted_wrapper)
        self.room.on("data_received")(self._on_data_received_wrapper)
        self.room.on("connected")(self._on_connected_wrapper)
        self.room.on("disconnected")(self._on_disconnected_wrapper)
        self.room.on("sip_dtmf_received")(self._on_sip_dtmf_received_wrapper)
        self.room.on("active_speakers_changed")(self._on_active_speakers_changed_wrapper)

    async def cleanup(self):
        """Cleanup client resources."""
        await self.disconnect()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def connect(self):
        """Connect to the LiveKit room with retry logic."""
        async with self._async_lock:
            if self._connected:
                # Increment disconnect counter if already connected.
                self._disconnect_counter += 1
                return

            logger.info(f"Connecting to {self._room_name}")
            self._active_speaker_id = None

            try:
                await self.room.connect(
                    self._url,
                    self._token,
                    options=rtc.RoomOptions(auto_subscribe=True),
                )
                self._participant_id = self.room.local_participant.identity
                logger.info(f"Connected to {self._room_name} as {self._participant_id}")

                # Set up audio source and track
                self._audio_source = rtc.AudioSource(
                    self._out_sample_rate,
                    self._params.audio_out_channels,
                    queue_size_ms=self._params.audio_out_queue_size_ms,
                )
                self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                    "pipecat-audio", self._audio_source
                )
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_MICROPHONE
                await self.room.local_participant.publish_track(self._audio_track, options)

                # Set up video source and track (only if video output is
                # enabled; unlike audio, which is always published).
                if self._params.video_out_enabled:
                    self._video_source = rtc.VideoSource(
                        self._params.video_out_width, self._params.video_out_height
                    )
                    self._video_track = rtc.LocalVideoTrack.create_video_track(
                        "pipecat-video", self._video_source
                    )
                    video_options = self._video_publish_options()
                    await self.room.local_participant.publish_track(
                        self._video_track, video_options
                    )

                # Only mark the client connected once its tracks are
                # published, so a retry after a failed publish starts over
                # instead of returning early without tracks.
                self._connected = True
                # Increment disconnect counter if we successfully connected.
                self._disconnect_counter += 1

                await self._callbacks.on_connected()

                # Participants already in the room raise the same events as
                # those who join later.
                for participant_id in self.get_participants():
                    await self._participant_connected(participant_id)
            except Exception as e:
                logger.error(f"Error connecting to {self._room_name}: {e}")
                if not self._connected:
                    await self._rollback_partial_connect()
                raise

    def _video_publish_options(self) -> rtc.TrackPublishOptions:
        """Build the publish options for the video track from the params."""
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA

        # LiveKit requires both fields of a video encoding, so the framerate
        # is only sent along with a bitrate.
        if self._params.video_out_max_bitrate is not None:
            options.video_encoding.max_bitrate = self._params.video_out_max_bitrate
            options.video_encoding.max_framerate = self._params.video_out_framerate

        codec = self._params.video_out_codec
        if codec:
            try:
                options.video_codec = rtc.VideoCodec.Value(codec.upper())
            except ValueError:
                logger.warning(
                    f"{self} unsupported video codec for LiveKit output: {codec!r}, "
                    f"expected one of {list(rtc.VideoCodec.keys())}"
                )

        return options

    async def _rollback_partial_connect(self):
        """Undo a connection attempt that failed before it completed."""
        await self._close_output_sources()
        try:
            await self.room.disconnect()
        except Exception as e:
            logger.warning(f"{self} error disconnecting after failed connect: {e}")

    async def disconnect(self):
        """Disconnect from the LiveKit room."""
        async with self._async_lock:
            # Decrement leave counter when leaving.
            self._disconnect_counter -= 1

            if not self._connected or self._disconnect_counter > 0:
                return

            logger.info(f"Disconnecting from {self._room_name}")
            await self._callbacks.on_before_disconnect()
            # Mark the client disconnected before the room disconnects, so the
            # room's own "disconnected" event does not report it a second time.
            self._connected = False
            await self.room.disconnect()
            await self._close_output_sources()
            # Close any remaining per-participant streams and cancel their
            # producer tasks so they do not outlive the connection.
            await self._close_all_streams()
            logger.info(f"Disconnected from {self._room_name}")
            await self._callbacks.on_disconnected()

    async def _close_output_sources(self):
        """Close the published audio and video sources.

        ``room.disconnect()`` does not release the native source handles, so
        each connection would otherwise leave them behind.
        """
        audio_source, self._audio_source, self._audio_track = self._audio_source, None, None
        video_source, self._video_source, self._video_track = self._video_source, None, None
        for source in (audio_source, video_source):
            if source is None:
                continue
            try:
                await source.aclose()
            except Exception as e:
                logger.warning(f"{self} error closing output source: {e}")

    async def send_data(self, data: bytes, participant_id: str | None = None):
        """Send data to participants in the room.

        Args:
            data: The data bytes to send.
            participant_id: Optional specific participant to send to.
        """
        if not self._connected:
            return

        try:
            if participant_id:
                await self.room.local_participant.publish_data(
                    data, reliable=True, destination_identities=[participant_id]
                )
            else:
                await self.room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            logger.error(f"Error sending data: {e}")

    async def send_dtmf(self, digit: str):
        r"""Send DTMF tone to the room.

        Args:
            digit: The DTMF digit to send (0-9, \*, #).
        """
        if not self._connected:
            return

        if digit not in DTMF_CODE_MAP:
            logger.warning(f"Invalid DTMF digit: {digit}")
            return

        code = DTMF_CODE_MAP[digit]

        try:
            await self.room.local_participant.publish_dtmf(code=code, digit=digit)
        except Exception as e:
            logger.error(f"Error sending DTMF tone {digit}: {e}")

    async def publish_audio(self, audio_frame: rtc.AudioFrame) -> bool:
        """Publish an audio frame to the room.

        Args:
            audio_frame: The LiveKit audio frame to publish.
        """
        if not self._connected or not self._audio_source:
            return False

        try:
            await self._audio_source.capture_frame(audio_frame)
            return True
        except Exception as e:
            # When using an audio mixer, the base output transport's
            # with_mixer() generator continuously yields frames (mixed with
            # background audio) even when no TTS audio is queued. During
            # interruptions, the audio task is cancelled and recreated, but
            # there is a brief window where the native LiveKit AudioSource
            # rejects capture_frame() with an InvalidState error. This is a
            # transient condition — the mixer will produce a new frame within
            # milliseconds, so we silently drop these frames.
            if "InvalidState" not in str(e):
                logger.error(f"Error publishing audio: {e}")
            return False

    async def publish_video(self, video_frame: rtc.VideoFrame) -> bool:
        """Publish a video frame to the room.

        Args:
            video_frame: The LiveKit video frame to publish.

        Returns:
            True if the video frame was published successfully, False otherwise.
        """
        if not self._connected or not self._video_source:
            return False

        try:
            # Unlike ``AudioSource.capture_frame``, ``VideoSource.capture_frame``
            # is synchronous in livekit-rtc.
            self._video_source.capture_frame(video_frame)
            return True
        except Exception as e:
            logger.error(f"Error publishing video: {e}")
            return False

    def get_participants(self) -> list[str]:
        """Get list of participant IDs in the room.

        Returns:
            List of participant LiveKit identities.
        """
        return [p.identity for p in self.room.remote_participants.values()]

    async def get_participant_metadata(self, participant_id: str) -> dict:
        """Get metadata for a specific participant.

        Args:
            participant_id: LiveKit identity of the participant to get metadata for.

        Returns:
            Dictionary containing participant metadata.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            return {
                "id": participant.identity,
                "name": participant.name,
                "metadata": participant.metadata,
            }
        return {}

    async def set_participant_metadata(self, metadata: str):
        """Set metadata for the local participant.

        Args:
            metadata: Metadata string to set.
        """
        await self.room.local_participant.set_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        """Stop receiving a specific participant's audio.

        LiveKit doesn't let one participant force-mute another's microphone;
        this unsubscribes the bot from their audio track instead.

        Args:
            participant_id: LiveKit identity of the participant to stop
                receiving audio from.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    publication.set_subscribed(False)

    async def unmute_participant(self, participant_id: str):
        """Resume receiving a specific participant's audio.

        Args:
            participant_id: LiveKit identity of the participant to resume
                receiving audio from.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    publication.set_subscribed(True)

    # Wrapper methods for event handlers
    def _on_participant_connected_wrapper(self, participant: rtc.RemoteParticipant):
        """Wrapper for participant connected events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_participant_connected(participant),
            f"{self}::_async_on_participant_connected",
        )

    def _on_participant_disconnected_wrapper(self, participant: rtc.RemoteParticipant):
        """Wrapper for participant disconnected events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_participant_disconnected(participant),
            f"{self}::_async_on_participant_disconnected",
        )

    def _on_track_subscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Wrapper for track subscribed events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_track_subscribed(track, publication, participant),
            f"{self}::_async_on_track_subscribed",
        )

    def _on_track_unsubscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Wrapper for track unsubscribed events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_track_unsubscribed(track, publication, participant),
            f"{self}::_async_on_track_unsubscribed",
        )

    def _on_track_muted_wrapper(
        self,
        participant: rtc.RemoteParticipant,
        publication: rtc.RemoteTrackPublication,
    ):
        """Wrapper for track muted events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_track_muted(participant, publication),
            f"{self}::_async_on_track_muted",
        )

    async def _async_on_track_muted(
        self, participant: rtc.RemoteParticipant, publication: rtc.RemoteTrackPublication
    ):
        """Handle track muted events."""
        if publication.kind == rtc.TrackKind.KIND_VIDEO:
            video_source = _video_source(publication)
            logger.info(f"Video track muted: {video_source} from {participant.identity}")
            await self._callbacks.on_video_track_muted(participant.identity, video_source)

    def _on_data_received_wrapper(self, data: rtc.DataPacket):
        """Wrapper for data received events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_data_received(data),
            f"{self}::_async_on_data_received",
        )

    def _on_connected_wrapper(self):
        """Wrapper for connected events."""
        assert self._task_manager is not None

        self._task_manager.create_task(self._async_on_connected(), f"{self}::_async_on_connected")

    def _on_disconnected_wrapper(self):
        """Wrapper for disconnected events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_disconnected(), f"{self}::_async_on_disconnected"
        )

    def _on_sip_dtmf_received_wrapper(self, dtmf: rtc.SipDTMF):
        """Wrapper for inbound SIP DTMF events."""
        assert self._task_manager is not None

        self._task_manager.create_task(
            self._async_on_sip_dtmf_received(dtmf),
            f"{self}::_async_on_sip_dtmf_received",
        )

    def _on_active_speakers_changed_wrapper(self, speakers: list[rtc.Participant]):
        """Wrapper for active speakers changed events.

        LiveKit lists only the speakers whose level changed, quietest first, so the last
        entry is the loudest and an empty list doesn't mean silence: report the loudest
        speaker only when it changes, and keep the current one through empty updates.
        """
        assert self._task_manager is not None

        if not speakers:
            return
        speaker_id = speakers[-1].identity
        if speaker_id == self._active_speaker_id:
            return
        self._active_speaker_id = speaker_id
        self._task_manager.create_task(
            self._async_on_active_speaker_changed(speaker_id),
            f"{self}::_async_on_active_speaker_changed",
        )

    # Async methods for event handling
    async def _async_on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connected events."""
        await self._participant_connected(participant.identity)

    async def _participant_connected(self, participant_id: str):
        """Report a participant in the room, and whether they're the first."""
        logger.info(f"Participant connected: {participant_id}")
        await self._callbacks.on_participant_connected(participant_id)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(participant_id)

    async def _async_on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnected events."""
        logger.info(f"Participant disconnected: {participant.identity}")
        await self._callbacks.on_participant_disconnected(participant.identity)
        if len(self.get_participants()) == 0:
            self._other_participant_has_joined = False

    async def _async_on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track subscribed events."""
        assert self._task_manager is not None

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(
                f"Audio track subscribed: {track.sid} from participant {participant.identity}"
            )
            # If the participant is re-publishing (e.g. mute/unmute cycle),
            # close + cancel the previous stream/task before replacing the
            # registry entry, so two producers never feed ``_audio_queue``
            # for the same participant.
            await self._close_audio_stream(participant.identity)
            self._audio_tracks[participant.identity] = track
            audio_stream = rtc.AudioStream(track)
            task = self._task_manager.create_task(
                self._process_audio_stream(audio_stream, participant.identity),
                f"{self}::_process_audio_stream",
            )
            self._audio_streams[participant.identity] = (audio_stream, task)
            await self._callbacks.on_audio_track_subscribed(participant.identity)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            video_source = _video_source(publication)
            logger.info(
                f"Video track subscribed: {track.sid} ({video_source}) from participant "
                f"{participant.identity}"
            )
            # Clean up any prior video stream/task for the same participant and
            # source before replacing.
            await self._close_video_stream(participant.identity, video_source)
            self._video_tracks[(participant.identity, video_source)] = track
            self._video_publications[(participant.identity, video_source)] = publication
            # Only process video stream if video input is enabled to prevent
            # unbounded queue growth when there is no consumer for video frames.
            if self._params.video_in_enabled:
                video_stream = rtc.VideoStream(track)
                task = self._task_manager.create_task(
                    self._process_video_stream(video_stream, participant.identity, video_source),
                    f"{self}::_process_video_stream",
                )
                self._video_streams[(participant.identity, video_source)] = (video_stream, task)
            await self._callbacks.on_video_track_subscribed(participant.identity)

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track unsubscribed events."""
        logger.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            await self._close_audio_stream(participant.identity)
            await self._callbacks.on_audio_track_unsubscribed(participant.identity)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            video_source = _video_source(publication)
            self._video_tracks.pop((participant.identity, video_source), None)
            self._video_publications.pop((participant.identity, video_source), None)
            await self._close_video_stream(participant.identity, video_source)
            await self._callbacks.on_video_track_unsubscribed(participant.identity, video_source)

    async def _close_audio_stream(self, participant_id: str) -> None:
        """Close a participant's owned audio stream and cancel its producer task.

        Idempotent: no-op when there is no registered stream for the
        participant.
        """
        entry = self._audio_streams.pop(participant_id, None)
        if entry is None:
            return
        stream, task = entry
        try:
            await asyncio.wait_for(stream.aclose(), timeout=2.0)
        except Exception as e:
            logger.warning(f"AudioStream.aclose failed for {participant_id}: {e}")
        if task is not None and not task.done():
            task.cancel()

    async def _close_video_stream(self, participant_id: str, video_source: str) -> None:
        """Close a participant's owned video stream and cancel its producer task.

        Idempotent: no-op when there is no registered stream for the
        participant and video source.
        """
        entry = self._video_streams.pop((participant_id, video_source), None)
        if entry is None:
            return
        stream, task = entry
        try:
            await asyncio.wait_for(stream.aclose(), timeout=2.0)
        except Exception as e:
            logger.warning(f"VideoStream.aclose failed for {participant_id} ({video_source}): {e}")
        if task is not None and not task.done():
            task.cancel()

    async def _close_all_streams(self) -> None:
        """Close every per-participant audio/video stream and cancel its task.

        Idempotent: no-op when no streams are registered.
        """
        for participant_id in list(self._audio_streams.keys()):
            await self._close_audio_stream(participant_id)
        for participant_id, video_source in list(self._video_streams.keys()):
            await self._close_video_stream(participant_id, video_source)

    async def _async_on_data_received(self, data: rtc.DataPacket):
        """Handle data received events."""
        # LiveKit delivers packets sent by a server SDK with no participant.
        sender = data.participant.identity if data.participant else None
        await self._callbacks.on_data_received(data.data, sender)

    async def _async_on_connected(self):
        """Handle connected events."""
        await self._callbacks.on_connected()

    async def _async_on_disconnected(self, reason=None):
        """Handle disconnected events.

        Only a disconnect of a connected client is reported. ``disconnect()``
        and a failed ``connect()`` clear the connected flag before disconnecting
        the room, so the room's event for those is ignored.
        """
        if not self._connected:
            return
        self._connected = False
        logger.info(f"Disconnected from {self._room_name}. Reason: {reason}")
        await self._callbacks.on_disconnected()

    async def _async_on_sip_dtmf_received(self, dtmf: rtc.SipDTMF):
        """Handle inbound SIP DTMF events from LiveKit telephony."""
        participant = getattr(dtmf, "participant", None)
        participant_id = getattr(participant, "identity", None) if participant else None
        data = {
            "tone": dtmf.digit,
            "digit": dtmf.digit,
            "code": dtmf.code,
            "participant_id": participant_id,
        }
        logger.debug(f"{self} SIP DTMF event: {data}")
        await self._callbacks.on_dtmf_event(data)

    async def _async_on_active_speaker_changed(self, speaker_id: str):
        """Handle a change of the room's loudest speaker."""
        await self._callbacks.on_active_speaker_changed(speaker_id)

    async def _process_audio_stream(self, audio_stream: rtc.AudioStream, participant_id: str):
        """Process incoming audio stream from a participant."""
        logger.info(f"Started processing audio stream for participant {participant_id}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                await self._audio_queue.put((event, participant_id))
            else:
                logger.warning(f"Received unexpected event type: {type(event)}")

    async def get_next_audio_frame(self):
        """Get the next audio frame from the queue."""
        while True:
            frame, participant_id = await self._audio_queue.get()
            yield frame, participant_id

    def video_source_enabled(self, participant_id: str, video_source: str) -> bool:
        """Whether a participant is sending a video source.

        A participant who turns their camera off usually mutes the track rather
        than unpublishing it, so a subscribed track can still send nothing.

        Args:
            participant_id: The participant's identity.
            video_source: The video source, ``"camera"`` or ``"screenVideo"``.

        Returns:
            Whether the source's track is subscribed and not muted.
        """
        publication = self._video_publications.get((participant_id, video_source))
        return publication is not None and not publication.muted

    async def _process_video_stream(
        self, video_stream: rtc.VideoStream, participant_id: str, video_source: str
    ):
        """Process incoming video stream from a participant."""
        logger.info(
            f"Started processing {video_source} video stream for participant {participant_id}"
        )
        async for event in video_stream:
            if isinstance(event, rtc.VideoFrameEvent):
                await self._video_queue.put((event, participant_id, video_source))
            else:
                logger.warning(f"Received unexpected event type: {type(event)}")

    async def get_next_video_frame(self):
        """Get the next video frame from the queue."""
        while True:
            frame, participant_id, video_source = await self._video_queue.get()
            yield frame, participant_id, video_source

    def __str__(self):
        """String representation of the LiveKit transport client."""
        return f"{self._transport_name}::LiveKitTransportClient"


class _ParticipantAudioMixer:
    """Mixes participants' audio, as it arrives, into a single stream.

    Each participant's audio is buffered and mixed in fixed-size chunks once
    every participant has a chunk buffered, so the output keeps the pace of the
    participants' streams. A participant who stops sending audio holds the mix
    back until another participant has ``max_wait_ms`` buffered; then they are
    left out of the mix until their audio arrives again.
    """

    def __init__(
        self, *, sample_rate: int, num_channels: int, chunk_ms: int = 10, max_wait_ms: int = 50
    ):
        """Initialize the mixer.

        Args:
            sample_rate: Sample rate of the participants' audio.
            num_channels: Number of channels of the participants' audio.
            chunk_ms: Duration of each mixed chunk, in milliseconds.
            max_wait_ms: How much audio another participant buffers before the
                mix stops waiting for a participant with no audio buffered.
        """
        bytes_per_ms = sample_rate * num_channels * 2 // 1000
        self._chunk_bytes = chunk_ms * bytes_per_ms
        self._max_wait_bytes = max_wait_ms * bytes_per_ms
        self._buffers: dict[str, bytearray] = {}

    def add(self, participant_id: str, audio: bytes) -> list[bytes]:
        """Add a participant's audio and return the chunks it completes.

        Args:
            participant_id: The participant the audio belongs to.
            audio: 16-bit PCM audio.

        Returns:
            The mixed chunks that are ready, oldest first.
        """
        self._buffers.setdefault(participant_id, bytearray()).extend(audio)
        chunks = []
        while True:
            if any(len(buffer) < self._chunk_bytes for buffer in self._buffers.values()):
                if all(len(buffer) < self._max_wait_bytes for buffer in self._buffers.values()):
                    break
                for silent_id in [p for p, buffer in self._buffers.items() if not buffer]:
                    del self._buffers[silent_id]
            chunks.append(self._mix_chunk())
        return chunks

    def _mix_chunk(self) -> bytes:
        parts = []
        for buffer in self._buffers.values():
            parts.append(bytes(buffer[: self._chunk_bytes]))
            del buffer[: self._chunk_bytes]
        return functools.reduce(mix_audio, parts)


class LiveKitInputTransport(BaseInputTransport):
    """Handles incoming media streams and events from LiveKit rooms.

    Processes incoming audio streams from room participants and forwards them
    as Pipecat frames, including audio resampling and VAD integration.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: LiveKitTransportClient,
        params: LiveKitParams,
        **kwargs,
    ):
        """Initialize the LiveKit input transport.

        Args:
            transport: The parent transport instance.
            client: LiveKitTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client

        self._audio_in_task = None
        self._video_in_task = None
        # One resampler per participant: a stream resampler keeps state between
        # calls, so sharing one would mix the audio of participants who talk at once.
        self._resamplers: dict[str, BaseAudioResampler] = {}
        self._video_samplers = _VideoInSamplers()
        self._audio_in_user_tracks = params.audio_in_user_tracks
        self._mixer: _ParticipantAudioMixer | None = None
        self._mixed_resampler = create_stream_resampler()

    def _supports_video_in_source(self, video_source: str) -> bool:
        """Whether this transport captures a video source listed in ``video_in_sources``.

        Args:
            video_source: The video source.

        Returns:
            Whether the source is the camera or the screen share.
        """
        return video_source in (CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE)

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the input transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)

        await self._client.setup(setup)

        await self._client.connect()

        logger.info("LiveKitInputTransport connected")

    async def cleanup(self):
        """Release input transport resources at teardown."""
        await super().cleanup()
        await self._teardown()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        """Start receiving media from the LiveKit room.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if not self._audio_in_task and self._params.audio_in_enabled:
            self._audio_in_task = self.create_task(self._audio_in_task_handler())
        if not self._video_in_task and self._params.video_in_enabled:
            self._video_in_task = self.create_task(self._video_in_task_handler())

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the input transport and disconnect from LiveKit room.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._teardown()
        logger.info("LiveKitInputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and disconnect from LiveKit room.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._teardown()

    async def _teardown(self):
        """Disconnect the client and cancel the media input tasks.

        Single idempotent teardown body shared by ``stop``, ``cancel`` and
        ``cleanup``.
        """
        await self._client.disconnect()
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
            self._audio_in_task = None
        if self._video_in_task:
            await self.cancel_task(self._video_in_task)
            self._video_in_task = None
        self._video_samplers.clear()

    async def push_app_message(self, message: Any, sender: str | None):
        """Push an application message as an urgent transport frame.

        Broadcast both upstream and downstream so it reaches ``RTVIProcessor``
        regardless of where it sits relative to the transport in the pipeline.

        Args:
            message: The message data to send.
            sender: ID of the message sender, or None if it was sent
                unattributed by a server SDK.
        """
        await self.broadcast_frame(
            LiveKitInputTransportMessageFrame, message=message, participant_id=sender
        )

    def release_participant(self, participant_id: str):
        """Drop a participant's resampler once their audio track is gone."""
        self._resamplers.pop(participant_id, None)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, answering user image requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int | None = 30,
        video_source: str = CAM_VIDEO_SOURCE,
        *,
        on_request_only: bool = False,
    ):
        """Capture a participant's video source at a framerate.

        This takes precedence over ``video_in_sources`` for the source.

        Args:
            participant_id: The participant's identity.
            framerate: Frames per second to pass on, or ``None`` for every frame. It
                doesn't apply with ``on_request_only``.
            video_source: The video source, ``"camera"`` or ``"screenVideo"``.
            on_request_only: Pass on only the frames that answer image requests.
        """
        framerate = _capture_framerate(
            framerate, on_request_only, "LiveKitTransport.capture_participant_video"
        )
        self._video_samplers.capture(participant_id, video_source, framerate)

    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Request a video frame from a specific participant.

        Args:
            frame: The user image request frame.
        """
        video_source = frame.video_source or CAM_VIDEO_SOURCE
        if not self._client.video_source_enabled(frame.user_id, video_source):
            error = f"{frame.user_id} isn't sending {video_source} video."
            await self._answer_image_requests([frame], error)
            return

        self._capture_configured_video(frame.user_id, video_source)
        if not self._video_samplers.add_request(frame.user_id, video_source, frame):
            error = f"No {video_source} video is being captured from {frame.user_id}."
            await self._answer_image_requests([frame], error)

    async def stop_participant_video(self, participant_id: str, video_source: str):
        """Answer the image requests waiting on a video source that stopped.

        The source stays captured, so its frames are sampled again if it restarts.

        Args:
            participant_id: The participant's identity.
            video_source: The video source that stopped.
        """
        requests = self._video_samplers.take_requests(participant_id, video_source)
        error = f"{participant_id} stopped sending {video_source} video."
        await self._answer_image_requests(requests, error)

    async def remove_participant_video(self, participant_id: str):
        """Stop sampling video from a participant who left.

        Args:
            participant_id: The participant's identity.
        """
        requests = self._video_samplers.remove_participant(participant_id)
        await self._answer_image_requests(requests, f"{participant_id} left.")

    async def remove_all_video(self):
        """Stop sampling video from every participant, e.g. after leaving the room."""
        requests = self._video_samplers.clear()
        await self._answer_image_requests(requests, "The bot left the room.")

    def _capture_configured_video(self, participant_id: str, video_source: str):
        """Capture a source as ``video_in_sources`` configures it, unless it already is.

        Without ``video_in_sources``, every source is captured at every frame.
        With it, only the listed sources are, each at its own framerate.
        """
        if self._video_samplers.is_capturing(participant_id, video_source):
            return
        if not self._params.video_in_sources:
            self._video_samplers.capture(participant_id, video_source, None)
        elif source_params := self._params.video_in_sources.get(video_source):
            framerate = 0 if source_params.on_request_only else source_params.framerate
            self._video_samplers.capture(participant_id, video_source, framerate)

    async def _audio_in_task_handler(self):
        """Handle incoming audio frames from participants."""
        logger.info("Audio input task started")
        audio_iterator = self._client.get_next_audio_frame()
        async for audio_data in audio_iterator:
            if not audio_data:
                continue
            audio_frame_event, participant_id = audio_data

            if not self._audio_in_user_tracks:
                await self._push_mixed_audio(audio_frame_event.frame, participant_id)
                continue

            pipecat_audio_frame = await self._convert_livekit_audio_to_pipecat(
                audio_frame_event, participant_id
            )

            # Skip frames with no audio data
            if len(pipecat_audio_frame.audio) == 0:
                continue

            input_audio_frame = UserAudioRawFrame(
                user_id=participant_id,
                audio=pipecat_audio_frame.audio,
                sample_rate=pipecat_audio_frame.sample_rate,
                num_channels=pipecat_audio_frame.num_channels,
            )
            await self.push_audio_frame(input_audio_frame)

    async def _push_mixed_audio(self, audio_frame: rtc.AudioFrame, participant_id: str):
        """Mix a participant's audio into the room's stream and push what is ready."""
        if not self._mixer:
            self._mixer = _ParticipantAudioMixer(
                sample_rate=audio_frame.sample_rate, num_channels=audio_frame.num_channels
            )
        for chunk in self._mixer.add(participant_id, audio_frame.data.tobytes()):
            audio = await self._mixed_resampler.resample(
                chunk, audio_frame.sample_rate, self.sample_rate
            )
            if not audio:
                continue
            await self.push_audio_frame(
                InputAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=audio_frame.num_channels,
                )
            )

    async def _video_in_task_handler(self):
        """Handle incoming video frames from participants."""
        logger.info("Video input task started")
        video_iterator = self._client.get_next_video_frame()
        async for video_data in video_iterator:
            if video_data:
                video_frame_event, participant_id, video_source = video_data
                self._capture_configured_video(participant_id, video_source)
                due, request = self._video_samplers.sample(participant_id, video_source)
                if not due:
                    continue

                pipecat_video_frame = await self._convert_livekit_video_to_pipecat(
                    video_frame_event=video_frame_event
                )

                # Skip frames with no video data
                if len(pipecat_video_frame.image) == 0:
                    continue

                input_video_frame = UserImageRawFrame(
                    user_id=participant_id,
                    image=pipecat_video_frame.image,
                    size=pipecat_video_frame.size,
                    format=pipecat_video_frame.format,
                    text=request.text if request else None,
                    append_to_context=request.append_to_context if request else None,
                    request=request,
                )
                input_video_frame.transport_source = video_source
                await self.push_video_frame(input_video_frame)

    async def _convert_livekit_audio_to_pipecat(
        self, audio_frame_event: rtc.AudioFrameEvent, participant_id: str
    ) -> AudioRawFrame:
        """Convert a participant's LiveKit audio frame to a Pipecat audio frame."""
        audio_frame = audio_frame_event.frame

        if participant_id not in self._resamplers:
            self._resamplers[participant_id] = create_stream_resampler()
        audio_data = await self._resamplers[participant_id].resample(
            audio_frame.data.tobytes(), audio_frame.sample_rate, self.sample_rate
        )

        return AudioRawFrame(
            audio=audio_data,
            sample_rate=self.sample_rate,
            num_channels=audio_frame.num_channels,
        )

    async def _convert_livekit_video_to_pipecat(
        self,
        video_frame_event: rtc.VideoFrameEvent,
    ) -> ImageRawFrame:
        """Convert LiveKit video frame to Pipecat video frame."""
        rgb_frame = video_frame_event.frame.convert(proto_video_frame.VideoBufferType.RGB24)
        image_frame = ImageRawFrame(
            image=bytes(rgb_frame.data),
            size=(rgb_frame.width, rgb_frame.height),
            format="RGB",
        )
        return image_frame


class LiveKitOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events to LiveKit rooms.

    Manages sending audio and video frames and data messages to LiveKit room
    participants, including audio/video format conversion for LiveKit
    compatibility. Video output publishes to a single default camera track
    when ``LiveKitParams.video_out_enabled`` is set.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: LiveKitTransportClient,
        params: LiveKitParams,
        **kwargs,
    ):
        """Initialize the LiveKit output transport.

        Args:
            transport: The parent transport instance.
            client: LiveKitTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        # Formats already reported as unsupported, so the error is logged once
        # per format instead of once per frame.
        self._unsupported_video_formats: set[str | None] = set()

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the output transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)

        await self._client.setup(setup)

        await self._client.connect()

        logger.info("LiveKitOutputTransport connected")

    async def cleanup(self):
        """Release output transport resources at teardown."""
        await super().cleanup()
        await self._client.disconnect()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and disconnect from LiveKit room.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.disconnect()
        logger.info("LiveKitOutputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and disconnect from LiveKit room.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, clearing the LiveKit AudioSource buffer on interruption.

        When an InterruptionFrame arrives, any audio already submitted to the
        LiveKit AudioSource (but not yet played out) is cleared immediately so
        the bot stops speaking without delay.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame) and self._client._audio_source is not None:
            self._client._audio_source.clear_queue()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message to participants.

        Args:
            frame: The transport message frame to send.
        """
        message = frame.message
        if isinstance(message, dict):
            # fix message encoding for dict-like messages, e.g. RTVI messages.
            message = json.dumps(message, ensure_ascii=False)
        if isinstance(
            frame, (LiveKitOutputTransportMessageFrame, LiveKitOutputTransportMessageUrgentFrame)
        ):
            await self._client.send_data(message.encode(), frame.participant_id)
        else:
            await self._client.send_data(message.encode())

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the LiveKit room.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        livekit_audio = self._convert_pipecat_audio_to_livekit(frame.audio)
        return await self._client.publish_audio(livekit_audio)

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the LiveKit room's published camera track.

        Publishes to the single default video track set up in
        ``LiveKitTransportClient.connect`` (mirroring how audio always
        publishes to one microphone track). Per-destination routing to
        multiple named video tracks is not supported yet, so
        ``frame.transport_destination`` is ignored.

        Args:
            frame: The video frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        livekit_video = self._convert_pipecat_video_to_livekit(frame)
        if livekit_video is None:
            return False
        return await self._client.publish_video(livekit_video)

    def _supports_native_dtmf(self) -> bool:
        """LiveKit supports native DTMF via telephone events.

        Returns:
            True, as LiveKit supports native DTMF transmission.
        """
        return True

    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Use LiveKit's native publish_dtmf method for telephone events.

        LiveKit's DTMF API sends a single tone per call, so when
        ``frame.buttons`` contains multiple entries only the first one is
        sent.

        Args:
            frame: The DTMF frame to write.
        """
        if not frame.buttons:
            return
        await self._client.send_dtmf(frame.buttons[0].value)

    def _convert_pipecat_audio_to_livekit(self, pipecat_audio: bytes) -> rtc.AudioFrame:
        """Convert Pipecat audio data to LiveKit audio frame."""
        bytes_per_sample = 2  # Assuming 16-bit audio
        total_samples = len(pipecat_audio) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        return rtc.AudioFrame(
            data=pipecat_audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
            samples_per_channel=samples_per_channel,
        )

    def _convert_pipecat_video_to_livekit(
        self, frame: OutputImageRawFrame
    ) -> rtc.VideoFrame | None:
        """Convert a Pipecat output video frame to a LiveKit video frame.

        Returns:
            The converted ``rtc.VideoFrame``, or None if ``frame.format`` has
            no known LiveKit ``VideoBufferType`` mapping or the image length
            does not match its size.
        """
        buffer_info = LIVEKIT_VIDEO_BUFFER_TYPES.get(frame.format) if frame.format else None
        if buffer_info is None:
            if frame.format not in self._unsupported_video_formats:
                self._unsupported_video_formats.add(frame.format)
                logger.error(
                    f"{self} unsupported video color format for LiveKit output: {frame.format!r}"
                )
            return None

        buffer_type, bytes_per_pixel = buffer_info
        width, height = frame.size
        # LiveKit reads ``width * height * bytes_per_pixel`` bytes from the
        # buffer without checking its length, so a short buffer crashes the
        # process.
        expected_length = width * height * bytes_per_pixel
        if len(frame.image) != expected_length:
            logger.error(
                f"{self} video frame of size {width}x{height} and format {frame.format!r} "
                f"has {len(frame.image)} bytes, expected {expected_length}"
            )
            return None

        return rtc.VideoFrame(width, height, buffer_type, frame.image)


class LiveKitTransport(BaseTransport):
    """Transport implementation for LiveKit real-time communication.

    Provides comprehensive LiveKit integration including audio streaming, data
    messaging, participant management, and room event handling for conversational
    AI applications.

    Every ``participant_id`` surfaced by this transport (event args, frame
    fields, ``get_participants()``, and the ``get_participant_metadata``/
    ``mute_participant``/``unmute_participant`` methods) is the participant's
    LiveKit *identity* (``rtc.Participant.identity``) — the value set when
    minting its access token, and what LiveKit itself keys
    ``room.remote_participants`` by and expects in ``destination_identities``.
    It is not the participant's *SID* (``rtc.Participant.sid``), a
    per-connection session id that changes on every reconnect.

    Event handlers available:

    - on_connected: Called when the bot connects to the room.
    - on_disconnected: Called when the bot disconnects from the room.
    - on_before_disconnect: [sync] Called just before the bot disconnects.
    - on_call_state_updated: Called when the call state changes. Args: (state: str)
    - on_first_participant_joined: Called when the first participant joins.
      Args: (participant_id: str)
    - on_participant_connected: Called when a participant connects.
      Args: (participant_id: str)
    - on_participant_disconnected: Called when a participant disconnects.
      Args: (participant_id: str)
    - on_participant_left: Called when a participant leaves.
      Args: (participant_id: str, reason: str)
    - on_client_connected: Called when a participant connects (alias for
      on_participant_connected). Args: (participant: dict)
    - on_client_disconnected: Called when a participant disconnects (alias for
      on_participant_disconnected). Args: (participant: dict)
    - on_audio_track_subscribed: Called when an audio track is subscribed.
      Args: (participant_id: str)
    - on_audio_track_unsubscribed: Called when an audio track is unsubscribed.
      Args: (participant_id: str)
    - on_video_track_subscribed: Called when a video track is subscribed.
      Args: (participant_id: str)
    - on_video_track_unsubscribed: Called when a video track is unsubscribed.
      Args: (participant_id: str)
    - on_app_message: Called when data is received from a participant. RTVI-compatible version of on_data_received.
      Args: (message: Any, sender: str)
    - on_data_received: Called when data is received. The participant ID is None
      for packets sent by a server SDK, which LiveKit delivers unattributed.
      Args: (data: bytes, participant_id: str | None)
    - on_dtmf_event: Called when a SIP DTMF tone is received from a participant.
      Args: (data: dict) with keys ``tone``/``digit``, ``code``, and
      ``participant_id``. Also pushes an ``InputDTMFFrame`` so
      ``DTMFAggregator`` works on LiveKit SIP calls.
    - on_active_speaker_changed: Called when the room's loudest speaker changes, as on
      Daily. The bot can be the speaker while it talks.
      Args: (participant: dict) with key ``id``, the speaker's identity.

    Example::

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant_id):
            await task.queue_frame(TTSSpeakFrame("Hello!"))

        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant_id):
            await task.queue_frame(EndFrame())
    """

    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: LiveKitParams | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the LiveKit transport.

        Args:
            url: LiveKit server URL to connect to.
            token: Authentication token for the room.
            room_name: Name of the LiveKit room to join.
            params: Configuration parameters for the transport.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        callbacks = LiveKitCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_before_disconnect=self._on_before_disconnect,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_video_track_subscribed=self._on_video_track_subscribed,
            on_video_track_unsubscribed=self._on_video_track_unsubscribed,
            on_video_track_muted=self._on_video_track_muted,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
            on_dtmf_event=self._on_dtmf_event,
            on_active_speaker_changed=self._on_active_speaker_changed,
        )
        self._params = params or LiveKitParams()

        self._client = LiveKitTransportClient(
            url, token, room_name, self._params, callbacks, self.name
        )
        self._input: LiveKitInputTransport | None = None
        self._output: LiveKitOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_video_track_subscribed")
        self._register_event_handler("on_video_track_unsubscribed")
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_before_disconnect", sync=True)
        self._register_event_handler("on_dtmf_event")
        self._register_event_handler("on_active_speaker_changed")

    def get_client_id(self, client: Any) -> str:
        """The id of a client, as passed to ``on_client_connected``.

        Args:
            client: The client, as passed to the transport's client events.

        Returns:
            The participant's identity.
        """
        return client["id"]

    def input(self) -> LiveKitInputTransport:
        """Get the input transport for receiving media and events.

        Returns:
            The LiveKit input transport instance.
        """
        if not self._input:
            self._input = LiveKitInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> LiveKitOutputTransport:
        """Get the output transport for sending media and events.

        Returns:
            The LiveKit output transport instance.
        """
        if not self._output:
            self._output = LiveKitOutputTransport(
                self, self._client, self._params, name=self._output_name
            )
        return self._output

    @property
    def participant_id(self) -> str:
        """Get the participant ID for this transport.

        Returns:
            The participant ID assigned by LiveKit.
        """
        return self._client.participant_id

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame to the LiveKit room.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> list[str]:
        """Get list of participant IDs in the room.

        Returns:
            List of participant LiveKit identities.
        """
        return self._client.get_participants()

    async def get_participant_metadata(self, participant_id: str) -> dict:
        """Get metadata for a specific participant.

        Args:
            participant_id: LiveKit identity of the participant to get metadata for.

        Returns:
            Dictionary containing participant metadata.
        """
        return await self._client.get_participant_metadata(participant_id)

    async def set_metadata(self, metadata: str):
        """Set metadata for the local participant.

        Args:
            metadata: Metadata string to set.
        """
        await self._client.set_participant_metadata(metadata)

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int | None = 30,
        video_source: str = CAM_VIDEO_SOURCE,
        *,
        on_request_only: bool = False,
    ):
        """Capture a participant's video source at a framerate.

        This takes precedence over ``video_in_sources`` for the source.

        Args:
            participant_id: The participant's identity.
            framerate: Frames per second to pass on, or ``None`` for every frame. It
                doesn't apply with ``on_request_only``.
            video_source: The video source, ``"camera"`` or ``"screenVideo"``.
            on_request_only: Pass on only the frames that answer image requests.
        """
        framerate = _capture_framerate(
            framerate, on_request_only, "LiveKitTransport.capture_participant_video"
        )
        if self._input:
            await self._input.capture_participant_video(
                participant_id, framerate, video_source, on_request_only=framerate == 0
            )

    async def mute_participant(self, participant_id: str):
        """Stop receiving a specific participant's audio.

        Args:
            participant_id: LiveKit identity of the participant to stop
                receiving audio from.
        """
        await self._client.mute_participant(participant_id)

    async def unmute_participant(self, participant_id: str):
        """Resume receiving a specific participant's audio.

        Args:
            participant_id: LiveKit identity of the participant to resume
                receiving audio from.
        """
        await self._client.unmute_participant(participant_id)

    async def _on_connected(self):
        """Handle room connected events."""
        await self._call_event_handler("on_connected")
        if self._input:
            await self._input.push_frame(BotConnectedFrame())

    async def _on_disconnected(self):
        """Handle room disconnected events."""
        if self._input:
            await self._input.remove_all_video()
        await self._call_event_handler("on_disconnected")

    async def _on_before_disconnect(self):
        """Handle before disconnection room events."""
        await self._call_event_handler("on_before_disconnect")

    async def _on_participant_connected(self, participant_id: str):
        """Handle participant connected events."""
        await self._call_event_handler("on_participant_connected", participant_id)
        # Also call on_client_connected for compatibility with other transports.
        # Wrapped as a dict (matching Daily's Mapping[str, Any] shape) so
        # drop-in bot templates reading client["id"] work across transports.
        await self._call_event_handler("on_client_connected", {"id": participant_id})
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_participant_disconnected(self, participant_id: str):
        """Handle participant disconnected events."""
        if self._input:
            await self._input.remove_participant_video(participant_id)
        await self._call_event_handler("on_participant_disconnected", participant_id)
        await self._call_event_handler("on_participant_left", participant_id, "disconnected")
        # Also call on_client_disconnected for compatibility with other transports
        await self._call_event_handler("on_client_disconnected", {"id": participant_id})

    async def _on_audio_track_subscribed(self, participant_id: str):
        """Handle audio track subscribed events."""
        await self._call_event_handler("on_audio_track_subscribed", participant_id)

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        """Handle audio track unsubscribed events."""
        if self._input:
            self._input.release_participant(participant_id)
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_video_track_subscribed(self, participant_id: str):
        """Handle video track subscribed events."""
        await self._call_event_handler("on_video_track_subscribed", participant_id)

    async def _on_video_track_muted(self, participant_id: str, video_source: str):
        """Handle video track muted events."""
        if self._input:
            await self._input.stop_participant_video(participant_id, video_source)

    async def _on_video_track_unsubscribed(self, participant_id: str, video_source: str):
        """Handle video track unsubscribed events."""
        if self._input:
            await self._input.stop_participant_video(participant_id, video_source)
        await self._call_event_handler("on_video_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str | None):
        """Handle data received events."""
        try:
            message = json.loads(data.decode())
            if not isinstance(message, dict):
                logger.debug(f"{self} Ignoring non-object JSON data: {message!r}")
                message = None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.debug(f"{self} Ignoring non-JSON data from {participant_id}: {e}")
            message = None

        if message is not None:
            if self._input:
                await self._input.push_app_message(message, participant_id)
            # RTVI compatibility:
            await self._call_event_handler("on_app_message", message, participant_id)
        # Backwards compatibility with older transports that used on_data_received for app messages
        await self._call_event_handler("on_data_received", data, participant_id)

    async def _on_active_speaker_changed(self, participant_id: str):
        """Handle a change of the room's loudest speaker.

        Wrapped as a dict, matching Daily's event and LiveKit's on_client_connected.
        """
        await self._call_event_handler("on_active_speaker_changed", {"id": participant_id})

    async def _on_dtmf_event(self, data: Any):
        """Handle inbound SIP DTMF events.

        Mirrors Daily transport behavior: expose ``on_dtmf_event`` to user code
        and push ``InputDTMFFrame`` so ``DTMFAggregator`` can consume digits.
        """
        logger.debug(f"{self} DTMF event: {data}")
        await self._call_event_handler("on_dtmf_event", data)

        tone = data.get("tone") if isinstance(data, dict) else None
        if tone is None or self._input is None:
            return

        try:
            button = KeypadEntry(tone)
        except ValueError:
            logger.warning(f"{self} Ignoring unsupported DTMF tone: {tone!r}")
            return

        await self._input.push_frame(InputDTMFFrame(button=button))

    async def send_message(self, message: str, participant_id: str | None = None):
        """Send a message to participants in the room.

        Args:
            message: The message string to send.
            participant_id: Optional specific participant to send to.
        """
        if self._output:
            frame = LiveKitOutputTransportMessageFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def send_message_urgent(self, message: str, participant_id: str | None = None):
        """Send an urgent message to participants in the room.

        Args:
            message: The urgent message string to send.
            participant_id: Optional specific participant to send to.
        """
        if self._output:
            frame = LiveKitOutputTransportMessageUrgentFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def on_room_event(self, event):
        """Handle room events.

        Args:
            event: The room event to handle.
        """
        # Handle room events
        pass

    async def on_participant_event(self, event):
        """Handle participant events.

        Args:
            event: The participant event to handle.
        """
        # Handle participant events
        pass

    async def on_track_event(self, event):
        """Handle track events.

        Args:
            event: The track event to handle.
        """
        # Handle track events
        pass

    async def _on_call_state_updated(self, state: str):
        """Handle call state update events."""
        await self._call_event_handler("on_call_state_updated", state)

    async def _on_first_participant_joined(self, participant_id: str):
        """Handle first participant joined events."""
        await self._call_event_handler("on_first_participant_joined", participant_id)
