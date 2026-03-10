#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TeleCMI transport implementation for Pipecat.

This transport provides comprehensive TeleCMI integration for voice and video.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    BotConnectedFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    ImageRawFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from livekit import rtc
    from livekit.rtc._proto import video_frame_pb2 as proto_video_frame
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logger.error(
        "In order to use TeleCMI transport, you need to `pip install pipecat-ai[telecmi]`."
    )
    raise Exception(f"Missing module: {e}")

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


@dataclass
class TelecmiOutputTransportMessageFrame(OutputTransportMessageFrame):
    """Frame for transport messages in TeleCMI rooms.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


@dataclass
class TelecmiOutputTransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in TeleCMI rooms.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


class TelecmiParams(TransportParams):
    """Configuration parameters for TeleCMI transport."""

    pass


class TelecmiCallbacks(BaseModel):
    """Callback handlers for TeleCMI events."""

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_before_disconnect: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_video_track_subscribed: Callable[[str], Awaitable[None]]
    on_video_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class TelecmiTransportClient:
    """Core client for interacting with TeleCMI rooms.

    This client manage the connection to a TeleCMI room and handles all
    synchronous and asynchronous events.
    """

    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: TelecmiParams,
        callbacks: TelecmiCallbacks,
        transport_name: str,
    ):
        """Initialize the TeleCMI transport client.

        Args:
            url: TeleCMI server URL.
            token: Authentication token.
            room_name: Name of the room.
            params: Transport parameters.
            callbacks: Callback handlers.
            transport_name: Name of the transport.
        """
        self._url = url
        self._token = token
        self._room_name = room_name
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name
        self._room: Optional[rtc.Room] = None
        self._participant_id: str = ""
        self._connected = False
        self._disconnect_counter = 0
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._audio_tracks = {}
        self._audio_queue = asyncio.Queue()
        self._video_tracks = {}
        self._video_queue = asyncio.Queue()
        self._other_participant_has_joined = False
        self._task_manager: Optional[BaseTaskManager] = None
        self._async_lock = asyncio.Lock()

    @property
    def participant_id(self) -> str:
        """Returns the current participant ID.

        Returns:
            The participant ID as a string.
        """
        return self._participant_id

    @property
    def room(self) -> rtc.Room:
        """Returns the LiveKit room object.

        Returns:
            The LiveKit room object.

        Raises:
            Exception: If the room object is not yet available.
        """
        if not self._room:
            raise Exception(f"{self}: missing room object (pipeline not started?)")
        return self._room

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client using the provided setup configuration.

        Args:
            setup: The frame processor setup.
        """
        if self._task_manager:
            return

        self._task_manager = setup.task_manager
        self._room = rtc.Room(loop=self._task_manager.get_event_loop())

        # Set up room event handlers
        self.room.on("participant_connected")(self._on_participant_connected_wrapper)
        self.room.on("participant_disconnected")(self._on_participant_disconnected_wrapper)
        self.room.on("track_subscribed")(self._on_track_subscribed_wrapper)
        self.room.on("track_unsubscribed")(self._on_track_unsubscribed_wrapper)
        self.room.on("data_received")(self._on_data_received_wrapper)
        self.room.on("connected")(self._on_connected_wrapper)
        self.room.on("disconnected")(self._on_disconnected_wrapper)

    async def cleanup(self):
        """Clean up resources and disconnect."""
        await self.disconnect()

    async def start(self, frame: StartFrame):
        """Start the client based on the StartFrame.

        Args:
            frame: The start frame containing stream details.
        """
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def connect(self):
        """Connect to the TeleCMI room."""
        async with self._async_lock:
            if self._connected:
                self._disconnect_counter += 1
                return

            logger.info(f"Connecting to {self._room_name}")

            try:
                await self.room.connect(
                    self._url,
                    self._token,
                    options=rtc.RoomOptions(auto_subscribe=True),
                )
                self._connected = True
                self._disconnect_counter += 1

                self._participant_id = self.room.local_participant.sid
                logger.info(f"Connected to {self._room_name}")

                # Set up audio source and track
                self._audio_source = rtc.AudioSource(
                    self._out_sample_rate, self._params.audio_out_channels
                )
                self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                    "pipecat-audio", self._audio_source
                )
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_MICROPHONE
                await self.room.local_participant.publish_track(self._audio_track, options)

                await self._callbacks.on_connected()

                participants = self.get_participants()
                if participants and not self._other_participant_has_joined:
                    self._other_participant_has_joined = True
                    await self._callbacks.on_first_participant_joined(participants[0])
            except Exception as e:
                logger.error(f"Error connecting to {self._room_name}: {e}")
                raise

    async def disconnect(self):
        """Disconnect from the TeleCMI room."""
        async with self._async_lock:
            self._disconnect_counter -= 1

            if not self._connected or self._disconnect_counter > 0:
                return

            logger.info(f"Disconnecting from {self._room_name}")
            await self._callbacks.on_before_disconnect()
            await self.room.disconnect()
            self._connected = False
            logger.info(f"Disconnected from {self._room_name}")
            await self._callbacks.on_disconnected()

    async def send_data(self, data: bytes, participant_id: Optional[str] = None):
        """Send data to the room or a specific participant.

        Args:
            data: The data to send.
            participant_id: Optional ID of the participant to send data to.
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
        """Send a DTMF tone to the room.

        Args:
            digit: The DTMF digit to send (0-9, *, #).
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
            audio_frame: The audio frame to publish.

        Returns:
            True if successful, False otherwise.
        """
        if not self._connected or not self._audio_source:
            return False

        try:
            await self._audio_source.capture_frame(audio_frame)
            return True
        except Exception as e:
            logger.error(f"Error publishing audio: {e}")
            return False

    def get_participants(self) -> List[str]:
        """Get a list of participant IDs in the room.

        Returns:
            A list of participant IDs.
        """
        return [p.sid for p in self.room.remote_participants.values()]

    async def get_participant_metadata(self, participant_id: str) -> dict:
        """Get metadata for a specific participant.

        Args:
            participant_id: The ID of the participant.

        Returns:
            A dictionary containing participant metadata.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            return {
                "id": participant.sid,
                "name": participant.name,
                "metadata": participant.metadata,
                "is_speaking": participant.is_speaking,
            }
        return {}

    async def set_participant_metadata(self, metadata: str):
        """Set metadata for the local participant.

        Args:
            metadata: The metadata to set.
        """
        await self.room.local_participant.set_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        """Mute a specific participant.

        Args:
            participant_id: The ID of the participant to mute.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(False)

    async def unmute_participant(self, participant_id: str):
        """Unmute a specific participant.

        Args:
            participant_id: The ID of the participant to unmute.
        """
        participant = self.room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(True)

    def _on_participant_connected_wrapper(self, participant: rtc.RemoteParticipant):
        self._task_manager.create_task(
            self._async_on_participant_connected(participant),
            f"{self}::_async_on_participant_connected",
        )

    def _on_participant_disconnected_wrapper(self, participant: rtc.RemoteParticipant):
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
        self._task_manager.create_task(
            self._async_on_track_unsubscribed(track, publication, participant),
            f"{self}::_async_on_track_unsubscribed",
        )

    def _on_data_received_wrapper(self, data: rtc.DataPacket):
        self._task_manager.create_task(
            self._async_on_data_received(data),
            f"{self}::_async_on_data_received",
        )

    def _on_connected_wrapper(self):
        self._task_manager.create_task(self._async_on_connected(), f"{self}::_async_on_connected")

    def _on_disconnected_wrapper(self):
        self._task_manager.create_task(
            self._async_on_disconnected(), f"{self}::_async_on_disconnected"
        )

    async def _async_on_participant_connected(self, participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")
        await self._callbacks.on_participant_connected(participant.sid)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(participant.sid)

    async def _async_on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        await self._callbacks.on_participant_disconnected(participant.sid)
        if len(self.get_participants()) == 0:
            self._other_participant_has_joined = False

    async def _async_on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed: {track.sid} from participant {participant.sid}")
            self._audio_tracks[participant.sid] = track
            audio_stream = rtc.AudioStream(track)
            self._task_manager.create_task(
                self._process_audio_stream(audio_stream, participant.sid),
                f"{self}::_process_audio_stream",
            )
            await self._callbacks.on_audio_track_subscribed(participant.sid)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"Video track subscribed: {track.sid} from participant {participant.sid}")
            self._video_tracks[participant.sid] = track
            if self._params.video_in_enabled:
                video_stream = rtc.VideoStream(track)
                self._task_manager.create_task(
                    self._process_video_stream(video_stream, participant.sid),
                    f"{self}::_process_video_stream",
                )
            await self._callbacks.on_video_track_subscribed(participant.sid)

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            await self._callbacks.on_audio_track_unsubscribed(participant.sid)
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            await self._callbacks.on_video_track_unsubscribed(participant.sid)

    async def _async_on_data_received(self, data: rtc.DataPacket):
        await self._callbacks.on_data_received(data.data, data.participant.sid)

    async def _async_on_connected(self):
        await self._callbacks.on_connected()

    async def _async_on_disconnected(self, reason=None):
        self._connected = False
        logger.info(f"Disconnected from {self._room_name}. Reason: {reason}")
        await self._callbacks.on_disconnected()

    async def _process_audio_stream(self, audio_stream: rtc.AudioStream, participant_id: str):
        logger.info(f"Started processing audio stream for participant {participant_id}")
        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                await self._audio_queue.put((event, participant_id))
            else:
                logger.warning(f"Received unexpected event type: {type(event)}")

    async def get_next_audio_frame(self):
        """Asynchronous generator for audio frames.

        Yields:
            A tuple containing the audio frame and the participant ID.
        """
        while True:
            frame, participant_id = await self._audio_queue.get()
            yield frame, participant_id

    async def _process_video_stream(self, video_stream: rtc.VideoStream, participant_id: str):
        logger.info(f"Started processing video stream for participant {participant_id}")
        async for event in video_stream:
            if isinstance(event, rtc.VideoFrameEvent):
                await self._video_queue.put((event, participant_id))
            else:
                logger.warning(f"Received unexpected event type: {type(event)}")

    async def get_next_video_frame(self):
        """Asynchronous generator for video frames.

        Yields:
            A tuple containing the video frame and the participant ID.
        """
        while True:
            frame, participant_id = await self._video_queue.get()
            yield frame, participant_id

    def __str__(self):
        return f"{self._transport_name}::TelecmiTransport"


class TelecmiInputTransport(BaseInputTransport):
    """Handles incoming media streams and events for TeleCMI.

    This transport receives audio and video frames from TeleCMI and pushes
    them to the Pipecat pipeline.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: TelecmiTransportClient,
        params: TelecmiParams,
        **kwargs,
    ):
        """Initialize the TeleCMI input transport.

        Args:
            transport: The parent transport.
            client: The TeleCMI transport client.
            params: Transport parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client

        self._audio_in_task = None
        self._video_in_task = None
        self._vad_analyzer: Optional[VADAnalyzer] = params.vad_analyzer
        self._resampler = create_stream_resampler()

        self._initialized = False

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        """Get the VAD analyzer used by this transport.

        Returns:
            The VAD analyzer instance or None.
        """
        return self._vad_analyzer

    async def start(self, frame: StartFrame):
        """Start the input transport.

        Args:
            frame: The start frame.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()
        if not self._audio_in_task and self._params.audio_in_enabled:
            self._audio_in_task = self.create_task(self._audio_in_task_handler())
        if not self._video_in_task and self._params.video_in_enabled:
            self._video_in_task = self.create_task(self._video_in_task_handler())
        await self.set_transport_ready(frame)
        logger.info("TelecmiInputTransport started")

    async def stop(self, frame: EndFrame):
        """Stop the input transport.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._client.disconnect()
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task:
            await self.cancel_task(self._video_in_task)
        logger.info("TelecmiInputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._client.disconnect()
        if self._audio_in_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task and self._params.video_in_enabled:
            await self.cancel_task(self._video_in_task)

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the input transport.

        Args:
            setup: The frame processor setup.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def push_app_message(self, message: Any, sender: str):
        """Push an application message as a frame.

        Args:
            message: The message to push.
            sender: The ID of the sender.
        """
        frame = TelecmiOutputTransportMessageUrgentFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    async def _audio_in_task_handler(self):
        logger.info("Audio input task started")
        audio_iterator = self._client.get_next_audio_frame()
        async for audio_data in audio_iterator:
            if audio_data:
                audio_frame_event, participant_id = audio_data
                pipecat_audio_frame = await self._convert_telecmi_audio_to_pipecat(
                    audio_frame_event
                )

                if len(pipecat_audio_frame.audio) == 0:
                    continue

                input_audio_frame = UserAudioRawFrame(
                    user_id=participant_id,
                    audio=pipecat_audio_frame.audio,
                    sample_rate=pipecat_audio_frame.sample_rate,
                    num_channels=pipecat_audio_frame.num_channels,
                )
                await self.push_audio_frame(input_audio_frame)

    async def _video_in_task_handler(self):
        logger.info("Video input task started")
        video_iterator = self._client.get_next_video_frame()
        async for video_data in video_iterator:
            if video_data:
                video_frame_event, participant_id = video_data
                pipecat_video_frame = await self._convert_telecmi_video_to_pipecat(
                    video_frame_event=video_frame_event
                )

                if len(pipecat_video_frame.image) == 0:
                    continue

                input_video_frame = UserImageRawFrame(
                    user_id=participant_id,
                    image=pipecat_video_frame.image,
                    size=pipecat_video_frame.size,
                    format=pipecat_video_frame.format,
                )
                await self.push_video_frame(input_video_frame)

    async def _convert_telecmi_audio_to_pipecat(
        self, audio_frame_event: rtc.AudioFrameEvent
    ) -> AudioRawFrame:
        audio_frame = audio_frame_event.frame

        audio_data = await self._resampler.resample(
            audio_frame.data.tobytes(), audio_frame.sample_rate, self.sample_rate
        )

        return AudioRawFrame(
            audio=audio_data,
            sample_rate=self.sample_rate,
            num_channels=audio_frame.num_channels,
        )

    async def _convert_telecmi_video_to_pipecat(
        self,
        video_frame_event: rtc.VideoFrameEvent,
    ) -> ImageRawFrame:
        rgb_frame = video_frame_event.frame.convert(proto_video_frame.VideoBufferType.RGB24)
        image_frame = ImageRawFrame(
            image=rgb_frame.data,
            size=(rgb_frame.width, rgb_frame.height),
            format="RGB",
        )
        return image_frame


class TelecmiOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events for TeleCMI.

    This transport sends audio frames and DTMF tones to the TeleCMI room.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: TelecmiTransportClient,
        params: TelecmiParams,
        **kwargs,
    ):
        """Initialize the TeleCMI output transport.

        Args:
            transport: The parent transport.
            client: The TeleCMI transport client.
            params: Transport parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The start frame.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()
        await self.set_transport_ready(frame)
        logger.info("TelecmiOutputTransport started")

    async def stop(self, frame: EndFrame):
        """Stop the output transport.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._client.disconnect()
        logger.info("TelecmiOutputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the output transport.

        Args:
            setup: The frame processor setup.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a data message to the room.

        Args:
            frame: The message frame to send.
        """
        message = frame.message
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if isinstance(
            frame, (TelecmiOutputTransportMessageFrame, TelecmiOutputTransportMessageUrgentFrame)
        ):
            await self._client.send_data(message.encode(), frame.participant_id)
        else:
            await self._client.send_data(message.encode())

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the room.

        Args:
            frame: The audio frame to write.

        Returns:
            True if successful, False otherwise.
        """
        telecmi_audio = self._convert_pipecat_audio_to_telecmi(frame.audio)
        return await self._client.publish_audio(telecmi_audio)

    def _supports_native_dtmf(self) -> bool:
        return True

    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        await self._client.send_dtmf(frame.button.value)

    def _convert_pipecat_audio_to_telecmi(self, pipecat_audio: bytes) -> rtc.AudioFrame:
        bytes_per_sample = 2
        total_samples = len(pipecat_audio) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        return rtc.AudioFrame(
            data=pipecat_audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
            samples_per_channel=samples_per_channel,
        )


class TelecmiTransport(BaseTransport):
    """Transport implementation for TeleCMI.

    Provides comprehensive TeleCMI integration including audio streaming, data
    messaging, and room event handling.
    """

    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: Optional[TelecmiParams] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the TeleCMI transport.

        Args:
            url: TeleCMI server URL to connect to.
            token: Authentication token for the room.
            room_name: Name of the TeleCMI room to join.
            params: Configuration parameters for the transport.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        callbacks = TelecmiCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_before_disconnect=self._on_before_disconnect,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_video_track_subscribed=self._on_video_track_subscribed,
            on_video_track_unsubscribed=self._on_video_track_unsubscribed,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
        )
        self._params = params or TelecmiParams()

        self._client = TelecmiTransportClient(
            url, token, room_name, self._params, callbacks, self.name
        )
        self._input: Optional[TelecmiInputTransport] = None
        self._output: Optional[TelecmiOutputTransport] = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_video_track_subscribed")
        self._register_event_handler("on_video_track_unsubscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_before_disconnect", sync=True)

    def input(self) -> TelecmiInputTransport:
        """Get the input transport.

        Returns:
            The TelecmiInputTransport instance.
        """
        if not self._input:
            self._input = TelecmiInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> TelecmiOutputTransport:
        """Get the output transport.

        Returns:
            The TelecmiOutputTransport instance.
        """
        if not self._output:
            self._output = TelecmiOutputTransport(
                self, self._client, self._params, name=self._output_name
            )
        return self._output

    @property
    def participant_id(self) -> str:
        """Get the ID of the local participant.

        Returns:
            The local participant ID.
        """
        return self._client.participant_id

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Queue an audio frame for sending.

        Args:
            frame: The audio raw frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> List[str]:
        """Get a list of remote participant IDs.

        Returns:
            A list of participant IDs.
        """
        return self._client.get_participants()

    async def get_participant_metadata(self, participant_id: str) -> dict:
        """Get metadata for a specific participant.

        Args:
            participant_id: ID of the participant.

        Returns:
            A dictionary containing participant metadata.
        """
        return await self._client.get_participant_metadata(participant_id)

    async def set_metadata(self, metadata: str):
        """Set metadata for the local participant.

        Args:
            metadata: The metadata string to set.
        """
        await self._client.set_participant_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        """Mute a specific remote participant.

        Args:
            participant_id: ID of the participant to mute.
        """
        await self._client.mute_participant(participant_id)

    async def unmute_participant(self, participant_id: str):
        """Unmute a specific remote participant.

        Args:
            participant_id: ID of the participant to unmute.
        """
        await self._client.unmute_participant(participant_id)

    async def _on_connected(self):
        await self._call_event_handler("on_connected")
        if self._input:
            await self._input.push_frame(BotConnectedFrame())

    async def _on_disconnected(self):
        await self._call_event_handler("on_disconnected")

    async def _on_before_disconnect(self):
        await self._call_event_handler("on_before_disconnect")

    async def _on_participant_connected(self, participant_id: str):
        await self._call_event_handler("on_participant_connected", participant_id)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_participant_disconnected(self, participant_id: str):
        await self._call_event_handler("on_participant_disconnected", participant_id)
        await self._call_event_handler("on_participant_left", participant_id, "disconnected")

    async def _on_audio_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_subscribed", participant_id)
        participant = self._client.room.remote_participants.get(participant_id)
        if participant:
            for publication in participant.audio_tracks.values():
                self._client._on_track_subscribed_wrapper(
                    publication.track, publication, participant
                )

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_video_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_video_track_subscribed", participant_id)
        participant = self._client.room.remote_participants.get(participant_id)
        if participant:
            for publication in participant.video_tracks.values():
                self._client._on_track_subscribed_wrapper(
                    publication.track, publication, participant
                )

    async def _on_video_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_video_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str):
        if self._input:
            await self._input.push_app_message(data.decode(), participant_id)
        await self._call_event_handler("on_data_received", data, participant_id)

    async def send_message(self, message: str, participant_id: Optional[str] = None):
        """Sends a data message to one or all participants.

        Args:
            message: The message to send.
            participant_id: The ID of the participant to send the message to.
                If None, the message is sent to all participants.
        """
        if self._output:
            frame = TelecmiOutputTransportMessageFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def send_message_urgent(self, message: str, participant_id: Optional[str] = None):
        """Sends an urgent data message to one or all participants.

        Args:
            message: The message to send.
            participant_id: The ID of the participant to send the message to.
                If None, the message is sent to all participants.
        """
        if self._output:
            frame = TelecmiOutputTransportMessageUrgentFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", state)

    async def _on_first_participant_joined(self, participant_id: str):
        await self._call_event_handler("on_first_participant_joined", participant_id)
