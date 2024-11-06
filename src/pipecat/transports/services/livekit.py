#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List

from pydantic import BaseModel

from pipecat.audio.utils import resample_audio
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    from livekit import rtc
    from tenacity import retry, stop_after_attempt, wait_exponential
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use LiveKit, you need to `pip install pipecat-ai[livekit]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class LiveKitTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class LiveKitTransportMessageUrgentFrame(TransportMessageUrgentFrame):
    participant_id: str | None = None


class LiveKitParams(TransportParams):
    pass


class LiveKitCallbacks(BaseModel):
    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class LiveKitTransportClient:
    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: LiveKitParams,
        callbacks: LiveKitCallbacks,
        loop: asyncio.AbstractEventLoop,
    ):
        self._url = url
        self._token = token
        self._room_name = room_name
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._room = rtc.Room(loop=loop)
        self._participant_id: str = ""
        self._connected = False
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._audio_tracks = {}
        self._audio_queue = asyncio.Queue()
        self._other_participant_has_joined = False

        # Set up room event handlers
        self._room.on("participant_connected")(self._on_participant_connected_wrapper)
        self._room.on("participant_disconnected")(self._on_participant_disconnected_wrapper)
        self._room.on("track_subscribed")(self._on_track_subscribed_wrapper)
        self._room.on("track_unsubscribed")(self._on_track_unsubscribed_wrapper)
        self._room.on("data_received")(self._on_data_received_wrapper)
        self._room.on("connected")(self._on_connected_wrapper)
        self._room.on("disconnected")(self._on_disconnected_wrapper)

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def connect(self):
        if self._connected:
            return

        logger.info(f"Connecting to {self._room_name}")

        try:
            await self._room.connect(
                self._url,
                self._token,
                options=rtc.RoomOptions(auto_subscribe=True),
            )
            self._connected = True
            self._participant_id = self._room.local_participant.sid
            logger.info(f"Connected to {self._room_name}")

            # Set up audio source and track
            self._audio_source = rtc.AudioSource(
                self._params.audio_out_sample_rate, self._params.audio_out_channels
            )
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                "pipecat-audio", self._audio_source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self._room.local_participant.publish_track(self._audio_track, options)

            await self._callbacks.on_connected()

            # Check if there are already participants in the room
            participants = self.get_participants()
            if participants and not self._other_participant_has_joined:
                self._other_participant_has_joined = True
                await self._callbacks.on_first_participant_joined(participants[0])
        except Exception as e:
            logger.error(f"Error connecting to {self._room_name}: {e}")
            raise

    async def disconnect(self):
        if not self._connected:
            return

        logger.info(f"Disconnecting from {self._room_name}")
        await self._room.disconnect()
        self._connected = False
        logger.info(f"Disconnected from {self._room_name}")
        await self._callbacks.on_disconnected()

    async def send_data(self, data: bytes, participant_id: str | None = None):
        if not self._connected:
            return

        try:
            if participant_id:
                await self._room.local_participant.publish_data(
                    data, reliable=True, destination_identities=[participant_id]
                )
            else:
                await self._room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            logger.error(f"Error sending data: {e}")

    async def publish_audio(self, audio_frame: rtc.AudioFrame):
        if not self._connected or not self._audio_source:
            return

        try:
            await self._audio_source.capture_frame(audio_frame)
        except Exception as e:
            logger.error(f"Error publishing audio: {e}")

    def get_participants(self) -> List[str]:
        return [p.sid for p in self._room.remote_participants.values()]

    async def get_participant_metadata(self, participant_id: str) -> dict:
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            return {
                "id": participant.sid,
                "name": participant.name,
                "metadata": participant.metadata,
                "is_speaking": participant.is_speaking,
            }
        return {}

    async def set_participant_metadata(self, metadata: str):
        await self._room.local_participant.set_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(False)

    async def unmute_participant(self, participant_id: str):
        participant = self._room.remote_participants.get(participant_id)
        if participant:
            for track in participant.tracks.values():
                if track.kind == "audio":
                    await track.set_enabled(True)

    # Wrapper methods for event handlers
    def _on_participant_connected_wrapper(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._async_on_participant_connected(participant))

    def _on_participant_disconnected_wrapper(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._async_on_participant_disconnected(participant))

    def _on_track_subscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(self._async_on_track_subscribed(track, publication, participant))

    def _on_track_unsubscribed_wrapper(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(self._async_on_track_unsubscribed(track, publication, participant))

    def _on_data_received_wrapper(self, data: rtc.DataPacket):
        asyncio.create_task(self._async_on_data_received(data))

    def _on_connected_wrapper(self):
        asyncio.create_task(self._async_on_connected())

    def _on_disconnected_wrapper(self):
        asyncio.create_task(self._async_on_disconnected())

    # Async methods for event handling
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
            asyncio.create_task(self._process_audio_stream(audio_stream, participant.sid))

    async def _async_on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            await self._callbacks.on_audio_track_unsubscribed(participant.sid)

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

    async def cleanup(self):
        await self.disconnect()

    async def get_next_audio_frame(self):
        frame, participant_id = await self._audio_queue.get()
        return frame, participant_id


class LiveKitInputTransport(BaseInputTransport):
    def __init__(self, client: LiveKitTransportClient, params: LiveKitParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._audio_in_task = None
        self._vad_analyzer: VADAnalyzer | None = params.vad_analyzer

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.connect()
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task = asyncio.create_task(self._audio_in_task_handler())
        logger.info("LiveKitInputTransport started")

    async def stop(self, frame: EndFrame):
        if self._audio_in_task:
            self._audio_in_task.cancel()
            try:
                await self._audio_in_task
            except asyncio.CancelledError:
                pass
        await super().stop(frame)
        await self._client.disconnect()
        logger.info("LiveKitInputTransport stopped")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, EndFrame):
            await self.stop(frame)
        else:
            await super().process_frame(frame, direction)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()
        if self._audio_in_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_in_task.cancel()
            await self._audio_in_task

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._vad_analyzer

    async def push_app_message(self, message: Any, sender: str):
        frame = LiveKitTransportMessageFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    async def _audio_in_task_handler(self):
        logger.info("Audio input task started")
        while True:
            try:
                audio_data = await self._client.get_next_audio_frame()
                if audio_data:
                    audio_frame_event, participant_id = audio_data
                    pipecat_audio_frame = self._convert_livekit_audio_to_pipecat(audio_frame_event)
                    input_audio_frame = InputAudioRawFrame(
                        audio=pipecat_audio_frame.audio,
                        sample_rate=pipecat_audio_frame.sample_rate,
                        num_channels=pipecat_audio_frame.num_channels,
                    )
                    await self.push_frame(
                        pipecat_audio_frame
                    )  # TODO: ensure audio frames are pushed with the default BaseInputTransport.push_audio_frame()
                    await self.push_audio_frame(input_audio_frame)
            except asyncio.CancelledError:
                logger.info("Audio input task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in audio input task: {e}")

    def _convert_livekit_audio_to_pipecat(
        self, audio_frame_event: rtc.AudioFrameEvent
    ) -> AudioRawFrame:
        audio_frame = audio_frame_event.frame
        audio_data = audio_frame.data
        original_sample_rate = audio_frame.sample_rate

        if original_sample_rate != self._params.audio_in_sample_rate:
            audio_data = resample_audio(
                audio_data, original_sample_rate, self._params.audio_in_sample_rate
            )

        return AudioRawFrame(
            audio=audio_data,
            sample_rate=self._params.audio_in_sample_rate,
            num_channels=audio_frame.num_channels,
        )


class LiveKitOutputTransport(BaseOutputTransport):
    def __init__(self, client: LiveKitTransportClient, params: LiveKitParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.connect()
        logger.info("LiveKitOutputTransport started")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()
        logger.info("LiveKitOutputTransport stopped")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, EndFrame):
            await self.stop(frame)
        else:
            await super().process_frame(frame, direction)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        if isinstance(frame, (LiveKitTransportMessageFrame, LiveKitTransportMessageUrgentFrame)):
            await self._client.send_data(frame.message.encode(), frame.participant_id)
        else:
            await self._client.send_data(frame.message.encode())

    async def write_raw_audio_frames(self, frames: bytes):
        livekit_audio = self._convert_pipecat_audio_to_livekit(frames)
        await self._client.publish_audio(livekit_audio)

    def _convert_pipecat_audio_to_livekit(self, pipecat_audio: bytes) -> rtc.AudioFrame:
        bytes_per_sample = 2  # Assuming 16-bit audio
        total_samples = len(pipecat_audio) // bytes_per_sample
        samples_per_channel = total_samples // self._params.audio_out_channels

        return rtc.AudioFrame(
            data=pipecat_audio,
            sample_rate=self._params.audio_out_sample_rate,
            num_channels=self._params.audio_out_channels,
            samples_per_channel=samples_per_channel,
        )


class LiveKitTransport(BaseTransport):
    def __init__(
        self,
        url: str,
        token: str,
        room_name: str,
        params: LiveKitParams = LiveKitParams(),
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        callbacks = LiveKitCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
        )
        self._params = params

        self._client = LiveKitTransportClient(
            url, token, room_name, self._params, callbacks, self._loop
        )
        self._input: LiveKitInputTransport | None = None
        self._output: LiveKitOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_call_state_updated")

    def input(self) -> LiveKitInputTransport:
        if not self._input:
            self._input = LiveKitInputTransport(self._client, self._params, name=self._input_name)
        return self._input

    def output(self) -> LiveKitOutputTransport:
        if not self._output:
            self._output = LiveKitOutputTransport(
                self._client, self._params, name=self._output_name
            )
        return self._output

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_audio(self, frame: OutputAudioRawFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> List[str]:
        return self._client.get_participants()

    async def get_participant_metadata(self, participant_id: str) -> dict:
        return await self._client.get_participant_metadata(participant_id)

    async def set_metadata(self, metadata: str):
        await self._client.set_participant_metadata(metadata)

    async def mute_participant(self, participant_id: str):
        await self._client.mute_participant(participant_id)

    async def unmute_participant(self, participant_id: str):
        await self._client.unmute_participant(participant_id)

    async def _on_connected(self):
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        await self._call_event_handler("on_disconnected")
        # Attempt to reconnect
        try:
            await self._client.connect()
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")

    async def _on_participant_connected(self, participant_id: str):
        await self._call_event_handler("on_participant_connected", participant_id)

    async def _on_participant_disconnected(self, participant_id: str):
        await self._call_event_handler("on_participant_disconnected", participant_id)
        await self._call_event_handler("on_participant_left", participant_id, "disconnected")

    async def _on_audio_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_subscribed", participant_id)
        participant = self._client._room.remote_participants.get(participant_id)
        if participant:
            for publication in participant.audio_tracks.values():
                self._client._on_track_subscribed_wrapper(
                    publication.track, publication, participant
                )

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str):
        if self._input:
            await self._input.push_app_message(data.decode(), participant_id)
        await self._call_event_handler("on_data_received", data, participant_id)

    async def send_message(self, message: str, participant_id: str | None = None):
        if self._output:
            frame = LiveKitTransportMessageFrame(message=message, participant_id=participant_id)
            await self._output.send_message(frame)

    async def send_message_urgent(self, message: str, participant_id: str | None = None):
        if self._output:
            frame = LiveKitTransportMessageUrgentFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def cleanup(self):
        if self._input:
            await self._input.cleanup()
        if self._output:
            await self._output.cleanup()
        await self._client.disconnect()

    async def on_room_event(self, event):
        # Handle room events
        pass

    async def on_participant_event(self, event):
        # Handle participant events
        pass

    async def on_track_event(self, event):
        # Handle track events
        pass

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", self, state)

    async def _on_first_participant_joined(self, participant_id: str):
        await self._call_event_handler("on_first_participant_joined", participant_id)
