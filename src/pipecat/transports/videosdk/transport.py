#
# Copyright (c) 2024–2025, Pipecat AI
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
import numpy as np
import av
from typing import Optional, Dict, Any, List
from fractions import Fraction
import os

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    InterruptionFrame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from videosdk import (
        MeetingConfig,
        VideoSDK,
        MeetingEventHandler,
        ParticipantEventHandler,
        CustomAudioTrack,
        Participant,
        Stream,
        PubSubSubscribeConfig,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the VideoSDK transport, you need to `pip install pipecat-ai[videosdk]`."
    )
    raise Exception(f"Missing module: {e}")

AUDIO_PTIME = 0.02 


class VideoSDKParams(TransportParams):
    """Configuration parameters for VideoSDK transport.
    Args:
        token: Authentication token for the meeting.
        meeting_id: ID of the meeting to join.
        name: Name of the participant in the meeting.
        mic_enabled: Whether to enable the microphone.
        webcam_enabled: Whether to enable the webcam.
        pubsub_topic: Topic to subscribe to for chat messages.
        audio_out_sample_rate: Sample rate for the audio output.
        audio_out_channels: Number of channels for the audio output.
    """
    token: str = None
    meeting_id: str = None
    name: str
    mic_enabled: bool = True
    webcam_enabled: bool = False
    pubsub_topic: str = "CHAT"
    audio_out_sample_rate: int = 24000
    audio_out_channels: int = 1


class PipecatAudioTrack(CustomAudioTrack):
    """
    Custom Audio Track for sending audio to VideoSDK.
    Maintains strict timing to prevent WebRTC drift or artifacts.
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self._start: float | None = None
        self._timestamp = 0
        self.audio_data_buffer = bytearray()
        self.frame_buffer: List[av.AudioFrame] = []
        
        self.sample_width = 2 
        self.time_base_fraction = Fraction(1, self.sample_rate)
        self.samples_per_frame = int(AUDIO_PTIME * self.sample_rate)
        self.bytes_per_frame = int(self.samples_per_frame * self.channels * self.sample_width)
        
        self.running = True
        self._lock = asyncio.Lock()

    def stop(self):
        self.running = False

    async def interrupt(self):
        """Clear all buffered audio immediately."""
        async with self._lock:
            self.frame_buffer.clear()
            self.audio_data_buffer.clear()

    async def add_audio(self, audio_data: bytes):
        """Add audio data to the buffer and build frames."""
        async with self._lock:
            self.audio_data_buffer += audio_data

            while len(self.audio_data_buffer) >= self.bytes_per_frame:
                chunk = self.audio_data_buffer[: self.bytes_per_frame]
                self.audio_data_buffer = self.audio_data_buffer[self.bytes_per_frame :]
                try:
                    audio_frame = self._build_audio_frame(chunk)
                    self.frame_buffer.append(audio_frame)
                except Exception as e:
                    logger.error(f"Error building audio frame: {e}")
                    break

    def _build_audio_frame(self, chunk: bytes) -> av.AudioFrame: 
        """Build an AudioFrame from PCM bytes."""
        data = np.frombuffer(chunk, dtype=np.int16)
        data = data.reshape(-1, self.channels)
        layout = "mono" if self.channels == 1 else "stereo"
        
        audio_frame = av.AudioFrame.from_ndarray(data.T, format="s16", layout=layout)
        return audio_frame

    def _create_silence_frame(self) -> av.AudioFrame:
        """Generates a silence frame to keep the transport alive."""
        frame = av.AudioFrame(
            format='s16', 
            layout='mono' if self.channels == 1 else 'stereo', 
            samples=self.samples_per_frame
        )
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self.sample_rate
        return frame

    def _next_timestamp(self):
        """Get the next PTS and time_base for a frame."""
        pts = self._timestamp
        time_base = self.time_base_fraction
        self._timestamp += self.samples_per_frame
        return pts, time_base

    async def recv(self) -> av.AudioFrame:
        """Called by VideoSDK media loop to request the next audio frame."""
        try:
            if self._start is None:
                self._start = time.time()
                self._timestamp = 0

            expected_time = self._start + (self._timestamp / self.sample_rate)
            wait = expected_time - time.time()
            
            if wait > 0:
                await asyncio.sleep(wait)

            if not self.running:
                frame = self._create_silence_frame()
            else:
                async with self._lock:
                    if len(self.frame_buffer) > 0:
                        frame = self.frame_buffer.pop(0)
                    else:
                        frame = self._create_silence_frame()

            pts, time_base = self._next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            frame.sample_rate = self.sample_rate
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in PipecatAudioTrack recv: {e}")
            await asyncio.sleep(0.01)
            return self._create_silence_frame()


class VideoSDKInputTransport(BaseInputTransport):
    """Handles incoming media streams from VideoSDK meetings."""

    def __init__(self, transport: "VideoSDKTransport", params: VideoSDKParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._params = params
        self._participant_handlers = {}
        self._audio_tasks = {}

    class ParticipantHandler(ParticipantEventHandler):
        def __init__(self, transport, participant_id):
            super().__init__()
            self.transport = transport
            self.participant_id = participant_id

        def on_stream_enabled(self, stream: Stream):
            if stream.kind == "audio":
                self.transport._start_receiving_audio(self.participant_id, stream)

        def on_stream_disabled(self, stream: Stream):
            if stream.kind == "audio":
                self.transport._stop_receiving_audio(self.participant_id)

    def _start_receiving_audio(self, participant_id, stream):
        if participant_id in self._audio_tasks:
            return
        
        logger.info(f"Subscribing to audio from {participant_id}")
        task = self.create_task(self._read_audio_stream(participant_id, stream))
        self._audio_tasks[participant_id] = task

    def _stop_receiving_audio(self, participant_id):
        if participant_id in self._audio_tasks:
            logger.info(f"Unsubscribing from audio of {participant_id}")
            self._audio_tasks[participant_id].cancel()
            del self._audio_tasks[participant_id]

    async def _read_audio_stream(self, participant_id, stream: Stream):
        """Read audio frames from a participant's stream."""
        
        # Resampler to convert VideoSDK's audio (48k/Opus decoded) to Pipecat (16k PCM)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        
        try:
            while True:
                frame = await stream.track.recv()
                
                resampled_frames = resampler.resample(frame)
                
                for resampled_frame in resampled_frames:
                    audio_data = resampled_frame.to_ndarray()[0]
                    pcm_frame = audio_data.flatten().astype(np.int16).tobytes()
                    
                    pipecat_frame = UserAudioRawFrame(
                        user_id=participant_id,
                        audio=pcm_frame,
                        sample_rate=16000,
                        num_channels=1,
                    )
                    await self.push_audio_frame(pipecat_frame)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error reading audio stream from {participant_id}: {e}")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._transport.connect(frame)
        await self.set_transport_ready(frame)
        
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        for task in self._audio_tasks.values():
            task.cancel()
        self._audio_tasks.clear()
        await self._transport.leave()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        for task in self._audio_tasks.values():
            task.cancel()
        self._audio_tasks.clear()
        await self._transport.leave()


class VideoSDKOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams to VideoSDK meetings."""

    def __init__(self, transport: "VideoSDKTransport", params: VideoSDKParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._params = params
        self._audio_track: Optional[PipecatAudioTrack] = None

    def set_audio_track(self, track: PipecatAudioTrack):
        self._audio_track = track

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._audio_track:
            await self._audio_track.add_audio(frame.audio)
            return True
        return False
    
    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        logger.warning("Video output not yet implemented for VideoSDK transport")
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, InterruptionFrame) and self._audio_track:
            await self._audio_track.interrupt()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._transport.connect(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._audio_track:
            self._audio_track.stop()
        await self._transport.leave()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._audio_track:
            self._audio_track.stop()
        await self._transport.leave()


class VideoSDKTransport(BaseTransport):
    """
    Transport implementation for VideoSDK.live (https://www.videosdk.live).

    This class provides seamless integration with VideoSDK for conversational AI applications. 
    It enables the Pipecat-AI bot to:
        - Join and participate in VideoSDK meetings as a virtual agent
        - Receive real-time audio streams and chat messages from meeting participants to process with pipecat-ai bot.
        - Send audio responses back into the meeting.
    Args:
        params: Configuration parameters for the transport. 
        input_name: Optional name for the input transport.
        output_name: Optional name for the output transport.
    """

    def __init__(
        self,
        params: VideoSDKParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params
        self._meeting = None
        self._input: Optional[VideoSDKInputTransport] = None
        self._output: Optional[VideoSDKOutputTransport] = None
        self._audio_track: Optional[PipecatAudioTrack] = None
        self._loop = None
        self._first_participant_joined = False
        self._joined = False

        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_meeting_joined")
        self._register_event_handler("on_meeting_left")
        self._register_event_handler("on_pubsub_message_received")

    def input(self) -> VideoSDKInputTransport:
        if not self._input:
            self._input = VideoSDKInputTransport(self, self._params, name=self._input_name)
        return self._input

    def output(self) -> VideoSDKOutputTransport:
        if not self._output:
            self._output = VideoSDKOutputTransport(self, self._params, name=self._output_name)
        return self._output

    class MeetingHandler(MeetingEventHandler):
        def __init__(self, transport):
            super().__init__()
            self.transport = transport

        def on_meeting_joined(self, data):
            logger.info("VideoSDK Meeting joined")
            if self.transport._loop:
                asyncio.run_coroutine_threadsafe(
                    self.transport._on_meeting_joined(data),
                    self.transport._loop
                )

        def on_meeting_left(self, data):
            logger.info("VideoSDK Meeting left")
            if self.transport._loop:
                asyncio.run_coroutine_threadsafe(
                    self.transport._call_event_handler("on_meeting_left", data),
                    self.transport._loop
                )

        def on_participant_joined(self, participant: Participant):
            logger.debug(f"Participant joined: {participant.id}")
            
            handler = self.transport.input().ParticipantHandler(
                self.transport.input(), 
                participant.id
            )
            participant.add_event_listener(handler)
            self.transport.input()._participant_handlers[participant.id] = handler
            
            if self.transport._loop:
                asyncio.run_coroutine_threadsafe(
                    self.transport._handle_participant_joined(participant), 
                    self.transport._loop
                )

        def on_participant_left(self, participant: Participant):
            logger.debug(f"Participant left: {participant.id}")
            if participant.id in self.transport.input()._participant_handlers:
                del self.transport.input()._participant_handlers[participant.id]
            
            if self.transport._loop:
                asyncio.run_coroutine_threadsafe(
                    self.transport._call_event_handler("on_participant_left", participant), 
                    self.transport._loop
                )

    async def _on_meeting_joined(self, data):
        self._joined = True
        await self._call_event_handler("on_meeting_joined", data)
        
        if self._params.pubsub_topic and self._meeting:
            pubsub_config = PubSubSubscribeConfig(
                topic=self._params.pubsub_topic,
                cb=self._on_pubsub_message
            )
            asyncio.create_task(self._subscribe_to_pubsub(pubsub_config))
    
    async def _subscribe_to_pubsub(self, pubsub_config):
        try:
            await self._meeting.pubsub.subscribe(pubsub_config)
        except Exception as e:
            logger.error(f"Error subscribing to PubSub: {e}")
    
    def _on_pubsub_message(self, message):
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._handle_pubsub_message(message),
                self._loop
            )
    
    async def _handle_pubsub_message(self, message):
        try:
            sender_id = message.get("senderId", "unknown")
            if self._meeting and sender_id == self._meeting.local_participant.id:
                return
            await self._call_event_handler("on_pubsub_message_received", message, sender_id)
        except Exception as e:
            logger.error(f"Error handling PubSub message: {e}")

    async def _handle_participant_joined(self, participant):
        await self._call_event_handler("on_participant_joined", participant)
        if not self._first_participant_joined:
            self._first_participant_joined = True
            await self._call_event_handler("on_first_participant_joined", participant)

    async def connect(self, frame: StartFrame):
        if self._meeting:
            return

        self._loop = asyncio.get_running_loop()
        output_sample_rate = self._params.audio_out_sample_rate or 24000
        
        self._audio_track = PipecatAudioTrack(
            sample_rate=output_sample_rate,
            channels=self._params.audio_out_channels
        )
        
        if self._output:
            self._output.set_audio_track(self._audio_track)

        meeting_config = MeetingConfig(
            meeting_id=self._params.meeting_id,
            name=self._params.name,
            mic_enabled=self._params.mic_enabled,
            webcam_enabled=self._params.webcam_enabled,
            token=self._params.token,
            custom_microphone_audio_track=self._audio_track if self._params.audio_out_enabled else None
        )

        self._meeting = VideoSDK.init_meeting(**meeting_config)
        self._meeting.add_event_listener(self.MeetingHandler(self))

        logger.info(f"Joining VideoSDK meeting: {self._params.meeting_id}")
        await self._meeting.async_join()

    async def leave(self):
        if self._meeting and self._joined:
            logger.info(f"Leaving VideoSDK meeting: {self._params.meeting_id}")
            self._meeting.leave()
            self._joined = False
            await asyncio.sleep(0.1)

    async def cleanup(self):
        await self.leave()
        if self._audio_track:
            self._audio_track.stop()
        self._meeting = None