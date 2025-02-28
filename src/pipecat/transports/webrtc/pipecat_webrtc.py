#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from collections import deque
from typing import Awaitable, Callable, Optional

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VideoFrame
from av import AudioFrame, AudioResampler
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputImageRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.webrtc.webrtc_connection import PipecatWebRTCConnection


class PipecatWebRTCCallbacks(BaseModel):
    on_client_connected: Callable[[PipecatWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[PipecatWebRTCConnection], Awaitable[None]]
    on_client_closed: Callable[[PipecatWebRTCConnection], Awaitable[None]]


class RawAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = self.sample_rate // 50  # 20ms per frame
        self.time = 0
        self.audio_buffer = deque()  # Efficient buffer storage

    def add_audio_bytes(self, audio_bytes: bytes):
        """
        Adds bytes to the audio buffer.
        Ensures that only full 16-bit samples are stored.
        """
        if len(audio_bytes) % 2 != 0:
            raise ValueError("Audio bytes length must be even (16-bit samples).")
        self.audio_buffer.append(audio_bytes)

    async def recv(self):
        """
        Returns the next audio frame, generating silence if needed.
        """

        await asyncio.sleep(0.02)  # Simulate real-time delay (20ms)

        # Check if we have enough data
        needed_bytes = self.samples_per_frame * 2  # 16-bit (2 bytes per sample)
        if sum(map(len, self.audio_buffer)) >= needed_bytes:
            # Extract data from deque
            chunk = bytearray()
            while len(chunk) < needed_bytes:
                chunk.extend(self.audio_buffer.popleft())

            # Trim excess bytes in case the last deque element exceeded needed_bytes
            chunk = bytes(chunk[:needed_bytes])

        else:
            # Not enough data, generate silent frame
            chunk = bytes(needed_bytes)

        # Convert the byte data to an ndarray of int16 samples
        samples = np.frombuffer(chunk, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.time

        self.time += self.samples_per_frame
        return frame


class PipecatWebRTCClient:
    def __init__(
        self, webrtc_connection: PipecatWebRTCConnection, callbacks: PipecatWebRTCCallbacks
    ):
        self._webrtcConnection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._audio_input_track = None
        self._video_input_track = None

        self._audio_in_channels = None
        self._in_sample_rate = None
        self._out_sample_rate = None

        # We are always resampling it for 16000 if the sample_rate that we receive is bigger than that.
        # otherwise we face issues with Silero VAD
        self._pipecat_resampler = AudioResampler("s16", "mono", 16000)

        @self._webrtcConnection.on("connected")
        async def on_connected():
            logger.info("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtcConnection.on("disconnected")
        async def on_disconnected():
            logger.info("Peer connection lost.")
            await self._handle_client_disconnected()

        @self._webrtcConnection.on("closed")
        async def on_closed():
            logger.info("Client connection closed.")
            await self._handle_client_closed()

    async def read_video_frame(self):
        """
        Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and returns it as a NumPy array.
        """
        while self._video_input_track is not None:
            frame = await self._video_input_track.recv()  # Get a video frame

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            # If you want to see the frame size or other information, you can print it
            print(f"Received video frame: {frame.width}x{frame.height}, timestamp: {frame.time}, format: {frame.format.name}")

            image_frame = InputImageRawFrame(
                image=frame.to_rgb().to_image().tobytes(),
                size=(frame.width, frame.height),
                format="RGB",
            )

            yield image_frame

    async def read_audio_frame(self):
        """
        Reads 20ms of audio from the given MediaStreamTrack and returns raw PCM bytes.
        """
        while self._audio_input_track is not None:
            frame = await self._audio_input_track.recv()  # Get an audio frame

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            if frame.sample_rate > self._in_sample_rate:
                resampled_frames = self._pipecat_resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # 16-bit PCM bytes
                    pcm_bytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    audio_frame = InputAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=resampled_frame.sample_rate,
                        num_channels=self._audio_in_channels,
                    )
                    yield audio_frame
            else:
                # 16-bit PCM bytes
                pcm_bytes = frame.to_ndarray().astype(np.int16).tobytes()
                audio_frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=frame.sample_rate,
                    num_channels=self._audio_in_channels,
                )
                yield audio_frame

    async def write_raw_audio_frames(self, data: bytes):
        if self._can_send():
            self._audio_output_track.add_audio_bytes(data)

    async def setup(self, _params, frame):
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or frame.audio_out_sample_rate

    async def connect(self):
        if self._audio_output_track:
            # already initialized
            return

        logger.info(f"Connecting to Pipecat WebRTC")
        self._audio_output_track = RawAudioTrack(sample_rate=self._out_sample_rate)
        self._webrtcConnection.replace_audio_track(self._audio_output_track)

    async def disconnect(self):
        if self.is_connected and not self.is_closing:
            self._closing = True
            await self._webrtcConnection.close()
            await self._handle_client_disconnected()

    async def _handle_client_connected(self):
        self._audio_input_track = self._webrtcConnection.audio_input_track()
        self._video_input_track = self._webrtcConnection.video_input_track()
        await self._callbacks.on_client_connected(self._webrtcConnection)

    async def _handle_client_disconnected(self):
        await self._callbacks.on_client_disconnected(self._webrtcConnection)

    async def _handle_client_closed(self):
        await self._callbacks.on_client_closed(self._webrtcConnection)

    def _can_send(self):
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        return self._webrtcConnection.is_connected()

    @property
    def is_closing(self) -> bool:
        return self._closing


class PipecatWebRTCInputTransport(BaseInputTransport):
    def __init__(
        self,
        client: PipecatWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(self._params, frame)
        await self._client.connect()
        if not self._receive_audio_task:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task:
            self._receive_video_task = self.create_task(self._receive_video())

    async def _stop_tasks(self):
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        try:
            async for audio_frame in self._client.read_audio_frame():
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self):
        try:
            async for video_frame in self._client.read_video_frame():
                if video_frame:
                    await self.push_frame(video_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")


class PipecatWebRTCOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        client: PipecatWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(self._params, frame)
        await self._client.connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        # TODO: implement it, we should send through the datachannel
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        await self._client.write_raw_audio_frames(frames)


class PipecatWebRTCTransport(BaseTransport):
    def __init__(
        self,
        webrtc_connection: PipecatWebRTCConnection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = PipecatWebRTCCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_client_closed=self._on_client_closed,
        )

        self._client = PipecatWebRTCClient(webrtc_connection, self._callbacks)

        self._input = PipecatWebRTCInputTransport(self._client, self._params, name=self._input_name)
        self._output = PipecatWebRTCOutputTransport(
            self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_client_closed")

    def input(self) -> PipecatWebRTCInputTransport:
        if not self._input:
            self._input = PipecatWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> PipecatWebRTCOutputTransport:
        if not self._output:
            self._output = PipecatWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def _on_client_connected(self, webrtc_connection):
        await self._call_event_handler("on_client_connected", webrtc_connection)

    async def _on_client_disconnected(self, webrtc_connection):
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def _on_client_closed(self, webrtc_connection):
        await self._call_event_handler("on_client_closed", webrtc_connection)
