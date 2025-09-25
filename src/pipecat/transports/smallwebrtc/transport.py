#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Small WebRTC transport implementation for Pipecat.

This module provides a WebRTC transport implementation using aiortc for
real-time audio and video communication. It supports bidirectional media
streaming, application messaging, and client connection management.
"""

import asyncio
import fractions
import time
from collections import deque
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SpriteFrame,
    StartFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

try:
    import cv2
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the SmallWebRTC, you need to `pip install pipecat-ai[webrtc]`.")
    raise Exception(f"Missing module: {e}")

CAM_VIDEO_SOURCE = "camera"
SCREEN_VIDEO_SOURCE = "screenVideo"
MIC_AUDIO_SOURCE = "microphone"


class SmallWebRTCCallbacks(BaseModel):
    """Callback handlers for SmallWebRTC events.

    Parameters:
        on_app_message: Called when an application message is received.
        on_client_connected: Called when a client establishes connection.
        on_client_disconnected: Called when a client disconnects.
    """

    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_client_connected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[SmallWebRTCConnection], Awaitable[None]]


class RawAudioTrack(AudioStreamTrack):
    """Custom audio stream track for WebRTC output.

    Handles audio frame generation and timing for WebRTC transmission,
    supporting queued audio data with proper synchronization.
    """

    def __init__(self, sample_rate):
        """Initialize the raw audio track.

        Args:
            sample_rate: The audio sample rate in Hz.
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_10ms = sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit (2 bytes per sample)
        self._timestamp = 0
        self._start = time.time()
        # Queue of (bytes, future), broken into 10ms sub chunks as needed
        self._chunk_queue = deque()

    def add_audio_bytes(self, audio_bytes: bytes):
        """Add audio bytes to the buffer for transmission.

        Args:
            audio_bytes: Raw audio data to queue for transmission.

        Returns:
            A Future that completes when the data is processed.

        Raises:
            ValueError: If audio bytes are not a multiple of 10ms size.
        """
        if len(audio_bytes) % self._bytes_per_10ms != 0:
            raise ValueError("Audio bytes must be a multiple of 10ms size.")
        future = asyncio.get_running_loop().create_future()

        # Break input into 10ms chunks
        for i in range(0, len(audio_bytes), self._bytes_per_10ms):
            chunk = audio_bytes[i : i + self._bytes_per_10ms]
            # Only the last chunk carries the future to be resolved once fully consumed
            fut = future if i + self._bytes_per_10ms >= len(audio_bytes) else None
            self._chunk_queue.append((chunk, fut))

        return future

    async def recv(self):
        """Return the next audio frame for WebRTC transmission.

        Returns:
            An AudioFrame containing the next audio data or silence.
        """
        # Compute required wait time for synchronization
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        if self._chunk_queue:
            chunk, future = self._chunk_queue.popleft()
            if future and not future.done():
                future.set_result(True)
        else:
            chunk = bytes(self._bytes_per_10ms)  # silence

        # Convert the byte data to an ndarray of int16 samples
        samples = np.frombuffer(chunk, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_10ms
        return frame


class RawVideoTrack(VideoStreamTrack):
    """Custom video stream track for WebRTC output.

    Handles video frame queuing and conversion for WebRTC transmission.
    """

    def __init__(self, width, height):
        """Initialize the raw video track.

        Args:
            width: Video frame width in pixels.
            height: Video frame height in pixels.
        """
        super().__init__()
        self._width = width
        self._height = height
        self._video_buffer = asyncio.Queue()

    def add_video_frame(self, frame):
        """Add a video frame to the transmission buffer.

        Args:
            frame: The video frame to queue for transmission.
        """
        self._video_buffer.put_nowait(frame)

    async def recv(self):
        """Return the next video frame for WebRTC transmission.

        Returns:
            A VideoFrame ready for WebRTC transmission.
        """
        raw_frame = await self._video_buffer.get()

        # Convert bytes to NumPy array
        frame_data = np.frombuffer(raw_frame.image, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")

        # Assign timestamp
        frame.pts, frame.time_base = await self.next_timestamp()

        return frame


class SmallWebRTCClient:
    """WebRTC client implementation for handling connections and media streams.

    Manages WebRTC peer connections, audio/video streaming, and application
    messaging through the SmallWebRTCConnection interface.
    """

    FORMAT_CONVERSIONS = {
        "yuv420p": cv2.COLOR_YUV2RGB_I420,
        "yuvj420p": cv2.COLOR_YUV2RGB_I420,  # OpenCV treats both the same
        "nv12": cv2.COLOR_YUV2RGB_NV12,
        "gray": cv2.COLOR_GRAY2RGB,
    }

    def __init__(self, webrtc_connection: SmallWebRTCConnection, callbacks: SmallWebRTCCallbacks):
        """Initialize the WebRTC client.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            callbacks: Event callbacks for connection and message handling.
        """
        self._webrtc_connection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._video_output_track = None
        self._audio_input_track: Optional[AudioStreamTrack] = None
        self._video_input_track: Optional[VideoStreamTrack] = None
        self._screen_video_track: Optional[VideoStreamTrack] = None

        self._params = None
        self._audio_in_channels = None
        self._in_sample_rate = None
        self._out_sample_rate = None
        self._leave_counter = 0

        # We are always resampling it for 16000 if the sample_rate that we receive is bigger than that.
        # otherwise we face issues with Silero VAD
        self._pipecat_resampler = AudioResampler("s16", "mono", 16000)

        @self._webrtc_connection.event_handler("connected")
        async def on_connected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtc_connection.event_handler("disconnected")
        async def on_disconnected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection lost.")
            await self._handle_peer_disconnected()

        @self._webrtc_connection.event_handler("closed")
        async def on_closed(connection: SmallWebRTCConnection):
            logger.debug("Client connection closed.")
            await self._handle_client_closed()

        @self._webrtc_connection.event_handler("app-message")
        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
            await self._handle_app_message(message, connection.pc_id)

    def _convert_frame(self, frame_array: np.ndarray, format_name: str) -> np.ndarray:
        """Convert a video frame to RGB format based on the input format.

        Args:
            frame_array: The input frame as a NumPy array.
            format_name: The format of the input frame.

        Returns:
            The converted RGB frame as a NumPy array.

        Raises:
            ValueError: If the format is unsupported.
        """
        if format_name.startswith("rgb"):  # Already in RGB, no conversion needed
            return frame_array

        conversion_code = SmallWebRTCClient.FORMAT_CONVERSIONS.get(format_name)

        if conversion_code is None:
            raise ValueError(f"Unsupported format: {format_name}")

        return cv2.cvtColor(frame_array, conversion_code)

    async def read_video_frame(self, video_source: str):
        """Read video frames from the WebRTC connection.

        Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and creates an InputImageRawFrame.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").

        Yields:
            UserImageRawFrame objects containing video data from the peer.
        """
        while True:
            video_track = (
                self._video_input_track
                if video_source == CAM_VIDEO_SOURCE
                else self._screen_video_track
            )
            if video_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(video_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logger.warning("Timeout: No video frame received within the specified time.")
                    # self._webrtc_connection.ask_to_renegotiate()
                frame = None
            except MediaStreamError:
                logger.warning("Received an unexpected media stream error while reading the video.")
                frame = None

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            format_name = frame.format.name
            # Convert frame to NumPy array in its native format
            frame_array = frame.to_ndarray(format=format_name)
            frame_rgb = self._convert_frame(frame_array, format_name)
            del frame_array  # free intermediate array immediately
            image_bytes = frame_rgb.tobytes()
            del frame_rgb  # free RGB array immediately

            image_frame = UserImageRawFrame(
                user_id=self._webrtc_connection.pc_id,
                image=image_bytes,
                size=(frame.width, frame.height),
                format="RGB",
            )
            image_frame.transport_source = video_source

            del frame  # free original VideoFrame
            del image_bytes  # reference kept in image_frame

            yield image_frame

    async def read_audio_frame(self):
        """Read audio frames from the WebRTC connection.

        Reads 20ms of audio from the given MediaStreamTrack and creates an InputAudioRawFrame.

        Yields:
            InputAudioRawFrame objects containing audio data from the peer.
        """
        while True:
            if self._audio_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._audio_input_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logger.warning("Timeout: No audio frame received within the specified time.")
                frame = None
            except MediaStreamError:
                logger.warning("Received an unexpected media stream error while reading the audio.")
                frame = None

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            if frame.sample_rate > self._in_sample_rate:
                resampled_frames = self._pipecat_resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # 16-bit PCM bytes
                    pcm_array = resampled_frame.to_ndarray().astype(np.int16)
                    pcm_bytes = pcm_array.tobytes()
                    del pcm_array  # free NumPy array immediately

                    audio_frame = InputAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=resampled_frame.sample_rate,
                        num_channels=self._audio_in_channels,
                    )
                    del pcm_bytes  # reference kept in audio_frame

                    yield audio_frame
            else:
                # 16-bit PCM bytes
                pcm_array = frame.to_ndarray().astype(np.int16)
                pcm_bytes = pcm_array.tobytes()
                del pcm_array  # free NumPy array immediately

                audio_frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=frame.sample_rate,
                    num_channels=self._audio_in_channels,
                )
                del pcm_bytes  # reference kept in audio_frame

                yield audio_frame

            del frame  # free original AudioFrame

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The audio frame to transmit.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if self._can_send() and self._audio_output_track:
            await self._audio_output_track.add_audio_bytes(frame.audio)
            return True
        return False

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        if self._can_send() and self._video_output_track:
            self._video_output_track.add_video_frame(frame)
            return True
        return False

    async def setup(self, _params: TransportParams, frame):
        """Set up the client with transport parameters.

        Args:
            _params: Transport configuration parameters.
            frame: The initialization frame containing setup data.
        """
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._params = _params
        self._leave_counter += 1

    async def connect(self):
        """Establish the WebRTC connection."""
        if self._webrtc_connection.is_connected():
            # already initialized
            return

        logger.info(f"Connecting to Small WebRTC")
        await self._webrtc_connection.connect()

    async def disconnect(self):
        """Disconnect from the WebRTC peer."""
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            logger.info(f"Disconnecting to Small WebRTC")
            self._closing = True
            await self._webrtc_connection.disconnect()
            await self._handle_peer_disconnected()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send an application message through the WebRTC connection.

        Args:
            frame: The message frame to send.
        """
        if self._can_send():
            self._webrtc_connection.send_app_message(frame.message)

    async def _handle_client_connected(self):
        """Handle client connection establishment."""
        # There is nothing to do here yet, the pipeline is still not ready
        if not self._params:
            return

        self._audio_input_track = self._webrtc_connection.audio_input_track()
        self._video_input_track = self._webrtc_connection.video_input_track()
        self._screen_video_track = self._webrtc_connection.screen_video_input_track()
        if self._params.audio_out_enabled:
            self._audio_output_track = RawAudioTrack(sample_rate=self._out_sample_rate)
            self._webrtc_connection.replace_audio_track(self._audio_output_track)

        if self._params.video_out_enabled:
            self._video_output_track = RawVideoTrack(
                width=self._params.video_out_width, height=self._params.video_out_height
            )
            self._webrtc_connection.replace_video_track(self._video_output_track)

        await self._callbacks.on_client_connected(self._webrtc_connection)

    async def _handle_peer_disconnected(self):
        """Handle peer disconnection cleanup."""
        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

    async def _handle_client_closed(self):
        """Handle client connection closure."""
        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._closing:
            await self._callbacks.on_client_disconnected(self._webrtc_connection)

    async def _handle_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        await self._callbacks.on_app_message(message, sender)

    def _can_send(self):
        """Check if the connection is ready for sending data."""
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        """Check if the WebRTC connection is established.

        Returns:
            True if connected to the peer.
        """
        return self._webrtc_connection.is_connected()

    @property
    def is_closing(self) -> bool:
        """Check if the connection is in the process of closing.

        Returns:
            True if the connection is closing.
        """
        return self._closing


class SmallWebRTCInputTransport(BaseInputTransport):
    """Input transport implementation for SmallWebRTC.

    Handles incoming audio and video streams from WebRTC peers,
    including user image requests and application message handling.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC input transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None
        self._receive_screen_video_task = None
        self._image_requests = {}

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames including user image requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    async def start(self, frame: StartFrame):
        """Start the input transport and establish WebRTC connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        await self.set_transport_ready(frame)
        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task and self._params.video_in_enabled:
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))

    async def _stop_tasks(self):
        """Stop all background tasks."""
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None

    async def stop(self, frame: EndFrame):
        """Stop the input transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        """Background task for receiving audio frames from WebRTC."""
        try:
            audio_iterator = self._client.read_audio_frame()
            async for audio_frame in audio_iterator:
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self, video_source: str):
        """Background task for receiving video frames from WebRTC.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").
        """
        try:
            video_iterator = self._client.read_video_frame(video_source)
            async for video_frame in video_iterator:
                if video_frame:
                    await self.push_video_frame(video_frame)

                    # Check if there are any pending image requests and create UserImageRawFrame
                    if self._image_requests:
                        for req_id, request_frame in list(self._image_requests.items()):
                            if request_frame.video_source == video_source:
                                # Create UserImageRawFrame using the current video frame
                                image_frame = UserImageRawFrame(
                                    user_id=request_frame.user_id,
                                    request=request_frame,
                                    image=video_frame.image,
                                    size=video_frame.size,
                                    format=video_frame.format,
                                )
                                image_frame.transport_source = video_source
                                # Push the frame to the pipeline
                                await self.push_video_frame(image_frame)
                                # Remove from pending requests
                                del self._image_requests[req_id]

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def push_app_message(self, message: Any):
        """Push an application message into the pipeline.

        Args:
            message: The application message to process.
        """
        logger.debug(f"Received app message inside SmallWebRTCInputTransport  {message}")
        frame = InputTransportMessageFrame(message=message)
        await self.push_frame(frame)

    # Add this method similar to DailyInputTransport.request_participant_image
    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Request an image frame from the participant's video stream.

        When a UserImageRequestFrame is received, this method will store the request
        and the next video frame received will be converted to a UserImageRawFrame.

        Args:
            frame: The user image request frame.
        """
        logger.debug(f"Requesting image from participant: {frame.user_id}")

        # Store the request
        request_id = f"{frame.function_name}:{frame.tool_call_id}"
        self._image_requests[request_id] = frame

        # Default to camera if no source specified
        if frame.video_source is None:
            frame.video_source = CAM_VIDEO_SOURCE
        # If we're not already receiving video, try to get a frame now
        if (
            frame.video_source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))
        elif (
            frame.video_source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE)
            )

    async def capture_participant_media(
        self,
        source: str = CAM_VIDEO_SOURCE,
    ):
        """Capture media from a specific participant.

        Args:
            source: Media source to capture from. ("camera", "microphone", or "screenVideo")
        """
        # If we're not already receiving video, try to get a frame now
        if (
            source == MIC_AUDIO_SOURCE
            and not self._receive_audio_task
            and self._params.audio_in_enabled
        ):
            # Start audio reception if it's not already running
            self._receive_audio_task = self.create_task(self._receive_audio())
        elif (
            source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(self._receive_video(CAM_VIDEO_SOURCE))
        elif (
            source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE)
            )


class SmallWebRTCOutputTransport(BaseOutputTransport):
    """Output transport implementation for SmallWebRTC.

    Handles outgoing audio and video streams to WebRTC peers,
    including transport message sending.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC output transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the output transport and establish WebRTC connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message through the WebRTC connection.

        Args:
            frame: The transport message frame to send.
        """
        await self._client.send_message(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The output audio frame to transmit.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        return await self._client.write_audio_frame(frame)

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The output video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return await self._client.write_video_frame(frame)


class SmallWebRTCTransport(BaseTransport):
    """WebRTC transport implementation for real-time communication.

    Provides bidirectional audio and video streaming over WebRTC connections
    with support for application messaging and connection event handling.
    """

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the WebRTC transport.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input: Optional[SmallWebRTCInputTransport] = None
        self._output: Optional[SmallWebRTCOutputTransport] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> SmallWebRTCInputTransport:
        """Get the input transport processor.

        Returns:
            The input transport for handling incoming media streams.
        """
        if not self._input:
            self._input = SmallWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> SmallWebRTCOutputTransport:
        """Get the output transport processor.

        Returns:
            The output transport for handling outgoing media streams.
        """
        if not self._output:
            self._output = SmallWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame through the transport.

        Args:
            frame: The image frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame through the transport.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def _on_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        if self._input:
            await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_client_connected(self, webrtc_connection):
        """Handle client connection events."""
        await self._call_event_handler("on_client_connected", webrtc_connection)

    async def _on_client_disconnected(self, webrtc_connection):
        """Handle client disconnection events."""
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def capture_participant_video(
        self,
        video_source: str = CAM_VIDEO_SOURCE,
    ):
        """Capture video from a specific participant.

        Args:
            video_source: Video source to capture from ("camera" or "screenVideo").
        """
        if self._input:
            await self._input.capture_participant_media(source=video_source)

    async def capture_participant_audio(
        self,
        audio_source: str = MIC_AUDIO_SOURCE,
    ):
        """Capture audio from a specific participant.

        Args:
            audio_source: Audio source to capture from. (currently, "microphone" is the only supported option)
        """
        if self._input:
            await self._input.capture_participant_media(source=audio_source)
