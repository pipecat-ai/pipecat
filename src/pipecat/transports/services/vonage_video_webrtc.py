# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebRTC transport."""

import asyncio
import json
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import bidirmodule
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        f"In order to use Vonage, you need to Vonage's native SDK wrapper for python installed."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class AudioData:
    """Represents audio data received from the Vonage session."""

    sample_buffer: memoryview
    bits_per_sample: int
    sample_rate: int
    number_of_channels: int
    number_of_frames: int


@dataclass
class SessionCallbacks:
    """Callbacks for session events."""

    on_error_cb: Callable[[str, str, int], None] = lambda session_id, description, code: None
    on_connected_cb: Callable[[str], None] = lambda session_id: None
    on_disconnected_cb: Callable[[str], None] = lambda session_id: None
    on_stream_received_cb: Callable[[str, str], None] = lambda session_id, stream_id: None
    on_stream_dropped_cb: Callable[[str, str], None] = lambda session_id, stream_id: None
    on_audio_data_cb: Callable[[str, AudioData], None] = lambda session_id, audio_data: None
    on_ready_to_publish_cb: Callable[[str], None] = lambda session_id: None


@dataclass
class SubscriberCallbacks:
    """Callbacks for subscriber events."""

    on_error_cb: Callable[[str, str, int], None] = lambda subscriber_id, description, code: None
    on_connected_cb: Callable[[str], None] = lambda subscriber_id: None
    on_disconnected_cb: Callable[[str], None] = lambda subscriber_id: None
    on_video_frame_cb: Callable[[str], None] = lambda subscriber_id: None
    on_audio_data_cb: Callable[[str, AudioData], None] = lambda session_id, audio_data: None


@dataclass
class PublisherSettings:
    """Settings for the publisher."""

    name: str
    publish_video: bool = False
    publish_audio: bool = True
    stereo_audio_mode: bool = True
    audio_noise_suppresion: bool = False
    audio_echo_cancellation: bool = False
    audio_processing: bool = False


@dataclass
class PublisherCallbacks:
    """Callbacks for publisher events."""

    on_error_cb: Callable[[str, str, int], None] = lambda publisher_id, description, code: None
    on_stream_created_cb: Callable[[str, str], None] = lambda publisher_id, stream_id: None
    on_stream_destroyed_cb: Callable[[str, str], None] = lambda publisher_id, stream_id: None


class VonageVideoWebrtcTransportParams(TransportParams):
    """Parameters for the Vonage transport."""

    audio_per_subcriber: bool = False
    publisher_name: str = ""
    publisher_video: bool = False
    publisher_audio: bool = True
    publisher_audio_noise_suppression: bool = False
    publisher_echo_cancellation: bool = False
    publisher_audio_processing: bool = False


class VonageException(Exception):
    """Exception raised for errors in the Vonage transport."""

    pass


@dataclass
class VonageClientListener:
    """Listener for Vonage client events."""

    on_session_audio_in: Callable[str, AudioData] = lambda _id, _audio: None
    on_subscriber_audio_in: Callable[str, AudioData] = lambda _id, _audio: None


@dataclass
class VonageClientParams:
    """Parameters for the Vonage client."""

    audio_sample_rate: int = 48000
    audio_channels: int = 2


class VonageClient:
    """Client for managing a Vonage session."""

    def __init__(
        self,
        session_str: str,
        params: VonageClientParams,
        publisher_settings: Optional[PublisherSettings] = None,
    ):
        """Initialize the Vonage client with session string and parameters."""
        self._session_str: str = session_str
        self._session_id: str = json.loads(session_str).get("sessionId", "")
        self._params = params
        self._connected: bool = False
        self._connection_counter: int = 0
        self._subscriber_callbacks: SubscriberCallbacks = SubscriberCallbacks(
            on_error_cb=self._on_subscriber_error_cb,
            on_connected_cb=self._on_subscriber_connected_cb,
            on_disconnected_cb=self._on_subscriber_disconnected_cb,
            on_video_frame_cb=self._on_subscriber_video_frame_cb,
            on_audio_data_cb=self._on_subscriber_audio_data_cb,
        )
        self._listeners: map[int, VonageClientListener] = {}
        self._publish_ready: Optional[asyncio.Future] = None
        self._publisher_settings: Optional[PublisherSettings] = publisher_settings
        self._publisher_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def get_params(self) -> VonageClientParams:
        """Return the parameters of the Vonage client."""
        return self._params

    async def connect(self, listener: VonageClientListener) -> int:
        """Connect to the Vonage session."""
        logger.info(f"Connecting with session string {self._session_id}")

        connect_id: int = self._connection_counter
        self._listeners[connect_id] = listener
        if self._connected:
            logger.info(f"Already connected to {self._session_id}")

            # if we've already connected refcount the times we've connected
            self._connection_counter += 1
            return connect_id

        if self._publish_ready is not None:
            logger.info(f"Already connecting to {self._session_id}")

            # if we already connecting, await for the publish ready event
            await self._publish_ready
            return connect_id

        session_callbacks = SessionCallbacks(
            on_error_cb=self._on_session_error_cb,
            on_connected_cb=self._on_session_connected_cb,
            on_disconnected_cb=self._on_session_disconnected_cb,
            on_stream_received_cb=self._on_stream_received_cb,
            on_stream_dropped_cb=self._on_stream_dropped_cb,
            on_audio_data_cb=self._on_session_audio_data_cb,
            on_ready_to_publish_cb=self._on_session_ready_to_publish_cb,
        )

        if self._publisher_settings:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self._publish_ready: asyncio.Future = loop.create_future()

        audio_settings = f'{{"sampleRate":{self._params.audio_sample_rate},"channels":{self._params.audio_channels}}}'
        if not bidirmodule.connect(
            session_info=self._session_str,
            session_callbacks=session_callbacks,
            session_audio_settings=audio_settings,
        ):
            logger.error(f"Could not connect to {self._session_id}")
            raise VonageException("Could not connect to session")

        logger.info(f"Connected to {self._session_id}")

        if self._publish_ready:
            await self._publish_ready

        self._connected = True
        return connect_id

    async def disconnect(self, connect_id: int):
        """Disconnect from the Vonage session."""
        self._connection_counter -= 1
        if not self._connected or self._connection_counter != 0:
            logger.info(f"Already disconnected from {self._session_id}")
            return

        logger.info(f"Disconnecting from {self._session_id}")

        if self._publisher_id:
            bidirmodule.unpublish(self._publisher_id)
            self._publisher_id = None

        # TODO adapt this when bidirmodule is not a singleton anymore
        bidirmodule.disconnect()

        self._listeners.pop(connect_id, None)

        logger.info(f"Disconnected from {self._session_id}")

    async def write_audio(self, raw_audio_frame: bytes):
        """Write audio data to the Vonage session."""
        frame_count = len(raw_audio_frame) // (self._params.audio_channels * 2)
        bidirmodule.add_audio(raw_audio_frame, frame_count)

    def _on_session_ready_to_publish_cb(self, session_id: str):
        logger.info(f"Session {session_id} ready to publish")
        if self._publish_ready:
            future = self._publish_ready
            self._publish_ready = None
            self._loop.call_soon_threadsafe(future.set_result, None)

    def _on_session_error_cb(self, session_id: str, description: str, code: int):
        logger.warning(f"Session error {session_id} code={code} description={description}")

    def _on_session_connected_cb(self, session_id: str):
        logger.info(f"Session connected {session_id}")
        bidirmodule.publish(
            settings=self._publisher_settings,
            callbacks=PublisherCallbacks(
                on_error_cb=self._on_publisher_error_cb,
                on_stream_created_cb=self._on_publisher_stream_created_cb,
                on_stream_destroyed_cb=self._on_publisher_stream_destroyed_cb,
            ),
        )

    def _on_session_disconnected_cb(self, session_id: str):
        logger.info(f"Session disconnected {session_id}")
        self._connected = False

    def _on_publisher_error_cb(self, publisher_id: str, description: str, code: int):
        logger.warning(
            f"Publisher error session={self._session_id} publisher={publisher_id} "
            f"code={code} description={description}"
        )

    def _on_publisher_stream_created_cb(self, publisher_id: str, stream_id: str):
        logger.info(
            f"Publisher stream created session={self._session_id} publisher={publisher_id} stream={stream_id}"
        )
        self._publisher_id = publisher_id

    def _on_publisher_stream_destroyed_cb(self, publisher_id: str, stream_id: str):
        logger.info(
            f"Publisher stream destroyed session={self._session_id} publisher={publisher_id} stream={stream_id}"
        )

    def _on_session_audio_data_cb(self, session_id: str, audio_data: AudioData):
        for listener in self._listeners.values():
            if listener.on_session_audio_in:
                listener.on_session_audio_in(session_id, audio_data)

    def _on_stream_received_cb(self, session_id: str, stream_id: str):
        logger.info(f"Stream received session={session_id} stream={stream_id}")
        # TODO If I reuse SubscriberCallbacks instance the wrapper may destroy it or something, it won't work
        bidirmodule.subscribe(
            stream_id,
            SubscriberCallbacks(
                on_error_cb=self._on_subscriber_error_cb,
                on_connected_cb=self._on_subscriber_connected_cb,
                on_disconnected_cb=self._on_subscriber_disconnected_cb,
                on_video_frame_cb=self._on_subscriber_video_frame_cb,
                on_audio_data_cb=self._on_subscriber_audio_data_cb,
            ),
        )

    def _on_stream_dropped_cb(self, session_id: str, stream_id: str):
        logger.info(f"Stream dropped session={session_id} stream={stream_id}")
        bidirmodule.unsubscribe(stream_id)

    def _on_subscriber_error_cb(self, subscriber_id: str, description: str, code: int):
        logger.info(
            f"Subscriber error session={self._session_id} subscriber={subscriber_id} "
            f"code={code} description={description}"
        )

    def _on_subscriber_connected_cb(self, subscriber_id: str):
        logger.info(f"Subscriber connected session={self._session_id} subscriber={subscriber_id} ")

    def _on_subscriber_disconnected_cb(self, subscriber_id: str):
        logger.info(
            f"Subscriber disconnected session={self._session_id} subscriber={subscriber_id} "
        )

    def _on_subscriber_video_frame_cb(self, subscriber_id: str):
        pass

    def _on_subscriber_audio_data_cb(self, subscriber_id: str, audio_data: AudioData):
        for listener in self._listeners.values():
            if listener.on_subscriber_audio_in:
                listener.on_subscriber_audio_in(subscriber_id, audio_data)


class VonageVideoWebrtcInputTransport(BaseInputTransport):
    """Input transport for Vonage, handling audio input from the Vonage session."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage input transport with the client and parameters."""
        super().__init__(params)

        self._initialized: bool = False
        self._sample_rate = 0
        self._client: VonageClient = client
        self._connect_id: Optional[int] = None
        self._resampler = create_default_resampler()

    async def start(self, frame: StartFrame):
        """Start the Vonage input transport."""
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_in_enabled:
            self._connect_id: int = await self._client.connect(
                VonageClientListener(on_session_audio_in=self._audio_in_cb)
            )

        await self.set_transport_ready(frame)

    def _audio_in_cb(self, identifier: str, audio: AudioData):
        if self._connect_id is not None and self._params.audio_in_enabled:
            dtype = np.int16 if audio.bits_per_sample == 16 else np.uint8
            # TODO: now the python wrapper also is losing reference to the passed audio properties such as sample rate
            # when the use them out of scope such inside the push_frame function, this makes us have to copy these
            # values here
            audio_sample_rate = audio.sample_rate
            number_of_channels = audio.number_of_channels

            # we need to copy the raw audio here as it is a memory view and it will be lost when processed async later
            raw_audio = np.frombuffer(audio.sample_buffer, dtype=dtype)

            async def push_frame():
                # ensure audio is mono
                if number_of_channels == 2:
                    raw_audio_channel_matrix = raw_audio.reshape(-1, 2)
                    raw_audio_mono = raw_audio_channel_matrix.mean(axis=1).astype(dtype)
                else:
                    raw_audio_mono = raw_audio

                # resample audio if needed
                if audio_sample_rate != self.sample_rate:
                    resampled_audio = await self._resampler.resample(
                        raw_audio_mono.tobytes(), audio_sample_rate, self.sample_rate
                    )
                else:
                    resampled_audio = raw_audio_mono.tobytes()

                if self._params.audio_per_subcriber:
                    frame = UserAudioRawFrame(
                        user_id=identifier,
                        audio=resampled_audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                else:
                    frame = InputAudioRawFrame(
                        audio=resampled_audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

                await self.push_audio_frame(frame)

            asyncio.run_coroutine_threadsafe(push_frame(), self.get_event_loop())

    async def stop(self, frame: EndFrame):
        """Stop the Vonage input transport."""
        await super().stop(frame)
        if self._connect_id is not None and self._params.audio_in_enabled:
            connect_id, self._connect_id = self._connect_id, None
            await self._client.disconnect(connect_id)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Vonage input transport."""
        await super().cancel(frame)
        if self._connect_id is not None and self._params.audio_in_enabled:
            connect_id, self._connect_id = self._connect_id, None
            await self._client.disconnect(connect_id)


class VonageVideoWebrtcOutputTransport(BaseOutputTransport):
    """Output transport for Vonage, handling audio output to the Vonage session."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage output transport with the client and parameters."""
        super().__init__(params)
        self._initialized: bool = False
        self._resampler = create_default_resampler()
        self._client = client
        self._connect_id: Optional[int] = None

    async def start(self, frame: StartFrame):
        """Start the Vonage output transport."""
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_out_enabled:
            self._connect_id: int = await self._client.connect(VonageClientListener())

        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the Vonage session."""
        if self._connect_id is not None and self._params.audio_out_enabled:
            params: VonageClientParams = self._client.get_params()

            audio = frame.audio

            # resample audio if needed
            if params.audio_sample_rate != frame.sample_rate:
                audio = await self._resampler.resample(
                    frame.audio, frame.sample_rate, params.audio_sample_rate
                )

            # ensure audio has the correct number of channels
            if params.audio_channels != frame.num_channels:
                audio = np.frombuffer(audio, dtype=np.int16)
                if params.audio_channels == 1:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    audio = audio.mean(axis=1).astype(dtype).tobytes()
                else:
                    audio = np.repeat(audio, params.audio_channels // frame.num_channels).tobytes()

            await self._client.write_audio(audio)

    async def stop(self, frame: EndFrame):
        """Stop the Vonage output transport."""
        await super().stop(frame)
        if self._connect_id is not None and self._params.audio_out_enabled:
            connect_id, self._connect_id = self._connect_id, None
            await self._client.disconnect(connect_id)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Vonage output transport."""
        await super().cancel(frame)
        if self._connect_id is not None and self._params.audio_out_enabled:
            connect_id, self._connect_id = self._connect_id, None
            await self._client.disconnect(connect_id)


class VonageVideoWebrtcTransport(BaseTransport):
    """Vonage transport implementation for Pipecat."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, session_str: str, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage transport with session string and parameters."""
        super().__init__()
        self._params = params

        vonage_params = VonageClientParams(
            audio_sample_rate=params.audio_out_sample_rate or 48000,
            audio_channels=params.audio_out_channels,
        )
        publisher_settings = (
            PublisherSettings(
                name=params.publisher_name,
                publish_video=params.publisher_video,
                publish_audio=params.publisher_audio,
                stereo_audio_mode=params.audio_out_channels == 2,
                audio_noise_suppresion=params.publisher_audio_noise_suppression,
                audio_echo_cancellation=params.publisher_echo_cancellation,
                audio_processing=params.publisher_audio_processing,
            )
            if params.audio_out_enabled
            else None
        )
        self._client = VonageClient(session_str, vonage_params, publisher_settings)

        self._input: Optional[VonageVideoWebrtcInputTransport] = None
        self._output: Optional[VonageVideoWebrtcOutputTransport] = None

    def input(self) -> FrameProcessor:
        """Return the input transport for Vonage."""
        if not self._input:
            self._input = VonageVideoWebrtcInputTransport(self._client, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Return the output transport for Vonage."""
        if not self._output:
            self._output = VonageVideoWebrtcOutputTransport(self._client, self._params)
        return self._output
