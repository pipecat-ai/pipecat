# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebRTC transport."""

import asyncio
import itertools
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

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
    import vonage_video
    from vonage_video.models import (
        AudioData,
        LoggingSettings,
        Publisher,
        PublisherAudioSettings,
        PublisherSettings,
        Session,
        SessionAudioSettings,
        SessionSettings,
        Stream,
        Subscriber,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        f"In order to use Vonage, you need to Vonage's native SDK wrapper for python installed."
    )
    raise Exception(f"Missing module: {e}")


class VonageVideoWebrtcTransportParams(TransportParams):
    """Parameters for the Vonage transport."""

    publisher_name: str = ""
    publisher_enable_opus_dtx: bool = False
    session_enable_migration: bool = False


class VonageException(Exception):
    """Exception raised for errors in the Vonage transport."""

    pass


async def async_noop(*args, **kwargs):
    """No operation async function."""
    pass


@dataclass
class VonageClientListener:
    """Listener for Vonage client events."""

    on_connected: Callable[[Session], Awaitable[None]] = async_noop
    on_disconnected: Callable[[Session], Awaitable[None]] = async_noop
    on_error: Callable[[Session, str, int], Awaitable[None]] = async_noop
    on_audio_in: Callable[[Session, AudioData], None] = lambda _session, _audio: None
    on_stream_received: Callable[[Session, Stream], Awaitable[None]] = async_noop
    on_stream_dropped: Callable[[Session, Stream], Awaitable[None]] = async_noop
    on_subscriber_connected: Callable[[Subscriber], Awaitable[None]] = async_noop
    on_subscriber_disconnected: Callable[[Subscriber], Awaitable[None]] = async_noop


@dataclass
class VonageClientParams:
    """Parameters for the Vonage client."""

    audio_sample_rate: int = 48000
    audio_channels: int = 2
    enable_migration: bool = False


class VonageClient:
    """Client for managing a Vonage session."""

    def __init__(
        self,
        application_id: str,
        session_id: str,
        token: str,
        params: VonageClientParams,
        publisher_settings: Optional[PublisherSettings] = None,
    ):
        """Initialize the Vonage client with session string and parameters."""
        self._client = vonage_video.VonageVideoClient()
        self._application_id: str = application_id
        self._session_id: str = session_id
        self._token: str = token
        self._params = params
        self._connected: bool = False
        self._connection_counter: int = 0
        self._listener_id_gen: itertools.count = itertools.count()
        self._listeners: map[int, VonageClientListener] = {}
        self._publish_ready: Optional[asyncio.Future] = None
        self._publisher_settings: Optional[PublisherSettings] = publisher_settings
        self._publisher: Optional[Publisher] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session = Session(id=session_id)

    def get_params(self) -> VonageClientParams:
        """Return the parameters of the Vonage client."""
        return self._params

    def add_listener(self, listener: VonageClientListener) -> int:
        """Add a listener to the Vonage client."""
        listener_id = next(self._listener_id_gen)
        self._listeners[listener_id] = listener
        return listener_id

    def remove_listener(self, listener_id: int):
        """Remove a listener from the Vonage client."""
        self._listeners.pop(listener_id, None)

    async def connect(self, listener: VonageClientListener) -> int:
        """Connect to the Vonage session."""
        logger.info(f"Connecting with session string {self._session_id}")

        listener_id: int = self.add_listener(listener)
        if self._connected:
            logger.info(f"Already connected to {self._session_id}")

            # if we've already connected refcount the times we've connected
            self._connection_counter += 1
            await listener.on_connected(self._session)
            return listener_id

        if self._publish_ready is not None:
            logger.info(f"Already connecting to {self._session_id}")

            # if we already connecting, await for the publish ready event
            await self._publish_ready
            return listener_id

        if self._publisher_settings:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self._publish_ready: asyncio.Future = loop.create_future()

        if not self._client.connect(
            application_id=self._application_id,
            session_id=self._session_id,
            token=self._token,
            session_settings=SessionSettings(
                audio=SessionAudioSettings(
                    sample_rate=self._params.audio_sample_rate,
                    number_of_channels=self._params.audio_channels,
                ),
                enable_migration=self._params.enable_migration,
                logging=LoggingSettings(level="INFO"),
            ),
            on_error_cb=self._on_session_error_cb,
            on_connected_cb=self._on_session_connected_cb,
            on_disconnected_cb=self._on_session_disconnected_cb,
            on_stream_received_cb=self._on_stream_received_cb,
            on_stream_dropped_cb=self._on_stream_dropped_cb,
            on_audio_data_cb=self._on_session_audio_data_cb,
            on_ready_for_audio_cb=self._on_session_ready_for_audio_cb,
        ):
            logger.error(f"Could not connect to {self._session_id}")
            raise VonageException("Could not connect to session")

        logger.info(f"Connected to {self._session_id}")

        if self._publish_ready:
            await self._publish_ready

        self._connected = True
        await self._on_session_connected()
        return listener_id

    async def disconnect(self, listener_id: int):
        """Disconnect from the Vonage session."""
        self._connection_counter -= 1
        if not self._connected or self._connection_counter != 0:
            logger.info(f"Already disconnected from {self._session_id}")
            return

        logger.info(f"Disconnecting from {self._session_id}")

        if self._publisher:
            self._client.unpublish()
            self._publisher = None

        self._client.disconnect()

        for listener in self._listeners.values():
            await listener.on_disconnected(self._session)

        self._listeners.pop(listener_id, None)

        logger.info(f"Disconnected from {self._session_id}")

    async def write_audio(self, raw_audio_frame: bytes):
        """Write audio data to the Vonage session."""
        frame_count = len(raw_audio_frame) // (self._params.audio_channels * 2)
        self._client.inject_audio(
            AudioData(
                sample_buffer=memoryview(raw_audio_frame).cast("h"),
                number_of_frames=frame_count,
                number_of_channels=self._params.audio_channels,
                sample_rate=self._params.audio_sample_rate,
            )
        )

    async def _on_session_connected(self):
        for listener in self._listeners.values():
            await listener.on_connected(self._session)

    def _on_session_ready_for_audio_cb(self, session: Session):
        logger.info(f"Session {session.id} ready to publish")
        if self._publish_ready:
            future = self._publish_ready
            self._publish_ready = None
            self._loop.call_soon_threadsafe(future.set_result, None)

    def _on_session_error_cb(self, session: Session, description: str, code: int):
        logger.warning(f"Session error {session.id} code={code} description={description}")
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_session_error_async_cb(session, description, code))
        )

    async def _on_session_error_async_cb(self, session: Session, description: str, code: int):
        for listener in self._listeners.values():
            await listener.on_error(session.id, description, code)

    def _on_session_connected_cb(self, session: Session):
        logger.info(f"Session connected {session.id}")
        self._session = session
        self._client.publish(
            settings=self._publisher_settings,
            on_error_cb=self._on_publisher_error_cb,
            on_stream_created_cb=self._on_publisher_stream_created_cb,
            on_stream_destroyed_cb=self._on_publisher_stream_destroyed_cb,
        )

    def _on_session_disconnected_cb(self, session: Session):
        logger.info(f"Session disconnected {session.id}")
        self._connected = False

    def _on_publisher_error_cb(self, publisher: Publisher, description: str, code: int):
        logger.warning(
            f"Publisher error session={self._session_id} publisher={publisher.stream.id} "
            f"code={code} description={description}"
        )

    def _on_publisher_stream_created_cb(self, publisher: Publisher, stream: Stream):
        logger.info(
            f"Publisher stream created session={self._session_id} publisher={publisher.stream.id}"
        )
        self._publisher = publisher

    def _on_publisher_stream_destroyed_cb(self, publisher: Publisher, stream: Stream):
        logger.info(
            f"Publisher stream destroyed session={self._session_id} publisher={publisher.stream.id}"
        )

    def _on_session_audio_data_cb(self, session: Session, audio_data: AudioData):
        for listener in self._listeners.values():
            if listener.on_audio_in:
                listener.on_audio_in(session, audio_data)

    def _on_stream_received_cb(self, session: Session, stream: Stream):
        logger.info(f"Stream received session={session.id} stream={stream.id}")
        self._client.subscribe(
            stream,
            on_error_cb=self._on_subscriber_error_cb,
            on_connected_cb=self._on_subscriber_connected_cb,
            on_disconnected_cb=self._on_subscriber_disconnected_cb,
        )
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_stream_received_async_cb(session, stream))
        )

    async def _on_stream_received_async_cb(self, session: Session, stream: Stream):
        for listener in self._listeners.values():
            await listener.on_stream_received(session, stream)

    def _on_stream_dropped_cb(self, session: Session, stream: Stream):
        logger.info(f"Stream dropped session={session.id} stream={stream.id}")
        self._client.unsubscribe(stream)
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_stream_dropped_async_cb(session, stream))
        )

    async def _on_stream_dropped_async_cb(self, session: Session, stream: Stream):
        for listener in self._listeners.values():
            await listener.on_stream_dropped(session, stream)

    def _on_subscriber_error_cb(self, subscriber: Subscriber, description: str, code: int):
        logger.info(
            f"Subscriber error session={self._session_id} subscriber={subscriber.stream.id} "
            f"code={code} description={description}"
        )

    def _on_subscriber_connected_cb(self, subscriber: Subscriber):
        logger.info(
            f"Subscriber connected session={self._session_id} subscriber={subscriber.stream.id} "
        )
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_subscriber_connected_async_cb(subscriber))
        )

    async def _on_subscriber_connected_async_cb(self, subscriber: Subscriber):
        for listener in self._listeners.values():
            await listener.on_subscriber_connected(subscriber)

    def _on_subscriber_disconnected_cb(self, subscriber: Subscriber):
        logger.info(
            f"Subscriber disconnected session={self._session_id} subscriber={subscriber.stream.id} "
        )
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_subscriber_disconnected_async_cb(subscriber))
        )

    async def _on_subscriber_disconnected_async_cb(self, subscriber: Subscriber):
        for listener in self._listeners.values():
            await listener.on_subscriber_disconnected(subscriber)


class VonageVideoWebrtcInputTransport(BaseInputTransport):
    """Input transport for Vonage, handling audio input from the Vonage session."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage input transport with the client and parameters."""
        super().__init__(params)

        self._initialized: bool = False
        self._sample_rate = 0
        self._client: VonageClient = client
        self._listener_id: Optional[int] = None
        self._resampler = create_default_resampler()

    async def start(self, frame: StartFrame):
        """Start the Vonage input transport."""
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_in_enabled:
            self._listener_id: int = await self._client.connect(
                VonageClientListener(on_audio_in=self._audio_in_cb)
            )

        await self.set_transport_ready(frame)

    def _audio_in_cb(self, _session: Session, audio: AudioData):
        if self._listener_id is not None and self._params.audio_in_enabled:
            bits_per_sample: int = (
                8 * len(audio.sample_buffer) // (audio.number_of_frames * audio.number_of_channels)
            )
            dtype = np.int16 if bits_per_sample == 16 else np.uint8
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
        if self._listener_id is not None and self._params.audio_in_enabled:
            listener_id, self._listener_id = self._listener_id, None
            await self._client.disconnect(listener_id)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Vonage input transport."""
        await super().cancel(frame)
        if self._listener_id is not None and self._params.audio_in_enabled:
            listener_id, self._listener_id = self._listener_id, None
            await self._client.disconnect(listener_id)


class VonageVideoWebrtcOutputTransport(BaseOutputTransport):
    """Output transport for Vonage, handling audio output to the Vonage session."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage output transport with the client and parameters."""
        super().__init__(params)
        self._initialized: bool = False
        self._resampler = create_default_resampler()
        self._client = client
        self._listener_id: Optional[int] = None

    async def start(self, frame: StartFrame):
        """Start the Vonage output transport."""
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_out_enabled:
            self._listener_id: int = await self._client.connect(VonageClientListener())

        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the Vonage session."""
        if self._listener_id is not None and self._params.audio_out_enabled:
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
        if self._listener_id is not None and self._params.audio_out_enabled:
            listener_id, self._listener_id = self._listener_id, None
            await self._client.disconnect(listener_id)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Vonage output transport."""
        await super().cancel(frame)
        if self._listener_id is not None and self._params.audio_out_enabled:
            listener_id, self._listener_id = self._listener_id, None
            await self._client.disconnect(listener_id)


class VonageVideoWebrtcTransport(BaseTransport):
    """Vonage transport implementation for Pipecat."""

    _params: VonageVideoWebrtcTransportParams

    def __init__(
        self,
        application_id: str,
        session_id: str,
        token: str,
        params: VonageVideoWebrtcTransportParams,
    ):
        """Initialize the Vonage transport with session string and parameters."""
        super().__init__()
        self._params = params

        vonage_params = VonageClientParams(
            audio_sample_rate=params.audio_out_sample_rate or 48000,
            audio_channels=params.audio_out_channels,
            enable_migration=params.session_enable_migration,
        )
        publisher_settings = (
            PublisherSettings(
                name=params.publisher_name,
                audio_settings=PublisherAudioSettings(
                    enable_stereo_mode=params.audio_out_channels == 2,
                    enable_opus_dtx=params.publisher_enable_opus_dtx,
                ),
            )
            if params.audio_out_enabled
            else None
        )
        self._client = VonageClient(
            application_id, session_id, token, vonage_params, publisher_settings
        )

        # Register supported handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_error")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

        self._client.add_listener(
            VonageClientListener(
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_error=self._on_error,
                on_stream_received=self._on_stream_received,
                on_stream_dropped=self._on_stream_dropped,
                on_subscriber_connected=self._on_subscriber_connected,
                on_subscriber_disconnected=self._on_subscriber_disconnected,
            )
        )

        self._input: Optional[VonageVideoWebrtcInputTransport] = None
        self._output: Optional[VonageVideoWebrtcOutputTransport] = None
        self._one_stream_received: bool = False

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

    async def _on_connected(self, session: Session):
        await self._call_event_handler("on_joined", {"sessionId": session.id})

    async def _on_disconnected(self, _session_id: Session):
        await self._call_event_handler("on_left")

    async def _on_error(self, _session: Session, description: str, _code: int):
        await self._call_event_handler("on_error", description)

    async def _on_stream_received(self, session: Session, stream: Stream):
        if not self._one_stream_received:
            self._one_stream_received = True
            await self._call_event_handler(
                "on_first_participant_joined", {"sessionId": session.id, "streamId": stream.id}
            )

        await self._call_event_handler(
            "on_participant_joined", {"sessionId": session.id, "streamId": stream.id}
        )

    async def _on_stream_dropped(self, session: Session, stream: Stream):
        await self._call_event_handler(
            "on_participant_left", {"sessionId": session.id, "streamId": stream.id}
        )

    async def _on_subscriber_connected(self, subscriber: Subscriber):
        await self._call_event_handler(
            "on_client_connected", {"subscriberId": subscriber.stream.id}
        )

    async def _on_subscriber_disconnected(self, subscriber: Subscriber):
        await self._call_event_handler(
            "on_client_disconnected", {"subscriberId": subscriber.stream.id}
        )
