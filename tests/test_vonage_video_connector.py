#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

# Mock the vonage_video module since it's not available in test environment
vonage_video_mock = MagicMock()
vonage_video_mock.VonageVideoClient = MagicMock()
vonage_video_mock.models = MagicMock()


# Create mock classes that match the expected interface


@dataclass(eq=True, frozen=True)
class MockAudioData:
    sample_buffer: memoryview
    sample_rate: int
    number_of_channels: int
    number_of_frames: int


@dataclass(eq=True, frozen=True)
class MockSession:
    id: str


@dataclass(eq=True, frozen=True)
class MockConnection:
    id: str
    creation_time: datetime
    data: str = ""


DUMMY_CONNECTION = MockConnection(id="dummy", creation_time=datetime.min)


@dataclass(eq=True, frozen=True)
class MockStream:
    id: str
    connection: MockConnection


@dataclass(eq=True, frozen=True)
class MockPublisher:
    stream: MockStream


@dataclass(eq=True, frozen=True)
class MockSubscriber:
    stream: Optional[MockStream] = None


@dataclass(eq=True, frozen=True)
class MockSessionAudioSettings:
    sample_rate: int = 48000
    number_of_channels: int = 2


@dataclass(eq=True, frozen=True)
class MockVideoResolution:
    width: int = 640
    height: int = 480


@dataclass(eq=True, frozen=True)
class MockSessionVideoPublisherSettings:
    resolution: MockVideoResolution
    fps: int = 30
    format: str = "YUV420P"


@dataclass(eq=True, frozen=True)
class MockSessionAVSettings:
    audio_publisher: Optional[MockSessionAudioSettings] = None
    audio_subscribers_mix: Optional[MockSessionAudioSettings] = None
    video_publisher: Optional[MockSessionVideoPublisherSettings] = None


@dataclass(eq=True, frozen=True)
class MockLoggingSettings:
    level: str = "WARN"


@dataclass(eq=True, frozen=True)
class MockSessionSettings:
    enable_migration: bool = False
    av: Optional[MockSessionAVSettings] = None
    logging: Optional[MockLoggingSettings] = None


@dataclass(eq=True, frozen=True)
class MockPublisherAudioSettings:
    enable_stereo_mode: bool = True
    enable_opus_dtx: bool = False


@dataclass(eq=True, frozen=True)
class MockPublisherSettings:
    name: str
    has_audio: bool
    has_video: bool
    audio_settings: Optional[MockPublisherAudioSettings] = None


@dataclass(eq=True, frozen=True)
class MockVideoFrame:
    frame_buffer: memoryview
    resolution: MockVideoResolution
    format: str = "YUV420P"


@dataclass(eq=True, frozen=True)
class MockSubscriberVideoSettings:
    preferred_resolution: Optional[MockVideoResolution] = None
    preferred_framerate: Optional[int] = None


@dataclass(eq=True, frozen=True)
class MockSubscriberSettings:
    subscribe_to_audio: bool = True
    subscribe_to_video: bool = True
    video_settings: Optional[MockSubscriberVideoSettings] = None


# Set up the mock module structure
vonage_video_mock.models.AudioData = MockAudioData
vonage_video_mock.models.Session = MockSession
vonage_video_mock.models.Connection = MockConnection
vonage_video_mock.models.Stream = MockStream
vonage_video_mock.models.Publisher = MockPublisher
vonage_video_mock.models.Subscriber = MockSubscriber
vonage_video_mock.models.LoggingSettings = MockLoggingSettings
vonage_video_mock.models.SessionAVSettings = MockSessionAVSettings
vonage_video_mock.models.SessionSettings = MockSessionSettings
vonage_video_mock.models.SessionAudioSettings = MockSessionAudioSettings
vonage_video_mock.models.PublisherAudioSettings = MockPublisherAudioSettings
vonage_video_mock.models.PublisherSettings = MockPublisherSettings
vonage_video_mock.models.SessionVideoPublisherSettings = MockSessionVideoPublisherSettings
vonage_video_mock.models.SubscriberSettings = MockSubscriberSettings
vonage_video_mock.models.SubscriberVideoSettings = MockSubscriberVideoSettings
vonage_video_mock.models.VideoResolution = MockVideoResolution
vonage_video_mock.models.VideoFrame = MockVideoFrame

# Mock the module in sys.modules so imports work
sys.modules["vonage_video_connector"] = vonage_video_mock
sys.modules["vonage_video_connector.models"] = vonage_video_mock.models

# Now we can import the transport classes since the vonage_video module is mocked
from pipecat.transports.vonage.client import (
    VonageClient,
    VonageClientListener,
)
from pipecat.transports.vonage.utils import (
    AudioProps,
    ImageFormat,
    check_audio_data,
    image_colorspace_conversion,
    process_audio,
    process_audio_channels,
)
from pipecat.transports.vonage.video_connector import (
    SubscribeSettings,
    VonageException,
    VonageVideoConnectorInputTransport,
    VonageVideoConnectorOutputTransport,
    VonageVideoConnectorTransport,
    VonageVideoConnectorTransportParams,
)


@dataclass(frozen=True)
class SubscriberCallbacks:
    on_error_cb: Callable[[MockSubscriber, str, int], None]
    on_connected_cb: Callable[[MockSubscriber], None]
    on_disconnected_cb: Callable[[MockSubscriber], None]
    on_render_frame_cb: Callable[[MockSubscriber, MockVideoFrame], None]


@dataclass(frozen=True)
class ConnectCallbacks:
    on_error_cb: Callable[[MockSession, str, int], None]
    on_disconnected_cb: Callable[[MockSession], None]
    on_ready_for_audio_cb: Callable[[MockSession], None]


class TestVonageVideoConnectorTransport:
    """Test cases for Vonage Video Connector transport classes."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.VonageClient = VonageClient
        self.VonageClientListener = VonageClientListener
        self.VonageVideoConnectorInputTransport = VonageVideoConnectorInputTransport
        self.VonageVideoConnectorOutputTransport = VonageVideoConnectorOutputTransport
        self.VonageVideoConnectorTransport = VonageVideoConnectorTransport
        self.VonageVideoConnectorTransportParams = VonageVideoConnectorTransportParams

        # Mock client instance
        self.mock_client_instance = Mock()
        vonage_video_mock.VonageVideoClient.return_value = self.mock_client_instance

        # Common test data
        self.application_id = "test-app-id"
        self.session_id = "test-session-id"
        self.token = "test-token"
        self._frame_processor_setup: Optional[FrameProcessorSetup] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # subscriber state
        self._connect_callbacks: Optional[ConnectCallbacks] = None
        self._subscriber_callbacks: dict[str, SubscriberCallbacks] = {}

    def _get_frame_processor_setup(self) -> FrameProcessorSetup:
        if self._frame_processor_setup is not None:
            return self._frame_processor_setup

        clock: SystemClock = SystemClock()  # type: ignore[no-untyped-call]
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        self._frame_processor_setup = FrameProcessorSetup(clock=clock, task_manager=task_manager)
        return self._frame_processor_setup

    async def _wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: timedelta = timedelta(seconds=1),
        check_interval: timedelta = timedelta(milliseconds=10),
    ) -> None:
        """Wait for a condition to become true with timeout.

        Args:
            condition: Callable that returns True when condition is met.
            timeout: Maximum time to wait.
            check_interval: How often to check the condition.

        Raises:
            asyncio.TimeoutError: If condition is not met within timeout.
        """
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout.total_seconds()
        check_interval_seconds = check_interval.total_seconds()

        while not condition():
            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                raise asyncio.TimeoutError(f"Condition not met within {timeout}")
            await asyncio.sleep(check_interval_seconds)

    def test_vonage_client_listener_defaults(self) -> None:
        """Test VonageClientListener default values."""
        listener = self.VonageClientListener()
        assert listener.on_connected is not None
        assert listener.on_disconnected is not None
        assert listener.on_error is not None
        assert listener.on_audio_in is not None
        assert listener.on_stream_received is not None
        assert listener.on_stream_dropped is not None
        assert listener.on_subscriber_connected is not None
        assert listener.on_subscriber_disconnected is not None

    def test_vonage_transport_params_defaults(self) -> None:
        """Test VonageVideoConnectorTransportParams default values."""
        params = self.VonageVideoConnectorTransportParams()
        assert params.publisher_name == "Bot"
        assert params.publisher_enable_opus_dtx is False
        assert params.session_enable_migration is False

    def test_vonage_client_initialization(self) -> None:
        """Test VonageClient initialization."""
        # Reset the mock for this specific test
        vonage_video_mock.VonageVideoClient.reset_mock()

        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        assert client._application_id == self.application_id
        assert client._session_id == self.session_id
        assert client._token == self.token
        assert client._params == params
        assert client._connected is False
        assert client._connection_counter == 0
        vonage_video_mock.VonageVideoClient.assert_called_once()

        # check getting the event loop before setup raises error
        with pytest.raises(Exception) as exc_info:
            _ = client._get_event_loop()

        assert "missing task manager" in str(exc_info.value)

        # check pushing events before setup raises error
        async def mock_coro() -> None:
            pass

        mock_task = mock_coro()
        with pytest.raises(Exception) as exc_info:
            client._sdk_event_cb_to_loop(mock_task)

        mock_task.close()
        assert "missing event queue" in str(exc_info.value)

    def test_vonage_client_add_remove_listener(self) -> None:
        """Test adding and removing listeners from VonageClient."""
        params = self.VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        listener = self.VonageClientListener()
        listener_id = client.add_listener(listener)

        assert isinstance(listener_id, int)
        assert listener_id in client._listeners
        assert client._listeners[listener_id] == listener

        client.remove_listener(listener_id)
        assert listener_id not in client._listeners

    def _setup_audio_ready_callback(self, client: VonageClient, call_ready_for_audio: bool) -> None:
        """Helper to set up the audio ready callback."""

        def connect_side_effect(
            *_: Any,
            on_error_cb: Callable[[MockSession, str, int], None],
            on_disconnected_cb: Callable[[MockSession], None],
            on_ready_for_audio_cb: Callable[[MockSession], None],
            **__: Any,
        ) -> bool:
            if call_ready_for_audio:
                on_ready_for_audio_cb(vonage_video_mock.models.Session(id="session"))

            self._connect_callbacks = ConnectCallbacks(
                on_error_cb=on_error_cb,
                on_disconnected_cb=on_disconnected_cb,
                on_ready_for_audio_cb=on_ready_for_audio_cb,
            )
            return True

        self.mock_client_instance.connect = MagicMock(side_effect=connect_side_effect)

    def _setup_subscriber_callbacks(self, client: VonageClient) -> None:
        def subscribe_side_effect(
            stream: MockStream,
            on_error_cb: Callable[[MockSubscriber, str, int], None],
            on_connected_cb: Callable[[MockSubscriber], None],
            on_disconnected_cb: Callable[[MockSubscriber], None],
            on_render_frame_cb: Callable[[MockSubscriber, MockVideoFrame], None],
            **__: Any,
        ) -> bool:
            self._subscriber_callbacks[stream.id] = SubscriberCallbacks(
                on_error_cb=on_error_cb,
                on_connected_cb=on_connected_cb,
                on_disconnected_cb=on_disconnected_cb,
                on_render_frame_cb=on_render_frame_cb,
            )
            return True

        self.mock_client_instance.subscribe = MagicMock(side_effect=subscribe_side_effect)

    async def _subscribe_n_handle_callbacks(
        self,
        client: VonageClient,
        stream_id: str,
        params: SubscribeSettings,
        callback: Callable[[SubscriberCallbacks], None],
    ) -> None:
        task = asyncio.create_task(client.subscribe_to_stream(stream_id, params))
        await self._wait_for_condition(
            lambda: stream_id in self._subscriber_callbacks,
            timeout=timedelta(seconds=2),
        )
        callback(self._subscriber_callbacks[stream_id])
        await task

    async def _create_client(
        self,
        params: Optional[VonageVideoConnectorTransportParams] = None,
        setup_connect_mock: bool = True,
    ) -> VonageClient:
        params = params or VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        if setup_connect_mock:
            self._setup_audio_ready_callback(client, call_ready_for_audio=True)

        self._setup_subscriber_callbacks(client)

        await client.setup(self._get_frame_processor_setup())

        return client

    async def _run_in_thread(self, callback: Callable[[], Any]) -> Any:
        """Helper to run a coroutine in a separate thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, callback)

    async def _wait_client_async_tasks(self, client: VonageClient) -> None:
        """Helper to wait for all async tasks in the client to complete."""
        # Wait for any pending tasks in the client's task manager
        drain_event = asyncio.Event()

        async def set_event_when_drained() -> None:
            # Wait for all queues to be joined (all tasks processed)
            if client._event_queue:
                await client._event_queue.join()
            if client._audio_queue:
                await client._audio_queue.join()
            if client._video_queue:
                await client._video_queue.join()
            drain_event.set()

        # Schedule the drain coroutine in the event loop
        asyncio.create_task(set_event_when_drained())
        await drain_event.wait()

    async def _create_output_transport(
        self, params: VonageVideoConnectorTransportParams
    ) -> VonageVideoConnectorOutputTransport:
        client = self.VonageClient(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        transport = self.VonageVideoConnectorOutputTransport(client, params)
        await transport.setup(self._get_frame_processor_setup())

        return transport

    async def _create_input_transport(
        self, params: VonageVideoConnectorTransportParams
    ) -> VonageVideoConnectorInputTransport:
        client = self.VonageClient(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        transport = self.VonageVideoConnectorInputTransport(client, params)
        await transport.setup(self._get_frame_processor_setup())

        return transport

    async def _create_transport(
        self, params: VonageVideoConnectorTransportParams
    ) -> VonageVideoConnectorTransport:
        transport = VonageVideoConnectorTransport(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        await transport.input().setup(self._get_frame_processor_setup())
        await transport.output().setup(self._get_frame_processor_setup())

        return transport

    @pytest.mark.asyncio
    async def test_vonage_client_setup_n_cleanup(self) -> None:
        """Test VonageClient setup and cleanup methods."""
        params = self.VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Before setup, task manager and queues should be None
        assert client._task_manager is None
        assert client._event_queue is None
        assert client._event_task is None
        assert client._audio_queue is None
        assert client._audio_task is None
        assert client._video_queue is None
        assert client._video_task is None

        # Setup the client
        setup = self._get_frame_processor_setup()
        await client.setup(setup)

        # Mock connection
        self.mock_client_instance.connect.return_value = True
        client._connected = True
        client._connection_counter = 1

        # After setup, task manager and queues should be initialized
        assert client._task_manager is not None
        assert client._task_manager == setup.task_manager
        assert client._event_queue is not None
        assert client._event_task is not None
        assert client._audio_queue is not None
        assert client._audio_task is not None
        assert client._video_queue is not None
        assert client._video_task is not None

        # Test that calling setup again doesn't recreate the task manager
        old_task_manager = client._task_manager
        old_event_queue = client._event_queue
        old_event_task = client._event_task
        await client.setup(setup)
        assert client._task_manager == old_task_manager
        assert client._event_queue == old_event_queue
        assert client._event_task == old_event_task

        # Test cleanup without being connected
        await client.cleanup()

        # After cleanup, tasks should be cancelled
        assert client._event_task is None
        assert client._audio_task is None
        assert client._video_task is None

        # Verify disconnect was called
        self.mock_client_instance.disconnect.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_audio, has_video",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    async def test_vonage_client_connect_first_time(self, has_audio: bool, has_video: bool) -> None:
        """Test VonageClient connect method for first connection."""
        params = self.VonageVideoConnectorTransportParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_channels = 1 if has_video else 2
        params.audio_out_channels = 2 if has_video else 1
        params.audio_in_sample_rate = 44100 if has_video else 22050
        params.audio_out_sample_rate = 22050 if has_video else 44100
        params.session_enable_migration = has_video
        params.video_out_color_format = "YCbCr" if has_audio else "RGB"
        params.video_out_framerate = 30 if has_audio else 15
        params.video_out_width = 1280 if has_audio else 640
        params.video_out_height = 720 if has_audio else 480
        params.audio_in_enabled = has_audio
        params.audio_out_enabled = has_audio
        params.video_in_enabled = has_video
        params.video_out_enabled = has_video
        params.video_connector_log_level = "WARN" if has_audio else "ERROR"

        client = await self._create_client(params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        listener = self.VonageClientListener()
        listener.on_connected = AsyncMock()
        # only set this callback if we have audio enabled
        self._setup_audio_ready_callback(client, has_audio)
        listener_id = client.add_listener(listener)
        await client.connect()

        assert isinstance(listener_id, int)
        self.mock_client_instance.connect.assert_called_once()

        # Verify connect was called with correct parameters
        call_args = self.mock_client_instance.connect.call_args
        assert call_args[1]["application_id"] == self.application_id
        assert call_args[1]["session_id"] == self.session_id
        assert call_args[1]["token"] == self.token
        assert call_args[1]["session_settings"] == MockSessionSettings(
            av=MockSessionAVSettings(
                audio_publisher=MockSessionAudioSettings(
                    sample_rate=params.audio_out_sample_rate,
                    number_of_channels=params.audio_out_channels,
                ),
                audio_subscribers_mix=MockSessionAudioSettings(
                    sample_rate=params.audio_in_sample_rate,
                    number_of_channels=params.audio_in_channels,
                ),
                video_publisher=MockSessionVideoPublisherSettings(
                    resolution=MockVideoResolution(
                        width=params.video_out_width,
                        height=params.video_out_height,
                    ),
                    fps=params.video_out_framerate,
                    format=client._video_out_color_format_vonage,
                ),
            ),
            enable_migration=params.session_enable_migration,
            logging=MockLoggingSettings(level=params.video_connector_log_level),
        )
        assert self._connect_callbacks is not None
        assert call_args[1]["on_audio_data_cb"] == client._on_session_audio_data_cb
        assert call_args[1]["on_error_cb"] == self._connect_callbacks.on_error_cb
        assert call_args[1]["on_connected_cb"] == client._on_session_connected_cb
        assert call_args[1]["on_disconnected_cb"] == self._connect_callbacks.on_disconnected_cb
        assert (
            call_args[1]["on_ready_for_audio_cb"] == self._connect_callbacks.on_ready_for_audio_cb
        )
        assert call_args[1]["on_stream_received_cb"] == client._on_stream_received_cb
        assert call_args[1]["on_stream_dropped_cb"] == client._on_stream_dropped_cb

        listener.on_connected.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_audio, has_video",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    async def test_vonage_client_publish_after_connect(
        self, has_audio: bool, has_video: bool
    ) -> None:
        """Test VonageClient publishes after being connected method for first connection."""
        params = self.VonageVideoConnectorTransportParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_enabled = has_audio
        params.audio_out_enabled = has_audio
        params.video_in_enabled = has_video
        params.video_out_enabled = has_video
        params.audio_out_channels = 2 if has_video else 1
        params.publisher_enable_opus_dtx = not has_video
        params.publisher_name = "test-audio" if has_audio else "test-video"

        client = await self._create_client(params)
        await client.connect()

        self.mock_client_instance.connect.assert_called_once()

        # trigger the _on_session_connected_cb to simulate being connected
        await self._run_in_thread(
            lambda: client._on_session_connected_cb(vonage_video_mock.models.Session(id="session"))
        )
        await self._wait_client_async_tasks(client)

        # if no audio and no video, publish should not be called
        if not has_audio and not has_video:
            self.mock_client_instance.publish.assert_not_called()
            return

        # Verify publish was called with correct parameters
        self.mock_client_instance.publish.assert_called_once()
        call_args = self.mock_client_instance.publish.call_args
        assert call_args[1]["settings"] == MockPublisherSettings(
            name=params.publisher_name,
            audio_settings=MockPublisherAudioSettings(
                enable_stereo_mode=params.audio_out_channels == 2,
                enable_opus_dtx=params.publisher_enable_opus_dtx,
            ),
            has_audio=has_audio,
            has_video=has_video,
        )
        assert call_args[1]["on_error_cb"] == client._on_publisher_error_cb
        assert call_args[1]["on_stream_created_cb"] == client._on_publisher_stream_created_cb
        assert call_args[1]["on_stream_destroyed_cb"] == client._on_publisher_stream_destroyed_cb

    @pytest.mark.asyncio
    async def test_vonage_client_connect_already_connected(self) -> None:
        """Test VonageClient connect when already connected."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        # add some listeners, tests multiple listeners are notified, test no notifiaction after removal too
        listener1 = self.VonageClientListener()
        listener1.on_connected = AsyncMock()
        client.add_listener(listener1)
        listener2 = self.VonageClientListener()
        listener2.on_connected = AsyncMock()
        client.add_listener(listener2)
        listener3 = self.VonageClientListener()
        listener3.on_connected = AsyncMock()
        listener_id3 = client.add_listener(listener3)
        client.remove_listener(listener_id3)

        # First connection, connection is performed
        await client.connect()
        self.mock_client_instance.connect.assert_called_once()
        listener1.on_connected.assert_called_once()
        listener2.on_connected.assert_called_once()

        # Second connection, should not trigger a new connect call or raised any events
        await client.connect()
        self.mock_client_instance.connect.assert_called_once()
        listener1.on_connected.assert_called_once()
        listener2.on_connected.assert_called_once()

        # the removed listener should not have received any events
        listener3.on_connected.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_connect_while_disconnecting(self) -> None:
        """Test VonageClient waits for disconnect to complete before connecting."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        self.mock_client_instance.disconnect = MagicMock()

        # Simulate disconnect in progress
        disconnect_future = asyncio.get_running_loop().create_future()
        client._disconnecting_future = disconnect_future

        # Start connect task - it should block waiting for disconnect
        connect_task = asyncio.create_task(client.connect())

        # Give control to the event loop to let connect task start
        await asyncio.sleep(0.2)

        self.mock_client_instance.connect.assert_not_called()

        # Resolve the disconnect future to unblock connect
        disconnect_future.set_result(None)

        # Wait for connect to complete
        await connect_task

        self.mock_client_instance.connect.assert_called_once()

        # Verify client state
        assert client._connected is True
        assert client._connection_counter == 1

    @pytest.mark.asyncio
    async def test_vonage_client_timeout_while_connecting(self) -> None:
        """Test VonageClient handles timeout during connection."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params, setup_connect_mock=False)

        # Create an event that will block but can be interrupted
        stop_event = threading.Event()

        # Mock the SDK connect method to block until interrupted
        def connect_blocks_forever(*args: Any, **kwargs: Any) -> bool:
            stop_event.wait(timeout=10)  # Wait max 10 seconds but can be interrupted
            return True

        self.mock_client_instance.connect.side_effect = connect_blocks_forever

        try:
            # Patch the timeout to be very short for fast test execution
            with patch(
                "pipecat.transports.vonage.client.VIDEO_CONNECTOR_TIMEOUT",
                timedelta(seconds=0.1),
            ):
                # Attempt to connect, should timeout
                with pytest.raises(asyncio.TimeoutError):
                    await client.connect()

                # Verify client state after timeout
                assert client._connected is False
                assert client._connection_counter == 0
                assert client._connecting_future is None
        finally:
            # Stop the blocking thread
            stop_event.set()

    @pytest.mark.asyncio
    async def test_vonage_client_concurrent_connects(self) -> None:
        """Test VonageClient concurrent connects."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        # Mock the connect method to return True and store the callback
        connecting_future: asyncio.Future[Callable[[Any], None]] = (
            asyncio.get_running_loop().create_future()
        )

        def connect_side_effect(
            *_: Any, on_ready_for_audio_cb: Optional[Callable[[Any], None]] = None, **__: Any
        ) -> bool:
            assert on_ready_for_audio_cb is not None
            connecting_future.set_result(on_ready_for_audio_cb)
            return True

        self.mock_client_instance.connect = MagicMock(side_effect=connect_side_effect)

        # create a listener
        listener = self.VonageClientListener()
        listener.on_connected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect1_task = asyncio.create_task(client.connect())
        connect2_task = asyncio.create_task(client.connect())

        audio_ready_cb = await connecting_future

        # Now both connects are waiting on the same promise, we can set it to complete
        audio_ready_cb(vonage_video_mock.models.Session(id="session"))

        # await for the connections to now complete
        await asyncio.gather(connect1_task, connect2_task)

        # SDK connect should only be called once
        self.mock_client_instance.connect.assert_called_once()
        listener.on_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_connect_failure(self) -> None:
        """Test VonageClient connect method when connection fails."""
        client = await self._create_client(setup_connect_mock=False)

        # Mock the connect method to return False
        self.mock_client_instance.connect.return_value = False

        with pytest.raises(Exception) as exc_info:
            await client.connect()

        assert "Could not connect to session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect_before_connecting(self) -> None:
        """Test VonageClient disconnect method before connecting."""
        client = await self._create_client()

        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        await client.disconnect()

        self.mock_client_instance.disconnect.assert_not_called()
        listener.on_disconnected.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect(self) -> None:
        """Test VonageClient disconnect method."""
        client = await self._create_client()

        # create a listener
        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect_promise1 = client.connect()
        connect_promise2 = client.connect()

        # await for the connections to now complete
        await asyncio.gather(connect_promise1, connect_promise2)

        # Add some items to the queues before disconnect
        assert client._event_queue is not None
        assert client._audio_queue is not None
        assert client._video_queue is not None

        async def mock_event_task() -> None:
            pass

        async def mock_audio_task() -> None:
            pass

        async def mock_video_task() -> None:
            pass

        await client._event_queue.put(mock_event_task())
        await client._audio_queue.put(mock_audio_task())
        await client._video_queue.put(mock_video_task())

        # Verify queues have items
        assert client._event_queue.qsize() == 1
        assert client._audio_queue.qsize() == 1
        assert client._video_queue.qsize() == 1

        # Mock the client's clear_media_buffers method
        self.mock_client_instance.clear_media_buffers = MagicMock()

        # send the first disconnect call, we should still be connected
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_not_called()
        self.mock_client_instance.clear_media_buffers.assert_not_called()
        listener.on_disconnected.assert_not_called()

        # Queues should still have items since we didn't actually disconnect
        assert client._event_queue.qsize() == 1
        assert client._audio_queue.qsize() == 1
        assert client._video_queue.qsize() == 1

        # check the second disconnect now disconnects for real
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        self.mock_client_instance.clear_media_buffers.assert_called_once()
        listener.on_disconnected.assert_called_once()

        # Verify queues are now empty after disconnect
        assert client._event_queue.qsize() == 0
        assert client._audio_queue.qsize() == 0
        assert client._video_queue.qsize() == 0

        # an extra disconnect should not do anything
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        self.mock_client_instance.clear_media_buffers.assert_called_once()
        listener.on_disconnected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect_while_connecting(self) -> None:
        """Test VonageClient waits for connect to complete before disconnecting."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        client._connected = True
        client._connection_counter = 1
        self.mock_client_instance.connect = MagicMock()

        # Simulate connect in progress
        connect_future = asyncio.get_running_loop().create_future()
        client._connecting_future = connect_future

        # Start disconnect task - it should block waiting for disconnect
        disconnect_task = asyncio.create_task(client.disconnect())

        # Give control to the event loop to let disconnect task start
        await asyncio.sleep(0.2)

        self.mock_client_instance.disconnect.assert_not_called()

        # Resolve the disconnect future to unblock connect
        connect_future.set_result(None)

        # Wait for connect to complete
        await disconnect_task

        self.mock_client_instance.disconnect.assert_called_once()

        # Verify client state
        assert client._connected is False
        assert client._connection_counter == 0

    @pytest.mark.asyncio
    async def test_vonage_client_timeout_while_disconnecting(self) -> None:
        """Test VonageClient handles timeout during disconnection."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params, setup_connect_mock=False)

        await client.connect()
        assert client._connection_counter == 1

        # Create an event that will block but can be interrupted
        stop_event = threading.Event()

        # Mock the SDK disconnect method to block until interrupted
        def disconnect_blocks_forever(*args: Any, **kwargs: Any) -> bool:
            stop_event.wait(timeout=10)  # Wait max 10 seconds but can be interrupted
            return True

        self.mock_client_instance.disconnect.side_effect = disconnect_blocks_forever
        try:
            # Patch the timeout to be very short for fast test execution
            with patch(
                "pipecat.transports.vonage.client.VIDEO_CONNECTOR_TIMEOUT",
                timedelta(seconds=0.1),
            ):
                # Attempt to connect, should timeout
                with pytest.raises(asyncio.TimeoutError):
                    await client.disconnect()

                # Verify client state after timeout
                assert client._connected is True
                assert client._connection_counter == 1
                assert client._disconnecting_future is None
        finally:
            # Stop the blocking thread
            stop_event.set()

    @pytest.mark.asyncio
    async def test_vonage_client_clear_media_buffers(self) -> None:
        """Test VonageClient clear_media_buffers method."""
        params = self.VonageVideoConnectorTransportParams(
            audio_out_channels=2, audio_out_sample_rate=48000
        )
        client = await self._create_client(params)

        # Add some items to the audio and video queues
        assert client._audio_queue is not None
        assert client._video_queue is not None

        # Create mock coroutines to add to queues
        async def mock_audio_task() -> None:
            pass

        async def mock_video_task() -> None:
            pass

        # Put some items in the queues
        await client._audio_queue.put(mock_audio_task())
        await client._audio_queue.put(mock_audio_task())
        await client._video_queue.put(mock_video_task())

        # Verify queues have items
        assert client._audio_queue.qsize() == 2
        assert client._video_queue.qsize() == 1

        # Mock the client's clear_media_buffers method
        self.mock_client_instance.clear_media_buffers = MagicMock()

        # Clear the buffers
        client.clear_media_buffers()

        # Verify queues are now empty
        assert client._audio_queue.qsize() == 0
        assert client._video_queue.qsize() == 0

        # Verify the SDK client's clear_media_buffers was called
        self.mock_client_instance.clear_media_buffers.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.VIDEO_QUEUE_MAXSIZE", 1)
    async def test_vonage_client_sdk_cb_to_loop_full_queue(self) -> None:
        """Test VonageClient SDK callback to loop filling up the queue."""
        params = self.VonageVideoConnectorTransportParams()
        client = await self._create_client(params)

        # Ensure the loop thread ID is set
        assert client._video_queue is not None
        assert client._loop_thread_id == threading.get_ident()

        # Create a mock coroutine to queue
        async def mock_task() -> None:
            pass

        # Fill queue to max size
        for _ in range(client._video_queue.maxsize):
            await client._video_queue.put(mock_task())

        # Queue should be full
        assert client._video_queue.qsize() == client._video_queue.maxsize
        # This should log an error and drop the event
        async_task = mock_task()
        client._sdk_cb_to_loop("test_event", client._video_queue, async_task)

        # Queue should still be full (no new item added)
        assert client._video_queue.qsize() == client._video_queue.maxsize
        # check the coroutine was closed and hence dropped
        assert inspect.getcoroutinestate(async_task) == "CORO_CLOSED"

        # Clean up the coroutine
        task = await client._video_queue.get()
        task.close()
        client._video_queue.task_done()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_vonage_client_get_audio_with_resampling(self, mock_resampler: MagicMock) -> None:
        """Test VonageClient get_audio method."""
        # Return resampled stereo data
        resampled_data = b"\x07\x06\x05\x04\x03\x02\x01\x00"
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_data)
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageVideoConnectorTransportParams(
            audio_in_channels=1,
            audio_in_sample_rate=48000,
            audio_in_enabled=True,
        )
        client = await self._create_client(params)
        listener = self.VonageClientListener()
        on_audio_in_mock = AsyncMock()
        listener.on_audio_in = on_audio_in_mock
        client.add_listener(listener)

        await client.connect()

        mock_audio_data = vonage_video_mock.models.AudioData(
            sample_buffer=memoryview(b"\x00\x01\x02\x03\x04\x05\x06\x07"),
            number_of_frames=4,
            number_of_channels=1,
            sample_rate=16000,
        )

        session = vonage_video_mock.models.Session(id="test_session")
        client._on_session_audio_data_cb(session, mock_audio_data)
        await self._wait_for_condition(lambda: on_audio_in_mock.call_count > 0)

        listener.on_audio_in.assert_called_once_with(session, ANY)
        frame = listener.on_audio_in.call_args[0][1]
        assert frame.audio == resampled_data
        assert frame.num_frames == 4
        assert frame.sample_rate == 48000
        assert frame.num_channels == 1

    @pytest.mark.asyncio
    async def test_vonage_client_get_video(self) -> None:
        """Test VonageClient get video."""
        pass

    @pytest.mark.asyncio
    async def test_vonage_client_write_audio(self) -> None:
        """Test VonageClient write_audio method."""
        params = self.VonageVideoConnectorTransportParams(
            audio_out_channels=2, audio_out_sample_rate=48000
        )
        client = await self._create_client(params)

        # Create mock audio data
        audio_data = OutputAudioRawFrame(
            audio=b"\x00\x01\x02\x03\x04\x05\x06\x07",
            sample_rate=48000,
            num_channels=2,
        )  # 4 frames of 2-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.add_audio.assert_called_once()
        call_args = self.mock_client_instance.add_audio.call_args[0][0]
        assert call_args.sample_buffer.tobytes() == audio_data.audio
        assert call_args.number_of_frames == 2  # 8 bytes / (2 channels * 2 bytes)
        assert call_args.number_of_channels == 2
        assert call_args.sample_rate == 48000

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_vonage_client_write_audio_with_resampling(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageClient write_audio method."""
        # Return resampled stereo data
        resampled_data = b"\x07\x06\x05\x04\x03\x02\x01\x00"
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_data)
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageVideoConnectorTransportParams(
            audio_out_channels=1, audio_out_sample_rate=16000
        )
        client = await self._create_client(params)

        # Create mock audio data
        audio_data = OutputAudioRawFrame(
            audio=b"\x00\x01\x02\x03\x04\x05\x06\x07",
            sample_rate=48000,
            num_channels=1,
        )  # 4 frames of 1-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.add_audio.assert_called_once()
        call_args = self.mock_client_instance.add_audio.call_args[0][0]
        assert call_args.sample_buffer.tobytes() == resampled_data
        assert call_args.number_of_frames == 4  # 8 bytes / (1 channel * 2 bytes)
        assert call_args.number_of_channels == 1
        assert call_args.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_vonage_client_write_video(self) -> None:
        """Test VonageClient write_video method."""
        params = self.VonageVideoConnectorTransportParams(
            video_out_width=640,
            video_out_height=480,
            video_out_color_format="RGB",
        )
        client = await self._create_client(params)

        # Create a test RGB image (640x480, 3 channels)
        width, height = 640, 480
        # Create RGB data: simple gradient pattern
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100  # R channel
        rgb_image[:, :, 1] = 150  # G channel
        rgb_image[:, :, 2] = 200  # B channel

        rgb_bytes = rgb_image.tobytes()

        # Create OutputImageRawFrame
        frame = OutputImageRawFrame(image=rgb_bytes, size=(width, height), format="RGB")

        # Mock the add_video method
        self.mock_client_instance.add_video = MagicMock(return_value=True)

        result = await client.write_video(frame)

        # Verify add_video was called
        assert result is True
        self.mock_client_instance.add_video.assert_called_once()

        # Get the VideoFrame argument
        call_args = self.mock_client_instance.add_video.call_args[0][0]

        # Verify the resolution
        assert call_args.resolution.width == width
        assert call_args.resolution.height == height

        # Verify the format
        assert call_args.format == "RGB24"

        # Verify BGR conversion happened correctly
        # Convert back from the buffer to verify
        bgr_buffer = bytes(call_args.frame_buffer)
        bgr_image = np.frombuffer(bgr_buffer, dtype=np.uint8).reshape(height, width, 3)

        # Check that RGB was converted to BGR (channels swapped)
        assert bgr_image[0, 0, 0] == 200  # B channel (was R=200 in RGB)
        assert bgr_image[0, 0, 1] == 150  # G channel (unchanged)
        assert bgr_image[0, 0, 2] == 100  # R channel (was B=100 in RGB)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_audio, has_video",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    async def test_vonage_client_subscribe_to_stream(
        self, has_audio: bool, has_video: bool
    ) -> None:
        """Test VonageClient subscribe_to_stream with a stream that exists in session."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        listener = self.VonageClientListener()
        client.add_listener(listener)
        on_subscriber_connected_mock = AsyncMock()
        listener.on_subscriber_connected = on_subscriber_connected_mock
        on_subscriber_disconnected_mock = AsyncMock()
        listener.on_subscriber_disconnected = on_subscriber_disconnected_mock

        await client.connect()

        # Add a stream to the session
        stream = vonage_video_mock.models.Stream(id="test_stream", connection=DUMMY_CONNECTION)
        client._session_streams["test_stream"] = stream

        # Setup subscriber callbacks
        self._setup_subscriber_callbacks(client)

        # Subscribe with audio and video
        subscribe_params = SubscribeSettings(
            subscribe_to_audio=has_audio,
            subscribe_to_video=has_video,
            preferred_resolution=(640, 480) if has_video else None,
            preferred_framerate=30 if has_video else None,
        )

        subscriber = vonage_video_mock.models.Subscriber(stream=stream)
        await self._subscribe_n_handle_callbacks(
            client,
            "test_stream",
            subscribe_params,
            lambda callbacks: callbacks.on_connected_cb(subscriber),
        )
        on_subscriber_connected_mock.assert_called_once_with(subscriber)

        # Verify subscribe was called with correct parameters
        self.mock_client_instance.subscribe.assert_called_once()
        call_args = self.mock_client_instance.subscribe.call_args

        expected_settings = MockSubscriberSettings(
            subscribe_to_audio=has_audio,
            subscribe_to_video=has_video,
            video_settings=MockSubscriberVideoSettings(
                preferred_resolution=MockVideoResolution(
                    width=subscribe_params.preferred_resolution[0],
                    height=subscribe_params.preferred_resolution[1],
                )
                if subscribe_params.preferred_resolution
                else None,
                preferred_framerate=subscribe_params.preferred_framerate,
            ),
        )

        assert call_args[0][0] == stream
        assert call_args[1]["settings"] == expected_settings

        # Verify subscription was stored
        assert "test_stream" in client._session_subscriptions
        assert client._session_subscriptions["test_stream"] == subscribe_params

        # check we can get a disconnect event from the subscriber
        self._subscriber_callbacks["test_stream"].on_disconnected_cb(subscriber)
        await self._wait_for_condition(lambda: on_subscriber_disconnected_mock.call_count > 0)
        on_subscriber_disconnected_mock.assert_called_once_with(subscriber)

    @pytest.mark.asyncio
    async def test_vonage_client_subscribe_to_stream_timeout(self) -> None:
        """Test VonageClient subscribe_to_stream when SDK subscribe times out."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        await client.connect()

        stream = vonage_video_mock.models.Stream(id="fail_stream", connection=DUMMY_CONNECTION)
        client._session_streams["fail_stream"] = stream

        self._setup_subscriber_callbacks(client)

        subscribe_params = SubscribeSettings(subscribe_to_audio=True, subscribe_to_video=False)

        # Patch the timeout to be very short for fast test execution
        # the call never gets on_connected_cb or any other callback, it will timeout
        with patch(
            "pipecat.transports.vonage.client.VIDEO_CONNECTOR_TIMEOUT",
            timedelta(seconds=0.1),
        ):
            with pytest.raises(asyncio.TimeoutError):
                await client.subscribe_to_stream("fail_stream", subscribe_params)

    @pytest.mark.asyncio
    async def test_vonage_client_subscribe_to_stream_fails(self) -> None:
        """Test VonageClient subscribe_to_stream when SDK subscribe fails."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        await client.connect()

        stream = vonage_video_mock.models.Stream(id="fail_stream", connection=DUMMY_CONNECTION)
        client._session_streams["fail_stream"] = stream

        self._setup_subscriber_callbacks(client)
        self.mock_client_instance.subscribe.side_effect = lambda *_, **__: False

        subscribe_params = SubscribeSettings(subscribe_to_audio=True, subscribe_to_video=False)
        with pytest.raises(VonageException) as exc_info:
            await client.subscribe_to_stream("fail_stream", subscribe_params)

        assert "Could not subscribe to stream" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vonage_client_subscribe_to_stream_subscriber_error(self) -> None:
        """Test VonageClient subscribe_to_stream when subscriber reports an error."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        await client.connect()

        stream = vonage_video_mock.models.Stream(id="error_stream", connection=DUMMY_CONNECTION)
        client._session_streams["error_stream"] = stream

        self._setup_subscriber_callbacks(client)

        subscribe_params = SubscribeSettings(subscribe_to_audio=True, subscribe_to_video=False)

        # Subscription should raise an exception
        with pytest.raises(VonageException) as exc_info:
            subscriber = vonage_video_mock.models.Subscriber(stream=stream)
            await self._subscribe_n_handle_callbacks(
                client,
                "error_stream",
                subscribe_params,
                lambda callbacks: callbacks.on_error_cb(subscriber, "Connection failed", 1500),
            )

        assert "Subscriber error" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)
        assert "(code 1500)" in str(exc_info.value)

        # Verify subscription was removed
        assert "error_stream" not in client._session_subscriptions

    @pytest.mark.asyncio
    async def test_vonage_client_subscribe_to_stream_subscriber_disconnected_before_connected(
        self,
    ) -> None:
        """Test VonageClient subscribe_to_stream when subscriber disconnects before connecting."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        await client.connect()

        stream = vonage_video_mock.models.Stream(id="dc_stream", connection=DUMMY_CONNECTION)
        client._session_streams["dc_stream"] = stream

        self._setup_subscriber_callbacks(client)

        # Add listener to track disconnection
        listener = self.VonageClientListener()
        on_subscriber_disconnected_mock = AsyncMock()
        listener.on_subscriber_disconnected = on_subscriber_disconnected_mock
        client.add_listener(listener)

        subscribe_params = SubscribeSettings(subscribe_to_audio=True, subscribe_to_video=False)

        # Subscription should raise an exception
        subscriber = vonage_video_mock.models.Subscriber(stream=stream)
        with pytest.raises(VonageException) as exc_info:
            await self._subscribe_n_handle_callbacks(
                client,
                "dc_stream",
                subscribe_params,
                lambda callbacks: callbacks.on_disconnected_cb(subscriber),
            )

        assert "disconnected before connecting" in str(exc_info.value)

        # Verify subscription was removed
        assert "dc_stream" not in client._session_subscriptions

        # Verify listener was called
        listener.on_subscriber_disconnected.assert_awaited_once_with(subscriber)

    @pytest.mark.asyncio
    async def test_vonage_client_on_stream_received_triggers_listeners(self) -> None:
        """Test that _on_stream_received_cb triggers on_stream_received listener callbacks."""
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=True, audio_in_auto_subscribe=False
        )
        client = await self._create_client(params)

        # Add multiple listeners
        listener1 = self.VonageClientListener()
        on_stream_received_mock1 = AsyncMock()
        listener1.on_stream_received = on_stream_received_mock1
        client.add_listener(listener1)

        listener2 = self.VonageClientListener()
        on_stream_received_mock2 = AsyncMock()
        listener2.on_stream_received = on_stream_received_mock2
        client.add_listener(listener2)

        await client.connect()

        session = vonage_video_mock.models.Session(id="test_session")
        stream = vonage_video_mock.models.Stream(id="test_stream", connection=DUMMY_CONNECTION)

        # Trigger the callback
        client._on_stream_received_cb(session, stream)

        # Wait for async processing
        await self._wait_for_condition(
            lambda: on_stream_received_mock1.await_count > 0
            and on_stream_received_mock2.await_count > 0
        )

        # Verify both listeners were called
        on_stream_received_mock1.assert_awaited_once_with(session, stream)
        on_stream_received_mock2.assert_awaited_once_with(session, stream)

        # Verify stream was added to session streams
        assert "test_stream" in client._session_streams
        assert client._session_streams["test_stream"] == stream

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "auto_audio, auto_video",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    async def test_vonage_client_on_stream_received_auto_subscribe(
        self, auto_audio: bool, auto_video: bool
    ) -> None:
        """Test that _on_stream_received_cb auto-subscribes when auto_subscribe is enabled."""
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=True,
            video_in_enabled=True,
            audio_in_auto_subscribe=auto_audio,
            video_in_auto_subscribe=auto_video,
            video_in_preferred_resolution=(640, 480) if auto_video else None,
            video_in_preferred_framerate=25 if auto_video else None,
        )
        client = await self._create_client(params)

        await client.connect()

        self._setup_subscriber_callbacks(client)

        session = vonage_video_mock.models.Session(id="test_session")
        stream = vonage_video_mock.models.Stream(id="auto_sub_stream", connection=DUMMY_CONNECTION)

        # Trigger the callback
        client._on_stream_received_cb(session, stream)
        await self._wait_for_condition(lambda: stream.id in client._session_streams)

        if not auto_audio and not auto_video:
            await self._wait_client_async_tasks(client)
            # No auto-subscribe should happen
            await self._wait_client_async_tasks(client)
            self.mock_client_instance.subscribe.assert_not_called()
            return

        # Wait for auto-subscribe to happen
        await self._wait_for_condition(lambda: "auto_sub_stream" in self._subscriber_callbacks)

        # Verify subscribe was called
        self.mock_client_instance.subscribe.assert_called_once()
        call_args = self.mock_client_instance.subscribe.call_args

        # Verify subscription settings
        expected_settings = MockSubscriberSettings(
            subscribe_to_audio=auto_audio,
            subscribe_to_video=auto_video,
            video_settings=MockSubscriberVideoSettings(
                preferred_resolution=MockVideoResolution(width=640, height=480)
                if auto_video
                else None,
                preferred_framerate=25 if auto_video else None,
            ),
        )
        assert call_args[0][0] == stream
        assert call_args[1]["settings"] == expected_settings

        # Verify subscription was stored
        assert "auto_sub_stream" in client._session_subscriptions
        assert client._session_subscriptions["auto_sub_stream"].subscribe_to_audio == auto_audio
        assert client._session_subscriptions["auto_sub_stream"].subscribe_to_video == auto_video

        self._subscriber_callbacks["auto_sub_stream"].on_connected_cb(MockSubscriber(stream=stream))
        await self._wait_client_async_tasks(client)

    @pytest.mark.asyncio
    async def test_vonage_client_on_stream_received_skips_existing_subscription(self) -> None:
        """Test that _on_stream_received_cb does not auto-subscribe if stream is already subscribed."""
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=True, audio_in_auto_subscribe=True
        )
        client = await self._create_client(params)

        listener = self.VonageClientListener()
        on_stream_received_mock = AsyncMock()
        listener.on_stream_received = on_stream_received_mock
        client.add_listener(listener)

        await client.connect()

        self._setup_subscriber_callbacks(client)

        session = vonage_video_mock.models.Session(id="test_session")
        stream = vonage_video_mock.models.Stream(id="existing_stream", connection=DUMMY_CONNECTION)

        # Manually add an existing subscription
        client._session_subscriptions["existing_stream"] = SubscribeSettings(
            subscribe_to_audio=True, subscribe_to_video=False
        )

        # Trigger the callback
        client._on_stream_received_cb(session, stream)

        # Wait for listener to be called
        await self._wait_for_condition(lambda: on_stream_received_mock.await_count > 0)
        on_stream_received_mock.assert_awaited_once_with(session, stream)

        # Wait to ensure no subscription happens
        await self._wait_client_async_tasks(client)

        # Verify subscribe was NOT called (because subscription already exists)
        self.mock_client_instance.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_events(self) -> None:
        """Test VonageClient events"""
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=48000,
            audio_in_channels=2,
        )
        client = await self._create_client(params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True
        self._setup_audio_ready_callback(client, call_ready_for_audio=True)
        self._setup_subscriber_callbacks(client)

        # create a listener
        listener = self.VonageClientListener()
        on_error_mock = AsyncMock()
        listener.on_error = on_error_mock
        on_audio_in_mock = AsyncMock()
        listener.on_audio_in = on_audio_in_mock
        on_stream_received_mock = AsyncMock()
        listener.on_stream_received = on_stream_received_mock
        on_stream_dropped_mock = AsyncMock()
        listener.on_stream_dropped = on_stream_dropped_mock
        on_subscriber_connected_mock = AsyncMock()
        listener.on_subscriber_connected = on_subscriber_connected_mock
        on_subscriber_disconnected_mock = AsyncMock()
        listener.on_subscriber_disconnected = on_subscriber_disconnected_mock

        client.add_listener(listener)

        # connect
        await client.connect()

        assert self._connect_callbacks is not None

        # Test _on_session_error_cb triggers on_error
        session = vonage_video_mock.models.Session(id="test_session")
        error_description = "Test error description"
        error_code = 500

        self._connect_callbacks.on_error_cb(session, error_description, error_code)
        await self._wait_for_condition(lambda: on_error_mock.await_count > 0)

        listener.on_error.assert_called_once_with(session, error_description, error_code)
        listener.on_error.reset_mock()

        # Test _on_session_audio_data_cb triggers on_audio_in
        audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
        mock_audio_data = vonage_video_mock.models.AudioData(
            sample_buffer=memoryview(audio_buffer),
            number_of_frames=2,
            number_of_channels=2,
            sample_rate=48000,
        )

        client._on_session_audio_data_cb(session, mock_audio_data)
        await self._wait_for_condition(lambda: on_audio_in_mock.await_count > 0)

        listener.on_audio_in.assert_awaited_once_with(session, ANY)
        frame = listener.on_audio_in.call_args[0][1]
        assert frame.audio == audio_buffer.tobytes()
        assert frame.sample_rate == 48000
        assert frame.num_channels == 2
        listener.on_audio_in.reset_mock()
        # Test _on_stream_received_cb triggers on_stream_received
        stream = vonage_video_mock.models.Stream(id="test_stream", connection=DUMMY_CONNECTION)

        client._on_stream_received_cb(session, stream)
        await self._wait_for_condition(lambda: on_stream_received_mock.await_count > 0)
        listener.on_stream_received.assert_awaited_once_with(session, stream)

        await self._wait_for_condition(lambda: stream.id in self._subscriber_callbacks)

        assert stream.id in self._subscriber_callbacks
        callbacks = self._subscriber_callbacks[stream.id]
        self.mock_client_instance.subscribe.assert_called_once_with(
            stream,
            settings=ANY,
            on_error_cb=callbacks.on_error_cb,
            on_connected_cb=callbacks.on_connected_cb,
            on_disconnected_cb=callbacks.on_disconnected_cb,
            on_render_frame_cb=client._on_subscriber_video_data_cb,
        )
        listener.on_stream_received.reset_mock()

        subscriber = vonage_video_mock.models.Subscriber(stream=stream)
        callbacks.on_connected_cb(subscriber)
        await self._wait_for_condition(lambda: on_subscriber_connected_mock.await_count > 0)
        listener.on_subscriber_connected.assert_awaited_once()
        listener.on_subscriber_connected.reset_mock()

        # Test _on_subscriber_disconnected_cb triggers on_subscriber_disconnected
        callbacks.on_disconnected_cb(subscriber)
        await self._wait_for_condition(lambda: on_subscriber_disconnected_mock.await_count > 0)

        listener.on_subscriber_disconnected.assert_awaited_once_with(subscriber)
        listener.on_subscriber_disconnected.reset_mock()

        # Test _on_stream_dropped_cb triggers on_stream_dropped
        self.mock_client_instance.unsubscribe = MagicMock()

        client._on_stream_dropped_cb(session, stream)
        await self._wait_for_condition(lambda: on_stream_dropped_mock.await_count > 0)

        listener.on_stream_dropped.assert_awaited_once_with(session, stream)
        self.mock_client_instance.unsubscribe.assert_not_called()
        listener.on_stream_dropped.reset_mock()

        # Test _on_subscriber_connected_cb triggers on_subscriber_connected
        subscriber_stream = vonage_video_mock.models.Stream(
            id="subscriber_stream", connection=DUMMY_CONNECTION
        )
        subscriber = vonage_video_mock.models.Subscriber(stream=subscriber_stream)

        # Test error callbacks are logged but don't trigger listener events
        # (these are internal error callbacks, not session errors)
        publisher_stream = vonage_video_mock.models.Stream(
            id="publisher_stream", connection=DUMMY_CONNECTION
        )
        publisher = vonage_video_mock.models.Publisher(stream=publisher_stream)

        # These should not raise exceptions
        client._on_publisher_error_cb(publisher, "publisher error", 400)
        callbacks.on_error_cb(subscriber, "subscriber error", 401)

        await self._wait_client_async_tasks(client)

    @pytest.mark.asyncio
    async def test_vonage_client_on_subscriber_video_data_cb_rgb_format(self) -> None:
        """Test _on_subscriber_video_data_cb with RGB format video frames."""
        from pipecat.frames.frames import UserImageRawFrame

        params = self.VonageVideoConnectorTransportParams(
            video_in_enabled=True,
            video_in_auto_subscribe=False,
        )
        client = await self._create_client(params)

        # Add listener to capture video frames
        listener = self.VonageClientListener()
        on_video_in_mock = AsyncMock()
        listener.on_video_in = on_video_in_mock
        client.add_listener(listener)

        await client.connect()

        # Create a test RGB video frame (4x4, 3 channels)
        width, height = 4, 4
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100  # R channel
        rgb_image[:, :, 1] = 150  # G channel
        rgb_image[:, :, 2] = 200  # B channel

        rgb_bytes = rgb_image.tobytes()

        # Create mock video frame with RGB24 format (which Vonage uses for RGB)
        mock_video_frame = vonage_video_mock.models.VideoFrame(
            frame_buffer=memoryview(rgb_bytes),
            resolution=MockVideoResolution(width=width, height=height),
            format="RGB24",
        )

        # Create mock subscriber
        stream = vonage_video_mock.models.Stream(id="video_stream", connection=DUMMY_CONNECTION)
        subscriber = vonage_video_mock.models.Subscriber(stream=stream)

        # Trigger the callback
        client._on_subscriber_video_data_cb(subscriber, mock_video_frame)

        # Wait for async processing
        await self._wait_for_condition(lambda: on_video_in_mock.await_count > 0)

        # Verify listener was called
        on_video_in_mock.assert_awaited_once()
        call_args = on_video_in_mock.call_args[0]
        assert call_args[0] == subscriber

        # Get the processed frame
        processed_frame: UserImageRawFrame = call_args[1]
        assert processed_frame.user_id == "video_stream"
        assert processed_frame.size == (width, height)
        assert processed_frame.format == "RGB"

        # Verify BGR to RGB conversion happened
        processed_image = np.frombuffer(processed_frame.image, dtype=np.uint8).reshape(
            height, width, 3
        )
        assert processed_image[0, 0, 0] == 200  # R channel (was B in BGR)
        assert processed_image[0, 0, 1] == 150  # G channel (unchanged)
        assert processed_image[0, 0, 2] == 100  # B channel (was R in BGR)

    @pytest.mark.asyncio
    async def test_vonage_input_transport_initialization(self) -> None:
        """Test VonageVideoConnectorInputTransport initialization."""
        params = self.VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoConnectorInputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_vonage_input_transport_start(self) -> None:
        """Test VonageVideoConnectorInputTransport start method."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        transport = self.VonageVideoConnectorInputTransport(client, params)

        # Mock the client connect method
        with (
            patch.object(client, "connect", AsyncMock(return_value=1)) as client_connect_mock,
            patch.object(transport, "set_transport_ready", AsyncMock()) as set_transport_ready_mock,
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            assert transport._initialized is True
            assert transport._connected is True
            client_connect_mock.assert_called_once()
            set_transport_ready_mock.assert_called_once_with(start_frame)

    @pytest.mark.asyncio
    async def test_vonage_input_transport_stop(self) -> None:
        """Test VonageVideoConnectorInputTransport stop method."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        transport = self.VonageVideoConnectorInputTransport(client, params)
        transport._listener_id = 1
        transport._connected = True

        with (
            patch.object(client, "disconnect", AsyncMock()) as client_disconnect_mock,
            patch.object(client, "remove_listener", MagicMock()) as remove_listener_mock,
        ):
            end_frame = EndFrame()
            await transport.stop(end_frame)

            client_disconnect_mock.assert_called_once()
            remove_listener_mock.assert_called_once_with(1)
            assert not transport._connected

    @pytest.mark.asyncio
    async def test_vonage_input_transport_cancel(self) -> None:
        """Test VonageVideoConnectorInputTransport cancel method."""
        params = self.VonageVideoConnectorTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport = self.VonageVideoConnectorInputTransport(client, params)
        transport._listener_id = 1
        transport._connected = True

        # Mock the client disconnect method
        with (
            patch.object(client, "disconnect", AsyncMock()) as client_disconnect_mock,
            patch.object(client, "remove_listener", MagicMock()) as remove_listener_mock,
        ):
            cancel_frame = CancelFrame()
            await transport.cancel(cancel_frame)

            client_disconnect_mock.assert_called_once()
            remove_listener_mock.assert_called_once_with(1)
            assert not transport._connected

    @pytest.mark.asyncio
    async def test_vonage_output_transport_initialization(self) -> None:
        """Test VonageVideoConnectorOutputTransport initialization."""
        params = self.VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoConnectorOutputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_start(self) -> None:
        """Test VonageVideoConnectorOutputTransport start method."""
        params = self.VonageVideoConnectorTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoConnectorOutputTransport(client, transport_params)

        with (
            patch.object(client, "connect", AsyncMock(return_value=1)) as client_connect_mock,
            patch.object(transport, "set_transport_ready", AsyncMock()) as set_transport_ready_mock,
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            assert transport._initialized is True
            client_connect_mock.assert_called_once()
            set_transport_ready_mock.assert_called_once_with(start_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_audio_frame(self) -> None:
        """Test VonageVideoConnectorOutputTransport write_audio_frame method."""

        params = self.VonageVideoConnectorTransportParams(
            audio_out_sample_rate=48000, audio_out_channels=2, audio_out_enabled=True
        )
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        with patch.object(client, "write_audio", AsyncMock()) as client_write_audio_mock:
            transport_params = self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
            transport = self.VonageVideoConnectorOutputTransport(client, transport_params)
            transport._connected = True

            # Create a mock audio frame
            audio_frame = OutputAudioRawFrame(
                audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
            )

            await transport.write_audio_frame(audio_frame)

            # Verify audio was written to client
            client_write_audio_mock.assert_called_once_with(audio_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_not_connected(self) -> None:
        """Test VonageVideoConnectorOutputTransport write_video_frame method."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(video_out_enabled=True)
        )
        client = transport._client

        # Create a test video frame
        width, height = 640, 480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100
        rgb_image[:, :, 1] = 150
        rgb_image[:, :, 2] = 200

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        with patch.object(client, "write_video", AsyncMock(return_value=True)) as write_video_mock:
            await transport.stop(EndFrame())
            result = await transport.write_video_frame(video_frame)

            # Should return False when not connected
            assert result is False
            write_video_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_connected(self) -> None:
        """Test VonageVideoConnectorOutputTransport write_video_frame method when connected."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="RGB",
            )
        )
        client = transport._client

        # Create a test video frame
        width, height = 640, 480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100
        rgb_image[:, :, 1] = 150
        rgb_image[:, :, 2] = 200

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        with patch.object(client, "write_video", AsyncMock(return_value=True)) as write_video_mock:
            transport._connected = True
            result = await transport.write_video_frame(video_frame)

            # Should return True and call write_video when connected
            assert result is True
            write_video_mock.assert_called_once_with(video_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_invalid_size(self) -> None:
        """Test VonageVideoConnectorOutputTransport write_video_frame with invalid frame size."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="RGB",
            )
        )

        # Create a video frame with incorrect size
        width, height = 320, 240  # Different from expected 640x480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        transport._connected = True
        result = await transport.write_video_frame(video_frame)

        # Should return False for invalid size
        assert result is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_invalid_format(self) -> None:
        """Test VonageVideoConnectorOutputTransport write_video_frame with invalid color format."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="YCbCr",
            )
        )

        # Create a video frame with incorrect size
        width, height = 320, 240  # Different from expected 640x480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        transport._connected = True
        result = await transport.write_video_frame(video_frame)

        # Should return False for invalid size
        assert result is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_with_interruption(self) -> None:
        """Test VonageVideoConnectorOutputTransport process_frame method with InterruptionFrame."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
        )
        client = transport._client

        with (
            patch.object(client, "clear_media_buffers") as clear_buffers_mock,
            patch.object(client, "connect", AsyncMock()),
        ):
            await transport.start(StartFrame())
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was called
            clear_buffers_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_without_interruption(self) -> None:
        """Test VonageVideoConnectorOutputTransport process_frame method with non-interruption frame."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
        )
        client = transport._client

        with patch.object(client, "clear_media_buffers") as clear_buffers_mock:
            audio_frame = OutputAudioRawFrame(
                audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
            )
            await transport.process_frame(audio_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was NOT called for non-interruption frames
            clear_buffers_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_when_not_connected(self) -> None:
        """Test VonageVideoConnectorOutputTransport process_frame method when not connected."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(audio_out_enabled=True)
        )
        await transport.stop(EndFrame())  # Ensure transport is not connected
        client = transport._client

        with patch.object(client, "clear_media_buffers") as clear_buffers_mock:
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was NOT called when not connected
            clear_buffers_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_interruption_with_clear_buffers_disabled(self) -> None:
        """Test VonageVideoConnectorOutputTransport with clear_buffers_on_interruption=False."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(
                audio_out_enabled=True, clear_buffers_on_interruption=False
            )
        )
        client = transport._client

        with (
            patch.object(client, "clear_media_buffers") as clear_buffers_mock,
            patch.object(client, "connect", AsyncMock()),
        ):
            await transport.start(StartFrame())
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was NOT called when clear_buffers_on_interruption is False
            clear_buffers_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_interruption_with_clear_buffers_enabled(self) -> None:
        """Test VonageVideoConnectorOutputTransport with clear_buffers_on_interruption=True (default)."""
        transport = await self._create_output_transport(
            params=self.VonageVideoConnectorTransportParams(
                audio_out_enabled=True, clear_buffers_on_interruption=True
            )
        )
        client = transport._client

        with (
            patch.object(client, "clear_media_buffers") as clear_buffers_mock,
            patch.object(client, "connect", AsyncMock()),
        ):
            await transport.start(StartFrame())
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was called when clear_buffers_on_interruption is True
            clear_buffers_mock.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transport_type", ["input", "output"])
    async def test_vonage_transport_sets_audio_sample_rates_from_start_frame(
        self, transport_type: str
    ) -> None:
        """Test transport sets audio sample rates from StartFrame when params are None."""
        # Create params with None sample rates
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=(transport_type == "input"),
            audio_out_enabled=(transport_type == "output"),
            audio_in_sample_rate=None,
            audio_out_sample_rate=None,
        )
        transport: VonageVideoConnectorInputTransport | VonageVideoConnectorOutputTransport
        if transport_type == "input":
            transport = await self._create_input_transport(params=params)
        else:
            transport = await self._create_output_transport(params=params)
        client = transport._client

        # Create a StartFrame with specific sample rates
        start_frame = StartFrame(audio_in_sample_rate=22050, audio_out_sample_rate=44100)

        with patch.object(client, "_sdk_connect", AsyncMock()):
            await transport.start(start_frame)

            # Verify both sample rates were set from the StartFrame
            assert client._audio_in_sample_rate == 22050
            assert client._audio_out_sample_rate == 44100

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transport_type", ["input", "output"])
    async def test_vonage_transport_doesnt_override_audio_sample_rates(
        self, transport_type: str
    ) -> None:
        """Test transport doesn't override audio sample rates when already set in params."""
        # Create params with specific sample rates
        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=(transport_type == "input"),
            audio_out_enabled=(transport_type == "output"),
            audio_in_sample_rate=48000,
            audio_out_sample_rate=16000,
        )
        transport: VonageVideoConnectorInputTransport | VonageVideoConnectorOutputTransport
        if transport_type == "input":
            transport = await self._create_input_transport(params=params)
        else:
            transport = await self._create_output_transport(params=params)
        client = transport._client

        # Create a StartFrame with different sample rates
        start_frame = StartFrame(audio_in_sample_rate=22050, audio_out_sample_rate=44100)

        with patch.object(client, "_sdk_connect", AsyncMock()):
            await transport.start(start_frame)

            # Verify sample rates remain as originally set in params
            assert client._audio_in_sample_rate == 48000
            assert client._audio_out_sample_rate == 16000

    @pytest.mark.asyncio
    async def test_vonage_transport_initialization(self) -> None:
        """Test VonageVideoConnectorTransport initialization."""
        params = self.VonageVideoConnectorTransportParams(
            audio_out_sample_rate=48000,
            audio_out_channels=2,
            audio_out_enabled=True,
            session_enable_migration=True,
            publisher_name="test-publisher",
            publisher_enable_opus_dtx=True,
        )
        transport = await self._create_transport(params=params)

        assert transport._client is not None
        assert transport._one_stream_received is False

        # Verify vonage client was initialized with correct parameters
        client_params = transport._client._params
        assert client_params.audio_out_sample_rate == 48000
        assert client_params.audio_out_channels == 2
        assert client_params.session_enable_migration is True

    @pytest.mark.asyncio
    async def test_vonage_transport_input_output_methods(self) -> None:
        """Test VonageVideoConnectorTransport input and output methods."""
        params = self.VonageVideoConnectorTransportParams()
        transport = self.VonageVideoConnectorTransport(
            self.application_id, self.session_id, self.token, params
        )

        # Test input method
        input_transport = transport.input()
        assert isinstance(input_transport, self.VonageVideoConnectorInputTransport)

        # Test output method
        output_transport = transport.output()
        assert isinstance(output_transport, self.VonageVideoConnectorOutputTransport)

        # Verify they return the same instances on subsequent calls
        assert transport.input() is input_transport
        assert transport.output() is output_transport

    @pytest.mark.asyncio
    async def test_vonage_input_audio_callback(self) -> None:
        """Test audio input callback processing."""

        params = self.VonageVideoConnectorTransportParams(
            audio_in_enabled=True,
        )
        transport = await self._create_input_transport(params)
        client = transport._client

        with (
            patch.object(transport, "push_audio_frame", AsyncMock()) as mock_push_audio_frame,
            patch.object(client, "connect", AsyncMock(return_value=1)),
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            # Create mock audio data
            audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
            audio_frame = InputAudioRawFrame(
                audio=audio_buffer.tobytes(), sample_rate=48000, num_channels=2
            )

            # Call the audio callback
            await transport._audio_in_cb(
                vonage_video_mock.models.Session(id="session"), audio_frame
            )

            mock_push_audio_frame.assert_called_once_with(audio_frame)

    @pytest.mark.asyncio
    async def test_vonage_input_video_callback(self) -> None:
        """Test video input callback processing."""

        params = self.VonageVideoConnectorTransportParams(
            video_in_enabled=True,
        )
        transport = await self._create_input_transport(params)
        client = transport._client

        with (
            patch.object(transport, "push_video_frame", AsyncMock()) as mock_push_video_frame,
            patch.object(client, "connect", AsyncMock(return_value=1)),
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            # Create mock video frame
            width, height = 640, 480
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = 100  # R channel
            rgb_image[:, :, 1] = 150  # G channel
            rgb_image[:, :, 2] = 200  # B channel

            video_frame = UserImageRawFrame(
                user_id="test-user", image=rgb_image.tobytes(), size=(width, height), format="RGB"
            )

            # Create mock subscriber
            stream = vonage_video_mock.models.Stream(id="video_stream", connection=DUMMY_CONNECTION)
            subscriber = vonage_video_mock.models.Subscriber(stream=stream)

            # Call the video callback
            await transport._video_in_cb(subscriber, video_frame)

            mock_push_video_frame.assert_called_once_with(video_frame)

    @pytest.mark.asyncio
    async def test_vonage_transport_event_handlers(self) -> None:
        """Test VonageVideoConnectorTransport event handlers."""
        params = self.VonageVideoConnectorTransportParams()
        transport = await self._create_transport(params)

        with patch.object(
            transport, "_call_event_handler", new_callable=AsyncMock
        ) as mock_call_event_handler:
            # Test session events
            mock_session = Mock()
            mock_session.id = "session-123"

            await transport._on_connected(mock_session)
            mock_call_event_handler.assert_called_with("on_joined", {"sessionId": "session-123"})

            await transport._on_disconnected(mock_session)
            mock_call_event_handler.assert_called_with("on_left", {"sessionId": "session-123"})

            await transport._on_error(mock_session, "test error", 500)
            mock_call_event_handler.assert_called_with("on_error", "test error")

            # Test stream events
            mock_connection = Mock()
            mock_connection.data = "connection-data-123"
            mock_stream = Mock()
            mock_stream.id = "stream-456"
            mock_stream.connection = mock_connection

            await transport._on_stream_received(mock_session, mock_stream)
            # Should call both first participant and participant joined events
            expected_calls = [
                call(
                    "on_first_participant_joined",
                    {
                        "sessionId": "session-123",
                        "streamId": "stream-456",
                        "connectionData": "connection-data-123",
                    },
                ),
                call(
                    "on_participant_joined",
                    {
                        "sessionId": "session-123",
                        "streamId": "stream-456",
                        "connectionData": "connection-data-123",
                    },
                ),
            ]
            mock_call_event_handler.assert_has_calls(expected_calls)

            await transport._on_stream_dropped(mock_session, mock_stream)
            mock_call_event_handler.assert_called_with(
                "on_participant_left",
                {
                    "sessionId": "session-123",
                    "streamId": "stream-456",
                    "connectionData": "connection-data-123",
                },
            )

            # Test subscriber events
            mock_subscriber = Mock()
            mock_subscriber.stream = Mock()
            mock_subscriber.stream.id = "subscriber-789"
            mock_subscriber.stream.connection = Mock()
            mock_subscriber.stream.connection.data = "subscriber-conn-data"

            await transport._on_subscriber_connected(mock_subscriber)
            mock_call_event_handler.assert_called_with(
                "on_client_connected",
                {
                    "subscriberId": "subscriber-789",
                    "streamId": "subscriber-789",
                    "connectionData": "subscriber-conn-data",
                },
            )

            await transport._on_subscriber_disconnected(mock_subscriber)
            mock_call_event_handler.assert_called_with(
                "on_client_disconnected",
                {
                    "subscriberId": "subscriber-789",
                    "streamId": "subscriber-789",
                    "connectionData": "subscriber-conn-data",
                },
            )

    @pytest.mark.asyncio
    async def test_vonage_transport_first_participant_flag(self) -> None:
        """Test that first participant event is only called once."""
        params = self.VonageVideoConnectorTransportParams()
        transport = await self._create_transport(params)

        with patch.object(
            transport, "_call_event_handler", new_callable=AsyncMock
        ) as mock_call_event_handler:
            mock_session = Mock()
            mock_session.id = "session-123"

            mock_connection1 = Mock()
            mock_connection1.data = "conn-data-1"
            mock_stream1 = Mock()
            mock_stream1.id = "stream-456"
            mock_stream1.connection = mock_connection1

            mock_connection2 = Mock()
            mock_connection2.data = "conn-data-2"
            mock_stream2 = Mock()
            mock_stream2.id = "stream-789"
            mock_stream2.connection = mock_connection2

            # First stream should trigger first participant event
            await transport._on_stream_received(mock_session, mock_stream1)
            assert transport._one_stream_received is True

            # Reset mock to check second stream
            mock_call_event_handler.reset_mock()

            # Second stream should not trigger first participant event
            await transport._on_stream_received(mock_session, mock_stream2)
            mock_call_event_handler.assert_called_once_with(
                "on_participant_joined",
                {
                    "sessionId": "session-123",
                    "streamId": "stream-789",
                    "connectionData": "conn-data-2",
                },
            )


class TestAudioNormalization:
    """Test cases for audio normalization functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.AudioProps = AudioProps
        self.process_audio_channels = process_audio_channels
        self.process_audio = process_audio
        self.check_audio_data = check_audio_data

    def test_audio_props_creation(self) -> None:
        """Test AudioProps dataclass creation."""
        props = self.AudioProps(sample_rate=48000, is_stereo=True)
        assert props.sample_rate == 48000
        assert props.is_stereo is True

        props_mono = self.AudioProps(sample_rate=16000, is_stereo=False)
        assert props_mono.sample_rate == 16000
        assert props_mono.is_stereo is False

    def test_process_audio_channels_mono_to_stereo(self) -> None:
        """Test converting mono audio to stereo."""
        # Create mono audio (4 samples)
        mono_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)

        result = self.process_audio_channels(mono_audio, current, target)

        # Should duplicate each sample
        expected = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_stereo_to_mono(self) -> None:
        """Test converting stereo audio to mono."""
        # Create stereo audio (2 frames, 4 samples total)
        stereo_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=48000, is_stereo=False)

        result = self.process_audio_channels(stereo_audio, current, target)

        # Should average each stereo pair: (100+200)/2=150, (300+400)/2=350
        expected = np.array([150, 350], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_same_format(self) -> None:
        """Test when source and target have the same channel format."""
        audio = np.array([100, 200, 300, 400], dtype=np.int16)

        # Test mono to mono
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=False)
        result = self.process_audio_channels(audio, current, target)
        np.testing.assert_array_equal(result, audio)

        # Test stereo to stereo
        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)
        result = self.process_audio_channels(audio, current, target)
        np.testing.assert_array_equal(result, audio)

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_process_audio_same_sample_rate(self, mock_resampler: MagicMock) -> None:
        """Test process_audio when sample rates are the same."""
        mock_resampler_instance = Mock()
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([100, 200, 300, 400], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should only do channel conversion, no resampling
        expected = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should not be called
        mock_resampler_instance.resample.assert_not_called()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates (mono)."""
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([150, 250, 350, 450], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=16000, is_stereo=False)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should resample the audio
        expected = np.array([100, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with correct parameters
        mock_resampler_instance.resample.assert_called_once_with(audio.tobytes(), 48000, 16000)

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_process_audio_different_sample_rate_stereo_to_mono(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates and channel conversion."""
        mock_resampler_instance = Mock()
        # Return resampled mono data
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        # Stereo audio: 2 frames with left/right channels
        audio = np.array([100, 200, 300, 400], dtype=np.int16)  # L1=100, R1=200, L2=300, R2=400
        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=16000, is_stereo=False)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should convert to mono first, then resample
        expected = np.array([100, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with mono audio
        expected_mono = np.array([150, 350], dtype=np.int16)  # (100+200)/2, (300+400)/2
        mock_resampler_instance.resample.assert_called_once_with(
            expected_mono.tobytes(), 48000, 16000
        )

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.client.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono_to_stereo(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates converting mono to stereo."""
        mock_resampler_instance = Mock()
        # Return resampled mono data
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([150, 250], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=16000, is_stereo=True)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should resample first (mono), then convert to stereo
        expected = np.array([100, 100, 200, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with mono audio
        mock_resampler_instance.resample.assert_called_once_with(audio.tobytes(), 48000, 16000)

    def test_check_audio_data_valid_mono_bytes(self) -> None:
        """Test check_audio_data with valid mono audio as bytes."""
        # 4 frames of mono 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)

    def test_check_audio_data_valid_stereo_bytes(self) -> None:
        """Test check_audio_data with valid stereo audio as bytes."""
        # 2 frames of stereo 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 2, 2)

    def test_check_audio_data_valid_memoryview(self) -> None:
        """Test check_audio_data with valid audio as memoryview."""
        # Create int16 memoryview (2 bytes per sample)
        array = np.array([100, 200, 300, 400], dtype=np.int16)
        buffer = memoryview(array)

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)  # 4 mono frames
        self.check_audio_data(buffer, 2, 2)  # 2 stereo frames

    def test_check_audio_data_invalid_channels(self) -> None:
        """Test check_audio_data with invalid number of channels."""
        buffer = b"\x00\x01\x02\x03"

        # Should raise ValueError for invalid channel counts
        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 3)  # 3 channels not supported
        assert "mono or stereo" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 0)  # 0 channels not supported
        assert "mono or stereo" in str(exc_info.value)

    def test_check_audio_data_invalid_bit_depth_bytes(self) -> None:
        """Test check_audio_data with invalid bit depth using bytes."""
        # 2 frames of mono audio with 1 byte per sample (8-bit)
        buffer = b"\x00\x01"

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        assert "16 bit PCM" in str(exc_info.value)
        assert "got 8 bit" in str(exc_info.value)

    def test_check_audio_data_invalid_bit_depth_memoryview(self) -> None:
        """Test check_audio_data with invalid bit depth using memoryview."""
        # Create uint8 memoryview (1 byte per sample)
        array = np.array([100, 200], dtype=np.uint8)
        buffer = memoryview(array)

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        assert "16 bit PCM" in str(exc_info.value)
        assert "got 8 bit" in str(exc_info.value)

    def test_check_audio_data_buffer_size_mismatch(self) -> None:
        """Test check_audio_data with buffer size that doesn't match expected size."""
        # 3 bytes total, but expecting 2 frames of mono 16-bit (should be 4 bytes)
        buffer = b"\x00\x01\x02"

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        # Should detect that 3 bytes / (2 frames * 1 channel) = 1.5 bytes per sample
        # which gets truncated to 1 byte per sample = 8 bit
        assert "16 bit PCM" in str(exc_info.value)


class TestColorspaceConversion:
    """Test cases for image colorspace conversion functions."""

    def test_same_format_no_conversion(self) -> None:
        """Test that conversion with same source and target format returns original image."""
        width, height = 4, 4
        image_data = np.random.randint(0, 256, width * height * 3, dtype=np.uint8).tobytes()

        # Test all formats with themselves
        for fmt in ImageFormat:
            result = image_colorspace_conversion(image_data, (width, height), fmt, fmt)
            assert result == image_data

    def test_rgb_to_bgr_conversion(self) -> None:
        """Test RGB to BGR conversion."""
        width, height = 2, 2
        # Create a simple RGB image with distinct colors
        rgb_image = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],  # Red, Green
                [[0, 0, 255], [255, 255, 0]],  # Blue, Yellow
            ],
            dtype=np.uint8,
        )

        result = image_colorspace_conversion(
            rgb_image.tobytes(),
            (width, height),
            ImageFormat.RGB,
            ImageFormat.BGR,
        )

        # Expected BGR: R and B channels swapped
        expected = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],  # Blue, Green (unchanged)
                [[255, 0, 0], [0, 255, 255]],  # Red, Cyan
            ],
            dtype=np.uint8,
        )

        assert result is not None
        result_array = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 3)
        np.testing.assert_array_equal(result_array, expected)

    def test_bgr_to_rgb_conversion(self) -> None:
        """Test BGR to RGB conversion (should be same as RGB to BGR)."""
        width, height = 2, 2
        bgr_image = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        )

        result = image_colorspace_conversion(
            bgr_image.tobytes(),
            (width, height),
            ImageFormat.BGR,
            ImageFormat.RGB,
        )

        # R and B channels should be swapped
        expected = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [0, 255, 255]],
            ],
            dtype=np.uint8,
        )

        assert result is not None
        result_array = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 3)
        np.testing.assert_array_equal(result_array, expected)

    def test_rgba_to_bgra_conversion(self) -> None:
        """Test RGBA to BGRA conversion."""
        width, height = 2, 2
        rgba_image = np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 200]],  # Red opaque, Green semi-transparent
                [[0, 0, 255, 150], [255, 255, 0, 100]],  # Blue, Yellow
            ],
            dtype=np.uint8,
        )

        result = image_colorspace_conversion(
            rgba_image.tobytes(),
            (width, height),
            ImageFormat.RGBA,
            ImageFormat.BGRA,
        )

        # Expected: R and B swapped, alpha unchanged
        expected = np.array(
            [
                [[0, 0, 255, 255], [0, 255, 0, 200]],
                [[255, 0, 0, 150], [0, 255, 255, 100]],
            ],
            dtype=np.uint8,
        )

        assert result is not None
        result_array = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 4)
        np.testing.assert_array_equal(result_array, expected)

    def test_bgra_to_rgba_conversion(self) -> None:
        """Test BGRA to RGBA conversion."""
        width, height = 2, 2
        bgra_image = np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 200]],
                [[0, 0, 255, 150], [255, 255, 0, 100]],
            ],
            dtype=np.uint8,
        )

        result = image_colorspace_conversion(
            bgra_image.tobytes(),
            (width, height),
            ImageFormat.BGRA,
            ImageFormat.RGBA,
        )

        expected = np.array(
            [
                [[0, 0, 255, 255], [0, 255, 0, 200]],
                [[255, 0, 0, 150], [0, 255, 255, 100]],
            ],
            dtype=np.uint8,
        )

        assert result is not None
        result_array = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 4)
        np.testing.assert_array_equal(result_array, expected)

    def test_planar_yuv420_to_packed_yuv444_conversion(self) -> None:
        """Test planar YUV420 to packed YUV444 conversion."""
        width, height = 4, 4

        # Create YUV420 planar data
        # Y plane: 4x4 = 16 bytes
        y_plane = np.array(
            [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
            dtype=np.uint8,
        )

        # U plane: 2x2 = 4 bytes (subsampled)
        u_plane = np.array([50, 60, 70, 80], dtype=np.uint8)

        # V plane: 2x2 = 4 bytes (subsampled)
        v_plane = np.array([90, 100, 110, 120], dtype=np.uint8)

        yuv420_data = np.concatenate([y_plane, u_plane, v_plane])

        result = image_colorspace_conversion(
            yuv420_data.tobytes(),
            (width, height),
            ImageFormat.PLANAR_YUV420,
            ImageFormat.PACKED_YUV444,
        )

        assert result is not None
        result_array = np.frombuffer(result, dtype=np.uint8).reshape(height, width, 3)

        # Check that Y plane values are preserved
        assert result_array[0, 0, 0] == 100
        assert result_array[0, 1, 0] == 110
        assert result_array[3, 3, 0] == 250

        # Check that U and V planes are upsampled (each 2x2 block should have same U/V values)
        # Top-left 2x2 block should have U=50, V=90
        assert result_array[0, 0, 1] == 50
        assert result_array[0, 0, 2] == 90
        assert result_array[0, 1, 1] == 50
        assert result_array[0, 1, 2] == 90
        assert result_array[1, 0, 1] == 50
        assert result_array[1, 0, 2] == 90
        assert result_array[1, 1, 1] == 50
        assert result_array[1, 1, 2] == 90

        # Top-right 2x2 block should have U=60, V=100
        assert result_array[0, 2, 1] == 60
        assert result_array[0, 2, 2] == 100

    def test_packed_yuv444_to_planar_yuv420_conversion(self) -> None:
        """Test packed YUV444 to planar YUV420 conversion."""
        width, height = 4, 4

        # Create packed YUV444 data (interleaved YUVYUVYUV...)
        # Each pixel has Y, U, V values
        packed_yuv444 = np.zeros((height, width, 3), dtype=np.uint8)

        # Set Y values
        packed_yuv444[:, :, 0] = np.arange(100, 100 + width * height, dtype=np.uint8).reshape(
            height, width
        )

        # Set U values (will be downsampled)
        packed_yuv444[0:2, 0:2, 1] = 50  # Top-left block
        packed_yuv444[0:2, 2:4, 1] = 60  # Top-right block
        packed_yuv444[2:4, 0:2, 1] = 70  # Bottom-left block
        packed_yuv444[2:4, 2:4, 1] = 80  # Bottom-right block

        # Set V values (will be downsampled)
        packed_yuv444[0:2, 0:2, 2] = 90
        packed_yuv444[0:2, 2:4, 2] = 100
        packed_yuv444[2:4, 0:2, 2] = 110
        packed_yuv444[2:4, 2:4, 2] = 120

        result = image_colorspace_conversion(
            packed_yuv444.tobytes(),
            (width, height),
            ImageFormat.PACKED_YUV444,
            ImageFormat.PLANAR_YUV420,
        )

        assert result is not None

        # Parse the planar YUV420 result
        y_plane_size = width * height
        uv_plane_size = (width // 2) * (height // 2)

        result_array = np.frombuffer(result, dtype=np.uint8)
        y_result = result_array[:y_plane_size]
        u_result = result_array[y_plane_size : y_plane_size + uv_plane_size]
        v_result = result_array[y_plane_size + uv_plane_size :]

        # Check Y plane is preserved
        expected_y = np.arange(100, 100 + width * height, dtype=np.uint8)
        np.testing.assert_array_equal(y_result, expected_y)

        # Check U plane is downsampled (should be 2x2)
        expected_u = np.array([50, 60, 70, 80], dtype=np.uint8)
        np.testing.assert_array_equal(u_result, expected_u)

        # Check V plane is downsampled
        expected_v = np.array([90, 100, 110, 120], dtype=np.uint8)
        np.testing.assert_array_equal(v_result, expected_v)

    def test_unsupported_conversion_returns_none(self) -> None:
        """Test that unsupported conversions return None."""
        width, height = 4, 4
        image_data = np.random.randint(0, 256, width * height * 3, dtype=np.uint8).tobytes()

        # Test some unsupported conversions
        result = image_colorspace_conversion(
            image_data,
            (width, height),
            ImageFormat.RGB,
            ImageFormat.PLANAR_YUV420,
        )
        assert result is None

        result = image_colorspace_conversion(
            image_data,
            (width, height),
            ImageFormat.RGBA,
            ImageFormat.RGB,
        )
        assert result is None

        result = image_colorspace_conversion(
            image_data,
            (width, height),
            ImageFormat.PLANAR_YUV420,
            ImageFormat.BGR,
        )
        assert result is None

    def test_conversion_with_different_sizes(self) -> None:
        """Test conversions work with different image sizes."""
        test_sizes = [(2, 2), (4, 4), (8, 8), (16, 16)]

        for width, height in test_sizes:
            # Test RGB to BGR
            rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            result = image_colorspace_conversion(
                rgb_image.tobytes(),
                (width, height),
                ImageFormat.RGB,
                ImageFormat.BGR,
            )
            assert result is not None
            assert len(result) == width * height * 3

    def test_yuv420_to_yuv444_roundtrip_preserves_y_plane(self) -> None:
        """Test that Y plane is preserved in YUV420 -> YUV444 -> YUV420 conversion."""
        width, height = 4, 4

        # Create original YUV420 data
        y_plane_orig = np.arange(0, width * height, dtype=np.uint8)
        u_plane_orig = np.array([50, 60, 70, 80], dtype=np.uint8)
        v_plane_orig = np.array([90, 100, 110, 120], dtype=np.uint8)
        yuv420_orig = np.concatenate([y_plane_orig, u_plane_orig, v_plane_orig])

        # Convert to YUV444
        yuv444 = image_colorspace_conversion(
            yuv420_orig.tobytes(),
            (width, height),
            ImageFormat.PLANAR_YUV420,
            ImageFormat.PACKED_YUV444,
        )
        assert yuv444 is not None

        # Convert back to YUV420
        yuv420_result = image_colorspace_conversion(
            yuv444,
            (width, height),
            ImageFormat.PACKED_YUV444,
            ImageFormat.PLANAR_YUV420,
        )
        assert yuv420_result is not None

        # Extract Y plane from result
        result_array = np.frombuffer(yuv420_result, dtype=np.uint8)
        y_plane_result = result_array[: width * height]

        # Y plane should be identical after roundtrip
        np.testing.assert_array_equal(y_plane_result, y_plane_orig)

    def test_rgb_bgr_roundtrip(self) -> None:
        """Test that RGB -> BGR -> RGB conversion preserves data."""
        width, height = 4, 4
        rgb_orig = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Convert to BGR
        bgr = image_colorspace_conversion(
            rgb_orig.tobytes(),
            (width, height),
            ImageFormat.RGB,
            ImageFormat.BGR,
        )
        assert bgr is not None

        # Convert back to RGB
        rgb_result = image_colorspace_conversion(
            bgr,
            (width, height),
            ImageFormat.BGR,
            ImageFormat.RGB,
        )
        assert rgb_result is not None

        result_array = np.frombuffer(rgb_result, dtype=np.uint8).reshape(height, width, 3)
        np.testing.assert_array_equal(result_array, rgb_orig)

    def test_rgba_bgra_roundtrip(self) -> None:
        """Test that RGBA -> BGRA -> RGBA conversion preserves data."""
        width, height = 4, 4
        rgba_orig = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

        # Convert to BGRA
        bgra = image_colorspace_conversion(
            rgba_orig.tobytes(),
            (width, height),
            ImageFormat.RGBA,
            ImageFormat.BGRA,
        )
        assert bgra is not None

        # Convert back to RGBA
        rgba_result = image_colorspace_conversion(
            bgra,
            (width, height),
            ImageFormat.BGRA,
            ImageFormat.RGBA,
        )
        assert rgba_result is not None

        result_array = np.frombuffer(rgba_result, dtype=np.uint8).reshape(height, width, 4)
        np.testing.assert_array_equal(result_array, rgba_orig)
