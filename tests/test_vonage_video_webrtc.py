# SPDX-License-Identifier: BSD 2-Clause License

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)

# Mock the vonage_video module since it's not available in test environment
vonage_video_mock = MagicMock()
vonage_video_mock.VonageVideoClient = MagicMock()
vonage_video_mock.models = MagicMock()


# Create mock classes that match the expected interface
class MockAudioData:
    def __init__(self, sample_buffer, number_of_frames, number_of_channels, sample_rate):
        self.sample_buffer = sample_buffer
        self.number_of_frames = number_of_frames
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate


class MockSession:
    def __init__(self, id="test_session"):
        self.id = id


class MockStream:
    def __init__(self, id="test_stream"):
        self.id = id


class MockPublisher:
    def __init__(self, stream=None):
        self.stream = stream or MockStream()


class MockSubscriber:
    def __init__(self, stream=None):
        self.stream = stream or MockStream()


# Set up the mock module structure
vonage_video_mock.models.AudioData = MockAudioData
vonage_video_mock.models.Session = MockSession
vonage_video_mock.models.Stream = MockStream
vonage_video_mock.models.Publisher = MockPublisher
vonage_video_mock.models.Subscriber = MockSubscriber
vonage_video_mock.models.LoggingSettings = MagicMock
vonage_video_mock.models.PublisherSettings = MagicMock
vonage_video_mock.models.PublisherAudioSettings = MagicMock
vonage_video_mock.models.SessionSettings = MagicMock
vonage_video_mock.models.SessionAudioSettings = MagicMock

# Mock the module in sys.modules so imports work
sys.modules["vonage_video_connector"] = vonage_video_mock
sys.modules["vonage_video_connector.models"] = vonage_video_mock.models


# Now we can import the transport classes since the vonage_video module is mocked
from pipecat.transports.vonage.video_webrtc import (
    AudioProps,
    VonageClient,
    VonageClientListener,
    VonageClientParams,
    VonageVideoWebrtcInputTransport,
    VonageVideoWebrtcOutputTransport,
    VonageVideoWebrtcTransport,
    VonageVideoWebrtcTransportParams,
    check_audio_data,
    process_audio,
    process_audio_channels,
)


class TestVonageVideoWebrtcTransport(unittest.IsolatedAsyncioTestCase):
    """Test cases for Vonage Video WebRTC transport classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.VonageClient = VonageClient
        self.VonageClientListener = VonageClientListener
        self.VonageClientParams = VonageClientParams
        self.VonageVideoWebrtcInputTransport = VonageVideoWebrtcInputTransport
        self.VonageVideoWebrtcOutputTransport = VonageVideoWebrtcOutputTransport
        self.VonageVideoWebrtcTransport = VonageVideoWebrtcTransport
        self.VonageVideoWebrtcTransportParams = VonageVideoWebrtcTransportParams

        # Mock client instance
        self.mock_client_instance = Mock()
        vonage_video_mock.VonageVideoClient.return_value = self.mock_client_instance

        # Common test data
        self.application_id = "test-app-id"
        self.session_id = "test-session-id"
        self.token = "test-token"

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_vonage_client_params_defaults(self):
        """Test VonageClientParams default values."""
        params = self.VonageClientParams()
        self.assertEqual(params.audio_in_sample_rate, 48000)
        self.assertEqual(params.audio_in_channels, 2)
        self.assertFalse(params.enable_migration)

    def test_vonage_client_params_custom_values(self):
        """Test VonageClientParams with custom values."""
        params = self.VonageClientParams(
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_sample_rate=22050,
            audio_out_channels=1,
            enable_migration=True,
        )
        self.assertEqual(params.audio_in_sample_rate, 16000)
        self.assertEqual(params.audio_in_channels, 1)
        self.assertEqual(params.audio_out_sample_rate, 22050)
        self.assertEqual(params.audio_out_channels, 1)
        self.assertTrue(params.enable_migration)

    def test_vonage_client_listener_defaults(self):
        """Test VonageClientListener default values."""
        listener = self.VonageClientListener()
        self.assertIsNotNone(listener.on_connected)
        self.assertIsNotNone(listener.on_disconnected)
        self.assertIsNotNone(listener.on_error)
        self.assertIsNotNone(listener.on_audio_in)
        self.assertIsNotNone(listener.on_stream_received)
        self.assertIsNotNone(listener.on_stream_dropped)
        self.assertIsNotNone(listener.on_subscriber_connected)
        self.assertIsNotNone(listener.on_subscriber_disconnected)

    def test_vonage_transport_params_defaults(self):
        """Test VonageVideoWebrtcTransportParams default values."""
        params = self.VonageVideoWebrtcTransportParams()
        self.assertEqual(params.publisher_name, "")
        self.assertFalse(params.publisher_enable_opus_dtx)
        self.assertFalse(params.session_enable_migration)

    def test_vonage_client_initialization(self):
        """Test VonageClient initialization."""
        # Reset the mock for this specific test
        vonage_video_mock.VonageVideoClient.reset_mock()

        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        self.assertEqual(client._application_id, self.application_id)
        self.assertEqual(client._session_id, self.session_id)
        self.assertEqual(client._token, self.token)
        self.assertEqual(client._params, params)
        self.assertFalse(client._connected)
        self.assertEqual(client._connection_counter, 0)
        vonage_video_mock.VonageVideoClient.assert_called_once()

    def test_vonage_client_add_remove_listener(self):
        """Test adding and removing listeners from VonageClient."""
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        listener = self.VonageClientListener()
        listener_id = client.add_listener(listener)

        self.assertIsInstance(listener_id, int)
        self.assertIn(listener_id, client._listeners)
        self.assertEqual(client._listeners[listener_id], listener)

        client.remove_listener(listener_id)
        self.assertNotIn(listener_id, client._listeners)

    async def test_vonage_client_connect_first_time(self):
        """Test VonageClient connect method for first connection."""
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        listener = self.VonageClientListener()
        listener_id = await client.connect(listener)

        self.assertIsInstance(listener_id, int)
        self.mock_client_instance.connect.assert_called_once()

        # Verify connect was called with correct parameters
        call_args = self.mock_client_instance.connect.call_args
        self.assertEqual(call_args[1]["application_id"], self.application_id)
        self.assertEqual(call_args[1]["session_id"], self.session_id)
        self.assertEqual(call_args[1]["token"], self.token)

    async def test_vonage_client_connect_already_connected(self):
        """Test VonageClient connect when already connected."""
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        # First connection
        listener1 = self.VonageClientListener()
        listener1.on_connected = AsyncMock()
        await client.connect(listener1)

        listener1.on_connected.assert_called_once()

        # Set connected state manually since we're mocking
        client._connected = True
        client._connection_counter = 1

        # Second connection
        listener2 = self.VonageClientListener()
        listener2.on_connected = AsyncMock()
        listener_id2 = await client.connect(listener2)

        self.assertIsInstance(listener_id2, int)
        self.assertEqual(client._connection_counter, 2)
        listener2.on_connected.assert_called_once()

        listener1.on_connected.assert_called_once()

    async def test_vonage_client_connect_failure(self):
        """Test VonageClient connect method when connection fails."""
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Mock the connect method to return False
        self.mock_client_instance.connect.return_value = False

        listener = self.VonageClientListener()

        with self.assertRaises(Exception) as context:
            await client.connect(listener)

        self.assertIn("Could not connect to session", str(context.exception))

    async def test_vonage_client_disconnect(self):
        """Test VonageClient disconnect method."""
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Mock connected state
        client._connected = True
        client._connection_counter = 1

        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        listener_id = client.add_listener(listener)

        await client.disconnect(listener_id)

        self.mock_client_instance.disconnect.assert_called_once()
        listener.on_disconnected.assert_called_once()

    async def test_vonage_client_write_audio(self):
        """Test VonageClient write_audio method."""
        params = self.VonageClientParams(audio_out_channels=2, audio_out_sample_rate=48000)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Create mock audio data
        audio_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"  # 4 frames of 2-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.inject_audio.assert_called_once()
        call_args = self.mock_client_instance.inject_audio.call_args[0][0]
        self.assertEqual(call_args.number_of_frames, 2)  # 8 bytes / (2 channels * 2 bytes)
        self.assertEqual(call_args.number_of_channels, 2)
        self.assertEqual(call_args.sample_rate, 48000)

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_initialization(self, mock_resampler):
        """Test VonageVideoWebrtcInputTransport initialization."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)

        self.assertEqual(transport._client, client)
        self.assertFalse(transport._initialized)
        mock_resampler.assert_called_once()

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_start(self, mock_resampler):
        """Test VonageVideoWebrtcInputTransport start method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)

        # Mock the client connect method
        client.connect = AsyncMock(return_value=1)
        transport.set_transport_ready = AsyncMock()

        start_frame = StartFrame()
        await transport.start(start_frame)

        self.assertTrue(transport._initialized)
        client.connect.assert_called_once()
        transport.set_transport_ready.assert_called_once_with(start_frame)

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_stop(self, mock_resampler):
        """Test VonageVideoWebrtcInputTransport stop method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
        transport._listener_id = 1

        # Mock the client disconnect method
        client.disconnect = AsyncMock()

        end_frame = EndFrame()
        await transport.stop(end_frame)

        client.disconnect.assert_called_once_with(1)
        self.assertIsNone(transport._listener_id)

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_cancel(self, mock_resampler):
        """Test VonageVideoWebrtcInputTransport cancel method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
        transport._listener_id = 1

        # Mock the client disconnect method
        client.disconnect = AsyncMock()

        cancel_frame = CancelFrame()
        await transport.cancel(cancel_frame)

        client.disconnect.assert_called_once_with(1)
        self.assertIsNone(transport._listener_id)

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_initialization(self, mock_resampler):
        """Test VonageVideoWebrtcOutputTransport initialization."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)

        self.assertEqual(transport._client, client)
        self.assertFalse(transport._initialized)
        mock_resampler.assert_called_once()

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_start(self, mock_resampler):
        """Test VonageVideoWebrtcOutputTransport start method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)

        # Mock the client connect method
        client.connect = AsyncMock(return_value=1)
        transport.set_transport_ready = AsyncMock()

        start_frame = StartFrame()
        await transport.start(start_frame)

        self.assertTrue(transport._initialized)
        client.connect.assert_called_once()
        transport.set_transport_ready.assert_called_once_with(start_frame)

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_write_audio_frame(self, mock_resampler):
        """Test VonageVideoWebrtcOutputTransport write_audio_frame method."""
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=b"\x00\x01\x02\x03")
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageClientParams(audio_out_sample_rate=48000, audio_out_channels=2)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        client.write_audio = AsyncMock()
        client.get_params = Mock(return_value=params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)
        transport._listener_id = 1

        # Create a mock audio frame
        audio_frame = OutputAudioRawFrame(
            audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
        )

        await transport.write_audio_frame(audio_frame)

        # Verify resampling was called
        mock_resampler_instance.resample.assert_called_once_with(audio_frame.audio, 16000, 48000)
        # Verify audio was written to client
        client.write_audio.assert_called_once()

    async def test_vonage_transport_initialization(self):
        """Test VonageVideoWebrtcTransport initialization."""
        params = self.VonageVideoWebrtcTransportParams(
            audio_out_sample_rate=48000,
            audio_out_channels=2,
            audio_out_enabled=True,
            session_enable_migration=True,
            publisher_name="test-publisher",
            publisher_enable_opus_dtx=True,
        )

        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        self.assertIsNotNone(transport._client)
        self.assertFalse(transport._one_stream_received)

        # Verify vonage client was initialized with correct parameters
        client_params = transport._client._params
        self.assertEqual(client_params.audio_out_sample_rate, 48000)
        self.assertEqual(client_params.audio_out_channels, 2)
        self.assertTrue(client_params.enable_migration)

    async def test_vonage_transport_input_output_methods(self):
        """Test VonageVideoWebrtcTransport input and output methods."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        # Test input method
        input_transport = transport.input()
        self.assertIsInstance(input_transport, self.VonageVideoWebrtcInputTransport)

        # Test output method
        output_transport = transport.output()
        self.assertIsInstance(output_transport, self.VonageVideoWebrtcOutputTransport)

        # Verify they return the same instances on subsequent calls
        self.assertIs(transport.input(), input_transport)
        self.assertIs(transport.output(), output_transport)

    @patch("pipecat.transports.vonage.video_webrtc.asyncio.run_coroutine_threadsafe")
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_audio_callback(self, mock_resampler, mock_run_coroutine):
        """Test audio input callback processing."""
        resampled_audio = b"\x00\x01\x02\x03"
        resampled_bitrate = 26000
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_audio)
        mock_resampler.return_value = mock_resampler_instance

        push_frame_coroutine = None

        # Mock the run_coroutine_threadsafe to capture the coroutine
        def mock_run_coro(coro, loop):
            nonlocal push_frame_coroutine
            push_frame_coroutine = coro
            # Return a mock task
            task = Mock()
            task.result.return_value = None
            return task

        mock_run_coroutine.side_effect = mock_run_coro

        params = self.VonageClientParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=resampled_bitrate,
        )
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
        transport._listener_id = 1
        transport.push_audio_frame = AsyncMock()
        transport.get_event_loop = Mock(return_value=asyncio.get_event_loop())

        # Mock the client connect method
        client.connect = AsyncMock(return_value=1)
        transport.set_transport_ready = AsyncMock()
        start_frame = StartFrame()
        await transport.start(start_frame)

        # Create mock audio data
        audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
        mock_audio_data = Mock()
        mock_audio_data.sample_buffer = audio_buffer.tobytes()
        mock_audio_data.number_of_frames = 2
        mock_audio_data.number_of_channels = 2
        mock_audio_data.sample_rate = 48000

        # Create mock session
        mock_session = Mock()

        # Call the audio callback
        transport._audio_in_cb(mock_session, mock_audio_data)

        # Execute the captured coroutine and check it does what we expect
        self.assertIsNotNone(push_frame_coroutine)
        await push_frame_coroutine

        transport.push_audio_frame.assert_called_once()
        # Verify run_coroutine_threadsafe was called
        mock_run_coroutine.assert_called_once()
        arg = transport.push_audio_frame.call_args[0][0]
        self.assertIsInstance(arg, InputAudioRawFrame)
        self.assertEqual(arg.audio, resampled_audio)
        self.assertEqual(arg.sample_rate, resampled_bitrate)
        self.assertEqual(arg.num_channels, 1)

    async def test_vonage_transport_event_handlers(self):
        """Test VonageVideoWebrtcTransport event handlers."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        # Mock the event handler calling mechanism
        transport._call_event_handler = AsyncMock()

        # Test session events
        mock_session = Mock()
        mock_session.id = "session-123"

        await transport._on_connected(mock_session)
        transport._call_event_handler.assert_called_with("on_joined", {"sessionId": "session-123"})

        await transport._on_disconnected(mock_session)
        transport._call_event_handler.assert_called_with("on_left")

        await transport._on_error(mock_session, "test error", 500)
        transport._call_event_handler.assert_called_with("on_error", "test error")

        # Test stream events
        mock_stream = Mock()
        mock_stream.id = "stream-456"

        await transport._on_stream_received(mock_session, mock_stream)
        # Should call both first participant and participant joined events
        expected_calls = [
            call(
                "on_first_participant_joined",
                {"sessionId": "session-123", "streamId": "stream-456"},
            ),
            call("on_participant_joined", {"sessionId": "session-123", "streamId": "stream-456"}),
        ]
        transport._call_event_handler.assert_has_calls(expected_calls)

        await transport._on_stream_dropped(mock_session, mock_stream)
        transport._call_event_handler.assert_called_with(
            "on_participant_left", {"sessionId": "session-123", "streamId": "stream-456"}
        )

        # Test subscriber events
        mock_subscriber = Mock()
        mock_subscriber.stream.id = "subscriber-789"

        await transport._on_subscriber_connected(mock_subscriber)
        transport._call_event_handler.assert_called_with(
            "on_client_connected", {"subscriberId": "subscriber-789"}
        )

        await transport._on_subscriber_disconnected(mock_subscriber)
        transport._call_event_handler.assert_called_with(
            "on_client_disconnected", {"subscriberId": "subscriber-789"}
        )

    async def test_vonage_transport_first_participant_flag(self):
        """Test that first participant event is only called once."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        transport._call_event_handler = AsyncMock()

        mock_session = Mock()
        mock_session.id = "session-123"
        mock_stream1 = Mock()
        mock_stream1.id = "stream-456"
        mock_stream2 = Mock()
        mock_stream2.id = "stream-789"

        # First stream should trigger first participant event
        await transport._on_stream_received(mock_session, mock_stream1)
        self.assertTrue(transport._one_stream_received)

        # Reset mock to check second stream
        transport._call_event_handler.reset_mock()

        # Second stream should not trigger first participant event
        await transport._on_stream_received(mock_session, mock_stream2)
        transport._call_event_handler.assert_called_once_with(
            "on_participant_joined", {"sessionId": "session-123", "streamId": "stream-789"}
        )


class TestAudioNormalization(unittest.IsolatedAsyncioTestCase):
    """Test cases for audio normalization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.AudioProps = AudioProps
        self.process_audio_channels = process_audio_channels
        self.process_audio = process_audio
        self.check_audio_data = check_audio_data

    def test_audio_props_creation(self):
        """Test AudioProps dataclass creation."""
        props = self.AudioProps(sample_rate=48000, is_stereo=True)
        self.assertEqual(props.sample_rate, 48000)
        self.assertTrue(props.is_stereo)

        props_mono = self.AudioProps(sample_rate=16000, is_stereo=False)
        self.assertEqual(props_mono.sample_rate, 16000)
        self.assertFalse(props_mono.is_stereo)

    def test_process_audio_channels_mono_to_stereo(self):
        """Test converting mono audio to stereo."""
        # Create mono audio (4 samples)
        mono_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)

        result = self.process_audio_channels(mono_audio, current, target)

        # Should duplicate each sample
        expected = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_stereo_to_mono(self):
        """Test converting stereo audio to mono."""
        # Create stereo audio (2 frames, 4 samples total)
        stereo_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=48000, is_stereo=False)

        result = self.process_audio_channels(stereo_audio, current, target)

        # Should average each stereo pair: (100+200)/2=150, (300+400)/2=350
        expected = np.array([150, 350], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_same_format(self):
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

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_same_sample_rate(self, mock_resampler):
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

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono(self, mock_resampler):
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

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_stereo_to_mono(self, mock_resampler):
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

    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono_to_stereo(self, mock_resampler):
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

    def test_check_audio_data_valid_mono_bytes(self):
        """Test check_audio_data with valid mono audio as bytes."""
        # 4 frames of mono 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)

    def test_check_audio_data_valid_stereo_bytes(self):
        """Test check_audio_data with valid stereo audio as bytes."""
        # 2 frames of stereo 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 2, 2)

    def test_check_audio_data_valid_memoryview(self):
        """Test check_audio_data with valid audio as memoryview."""
        # Create int16 memoryview (2 bytes per sample)
        array = np.array([100, 200, 300, 400], dtype=np.int16)
        buffer = memoryview(array)

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)  # 4 mono frames
        self.check_audio_data(buffer, 2, 2)  # 2 stereo frames

    def test_check_audio_data_invalid_channels(self):
        """Test check_audio_data with invalid number of channels."""
        buffer = b"\x00\x01\x02\x03"

        # Should raise ValueError for invalid channel counts
        with self.assertRaises(ValueError) as context:
            self.check_audio_data(buffer, 2, 3)  # 3 channels not supported
        self.assertIn("mono or stereo", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.check_audio_data(buffer, 2, 0)  # 0 channels not supported
        self.assertIn("mono or stereo", str(context.exception))

    def test_check_audio_data_invalid_bit_depth_bytes(self):
        """Test check_audio_data with invalid bit depth using bytes."""
        # 2 frames of mono audio with 1 byte per sample (8-bit)
        buffer = b"\x00\x01"

        with self.assertRaises(ValueError) as context:
            self.check_audio_data(buffer, 2, 1)
        self.assertIn("16 bit PCM", str(context.exception))
        self.assertIn("got 8 bit", str(context.exception))

    def test_check_audio_data_invalid_bit_depth_memoryview(self):
        """Test check_audio_data with invalid bit depth using memoryview."""
        # Create uint8 memoryview (1 byte per sample)
        array = np.array([100, 200], dtype=np.uint8)
        buffer = memoryview(array)

        with self.assertRaises(ValueError) as context:
            self.check_audio_data(buffer, 2, 1)
        self.assertIn("16 bit PCM", str(context.exception))
        self.assertIn("got 8 bit", str(context.exception))

    def test_check_audio_data_buffer_size_mismatch(self):
        """Test check_audio_data with buffer size that doesn't match expected size."""
        # 3 bytes total, but expecting 2 frames of mono 16-bit (should be 4 bytes)
        buffer = b"\x00\x01\x02"

        with self.assertRaises(ValueError) as context:
            self.check_audio_data(buffer, 2, 1)
        # Should detect that 3 bytes / (2 frames * 1 channel) = 1.5 bytes per sample
        # which gets truncated to 1 byte per sample = 8 bit
        self.assertIn("16 bit PCM", str(context.exception))


if __name__ == "__main__":
    unittest.main()
