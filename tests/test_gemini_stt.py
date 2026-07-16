#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    AudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.google.stt import GeminiSTTService, GeminiSTTSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.utils.time import time_now_iso8601


class _FakeSession:
    """Mock Gemini Live session that yields messages and records sent inputs."""

    def __init__(self):
        self.sent_inputs = []
        self.receive_queue = asyncio.Queue()

    async def send_realtime_input(self, **kwargs):
        self.sent_inputs.append(kwargs)

    def receive(self):
        """Minimal mock of session.receive() that returns an async generator."""

        class AsyncMessageIterator:
            def __init__(self, queue):
                self.queue = queue

            def __aiter__(self):
                return self

            async def __anext__(self):
                val = await self.queue.get()
                self.queue.task_done()
                if val is None:
                    raise StopAsyncIteration
                return val

        return AsyncMessageIterator(self.receive_queue)


def _make_server_message(text: str):
    """Create a mock LiveServerMessage with input_transcription."""

    class MockInputTranscription:
        def __init__(self, t):
            self.text = t

    class MockServerContent:
        def __init__(self, t):
            self.input_transcription = MockInputTranscription(t)
            self.interrupted = False
            self.model_turn = None
            self.output_transcription = None
            self.grounding_metadata = None
            self.turn_complete = False

    class MockLiveServerMessage:
        def __init__(self, t):
            self.server_content = MockServerContent(t)
            self.tool_call = None
            self.session_resumption_update = None
            self.usage_metadata = None

    return MockLiveServerMessage(text)


def _make_interim_server_message(text: str):
    """Create a mock LiveServerMessage with interim_input_transcription."""

    class MockInputTranscription:
        def __init__(self, t):
            self.text = t

    class MockServerContent:
        def __init__(self, t):
            self.interim_input_transcription = MockInputTranscription(t)
            self.interrupted = False
            self.model_turn = None
            self.output_transcription = None
            self.grounding_metadata = None
            self.turn_complete = False

    class MockLiveServerMessage:
        def __init__(self, t):
            self.server_content = MockServerContent(t)
            self.tool_call = None
            self.session_resumption_update = None
            self.usage_metadata = None

    return MockLiveServerMessage(text)


@pytest.mark.asyncio
@patch("pipecat.services.google.stt.Client")
async def test_gemini_stt_lifecycle_and_transcription(mock_client_class):
    # Setup the mock Client and session
    mock_client = MagicMock()
    mock_client._api_client = MagicMock()
    mock_client._api_client.vertexai = True
    mock_client_class.return_value = mock_client

    fake_session = _FakeSession()

    # Mock live.connect context manager
    class AsyncContextManagerMock:
        async def __aenter__(self):
            return fake_session

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_client.aio.live.connect.return_value = AsyncContextManagerMock()

    service = GeminiSTTService(
        api_key="test-key",
        settings=GeminiSTTService.Settings(
            model="latest",
            language=Language.EN_US,
        ),
    )

    # Store frame outputs
    frames_received = []

    async def push_frame(frame, direction=None):
        frames_received.append(frame)

    service.push_frame = push_frame

    # Setup TaskManager
    tm = TaskManager()
    service._task_manager = tm

    # Start the service
    start_frame = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
    await service.start(start_frame)
    await asyncio.sleep(0.05)

    # Verify connection was made with the right model/config
    mock_client.aio.live.connect.assert_called_once()
    args, kwargs = mock_client.aio.live.connect.call_args
    assert kwargs["model"] == "latest"
    assert kwargs["config"].response_modalities == ["TEXT"]
    assert kwargs["config"].input_audio_transcription.language_codes == ["en-US"]

    # Simulate sending audio
    audio_data = b"\x00\x00" * 160
    # Process audio chunk
    async for _ in service.run_stt(audio_data):
        pass

    # Give a brief moment for the send loop task to process the queued audio
    await asyncio.sleep(0.05)

    assert len(fake_session.sent_inputs) == 1
    assert fake_session.sent_inputs[0]["audio"].data == audio_data
    assert "audio/pcm;rate=16000" in fake_session.sent_inputs[0]["audio"].mime_type

    # Test transcription reception
    # 1. Send interim text
    await fake_session.receive_queue.put(_make_interim_server_message("Hello"))
    await asyncio.sleep(0.05)

    assert len(frames_received) == 1
    assert isinstance(frames_received[-1], InterimTranscriptionFrame)
    assert frames_received[-1].text == "Hello"

    # 2. Send final text
    await fake_session.receive_queue.put(_make_server_message("Hello world."))
    await asyncio.sleep(0.05)

    assert len(frames_received) == 2
    assert isinstance(frames_received[-1], TranscriptionFrame)
    assert frames_received[-1].text == "Hello world."
    assert frames_received[-1].finalized is True

    # Clean up and stop
    await service.stop(None)


@patch("pipecat.services.google.stt.Client")
def test_gemini_stt_vertex_client_args(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Case 1: passing project & location implies enterprise/vertexai
    service1 = GeminiSTTService(
        project="test-project",
        location="us-central1",
    )
    mock_client_class.assert_called_with(
        api_key=None,
        enterprise=True,
        project="test-project",
        location="us-central1",
        http_options=None,
    )

    # Case 2: passing enterprise/vertexai explicitly
    mock_client_class.reset_mock()
    service2 = GeminiSTTService(
        enterprise=True,
        project="test-project-2",
    )
    mock_client_class.assert_called_with(
        api_key=None,
        enterprise=True,
        project="test-project-2",
        location="global",
        http_options=None,
    )


@pytest.mark.asyncio
@patch("pipecat.services.google.stt.Client")
async def test_gemini_stt_interim_fallback(mock_client_class):
    # Setup mock Client and session
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    fake_session = _FakeSession()

    class AsyncContextManagerMock:
        async def __aenter__(self):
            return fake_session

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_client.aio.live.connect.return_value = AsyncContextManagerMock()

    service = GeminiSTTService(api_key="test-key")
    frames_received = []

    async def push_frame(frame, direction=None):
        frames_received.append(frame)

    service.push_frame = push_frame

    tm = TaskManager()
    service._task_manager = tm

    # Start the service
    await service.start(StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000))
    await asyncio.sleep(0.05)

    # Send interim text via interim_input_transcription field
    await fake_session.receive_queue.put(_make_interim_server_message("Hello from interim"))
    await asyncio.sleep(0.05)

    # Verify InterimTranscriptionFrame was pushed
    assert len(frames_received) == 1
    assert isinstance(frames_received[0], InterimTranscriptionFrame)
    assert frames_received[0].text == "Hello from interim"

    await service.stop(None)
