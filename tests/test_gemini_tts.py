#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.google.tts import GeminiTTSService
from pipecat.tests.utils import run_test


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_genai_backend_success(mock_genai_client_class):
    # Setup mocks for GenAI Client
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    # Mock chunk generator
    class MockPart:
        def __init__(self, data):
            class MockInlineData:
                def __init__(self, d):
                    self.data = d

            self.inline_data = MockInlineData(data)

    class MockContent:
        def __init__(self, data):
            self.parts = [MockPart(data)]

    class MockCandidate:
        def __init__(self, data):
            self.content = MockContent(data)

    class MockChunk:
        def __init__(self, data):
            self.candidates = [MockCandidate(data)]

    async def mock_generator(*args, **kwargs):
        yield MockChunk(b"\x00" * 4800)
        yield MockChunk(b"\x01" * 4800)

    mock_stream.side_effect = mock_generator

    # Initialize GeminiTTSService with api_key explicitly triggering GenAI
    tts = GeminiTTSService(
        api_key="test-api-key",
        settings=GeminiTTSService.Settings(
            model="gemini-3.1-flash-tts-preview",
            voice="Charon",
        ),
    )

    assert tts._use_genai
    assert tts._api_key == "test-api-key"

    # Run test pipeline
    frames_to_send = [
        TTSSpeakFrame(text="Hello world."),
    ]

    frames_received = await run_test(
        tts,
        frames_to_send=frames_to_send,
    )
    down_frames = frames_received[0]
    frame_types = [type(f) for f in down_frames]

    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types
    assert TTSAudioRawFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1
    for a_frame in audio_frames:
        assert a_frame.sample_rate == 24000


@patch("google.oauth2.service_account.Credentials.from_service_account_info")
@patch("google.cloud.texttospeech_v1.TextToSpeechAsyncClient")
def test_env_var_does_not_select_genai_backend(mock_gcp_client_class, mock_from_info):
    """GOOGLE_API_KEY in the env must not flip a credentialed service onto GenAI.

    The env var is commonly set for other Google services (e.g. the LLM), so it
    must not silently override an explicit ``credentials=`` GCP configuration.
    """
    mock_from_info.return_value = MagicMock()

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
        tts = GeminiTTSService(credentials='{"type": "service_account"}')

    assert not tts._use_genai
    # The key is only resolved for the GenAI backend; on GCP it stays unset.
    assert tts._api_key is None


@patch("google.genai.Client")
def test_use_genai_true_resolves_env_api_key(mock_genai_client_class):
    """``use_genai=True`` selects GenAI and resolves the key from the env."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
        tts = GeminiTTSService(use_genai=True)

    assert tts._use_genai
    assert tts._api_key == "env-key"


@patch("google.oauth2.service_account.Credentials.from_service_account_info")
@patch("google.cloud.texttospeech_v1.TextToSpeechAsyncClient")
def test_use_genai_false_overrides_api_key(mock_gcp_client_class, mock_from_info):
    """An explicit ``use_genai=False`` wins even when an ``api_key`` is passed."""
    mock_from_info.return_value = MagicMock()

    tts = GeminiTTSService(
        use_genai=False,
        api_key="explicit-key",
        credentials='{"type": "service_account"}',
    )

    assert not tts._use_genai
    assert tts._api_key is None


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_close_client_closes_genai_session(mock_genai_client_class):
    """The GenAI client's async session is closed; the GCP path is a no-op."""
    mock_client = MagicMock()
    mock_client.aio.aclose = AsyncMock()
    mock_genai_client_class.return_value = mock_client

    tts = GeminiTTSService(api_key="test-api-key")
    await tts._close_client()

    mock_client.aio.aclose.assert_awaited_once()


@patch("google.genai.Client")
@patch("pipecat.services.google.tts.logger")
def test_warns_for_unsupported_genai_settings_at_init(mock_logger, mock_genai_client_class):
    """GenAI ignores prompt/style and multi-speaker; warn once at construction."""
    mock_genai_client_class.return_value = MagicMock()

    GeminiTTSService(
        api_key="test-api-key",
        settings=GeminiTTSService.Settings(prompt="Speak clearly", multi_speaker=True),
    )

    warnings = " ".join(str(call.args[0]) for call in mock_logger.warning.call_args_list)
    assert "Prompt" in warnings
    assert "Multi-speaker" in warnings


@pytest.mark.asyncio
@patch("google.genai.Client")
@patch("pipecat.services.google.tts.logger")
async def test_update_settings_warns_for_unsupported_genai_prompt(
    mock_logger, mock_genai_client_class
):
    """A runtime settings change to an unsupported field warns on the GenAI backend."""
    mock_genai_client_class.return_value = MagicMock()

    tts = GeminiTTSService(api_key="test-api-key")
    mock_logger.reset_mock()

    await tts._update_settings(GeminiTTSService.Settings(prompt="Now use a style"))

    warnings = " ".join(str(call.args[0]) for call in mock_logger.warning.call_args_list)
    assert "Prompt" in warnings


@patch("google.oauth2.service_account.Credentials.from_service_account_info")
@patch("google.cloud.texttospeech_v1.TextToSpeechAsyncClient")
@patch("pipecat.services.google.tts.logger")
def test_no_genai_warning_on_gcp_backend(mock_logger, mock_gcp_client_class, mock_from_info):
    """The GCP backend supports prompt/multi-speaker, so it must not warn about them."""
    mock_from_info.return_value = MagicMock()

    GeminiTTSService(
        credentials='{"type": "service_account"}',
        settings=GeminiTTSService.Settings(prompt="Speak clearly", multi_speaker=True),
    )

    warnings = " ".join(str(call.args[0]) for call in mock_logger.warning.call_args_list)
    assert "Prompt" not in warnings
    assert "Multi-speaker" not in warnings


@pytest.mark.asyncio
@patch("google.oauth2.service_account.Credentials.from_service_account_info")
@patch("google.cloud.texttospeech_v1.TextToSpeechAsyncClient")
async def test_gcp_backend_success(mock_gcp_client_class, mock_from_info):
    # Setup mocks for GCP Client
    mock_creds = MagicMock()
    mock_from_info.return_value = mock_creds

    mock_client = MagicMock()
    mock_gcp_client_class.return_value = mock_client

    mock_streaming_synthesize = AsyncMock()
    mock_client.streaming_synthesize = mock_streaming_synthesize

    # Mock chunk responses
    class MockStreamingResponse:
        def __init__(self, audio_content):
            self.audio_content = audio_content

    async def mock_generator(*args, **kwargs):
        yield MockStreamingResponse(b"\x02" * 4800)
        yield MockStreamingResponse(b"\x03" * 4800)

    mock_streaming_synthesize.return_value = mock_generator()

    # Initialize GeminiTTSService using credentials triggering GCP client
    tts = GeminiTTSService(
        credentials='{"type": "service_account"}',
        settings=GeminiTTSService.Settings(
            model="gemini-3.1-flash-tts-preview",
            voice="Charon",
            prompt="Speak clearly",
        ),
    )

    assert not tts._use_genai

    # Run test pipeline
    frames_to_send = [
        TTSSpeakFrame(text="Hello world."),
    ]

    frames_received = await run_test(
        tts,
        frames_to_send=frames_to_send,
    )
    down_frames = frames_received[0]
    frame_types = [type(f) for f in down_frames]

    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types
    assert TTSAudioRawFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1
