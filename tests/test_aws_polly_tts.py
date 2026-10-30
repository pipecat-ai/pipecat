#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how AWSPollyTTSService manages its Polly client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# Two 20 ms frames of 16 kHz mono PCM.
AUDIO = b"\x01\x00" * 640


def _service() -> AWSPollyTTSService:
    return AWSPollyTTSService(
        api_key="secret", aws_access_key_id="key", region="us-east-1", sample_rate=16000
    )


def _polly_client(stream: MagicMock | None = None) -> tuple[MagicMock, MagicMock]:
    """A Polly client double, and the context manager that hands it out."""
    stream = stream or MagicMock(read=AsyncMock(return_value=AUDIO))
    client = AsyncMock()
    client.synthesize_speech.return_value = {"AudioStream": stream}
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    return client, context_manager


async def _speak(service: AWSPollyTTSService, text: str = "Hello there.") -> list:
    return [frame async for frame in service.run_tts(text, "ctx-1")]


@pytest.mark.asyncio
async def test_each_request_gets_its_own_client_before_setup():
    """Outside a pipeline nothing would close a kept client, so none is kept."""
    service = _service()
    service._sample_rate = 16000
    client, context_manager = _polly_client()

    with patch.object(
        service._aws_session, "create_client", return_value=context_manager
    ) as create:
        await _speak(service)
        await _speak(service)

    assert create.call_count == 2
    assert context_manager.__aexit__.await_count == 2
    assert client.synthesize_speech.await_count == 2


@pytest.mark.asyncio
async def test_requests_share_one_client_once_set_up():
    """After setup() every sentence goes through the same client until cleanup()."""
    service = _service()
    client, context_manager = _polly_client()

    with patch.object(
        service._aws_session, "create_client", return_value=context_manager
    ) as create:
        await service.setup(frame_processor_setup(TaskManager()))
        first = await _speak(service)
        second = await _speak(service)

        assert create.call_count == 1
        assert client.synthesize_speech.await_count == 2
        context_manager.__aexit__.assert_not_awaited()
        assert any(isinstance(frame, TTSAudioRawFrame) for frame in first + second)

        await service.cleanup()

    assert context_manager.__aexit__.await_count == 1


@pytest.mark.asyncio
async def test_client_keeps_idle_connections_open_between_sentences():
    """The client's connector holds a connection longer than aiobotocore's 12 s default."""
    service = _service()
    _, context_manager = _polly_client()

    with patch.object(
        service._aws_session, "create_client", return_value=context_manager
    ) as create:
        await service.setup(frame_processor_setup(TaskManager()))
        await service.cleanup()

    assert create.call_args.args == ("polly",)
    assert create.call_args.kwargs["config"].connector_args["keepalive_timeout"] > 12


@pytest.mark.asyncio
async def test_audio_stream_is_released_when_reading_it_fails():
    """A body that cannot be read to the end is closed rather than left holding a connection."""
    service = _service()
    stream = MagicMock(read=AsyncMock(side_effect=ConnectionResetError("reset")))
    _, context_manager = _polly_client(stream)

    with patch.object(service._aws_session, "create_client", return_value=context_manager):
        await service.setup(frame_processor_setup(TaskManager()))
        with pytest.raises(ConnectionResetError):
            await _speak(service)
        await service.cleanup()

    stream.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_a_response_without_audio_pushes_no_frames():
    """Polly answering without an audio stream yields nothing and keeps the client usable."""
    service = _service()
    client, context_manager = _polly_client()
    client.synthesize_speech.return_value = {}

    with patch.object(service._aws_session, "create_client", return_value=context_manager):
        await service.setup(frame_processor_setup(TaskManager()))
        frames = await _speak(service)
        await service.cleanup()

    assert not [f for f in frames if isinstance(f, (TTSAudioRawFrame, ErrorFrame))]
