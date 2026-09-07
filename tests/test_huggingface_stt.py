#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.huggingface.stt import (
    HuggingFaceSTTService,
    HuggingFaceSTTSettings,
)
from pipecat.transcriptions.language import Language


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {"text": "hello", "chunks": []}

    async def text(self):
        return '{"text":"hello","chunks":[]}'


class _FakeSession:
    def __init__(self):
        self.closed = False
        self.url = None
        self.payload = None
        self.headers = None

    def post(self, url, *, json, headers):
        self.url = url
        self.payload = json
        self.headers = headers
        return _FakeResponse()

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_huggingface_stt_sends_router_payload():
    session = _FakeSession()
    service = HuggingFaceSTTService(
        api_key="hf_test",
        aiohttp_session=session,
        base_url="https://router.huggingface.co/hf-inference",
        bill_to="demo-org",
        settings=HuggingFaceSTTSettings(
            model="openai/whisper-large-v3-turbo",
            return_timestamps=True,
            generation_parameters={"temperature": 0.0},
        ),
    )

    result = await service._transcribe_audio(b"RIFF")

    assert result["text"] == "hello"
    assert session.url == (
        "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
    )
    assert session.headers["Authorization"] == "Bearer hf_test"
    assert session.headers["X-HF-Bill-To"] == "demo-org"
    assert session.payload["inputs"] == base64.b64encode(b"RIFF").decode("utf-8")
    assert session.payload["parameters"] == {
        "return_timestamps": True,
        "generation_parameters": {"temperature": 0.0},
    }


@pytest.mark.asyncio
async def test_cleanup_does_not_close_a_borrowed_session():
    session = _FakeSession()
    service = HuggingFaceSTTService(api_key="hf_test", aiohttp_session=session)

    await service.cleanup()

    assert service._session is session
    assert not session.closed


@pytest.mark.asyncio
async def test_cleanup_closes_and_releases_an_owned_session():
    session = _FakeSession()
    service = HuggingFaceSTTService(api_key="hf_test")
    service._session = session

    await service.cleanup()

    assert session.closed
    assert service._session is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "expected"),
    [(Language.EN_US, "en"), ("custom-provider-code", "custom-provider-code")],
)
async def test_transcription_frame_preserves_language_metadata(language, expected):
    service = HuggingFaceSTTService(
        api_key="hf_test",
        settings=HuggingFaceSTTSettings(language=language),
    )
    service._transcribe_audio = AsyncMock(return_value={"text": "hello"})
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()

    frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].language == expected
