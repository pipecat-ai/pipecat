#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64

import pytest

from pipecat.services.huggingface.stt import (
    HuggingFaceSTTService,
    HuggingFaceSTTSettings,
)


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
    closed = False

    def __init__(self):
        self.url = None
        self.payload = None
        self.headers = None

    def post(self, url, *, json, headers):
        self.url = url
        self.payload = json
        self.headers = headers
        return _FakeResponse()


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
