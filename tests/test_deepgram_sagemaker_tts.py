#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DeepgramSageMakerTTSService."""

import asyncio
import json

import pytest

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from aws_sdk_sagemaker_runtime_http2.models import (  # noqa: E402
    ResponsePayloadPart,
    ResponseStreamEventPayloadPart,
)

import pipecat.services.deepgram.sagemaker.tts as deepgram_sagemaker_tts  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402


class _FakeBidiClient:
    """Answers each Speak message with audio and each Flush with Flushed."""

    def __init__(self, **kwargs):
        self.is_active = False
        self._responses: asyncio.Queue[ResponseStreamEventPayloadPart] = asyncio.Queue()

    async def start_session(self):
        self.is_active = True

    async def send_json(self, data: dict):
        if data["type"] == "Speak":
            self._respond(b"\x00\x01" * 1600)
        elif data["type"] == "Flush":
            self._respond(json.dumps({"type": "Flushed", "sequence_id": 0}).encode())

    async def receive_response(self) -> ResponseStreamEventPayloadPart:
        return await self._responses.get()

    async def close_session(self):
        self.is_active = False

    def _respond(self, payload: bytes):
        part = ResponseStreamEventPayloadPart(value=ResponsePayloadPart(bytes_=payload))
        self._responses.put_nowait(part)


@pytest.mark.asyncio
async def test_flushed_ends_the_audio_context(monkeypatch):
    """Each context ends when Deepgram reports its flush, before the stop frame timeout."""
    monkeypatch.setattr(deepgram_sagemaker_tts, "SageMakerBidiClient", _FakeBidiClient)
    service = DeepgramSageMakerTTSService(
        endpoint_name="endpoint", region="us-east-2", sample_rate=16000, stop_frame_timeout_s=10
    )

    down, _ = await asyncio.wait_for(
        run_test(service, frames_to_send=[TTSSpeakFrame("Hello."), TTSSpeakFrame("Bye.")]),
        timeout=5,
    )

    frame_types = (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)
    assert [type(f) for f in down if isinstance(f, frame_types)] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    ]
