#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for NvidiaSageMakerTTSService."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from aws_sdk_sagemaker_runtime_http2.models import (  # noqa: E402
    ResponsePayloadPart,
    ResponseStreamEventPayloadPart,
)

import pipecat.services.nvidia.sagemaker.tts as nvidia_sagemaker_tts  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.nvidia.sagemaker.tts import NvidiaSageMakerTTSService  # noqa: E402
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402


class _FakeBidiClient:
    """Answers each committed text with half a second of 16 kHz audio."""

    def __init__(self, **kwargs):
        self.is_active = False
        self._responses: asyncio.Queue[ResponseStreamEventPayloadPart] = asyncio.Queue()

    async def start_session(self):
        self.is_active = True

    async def send_json(self, data: dict):
        if data["type"] == "input_text.commit":
            audio = base64.b64encode(b"\x00\x01" * 8000).decode()
            self._respond({"type": "conversation.item.speech.data", "audio": audio})

    async def receive_response(self) -> ResponseStreamEventPayloadPart:
        return await self._responses.get()

    async def close_session(self):
        self.is_active = False

    def _respond(self, msg: dict):
        payload = ResponsePayloadPart(bytes_=json.dumps(msg).encode())
        self._responses.put_nowait(ResponseStreamEventPayloadPart(value=payload))


@pytest.mark.asyncio
async def test_audio_plays_in_its_audio_context(monkeypatch):
    """Each request's audio plays between its started and stopped frames, without errors."""
    monkeypatch.setattr(nvidia_sagemaker_tts, "SageMakerBidiClient", _FakeBidiClient)
    service = NvidiaSageMakerTTSService(
        endpoint_name="endpoint", region="us-west-2", sample_rate=16000, stop_frame_timeout_s=0.2
    )

    down, up = await asyncio.wait_for(
        run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame("One."),
                TTSSpeakFrame("Two."),
                TTSSpeakFrame("Three."),
                SleepFrame(sleep=0.5),
            ],
            expected_up_frames=None,
        ),
        timeout=5,
    )

    frame_types = (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)
    assert [type(f) for f in down if isinstance(f, frame_types)] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    ] * 3
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert service.is_usable
