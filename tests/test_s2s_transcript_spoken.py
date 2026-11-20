#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speech-to-speech services report their audio transcript as spoken text.

A speech-to-speech service produces audio itself, so its output transcript is
text the bot actually says. Like a TTS service, it announces each transcript
segment with a ``will_be_spoken`` ``AggregatedTextFrame`` and then reports it
spoken with a ``TTSTextFrame`` that references the announcement, so RTVI sends
the segment as ``new`` (clients add its text) and then ``completed``.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import AggregatedTextFrame, BotStartedSpeakingFrame, TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


class _FrameRecorder:
    """Records frames passed to a service's ``push_frame``."""

    def __init__(self):
        self.frames = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append(frame)


def _assert_announced_then_spoken(recorder: _FrameRecorder, text: str):
    """Assert the transcript is announced, then reported spoken for that segment."""
    frames = [f for f in recorder.frames if isinstance(f, AggregatedTextFrame)]
    assert len(frames) == 2, frames
    announcement, spoken = frames

    assert not isinstance(announcement, TTSTextFrame)
    assert announcement.text == text
    assert announcement.will_be_spoken
    assert not announcement.append_to_context

    assert isinstance(spoken, TTSTextFrame)
    assert spoken.text == text
    assert spoken.will_be_spoken
    assert spoken.append_to_context
    assert spoken.segment_id == announcement.id


@pytest.mark.asyncio
async def test_openai_realtime_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.openai.realtime.llm")
    service = mod.OpenAIRealtimeLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_azure_voice_live_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.azure.voice_live.llm")
    service = mod.AzureVoiceLiveLLMService(
        api_key="test-key", endpoint="https://my-resource.services.ai.azure.com"
    )
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_inworld_realtime_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.inworld.realtime.llm")
    service = mod.InworldRealtimeLLMService(api_key="test")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_xai_realtime_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.xai.realtime.llm")
    service = mod.GrokRealtimeLLMService(api_key="test")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_openai_live_assistant_text_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.openai.live.llm")
    service = mod.OpenAILiveLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_assistant_text("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_gemini_live_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.google.gemini_live.llm")
    service = mod.GeminiLiveLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcription_text_frames("Hello there.")

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_nova_sonic_transcript_is_announced_then_spoken():
    mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
    service = mod.AWSNovaSonicLLMService(
        secret_access_key="test", access_key_id="test", region="us-east-1"
    )
    service.push_frame = recorder = _FrameRecorder()

    service._pending_speculative_text = "Hello there."
    await service._report_tts_text()

    _assert_announced_then_spoken(recorder, "Hello there.")


@pytest.mark.asyncio
async def test_ultravox_agent_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.ultravox.llm")
    service = mod.UltravoxRealtimeLLMService(
        params=mod.OneShotInputParams(api_key="test", system_prompt="test")
    )
    service.push_frame = recorder = _FrameRecorder()

    await service._handle_agent_transcript("voice", None, "Hello", False)

    frames = [f for f in recorder.frames if isinstance(f, TTSTextFrame)]
    assert frames
    assert all(f.will_be_spoken for f in frames)


@pytest.mark.asyncio
async def test_rtvi_reports_the_transcript_as_new_then_completed():
    mod = pytest.importorskip("pipecat.services.openai.realtime.llm")
    service = mod.OpenAIRealtimeLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()
    await service._push_output_transcript_text_frames("Hello there.")

    observer = RTVIObserver(params=RTVIObserverParams())
    observer.send_rtvi_message = AsyncMock()
    transport = BaseOutputTransport(TransportParams())
    for frame in [BotStartedSpeakingFrame(), *recorder.frames]:
        await observer.on_push_frame(
            FramePushed(
                source=transport,
                destination=transport,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=True,
            )
        )

    outputs = [
        call.args[0].data
        for call in observer.send_rtvi_message.await_args_list
        if call.args[0].type == "bot-output"
    ]
    assert [(d.text, d.will_be_spoken, d.spoken_status) for d in outputs] == [
        ("Hello there.", True, "new"),
        ("Hello there.", True, "completed"),
    ]
    assert outputs[0].segment_id == outputs[1].segment_id
