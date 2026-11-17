"""Speech-to-speech services mark their audio transcript frames as spoken.

A speech-to-speech service produces audio itself, so its output transcript
``TTSTextFrame`` describes text the bot actually says. The frames must carry
``will_be_spoken=True`` or RTVI reports them as text that won't be spoken and
clients render the transcript dimmed forever (pipecat-ai/pipecat#6834).
"""

import pytest

from pipecat.frames.frames import TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection


class _FrameRecorder:
    """Records frames passed to a service's ``push_frame``."""

    def __init__(self):
        self.frames = []

    async def __call__(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append(frame)

    def of_types(self, *types):
        return [f for f in self.frames if isinstance(f, types)]


def _transcript_frames(recorder: _FrameRecorder) -> list[TTSTextFrame]:
    frames = recorder.of_types(TTSTextFrame)
    assert frames, "expected at least one transcript TTSTextFrame"
    return frames


@pytest.mark.asyncio
async def test_openai_realtime_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.openai.realtime.llm")
    service = mod.OpenAIRealtimeLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))


@pytest.mark.asyncio
async def test_azure_voice_live_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.azure.voice_live.llm")
    service = mod.AzureVoiceLiveLLMService(
        api_key="test-key", endpoint="https://my-resource.services.ai.azure.com"
    )
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))


@pytest.mark.asyncio
async def test_inworld_realtime_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.inworld.realtime.llm")
    service = mod.InworldRealtimeLLMService(api_key="test")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))


@pytest.mark.asyncio
async def test_xai_realtime_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.xai.realtime.llm")
    service = mod.GrokRealtimeLLMService(api_key="test")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_output_transcript_text_frames("Hello there.")

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))


@pytest.mark.asyncio
async def test_openai_live_assistant_text_is_spoken():
    mod = pytest.importorskip("pipecat.services.openai.live.llm")
    service = mod.OpenAILiveLLMService(api_key="test-key")
    service.push_frame = recorder = _FrameRecorder()

    await service._push_assistant_text("Hello there.")

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))


@pytest.mark.asyncio
async def test_ultravox_agent_transcript_is_spoken():
    mod = pytest.importorskip("pipecat.services.ultravox.llm")
    service = mod.UltravoxRealtimeLLMService(
        params=mod.OneShotInputParams(api_key="test", system_prompt="test")
    )
    service.push_frame = recorder = _FrameRecorder()

    await service._handle_agent_transcript("voice", None, "Hello", False)

    assert all(f.will_be_spoken for f in _transcript_frames(recorder))
