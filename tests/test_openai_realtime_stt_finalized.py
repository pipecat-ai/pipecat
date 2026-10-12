#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Finalized utterance signals from OpenAI realtime transcription events."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.openai.stt import OpenAIRealtimeSTTService


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_detection", [False, {"type": "server_vad"}])
async def test_completed_transcript_is_finalized(monkeypatch, turn_detection):
    service = OpenAIRealtimeSTTService(api_key="test-key", turn_detection=turn_detection)
    calls = []
    emit_usage = AsyncMock(side_effect=lambda: calls.append("usage"))
    push_frame = AsyncMock(side_effect=lambda frame: calls.append("transcript"))
    trace = AsyncMock(side_effect=lambda *args: calls.append("trace"))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", emit_usage)
    monkeypatch.setattr(service, "push_frame", push_frame)
    monkeypatch.setattr(service, "_handle_transcription_trace", trace)
    event = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "utterance-1",
        "transcript": "Hello world.",
    }

    await service._handle_transcription_completed(event)

    frame = push_frame.await_args.args[0]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "Hello world."
    assert frame.finalized is True
    assert frame.result is event
    assert calls == ["usage", "transcript", "trace"]
    trace.assert_awaited_once_with("Hello world.", True)


@pytest.mark.asyncio
async def test_transcription_delta_remains_interim(monkeypatch):
    service = OpenAIRealtimeSTTService(api_key="test-key")
    push_frame = AsyncMock()
    monkeypatch.setattr(service, "push_frame", push_frame)
    event = {"type": "conversation.item.input_audio_transcription.delta", "delta": "Hello"}

    await service._handle_transcription_delta(event)

    frame = push_frame.await_args.args[0]
    assert isinstance(frame, InterimTranscriptionFrame)
    assert frame.text == "Hello"
    assert frame.result is event


@pytest.mark.asyncio
async def test_empty_completed_transcript_emits_no_frame(monkeypatch):
    service = OpenAIRealtimeSTTService(api_key="test-key")
    push_frame = AsyncMock()
    monkeypatch.setattr(service, "push_frame", push_frame)

    await service._handle_transcription_completed(
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}
    )

    push_frame.assert_not_awaited()
