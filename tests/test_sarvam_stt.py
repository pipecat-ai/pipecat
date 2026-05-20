#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SarvamSTTService transcription-frame emission.

The Sarvam streaming endpoint emits a single ``data`` message per utterance
that is the explicit terminal transcript (it follows the ``END_SPEECH`` VAD
event). This test pins that the resulting TranscriptionFrame is marked with
``finalized=True`` so downstream user-turn-stop strategies can fast-path
through to LLM invocation rather than waiting on the STT-timeout fallback.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transcriptions.language import Language


def _make_service() -> SarvamSTTService:
    """Construct without touching the network (no WS connect happens in __init__)."""
    return SarvamSTTService(
        api_key="test-api-key",
        sample_rate=8000,
        input_audio_codec="wav",
    )


def _data_message(transcript: str, language_code: str = "en-IN"):
    """Build a fake Sarvam data-event message shaped like the SDK response."""
    return SimpleNamespace(
        type="data",
        data=SimpleNamespace(
            transcript=transcript,
            language_code=language_code,
        ),
        dict=lambda: {"type": "data", "transcript": transcript},
    )


@pytest.fixture
def service() -> SarvamSTTService:
    return _make_service()


@pytest.mark.asyncio
async def test_data_event_emits_finalized_transcription_frame(
    service: SarvamSTTService,
) -> None:
    """The terminal data-event TranscriptionFrame must have finalized=True.

    User-turn-stop strategies (TurnAnalyzerUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy) fast-path on finalized=True; without
    it they fall through to the SARVAM_TTFS_P99 (1.17s) STT-timeout
    fallback, adding ~1s to TTFB on every turn.
    """
    pushed = []

    async def capture(frame):
        pushed.append(frame)

    with (
        patch.object(service, "push_frame", side_effect=capture),
        patch.object(service, "_call_event_handler", new=AsyncMock()),
        patch.object(service, "stop_processing_metrics", new=AsyncMock()),
        patch.object(service, "_handle_transcription", new=AsyncMock()),
    ):
        await service._handle_message(_data_message("hello world"))

    frames = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert len(frames) == 1
    assert frames[0].finalized is True
    assert frames[0].text == "hello world"
    assert frames[0].language == Language.EN_IN


@pytest.mark.asyncio
async def test_empty_transcript_is_dropped(service: SarvamSTTService) -> None:
    """Whitespace-only transcripts emit no frame — preserves existing behavior."""
    pushed = []

    async def capture(frame):
        pushed.append(frame)

    with (
        patch.object(service, "push_frame", side_effect=capture),
        patch.object(service, "_call_event_handler", new=AsyncMock()),
        patch.object(service, "stop_processing_metrics", new=AsyncMock()),
        patch.object(service, "_handle_transcription", new=AsyncMock()),
    ):
        await service._handle_message(_data_message("   "))

    assert [f for f in pushed if isinstance(f, TranscriptionFrame)] == []


@pytest.mark.asyncio
async def test_language_falls_back_to_hi_in_when_message_lacks_code(
    service: SarvamSTTService,
) -> None:
    """No language_code in message and no configured language → Language.HI_IN."""
    pushed = []

    async def capture(frame):
        pushed.append(frame)

    msg = SimpleNamespace(
        type="data",
        data=SimpleNamespace(transcript="नमस्ते", language_code=None),
        dict=lambda: {"type": "data"},
    )

    with (
        patch.object(service, "push_frame", side_effect=capture),
        patch.object(service, "_call_event_handler", new=AsyncMock()),
        patch.object(service, "stop_processing_metrics", new=AsyncMock()),
        patch.object(service, "_handle_transcription", new=AsyncMock()),
    ):
        await service._handle_message(msg)

    frames = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert len(frames) == 1
    assert frames[0].language == Language.HI_IN
    assert frames[0].finalized is True


@pytest.mark.asyncio
async def test_events_message_does_not_push_transcription_frame(
    service: SarvamSTTService,
) -> None:
    """VAD events (START_SPEECH, END_SPEECH) go through broadcast_frame, not push_frame."""
    pushed = []

    async def capture(frame):
        pushed.append(frame)

    msg = SimpleNamespace(
        type="events",
        data=SimpleNamespace(signal_type="END_SPEECH", occured_at="t0"),
        dict=lambda: {"type": "events"},
    )

    with (
        patch.object(service, "push_frame", side_effect=capture),
        patch.object(service, "broadcast_frame", new=AsyncMock()),
        patch.object(service, "_call_event_handler", new=AsyncMock()),
        patch.object(service, "push_interruption_task_frame_and_wait", new=AsyncMock()),
        patch.object(service, "_start_metrics", new=AsyncMock()),
    ):
        await service._handle_message(msg)

    assert pushed == []
