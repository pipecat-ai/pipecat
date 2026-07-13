#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Google STT streaming response handling."""

import time
from types import SimpleNamespace

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.google.stt import GoogleSTTService


class AsyncResponses:
    """Minimal async iterator for Google streaming responses."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def result(*, transcript: str, is_final: bool):
    return SimpleNamespace(
        alternatives=[SimpleNamespace(transcript=transcript)],
        is_final=is_final,
    )


@pytest.mark.asyncio
async def test_google_final_result_emits_finalized_transcription_frame():
    service = object.__new__(GoogleSTTService)
    service._stream_start_time = int(time.time() * 1000)
    service._user_id = "user"
    service._last_transcript_was_final = False
    service._get_language_codes = lambda: ["en-US"]

    frames = []
    transcriptions = []

    async def push_frame(frame):
        frames.append(frame)

    async def stop_processing_metrics():
        pass

    async def handle_transcription(transcript, is_final, language=None):
        transcriptions.append((transcript, is_final, language))

    service.push_frame = push_frame
    service.stop_processing_metrics = stop_processing_metrics
    service._handle_transcription = handle_transcription

    responses = AsyncResponses(
        [
            SimpleNamespace(results=[result(transcript="hel", is_final=False)]),
            SimpleNamespace(results=[result(transcript="hello", is_final=True)]),
        ]
    )

    await service._process_responses(responses)

    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].finalized is True
    assert transcriptions == [("hello", True, "en-US")]
