#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Deepgram SageMaker STT service's query string."""

import asyncio
from urllib.parse import parse_qs

import pytest

pytest.importorskip("pipecat.services.deepgram.sagemaker.stt", exc_type=ImportError)

from pipecat.frames.frames import (
    TranscriptionFrame,  # noqa: E402
    TranscriptionSpeaker,
)
from pipecat.services.deepgram.sagemaker.stt import DeepgramSageMakerSTTService  # noqa: E402


def _query(settings: DeepgramSageMakerSTTService.Settings) -> dict[str, list[str]]:
    service = DeepgramSageMakerSTTService(
        endpoint_name="endpoint", region="us-east-1", sample_rate=16000, settings=settings
    )
    return parse_qs(service._build_query_string())


def test_list_settings_become_one_parameter_per_item():
    query = _query(DeepgramSageMakerSTTService.Settings(keyterm=["pipecat", "daily"]))

    assert query["keyterm"] == ["pipecat", "daily"]


def test_values_are_url_encoded():
    query = _query(
        DeepgramSageMakerSTTService.Settings(
            keyterm=["Q&A session"], extra={"tag": "a:b c", "flag": True}
        )
    )

    assert query["keyterm"] == ["Q&A session"]
    assert query["tag"] == ["a:b c"]
    assert query["flag"] == ["true"]


def test_speaker_format_is_not_sent():
    query = _query(DeepgramSageMakerSTTService.Settings(speaker_format="{speaker}: {text}"))

    assert "speaker_format" not in query


def test_diarized_final_response_emits_one_frame_per_speaker_run():
    service = DeepgramSageMakerSTTService(
        endpoint_name="endpoint",
        region="us-east-1",
        sample_rate=16000,
        settings=DeepgramSageMakerSTTService.Settings(diarize=True),
    )
    service._user_id = "session-user"
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame
    response = {
        "is_final": True,
        "channel": {
            "alternatives": [
                {
                    "transcript": "Hi. Hello.",
                    "words": [
                        {"word": "hi", "punctuated_word": "Hi.", "speaker": 0},
                        {"word": "hello", "punctuated_word": "Hello.", "speaker": 1},
                    ],
                }
            ]
        },
    }

    asyncio.run(service._handle_transcript_response(response))

    frames = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert [(f.speaker, f.text, f.user_id) for f in frames] == [
        (TranscriptionSpeaker("0", 1), "Hi.", "session-user"),
        (TranscriptionSpeaker("1", 2), "Hello.", "session-user"),
    ]


def _service_capturing_frames_below_the_stt_service(monkeypatch):
    """Capture frames below STTService.push_frame, so its finalize handling still runs."""
    from pipecat.services.ai_service import AIService

    pushed: list = []

    async def fake_push_frame(self, frame, direction=None):
        pushed.append(frame)

    monkeypatch.setattr(AIService, "push_frame", fake_push_frame)
    service = DeepgramSageMakerSTTService(
        endpoint_name="endpoint", region="us-east-1", sample_rate=16000
    )
    service._user_id = "session-user"
    return service, pushed


def _response(transcript: str, *, from_finalize: bool = False) -> dict:
    return {
        "is_final": True,
        "from_finalize": from_finalize,
        "channel": {"alternatives": [{"transcript": transcript}]},
    }


def test_empty_finalize_response_finalizes_the_transcript_before_it(monkeypatch):
    """Deepgram's endpointing can send the transcript before it is asked to finalize."""
    from pipecat.frames.frames import TranscriptionFinalizedFrame

    service, pushed = _service_capturing_frames_below_the_stt_service(monkeypatch)

    async def run():
        await service._handle_transcript_response(_response("Hello there."))
        # The VAD stop asks Deepgram to finalize.
        service.request_finalize()
        await service._handle_transcript_response(_response("", from_finalize=True))

    asyncio.run(run())

    transcript, confirmation = pushed
    assert (transcript.text, transcript.finalized) == ("Hello there.", False)
    assert isinstance(confirmation, TranscriptionFinalizedFrame)
    assert confirmation.user_id == "session-user"


def test_empty_finalize_response_nobody_asked_for_finalizes_nothing(monkeypatch):
    """A confirmation that arrives after the user spoke again belongs to no utterance."""
    service, pushed = _service_capturing_frames_below_the_stt_service(monkeypatch)

    asyncio.run(service._handle_transcript_response(_response("", from_finalize=True)))

    assert pushed == []
