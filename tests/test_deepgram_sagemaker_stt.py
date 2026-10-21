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

from pipecat.frames.frames import TranscriptionFrame  # noqa: E402
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
    assert [(f.speaker_label, f.text, f.user_id) for f in frames] == [
        ("0", "Hi.", "session-user"),
        ("1", "Hello.", "session-user"),
    ]
