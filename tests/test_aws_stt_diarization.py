#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWS Transcribe speaker diarization."""

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import TranscriptionFrame, TranscriptionSpeaker
from pipecat.services.aws import stt as aws_stt
from pipecat.services.aws.stt import AWSTranscribeSTTService


class _FakeWebsocket:
    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _service(**settings) -> AWSTranscribeSTTService:
    return AWSTranscribeSTTService(
        api_key="secret",
        aws_access_key_id="key-id",
        region="us-east-1",
        settings=AWSTranscribeSTTService.Settings(**settings),
    )


def _final_result(transcript: str, items: list[dict]) -> dict:
    return {
        "Transcript": {
            "Results": [
                {
                    "IsPartial": False,
                    "Alternatives": [{"Transcript": transcript, "Items": items}],
                }
            ]
        }
    }


def _word(content: str, speaker: str) -> dict:
    return {"Type": "pronunciation", "Content": content, "Speaker": speaker}


def _punctuation(content: str) -> dict:
    return {"Type": "punctuation", "Content": content}


async def _final_frames(service, payload, monkeypatch) -> list[TranscriptionFrame]:
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    monkeypatch.setattr(
        aws_stt, "decode_event", lambda message: ({":message-type": "event"}, message)
    )
    service.push_frame = fake_push_frame
    service._websocket = _FakeWebsocket([payload])

    await service._receive_messages()

    return [frame for frame in pushed if isinstance(frame, TranscriptionFrame)]


@pytest.mark.asyncio
async def test_diarized_final_result_emits_one_frame_per_speaker_run(monkeypatch):
    service = _service(show_speaker_label=True, speaker_format="{speaker}: {text}")
    service._user_id = "session-user"
    payload = _final_result(
        "Hi there. Hello.",
        [
            _word("Hi", "0"),
            _word("there", "0"),
            _punctuation("."),
            _word("Hello", "1"),
            _punctuation("."),
        ],
    )

    frames = await _final_frames(service, payload, monkeypatch)

    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("0", 1), "0: Hi there."),
        (TranscriptionSpeaker("1", 2), "1: Hello."),
    ]
    assert all(f.user_id == "session-user" for f in frames)


@pytest.mark.asyncio
async def test_final_result_is_not_split_without_speaker_labels(monkeypatch):
    service = _service()
    payload = _final_result("Hi. Hello.", [_word("Hi", "0"), _word("Hello", "1")])

    frames = await _final_frames(service, payload, monkeypatch)

    assert [(f.speaker, f.text) for f in frames] == [(None, "Hi. Hello.")]


@pytest.mark.asyncio
async def test_speaker_format_update_does_not_reconnect():
    service = _service()
    service._websocket = object()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    changed = await service._update_settings(
        AWSTranscribeSTTService.Settings(speaker_format="{speaker}: {text}")
    )

    assert "speaker_format" in changed
    service._disconnect.assert_not_awaited()
