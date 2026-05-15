#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.soniox.stt import END_TOKEN, SonioxSTTService, _language_from_tokens
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def test_language_from_tokens_uses_single_recognized_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " world", "language": "en"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_uses_most_common_language():
    tokens = [
        {"text": "Ik", "language": "nl"},
        {"text": " zoek", "language": "nl"},
        {"text": " computer", "language": "en"},
    ]

    assert _language_from_tokens(tokens) == Language.NL


def test_language_from_tokens_skips_unknown_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": "!", "language": "klingon"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_skips_missing_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " wereld"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_ignores_unknown_and_missing_languages():
    tokens = [
        {"text": "Hello", "language": "klingon"},
        {"text": " world"},
        {"text": "!"},
    ]

    assert _language_from_tokens(tokens) is None


def test_language_from_tokens_uses_first_language_on_tie():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " wereld", "language": "nl"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


@pytest.mark.asyncio
async def test_receive_messages_sets_final_transcription_language(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    pushed_frames = []
    traced_transcriptions = []

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

    async def fake_stop_processing_metrics():
        pass

    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Ik", "is_final": True, "language": "nl"},
                    {"text": " zoek", "is_final": True, "language": "nl"},
                    {"text": " computer", "is_final": True, "language": "en"},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]

    service._websocket = _FakeWebsocket(messages)
    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._receive_messages()

    final_frames = [frame for frame in pushed_frames if isinstance(frame, TranscriptionFrame)]
    assert len(final_frames) == 1
    assert final_frames[0].text == "Ik zoek computer"
    assert final_frames[0].language == Language.NL
    assert final_frames[0].finalized is True
    assert final_frames[0].result == [
        {"text": "Ik", "is_final": True, "language": "nl"},
        {"text": " zoek", "is_final": True, "language": "nl"},
        {"text": " computer", "is_final": True, "language": "en"},
    ]
    assert traced_transcriptions == [("Ik zoek computer", True, Language.NL)]


@pytest.mark.asyncio
async def test_receive_messages_allows_final_transcription_without_language(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    pushed_frames = []
    traced_transcriptions = []

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

    async def fake_stop_processing_metrics():
        pass

    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Tell", "is_final": True},
                    {"text": " me", "is_final": True},
                    {"text": " a", "is_final": True},
                    {"text": " joke.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]

    service._websocket = _FakeWebsocket(messages)
    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._receive_messages()

    final_frames = [frame for frame in pushed_frames if isinstance(frame, TranscriptionFrame)]
    assert len(final_frames) == 1
    assert final_frames[0].text == "Tell me a joke."
    assert final_frames[0].language is None
    assert final_frames[0].finalized is True
    assert traced_transcriptions == [("Tell me a joke.", True, None)]
