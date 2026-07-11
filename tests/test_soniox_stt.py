#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.soniox.stt import END_TOKEN, SonioxSTTService, _language_from_tokens
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


class _FakeWebsocket:
    def __init__(self, messages, *, state=State.OPEN, send_side_effect=None):
        self._messages = messages
        self.state = state
        self.send = AsyncMock(side_effect=send_side_effect)

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


@pytest.mark.asyncio
async def test_connect_failure_clears_stale_websocket_without_raising(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise RuntimeError("connection failed")

    monkeypatch.setattr("pipecat.services.soniox.stt.websocket_connect", fake_websocket_connect)

    service = SonioxSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket([], state=State.CLOSED)

    await service._connect_websocket()

    assert service._websocket is None


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


def _instrumented_service(monkeypatch, events, **kwargs):
    """Create a service that records pushes, broadcasts, and interruptions in order."""
    service = SonioxSTTService(api_key="test-key", **kwargs)

    async def fake_push_frame(frame, direction=None):
        events.append(("push", type(frame)))

    async def fake_broadcast_frame(frame_cls, **frame_kwargs):
        events.append(("broadcast", frame_cls))

    async def fake_broadcast_interruption():
        events.append(("interruption", None))

    async def fake_noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "broadcast_frame", fake_broadcast_frame)
    monkeypatch.setattr(service, "broadcast_interruption", fake_broadcast_interruption)
    monkeypatch.setattr(service, "_handle_transcription", fake_noop)
    monkeypatch.setattr(service, "start_processing_metrics", fake_noop)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_noop)
    return service


def test_service_metadata_recommends_external_turn_strategies_in_soniox_mode():
    service = SonioxSTTService(api_key="test-key", vad_force_turn_endpoint=False)
    frame = service.service_metadata_frame()
    assert isinstance(frame.user_turn_strategies, ExternalUserTurnStrategies)


def test_service_metadata_leaves_turn_strategies_unset_in_pipecat_mode():
    service = SonioxSTTService(api_key="test-key")
    frame = service.service_metadata_frame()
    assert frame.user_turn_strategies is None


@pytest.mark.asyncio
async def test_soniox_turn_detection_emits_turn_frames(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert events == [
        # Turn opens on the first token, before any transcription frames.
        ("broadcast", UserStartedSpeakingFrame),
        ("interruption", None),
        ("push", InterimTranscriptionFrame),
        # Endpoint: finalized transcript first, then the turn closes.
        ("push", TranscriptionFrame),
        ("broadcast", UserStoppedSpeakingFrame),
    ]
    assert service._user_turn_open is False


@pytest.mark.asyncio
async def test_soniox_turn_detection_should_interrupt_false(monkeypatch):
    events = []
    service = _instrumented_service(
        monkeypatch, events, vad_force_turn_endpoint=False, should_interrupt=False
    )

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert ("broadcast", UserStartedSpeakingFrame) in events
    assert ("interruption", None) not in events


@pytest.mark.asyncio
async def test_soniox_turn_detection_reopens_turn_after_end_token(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    # A single message can close one turn and start the next.
    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                    {"text": "And", "is_final": False},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert events == [
        ("broadcast", UserStartedSpeakingFrame),
        ("interruption", None),
        ("push", TranscriptionFrame),
        ("broadcast", UserStoppedSpeakingFrame),
        # Tokens after the endpoint open a new turn.
        ("broadcast", UserStartedSpeakingFrame),
        ("interruption", None),
        ("push", InterimTranscriptionFrame),
        # The finished message closes the still-open turn.
        ("broadcast", UserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_soniox_turn_detection_no_duplicate_started_across_messages(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps({"tokens": [{"text": "Hello", "is_final": True}]}),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    started = [event for event in events if event == ("broadcast", UserStartedSpeakingFrame)]
    assert len(started) == 1


@pytest.mark.asyncio
async def test_pipecat_mode_emits_no_turn_frames(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events)

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert events == [
        ("push", InterimTranscriptionFrame),
        ("push", TranscriptionFrame),
    ]


@pytest.mark.asyncio
async def test_vad_start_opens_turn_before_tokens(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    # The local VAD signal is the fast path: the turn opens before any token.
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert events == [
        ("push", VADUserStartedSpeakingFrame),  # re-pushed by the base STTService
        ("broadcast", UserStartedSpeakingFrame),
        ("interruption", None),
    ]

    # Tokens arriving later must not open a second turn.
    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    started = [event for event in events if event == ("broadcast", UserStartedSpeakingFrame)]
    assert len(started) == 1
    assert events[-2:] == [
        ("push", TranscriptionFrame),
        ("broadcast", UserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_vad_stop_does_not_close_turn(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    # The Soniox endpoint owns the turn close: a VAD stop (e.g. a mid-turn
    # pause) must not close it.
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert ("broadcast", UserStoppedSpeakingFrame) not in events
    assert service._user_turn_open is True


@pytest.mark.asyncio
async def test_pipecat_mode_vad_frames_emit_no_turn_frames(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events)

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert events == [
        ("push", VADUserStartedSpeakingFrame),
        ("push", VADUserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_soniox_turn_detection_error_closes_open_turn(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    async def fake_push_error(*args, **kwargs):
        events.append(("error", None))

    monkeypatch.setattr(service, "push_error", fake_push_error)

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps({"tokens": [], "error_code": 500, "error_message": "boom"}),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert events == [
        ("broadcast", UserStartedSpeakingFrame),
        ("interruption", None),
        ("push", InterimTranscriptionFrame),
        ("broadcast", UserStoppedSpeakingFrame),
        ("error", None),
    ]


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
