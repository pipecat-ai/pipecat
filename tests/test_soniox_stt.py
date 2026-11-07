#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.bot_speech_observer import BotSpeechObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.soniox.stt import (
    DEFAULT_AGENT_TEXT_LEAD_MS,
    END_TOKEN,
    FINALIZED_TOKEN,
    SonioxSTTService,
    _language_from_tokens,
    language_to_soniox_language,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.time import time_now_iso8601
from tests.frame_processor_helpers import frame_processor_setup


class _FakeWebsocket:
    def __init__(self, messages, *, state=State.OPEN, send_side_effect=None):
        self._messages = messages
        self.state = state
        self.closed = False
        self.send = AsyncMock(side_effect=send_side_effect)

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


@pytest.mark.asyncio
async def test_connect_sends_api_key_on_the_handshake(monkeypatch):
    captured = {}
    websocket = _FakeWebsocket([], state=State.OPEN)

    async def fake_websocket_connect(*args, **kwargs):
        captured.update(kwargs)
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = SonioxSTTService(api_key="test-key")
    await service._connect_websocket()

    assert captured["additional_headers"] == {"Authorization": "Bearer test-key"}
    # The handshake is the only carrier; the config message no longer repeats the key.
    assert "api_key" not in json.loads(websocket.send.await_args.args[0])


@pytest.mark.asyncio
async def test_connect_failure_clears_stale_websocket_without_raising(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise RuntimeError("connection failed")

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

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
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", InterimTranscriptionFrame),
        # Endpoint: finalized transcript first, then the turn closes.
        ("push", TranscriptionFrame),
        ("broadcast", ProposedUserStoppedSpeakingFrame),
    ]
    assert service._user_turn_open is False


@pytest.mark.asyncio
async def test_soniox_turn_detection_never_interrupts_directly(monkeypatch):
    """The service proposes turns; the strategies own the interruption."""
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    messages = [
        json.dumps({"tokens": [{"text": "Hel", "is_final": False}]}),
        json.dumps({"tokens": [], "finished": True}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert ("broadcast", ProposedUserStartedSpeakingFrame) in events
    assert ("interruption", None) not in events


def test_soniox_should_interrupt_rides_on_recommended_strategies():
    # should_interrupt no longer gates a local broadcast; it configures the
    # strategies the service recommends via its metadata frame.
    for should_interrupt in (True, False):
        service = SonioxSTTService(
            api_key="test-key",
            vad_force_turn_endpoint=False,
            should_interrupt=should_interrupt,
        )
        strategies = service.service_metadata_frame().user_turn_strategies
        assert isinstance(strategies, ExternalUserTurnStrategies)
        assert strategies.enable_interruptions is should_interrupt


def test_soniox_pipecat_mode_recommends_no_strategies():
    """In the default Pipecat mode Soniox proposes no turns, so the defaults stand."""
    service = SonioxSTTService(api_key="test-key", vad_force_turn_endpoint=True)
    assert service.service_metadata_frame().user_turn_strategies is None


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
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", TranscriptionFrame),
        ("broadcast", ProposedUserStoppedSpeakingFrame),
        # Tokens after the endpoint open a new turn.
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", InterimTranscriptionFrame),
        # The finished message closes the still-open turn.
        ("broadcast", ProposedUserStoppedSpeakingFrame),
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

    started = [
        event for event in events if event == ("broadcast", ProposedUserStartedSpeakingFrame)
    ]
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
@pytest.mark.parametrize("vad_force_turn_endpoint", [True, False])
async def test_connect_enables_endpoint_detection_in_both_modes(
    monkeypatch, vad_force_turn_endpoint
):
    """Endpoint detection ends the transcript of speech the local VAD misses."""
    websocket = _FakeWebsocket([])

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    service = SonioxSTTService(api_key="test-key", vad_force_turn_endpoint=vad_force_turn_endpoint)

    await service._connect_websocket()

    config = json.loads(websocket.send.call_args.args[0])
    assert config["enable_endpoint_detection"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        # Soniox answers a finalize with the endpoint and <fin> together...
        [
            [
                {"text": "Bye bye.", "is_final": True},
                {"text": END_TOKEN, "is_final": True},
                {"text": FINALIZED_TOKEN, "is_final": True},
            ]
        ],
        # ...or with <fin> alone when its endpoint came first.
        [
            [{"text": "Bye bye.", "is_final": True}, {"text": END_TOKEN, "is_final": True}],
            [{"text": FINALIZED_TOKEN, "is_final": True}],
        ],
    ],
    ids=["end-and-fin-together", "end-then-fin"],
)
async def test_pipecat_mode_endpoint_and_finalize_push_one_transcription(monkeypatch, answer):
    events = []
    service = _instrumented_service(monkeypatch, events)

    messages = [json.dumps({"tokens": [{"text": "Bye", "is_final": False}]})]
    messages += [json.dumps({"tokens": tokens}) for tokens in answer]
    messages.append(json.dumps({"tokens": [], "finished": True}))
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
        ("broadcast", ProposedUserStartedSpeakingFrame),
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

    started = [
        event for event in events if event == ("broadcast", ProposedUserStartedSpeakingFrame)
    ]
    assert len(started) == 1
    assert events[-2:] == [
        ("push", TranscriptionFrame),
        ("broadcast", ProposedUserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_vad_stop_does_not_close_turn(monkeypatch):
    events = []
    service = _instrumented_service(monkeypatch, events, vad_force_turn_endpoint=False)

    # The Soniox endpoint owns the turn close: a VAD stop (e.g. a mid-turn
    # pause) must not close it.
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert ("broadcast", ProposedUserStoppedSpeakingFrame) not in events
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
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", InterimTranscriptionFrame),
        ("broadcast", ProposedUserStoppedSpeakingFrame),
        ("error", None),
    ]


@pytest.mark.asyncio
async def test_endpoint_transcript_emits_usage_before_transcription_frame(monkeypatch):
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import STTUsageMetricsData

    service = SonioxSTTService(api_key="test-key")
    service._setup = frame_processor_setup(TaskManager(), enable_usage_metrics=True)
    pushed_frames = []

    async def fake_push_frame(frame, direction=None):
        pushed_frames.append(frame)

    async def fake_noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_noop)

    # Simulate audio previously submitted to the service.
    service._stt_usage_pending_seconds = 2.5

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

    frame_types = [type(frame) for frame in pushed_frames]
    assert MetricsFrame in frame_types
    assert TranscriptionFrame in frame_types
    # Usage must precede the transcript so tracing can attach it to the span
    # the finalized TranscriptionFrame closes.
    assert frame_types.index(MetricsFrame) < frame_types.index(TranscriptionFrame)

    metrics_frame = pushed_frames[frame_types.index(MetricsFrame)]
    data = metrics_frame.data[0]
    assert isinstance(data, STTUsageMetricsData)
    assert data.value.audio_seconds == 2.5
    assert service._stt_usage_pending_seconds == 0.0


@pytest.mark.asyncio
async def test_receive_messages_allows_final_transcription_without_language(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    pushed_frames = []
    traced_transcriptions = []

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

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

    await service._receive_messages()

    final_frames = [frame for frame in pushed_frames if isinstance(frame, TranscriptionFrame)]
    assert len(final_frames) == 1
    assert final_frames[0].text == "Tell me a joke."
    assert final_frames[0].language is None
    assert final_frames[0].finalized is True
    assert traced_transcriptions == [("Tell me a joke.", True, None)]


def _connected_service(**kwargs):
    """Build a service holding an open fake socket, without touching the network."""
    service = SonioxSTTService(api_key="test-key", **kwargs)
    service._setup = frame_processor_setup(TaskManager())
    websocket = _FakeWebsocket([])
    service._websocket = websocket
    return service, websocket


def _sent_payloads(websocket):
    return [call.args[0] for call in websocket.send.await_args_list]


@pytest.mark.asyncio
async def test_stop_sends_end_of_audio_while_the_socket_is_open():
    service, websocket = _connected_service()

    await service.stop(EndFrame())

    assert _sent_payloads(websocket) == [""]
    assert websocket.closed


@pytest.mark.asyncio
async def test_stop_disconnects_once():
    service, _ = _connected_service()
    service._disconnect_websocket = AsyncMock(wraps=service._disconnect_websocket)

    await service.stop(EndFrame())

    service._disconnect_websocket.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_does_not_send_end_of_audio():
    service, websocket = _connected_service()

    await service.cancel(CancelFrame())

    assert _sent_payloads(websocket) == []
    assert websocket.closed


@pytest.mark.asyncio
async def test_stop_completes_teardown_when_the_end_of_audio_send_fails():
    service, websocket = _connected_service()
    websocket.send = AsyncMock(side_effect=ConnectionResetError("transient send failure"))
    service._flush_stt_usage_metrics = AsyncMock(wraps=service._flush_stt_usage_metrics)

    await service.stop(EndFrame())

    # The send is best-effort: everything after it still has to run.
    service._flush_stt_usage_metrics.assert_awaited_once()
    assert service._disconnecting is True
    assert websocket.closed


@pytest.mark.asyncio
async def test_language_update_reconnects_with_that_language_as_the_hint(monkeypatch):
    websocket = _FakeWebsocket([])

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    service = SonioxSTTService(
        api_key="test-key", settings=SonioxSTTService.Settings(language_hints=[Language.FI])
    )
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())

    await service._update_settings(STTSettings(language=Language.SV))
    await service._connect_websocket()

    config = json.loads(websocket.send.call_args.args[0])
    assert config["language_hints"] == ["sv"]
    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_language_update_leaves_the_callers_delta_unchanged(monkeypatch):
    service = SonioxSTTService(
        api_key="test-key", settings=SonioxSTTService.Settings(language_hints=[Language.FI])
    )
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())
    delta = STTSettings(language=Language.SV)

    await service._update_settings(delta)

    assert delta.language == Language.SV
    assert not hasattr(delta, "language_hints")


@pytest.mark.asyncio
async def test_constructor_language_connects_with_that_language_as_the_hint(monkeypatch):
    websocket = _FakeWebsocket([])

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    service = SonioxSTTService(
        api_key="test-key", settings=SonioxSTTService.Settings(language=Language.SV)
    )

    await service._connect_websocket()

    config = json.loads(websocket.send.call_args.args[0])
    assert config["language_hints"] == ["sv"]


@pytest.mark.asyncio
async def test_language_update_outside_the_language_enum_does_not_raise(monkeypatch):
    service = SonioxSTTService(
        api_key="test-key", settings=SonioxSTTService.Settings(language_hints=[Language.FI])
    )
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())

    await service._update_settings(STTSettings(language="zh-Hant"))

    assert service._settings.language_hints == [Language.FI]


def test_language_hints_given_with_a_language_are_kept():
    service = SonioxSTTService(
        api_key="test-key",
        settings=SonioxSTTService.Settings(
            language=Language.FI, language_hints=[Language.FI, Language.SV]
        ),
    )

    assert service._settings.language_hints == [Language.FI, Language.SV]


@pytest.mark.asyncio
async def test_soniox_error_code_sets_category(monkeypatch):
    from pipecat.utils.errors import ErrorCategory

    events = []
    service = _instrumented_service(monkeypatch, events)
    categories = []

    async def fake_push_error(*args, **kwargs):
        categories.append(kwargs.get("category"))

    monkeypatch.setattr(service, "push_error", fake_push_error)

    messages = [
        json.dumps({"tokens": [], "error_code": 402, "error_message": "balance exhausted"}),
        json.dumps({"tokens": [], "error_code": 401, "error_message": "invalid key"}),
        json.dumps({"tokens": [], "error_message": "no code"}),
    ]
    service._websocket = _FakeWebsocket(messages)

    await service._receive_messages()

    assert categories == [ErrorCategory.QUOTA, ErrorCategory.AUTHENTICATION, None]


@pytest.mark.parametrize("language", [Language.NB, Language.NB_NO])
def test_norwegian_bokmal_maps_to_soniox_norwegian(language):
    assert language_to_soniox_language(language) == "no"


async def _connect_config(monkeypatch, **kwargs):
    """Connect a service to a fake socket and return the config it sent."""
    websocket = _FakeWebsocket([], state=State.OPEN)

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = SonioxSTTService(api_key="test-key", **kwargs)
    await service._connect_websocket()

    assert service._websocket is websocket
    return json.loads(websocket.send.await_args.args[0])


@pytest.mark.asyncio
async def test_connect_config_requests_agent_text_input_by_default(monkeypatch):
    config = await _connect_config(monkeypatch)

    assert config["model"] == "stt-rt-v6"
    assert config["enable_agent_text_input"] is True


@pytest.mark.asyncio
async def test_connect_config_requests_agent_text_input_for_echo_cancellation(monkeypatch):
    settings = SonioxSTTService.Settings(
        enable_agent_text_input=False, enable_echo_cancellation=True
    )
    config = await _connect_config(monkeypatch, settings=settings)

    assert config["enable_agent_text_input"] is True


@pytest.mark.parametrize(
    "settings",
    [
        SonioxSTTService.Settings(enable_agent_text_input=False),
        SonioxSTTService.Settings(model="stt-rt-v5"),
        SonioxSTTService.Settings(model="stt-rt-v5", enable_agent_text_input=True),
        SonioxSTTService.Settings(model="stt-rt-v5", enable_echo_cancellation=True),
    ],
)
@pytest.mark.asyncio
async def test_connect_config_leaves_out_agent_text_input_when_disabled_or_unsupported(
    monkeypatch, settings
):
    config = await _connect_config(monkeypatch, settings=settings)

    assert "enable_agent_text_input" not in config


@pytest.mark.parametrize(
    ("streams_while_it_plays", "sent"),
    [
        (False, [{"type": "agent_text", "text": "Lekadol is in stock.", "end_of_segment": True}]),
        (True, []),
    ],
)
@pytest.mark.asyncio
async def test_assistant_turn_is_sent_as_agent_text_unless_it_was_sent_while_playing(
    streams_while_it_plays, sent
):
    service, websocket = _connected_service()
    service._settings.enable_agent_text_input = True
    service._settings.enable_echo_cancellation = streams_while_it_plays
    if streams_while_it_plays:
        # Present while echo cancellation is on, and what makes the reply already sent.
        service._agent_text_observer = BotSpeechObserver()

    await service._process_assistant_turn("Lekadol is in stock.")

    assert [json.loads(p) for p in _sent_payloads(websocket)] == sent


@pytest.mark.asyncio
async def test_settings_update_starts_and_stops_observing_the_bots_speech(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    monkeypatch.setattr(service, "_connect", AsyncMock())
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())
    worker = _fake_worker()
    await service.setup(frame_processor_setup(TaskManager(), pipeline_worker=worker))
    assert not worker.add_observer.called

    await service._update_settings(SonioxSTTService.Settings(enable_echo_cancellation=True))
    observer = service._agent_text_observer
    assert isinstance(observer, BotSpeechObserver)
    worker.add_observer.assert_called_once_with(observer)

    await service._update_settings(SonioxSTTService.Settings(enable_echo_cancellation=False))
    assert service._agent_text_observer is None
    worker.remove_observer.assert_awaited_once_with(observer)


@pytest.mark.asyncio
async def test_assistant_turn_is_sent_as_agent_text_when_the_turn_ends():
    service, websocket = _connected_service()

    await service.process_frame(
        LLMContextAssistantTurnFrame(text="Lekadol is in stock.", timestamp=time_now_iso8601()),
        FrameDirection.DOWNSTREAM,
    )

    # The framework reports the turn, and the whole reply goes out as one segment.
    assert [json.loads(p) for p in _sent_payloads(websocket)] == [
        {"type": "agent_text", "text": "Lekadol is in stock.", "end_of_segment": True}
    ]


@pytest.mark.parametrize(
    "settings",
    [
        SonioxSTTService.Settings(enable_agent_text_input=False),
        SonioxSTTService.Settings(model="stt-rt-v5"),
    ],
)
@pytest.mark.asyncio
async def test_assistant_turn_is_not_sent_when_disabled_or_unsupported(settings):
    service, websocket = _connected_service(settings=settings)

    await service._process_assistant_turn("Lekadol is in stock.")

    assert _sent_payloads(websocket) == []


def _fake_worker():
    return SimpleNamespace(add_observer=MagicMock(), remove_observer=AsyncMock())


def _agent_text_service(lead_ms=0):
    """A connected service observing the bot's speech, as echo cancellation sets it up."""
    service, websocket = _connected_service()
    service._settings.enable_echo_cancellation = True
    service._settings.agent_text_lead_ms = lead_ms
    service._setup = frame_processor_setup(TaskManager(), pipeline_worker=_fake_worker())
    # The service schedules each word on its task manager.
    service._task_manager = service._setup.task_manager
    observer = BotSpeechObserver()
    observer.add_event_handler("on_bot_text", service._on_bot_text)
    observer.add_event_handler("on_bot_speech_ended", service._on_bot_speech_ended)
    service._agent_text_observer = observer
    return service, observer, websocket


def _pushed(frame, *, source=None, direction=FrameDirection.DOWNSTREAM, first_push=True):
    return FramePushed(
        # The TTS service, unless a test says otherwise.
        source=source or IdentityFilter(),
        destination=IdentityFilter(),
        frame=frame,
        direction=direction,
        timestamp=0,
        first_push=first_push,
    )


def _word(text, pts=1, text_type=TextType.WORD, interruptible=True):
    # A pts of 0 is how the rest of the framework says "untimed", so a timed
    # word in a test needs a real one.
    frame = TTSTextFrame(text, text_type=text_type)
    frame.pts = pts
    frame.interruptible = interruptible
    return frame


def _transport():
    return BaseOutputTransport(TransportParams())


# Captured before any test patches `asyncio.wait_for` to watch the observer wait.
_real_wait_for = asyncio.wait_for


async def _drain(service):
    """Let the words the service queued reach the socket.

    Bounded: a sender that died leaves the queue unjoinable, and waiting on it
    for ever would hang the suite rather than fail it.
    """
    await _real_wait_for(service._agent_text_words.join(), timeout=1)


@pytest.mark.asyncio
async def test_agent_text_sends_the_bots_words_and_closes_the_segment_when_it_stops():
    service, observer, websocket = _agent_text_service()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await observer.on_push_frame(_pushed(_word("there.")))
    await _drain(service)
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(_word("Again")))
    await _drain(service)

    assert [json.loads(p) for p in _sent_payloads(websocket)] == [
        {"type": "agent_text", "text": "Hello", "end_of_segment": False},
        {"type": "agent_text", "text": " there.", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
        {"type": "agent_text", "text": "Again", "end_of_segment": False},
    ]


@pytest.mark.asyncio
async def test_agent_text_is_sent_ahead_of_the_word_being_played(monkeypatch):
    service, observer, websocket = _agent_text_service(lead_ms=700)
    waited = []

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        waited.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    await observer.on_push_frame(_pushed(_word("Hello", pts=1_000_000_000)))
    await _drain(service)

    assert waited == [pytest.approx(0.3)]
    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)] == ["Hello"]


@pytest.mark.asyncio
async def test_agent_text_drops_the_words_the_caller_never_heard(monkeypatch):
    service, observer, websocket = _agent_text_service()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    await observer.on_push_frame(_pushed(_word("Hello")))
    await _drain(service)
    await observer.on_push_frame(_pushed(_word("there.", pts=10_000_000_000)))
    await observer.on_push_frame(_pushed(InterruptionFrame()))
    await _drain(service)

    # The caller heard "Hello" and cut in, so the turn closes there: the word
    # still waiting is dropped, because the transport drops its audio too.
    assert [json.loads(p) for p in _sent_payloads(websocket)] == [
        {"type": "agent_text", "text": "Hello", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]


@pytest.mark.asyncio
async def test_agent_text_still_sends_speech_the_interruption_cannot_stop(monkeypatch):
    service, observer, websocket = _agent_text_service()
    # Uninterruptible speech waits in `asyncio.sleep`, which the far-future pts
    # would make a real wait; the interruption has to end it instead.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    await observer.on_push_frame(
        _pushed(_word("Transferring you now.", pts=10_000_000_000, interruptible=False))
    )
    await observer.on_push_frame(_pushed(InterruptionFrame()))
    await _drain(service)

    # Uninterruptible audio keeps playing, so the caller hears these words.
    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)] == ["Transferring you now."]


@pytest.mark.asyncio
async def test_agent_text_closes_the_turn_on_an_interruption():
    service, observer, websocket = _agent_text_service()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await _drain(service)
    await observer.on_push_frame(_pushed(InterruptionFrame()))
    # The transport pushes no BotStoppedSpeakingFrame when it had not started,
    # and closing twice would open a segment nothing ever closes.
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    assert [json.loads(p) for p in _sent_payloads(websocket)] == [
        {"type": "agent_text", "text": "Hello", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]


@pytest.mark.asyncio
async def test_agent_text_scheduled_words_are_dropped_at_teardown():
    service, observer, websocket = _agent_text_service()

    await observer.on_push_frame(_pushed(_word("Hello", pts=10_000_000_000)))
    assert service._agent_text_sender
    await service._stop_agent_text_observer()

    assert not service._agent_text_sender
    assert _sent_payloads(websocket) == []


@pytest.mark.asyncio
async def test_agent_text_starts_a_new_sender_when_the_old_one_is_over():
    service, observer, websocket = _agent_text_service()

    await observer.on_push_frame(_pushed(_word("Hello")))
    await _drain(service)
    service._agent_text_sender.cancel()
    await asyncio.sleep(0.01)
    assert service._agent_text_sender.done()

    await observer.on_push_frame(_pushed(_word("there.")))
    # Bounded: with no live sender the word is never taken off the queue, and
    # an unbounded join would hang the suite rather than fail it.
    await asyncio.wait_for(_drain(service), timeout=1)

    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)] == ["Hello", " there."]


@pytest.mark.asyncio
async def test_agent_text_drops_the_words_timed_against_the_previous_connection():
    service, observer, websocket = _agent_text_service()

    await observer.on_push_frame(_pushed(_word("Hello", pts=10_000_000_000)))
    service._drop_agent_text_words()
    await _drain(service)

    assert _sent_payloads(websocket) == []


@pytest.mark.asyncio
async def test_agent_text_drops_even_uninterruptible_words_on_a_reconnect(monkeypatch):
    service, observer, websocket = _agent_text_service()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    # Uninterruptible speech ignores the turn ending, so only dropping the queue
    # keeps it off a connection that never carried the audio it was timed for.
    await observer.on_push_frame(
        _pushed(_word("Transferring you now.", pts=10_000_000_000, interruptible=False))
    )
    service._drop_agent_text_words()
    await _drain(service)

    assert _sent_payloads(websocket) == []


@pytest.mark.asyncio
async def test_agent_text_still_sends_text_from_a_tts_without_word_timestamps():
    service, observer, websocket = _agent_text_service()

    sentence = _word("A whole sentence.", pts=None)
    # Sent as the TTS service pushes it, not again as it passes the transport.
    await observer.on_push_frame(_pushed(sentence))
    assert len(_sent_payloads(websocket)) == 1
    await observer.on_push_frame(_pushed(BotStartedSpeakingFrame(), source=_transport()))
    await observer.on_push_frame(_pushed(sentence, source=_transport(), first_push=False))
    await observer.on_push_frame(_pushed(_word("And another.", pts=None)))

    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)] == [
        "A whole sentence.",
        " And another.",
    ]
    assert service._warned_untimed_text


@pytest.mark.asyncio
async def test_agent_text_keeps_the_order_the_words_arrived_in():
    service, observer, websocket = _agent_text_service()

    # The same timestamp twice, then an earlier one.
    for text, pts in [("help", 5), ("you", 5), ("with", 5), ("today?", 4)]:
        await observer.on_push_frame(_pushed(_word(text, pts=pts)))
    await _drain(service)

    sent = [json.loads(p)["text"] for p in _sent_payloads(websocket)]
    assert sent == ["help", " you", " with", " today?"]


@pytest.mark.asyncio
async def test_blank_assistant_turn_is_not_sent():
    service, websocket = _connected_service()

    await service._process_assistant_turn("  ")

    assert _sent_payloads(websocket) == []


@pytest.mark.asyncio
async def test_agent_text_starts_a_new_segment_after_a_reconnect(monkeypatch):
    service, observer, _ = _agent_text_service()
    await observer.on_push_frame(_pushed(_word("Hello")))
    await _drain(service)
    service._websocket = None

    websocket = _FakeWebsocket([], state=State.OPEN)

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    await service._connect_websocket()
    await observer.on_push_frame(_pushed(_word("there.")))
    await _drain(service)

    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)[1:]] == ["there."]


@pytest.mark.parametrize(
    ("words", "includes_spaces", "sent"),
    [
        (["こんにちは", "世界"], True, ["こんにちは", "世界"]),
        (["你", "好"], True, ["你", "好"]),
        (["สวัสดีครับ", "ยินดีต้อนรับ"], False, ["สวัสดีครับ", " ยินดีต้อนรับ"]),
    ],
)
@pytest.mark.asyncio
async def test_agent_text_follows_the_tts_spacing_for_scripts_without_spaces(
    words, includes_spaces, sent
):
    service, observer, websocket = _agent_text_service()

    for text in words:
        frame = _word(text)
        frame.includes_inter_frame_spaces = includes_spaces
        await observer.on_push_frame(_pushed(frame))
        await _drain(service)

    assert [json.loads(p)["text"] for p in _sent_payloads(websocket)] == sent


@pytest.mark.asyncio
async def test_agent_text_lead_follows_a_runtime_change(monkeypatch):
    service, observer, _ = _agent_text_service(lead_ms=700)
    waited = []

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        waited.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    service._settings.agent_text_lead_ms = 250
    await observer.on_push_frame(_pushed(_word("Hello", pts=1_000_000_000)))
    await _drain(service)

    assert waited == [pytest.approx(0.75)]
    assert SonioxSTTService(api_key="k")._settings.agent_text_lead_ms == (
        DEFAULT_AGENT_TEXT_LEAD_MS
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_setup_observes_the_pipeline_only_when_echo_cancellation_is_enabled(
    monkeypatch, enabled
):
    service = SonioxSTTService(
        api_key="test-key",
        settings=SonioxSTTService.Settings(enable_echo_cancellation=enabled),
    )
    monkeypatch.setattr(service, "_connect", AsyncMock())
    worker = SimpleNamespace(add_observer=MagicMock(), remove_observer=AsyncMock())

    await service.setup(frame_processor_setup(TaskManager(), pipeline_worker=worker))
    await service.cleanup()

    assert worker.add_observer.called is enabled
    assert worker.remove_observer.await_count == int(enabled)


@pytest.mark.asyncio
async def test_setup_does_not_observe_the_pipeline_for_a_model_without_agent_text_input(
    monkeypatch,
):
    service = SonioxSTTService(
        api_key="test-key",
        settings=SonioxSTTService.Settings(model="stt-rt-v5", enable_echo_cancellation=True),
    )
    monkeypatch.setattr(service, "_connect", AsyncMock())
    worker = _fake_worker()

    await service.setup(frame_processor_setup(TaskManager(), pipeline_worker=worker))

    assert not worker.add_observer.called
