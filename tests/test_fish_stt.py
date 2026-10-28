#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish realtime ASR protocol and conversation integration."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import ormsgpack
import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EagerTranscriptionFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserBackchannelFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.fish.stt import FishAudioContextMessage, FishAudioSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory


def service(**kwargs):
    stt = FishAudioSTTService(api_key="test", sample_rate=16000, **kwargs)
    stt._sample_rate = 16000
    stt.push_frame = AsyncMock()
    stt._call_event_handler = AsyncMock()
    return stt


def committed(text, *, index=0, user_turn=True, turn_kind="user", **kwargs):
    return {
        "type": "committed",
        "turn_index": index,
        "text": text,
        "user_turn": user_turn,
        "turn_kind": turn_kind,
        **kwargs,
    }


def pushed(stt):
    return [call.args[0] for call in stt.push_frame.await_args_list]


@pytest.mark.asyncio
async def test_backchannel_waits_for_committed_text_and_does_not_start_a_turn():
    stt = service()
    for message in (
        {"type": "user_backchannel", "turn_index": 0},
        {"type": "speech_start", "turn_index": 0},
        {"type": "draft", "turn_index": 0, "text": "yes"},
    ):
        await stt._handle_message(message)
    assert pushed(stt) == []

    await stt._handle_message(committed("Yes.", user_turn=False, turn_kind="backchannel"))
    assert [type(frame) for frame in pushed(stt)] == [UserBackchannelFrame]
    assert pushed(stt)[0].text == "Yes."
    assert list(stt._history) == []


@pytest.mark.asyncio
async def test_backchannel_can_become_a_user_turn():
    stt = service()
    await stt._handle_message({"type": "user_backchannel", "turn_index": 0})
    await stt._handle_message({"type": "user_turn_taken", "turn_index": 0})
    await stt._handle_message(committed("Yes, but I meant tomorrow."))
    types = [type(frame) for frame in pushed(stt)]
    assert types.count(ProposedUserStartedSpeakingFrame) == 2  # upstream and downstream
    assert types.count(ProposedUserStoppedSpeakingFrame) == 2
    assert types.count(TranscriptionFrame) == 1
    assert UserBackchannelFrame not in types


@pytest.mark.asyncio
async def test_empty_user_commit_ends_the_turn_without_inventing_text():
    stt = service()
    await stt._handle_message({"type": "user_turn_taken", "turn_index": 0})
    await stt._handle_message(committed("", degraded=True))
    types = [type(frame) for frame in pushed(stt)]
    assert types.count(ProposedUserStoppedSpeakingFrame) == 2
    assert TranscriptionFrame not in types
    assert UserBackchannelFrame not in types
    assert all(
        not strategy.wait_for_transcript
        for strategy in stt.service_metadata_frame().user_turn_strategies.stop
    )


@pytest.mark.asyncio
async def test_noise_is_ignored_and_user_turn_flag_takes_precedence_over_reason():
    stt = service()
    await stt._handle_message(
        committed("background television", user_turn=False, turn_kind="other", reason="finalize")
    )
    assert pushed(stt) == []
    assert list(stt._history) == []


@pytest.mark.asyncio
async def test_sound_tags_remain_in_raw_result_and_short_replies_keep_language():
    stt = service()
    result = committed("[laughs] こんにちは", language_code="ja")
    await stt._handle_message(result)
    await stt._handle_message(committed("はい", index=1))
    transcripts = [frame for frame in pushed(stt) if isinstance(frame, TranscriptionFrame)]
    assert [frame.text for frame in transcripts] == ["こんにちは", "はい"]
    assert all(frame.language == Language.JA for frame in transcripts)
    assert transcripts[0].result == result


@pytest.mark.asyncio
async def test_eager_withdrawal_and_empty_predictions():
    stt = service(enable_eager_end_of_turn=True)
    await stt._handle_message({"type": "user_turn_taken", "turn_index": 0})
    stt.push_frame.reset_mock()
    await stt._handle_message({"type": "eager_end_of_turn", "turn_index": 0, "text": "[sigh]"})
    assert pushed(stt) == []
    await stt._handle_message({"type": "eager_end_of_turn", "turn_index": 0, "text": "book it"})
    await stt._handle_message({"type": "turn_resumed", "turn_index": 0})
    assert [type(frame) for frame in pushed(stt)] == [
        EagerTranscriptionFrame,
        UserTurnResponseCancelFrame,
    ]


@pytest.mark.asyncio
async def test_playback_changes_are_queued_once_and_frames_are_forwarded():
    stt = service()
    stt._websocket = object()
    stt._ready.set()
    start, stop = BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()
    for frame in (start, start, stop):
        await stt.process_frame(frame, FrameDirection.UPSTREAM)
    messages = [stt._send_queue.get_nowait()[1] for _ in range(stt._send_queue.qsize())]
    assert messages == [
        {"type": "agent_speaking", "speaking": True},
        {"type": "agent_speaking", "speaking": False},
    ]
    assert pushed(stt) == [start, start, stop]


@pytest.mark.asyncio
async def test_context_keeps_both_roles_and_respects_server_limits():
    stt = service(context_history=[FishAudioContextMessage(role="assistant", text="Welcome")])
    await stt._handle_message(committed("I need help"))
    await stt._process_assistant_turn("a" * 2100)
    history = stt._session_config()["context"]["history"]
    assert [item["role"] for item in history] == ["assistant", "user", "assistant", "assistant"]
    assert [len(item["text"]) for item in history[-2:]] == [2000, 100]
    for _ in range(40):
        await stt._process_assistant_turn("reply")
    assert len(stt._session_config()["context"]["history"]) == 32


@pytest.mark.asyncio
async def test_context_added_during_handshake_is_sent_exactly_once():
    stt = service()
    websocket = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), response=None)

    async def connect(*args, **kwargs):
        await stt._process_assistant_turn("Before config")
        return websocket

    stt._websocket_connect = connect
    await stt._connect_websocket()
    await stt._process_assistant_turn("After config")
    await stt._handle_message({"type": "session_started"})
    config = ormsgpack.unpackb(websocket.send.call_args.args[0])
    assert config["context"]["history"] == [{"role": "assistant", "text": "Before config"}]
    _, update = stt._send_queue.get_nowait()
    assert update == {
        "type": "context_add",
        "items": [{"role": "assistant", "text": "After config"}],
    }
    assert stt._send_queue.empty()


@pytest.mark.asyncio
async def test_wire_order_preserves_pcm_around_playback_state_changes():
    stt = service()
    websocket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    stt._websocket = websocket
    stt._ready.set()
    await stt._queue_message({"type": "audio", "data": b"\x01\x00"})
    await stt.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await stt._queue_message({"type": "audio", "data": b"\x02\x00"})
    task = asyncio.create_task(stt._send_messages())
    try:
        await asyncio.wait_for(stt._send_queue.join(), timeout=1)
        assert [ormsgpack.unpackb(call.args[0]) for call in websocket.send.call_args_list] == [
            {"type": "audio", "data": b"\x01\x00"},
            {"type": "agent_speaking", "speaking": True},
            {"type": "audio", "data": b"\x02\x00"},
        ]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_old_socket_send_failure_does_not_clear_new_session_readiness():
    stt = service()
    new_socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())

    async def fail_after_reconnect(raw):
        stt._websocket = new_socket
        raise ConnectionError("old connection closed")

    old_socket = SimpleNamespace(
        send=AsyncMock(side_effect=fail_after_reconnect), close=AsyncMock()
    )
    stt._websocket = old_socket
    stt._ready.set()
    await stt._queue_message({"type": "audio", "data": b"\x01\x00"})
    task = asyncio.create_task(stt._send_messages())
    try:
        await asyncio.wait_for(stt._send_queue.join(), timeout=1)
        assert stt._ready.is_set()
        new_socket.close.assert_not_awaited()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_unconfirmed_session_drops_audio_and_unsupported_updates_retain_settings():
    stt = service()
    await stt.process_audio_frame(audio(), FrameDirection.DOWNSTREAM)
    assert stt._send_queue.empty()
    assert stt._stt_usage_pending_seconds == 0
    await stt._update_settings(stt.Settings(model="another-model", language=Language.JA))
    assert stt._settings.model == "transcribe-stream-1"
    assert stt._settings.language is None


@pytest.mark.asyncio
async def test_incompatible_audio_reports_one_permanent_error():
    stt = service()
    for _ in range(3):
        await stt.process_audio_frame(
            InputAudioRawFrame(b"\x00\x00" * 160, 8000, 1), FrameDirection.DOWNSTREAM
        )
    assert not stt.is_usable
    assert len([frame for frame in pushed(stt) if isinstance(frame, ErrorFrame)]) == 1
    assert stt._send_queue.empty()


class FakeFish:
    def __init__(self, responses=(), *, drop_first=False):
        self.responses = responses
        self.drop_first = drop_first
        self.configs = []
        self.messages = []

    async def handle(self, websocket):
        config = ormsgpack.unpackb(await websocket.recv())
        self.configs.append(config)
        connection = len(self.configs)
        await websocket.send(ormsgpack.packb({"type": "session_started", "request_id": "test"}))
        sent = False
        async for raw in websocket:
            message = ormsgpack.unpackb(raw)
            self.messages.append((connection, message))
            if message["type"] == "audio" and not sent:
                sent = True
                for response in self.responses:
                    await websocket.send(ormsgpack.packb(response))
                if self.drop_first and connection == 1:
                    await asyncio.sleep(0.02)
                    await websocket.close(code=1012)
                    return
            elif message["type"] == "audio_done":
                await websocket.send(ormsgpack.packb({"type": "session_done", "stats": {}}))
                return


def audio():
    return InputAudioRawFrame(b"\x00\x00" * 320, 16000, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "expected"), [("Hello", "Hello"), ("", ""), ("[laughs]", "")])
async def test_commits_drive_the_aggregator_without_waiting_for_its_watchdog(text, expected):
    fake = FakeFish(
        [
            {"type": "speech_start", "turn_index": 0},
            {"type": "user_turn_taken", "turn_index": 0},
            committed(text),
        ]
    )
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context, params=LLMUserAggregatorParams(empty_user_turn=None, user_turn_stop_timeout=10)
        )
        down, _ = await run_test(
            Pipeline([stt, aggregator]),
            frames_to_send=[SleepFrame(sleep=0.1), audio(), SleepFrame(sleep=0.2)],
        )
    assert any(isinstance(frame, UserStoppedSpeakingFrame) for frame in down)
    assert len([frame for frame in down if isinstance(frame, LLMContextFrame)]) == bool(expected)
    assert [message["content"] for message in context.messages] == ([expected] if expected else [])
    assert fake.messages[-1][1]["type"] == "audio_done"


@pytest.mark.asyncio
async def test_backchannel_does_not_interrupt_or_call_the_llm():
    fake = FakeFish(
        [
            {"type": "user_backchannel", "turn_index": 0},
            {"type": "speech_start", "turn_index": 0},
            committed("Mm-hm", user_turn=False, turn_kind="backchannel"),
        ]
    )
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        context = LLMContext()
        aggregator = LLMUserAggregator(context)
        down, up = await run_test(
            Pipeline([stt, aggregator]),
            frames_to_send=[SleepFrame(sleep=0.1), audio(), SleepFrame(sleep=0.1)],
        )
    assert any(isinstance(frame, UserBackchannelFrame) for frame in down)
    assert not any(
        isinstance(frame, (LLMContextFrame, InterruptionFrame)) for frame in [*down, *up]
    )
    assert context.messages == []


@pytest.mark.asyncio
async def test_eager_commit_writes_the_final_turn_without_a_second_inference():
    fake = FakeFish(
        [
            {"type": "user_turn_taken", "turn_index": 0},
            {"type": "eager_end_of_turn", "turn_index": 0, "text": "Hello"},
            committed("Hello"),
        ]
    )
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(
            api_key="test",
            url=f"ws://127.0.0.1:{port}",
            enable_eager_end_of_turn=True,
        )
        context = LLMContext()
        down, _ = await run_test(
            Pipeline([stt, LLMUserAggregator(context)]),
            frames_to_send=[SleepFrame(sleep=0.1), audio(), SleepFrame(sleep=0.1)],
        )
    contexts = [frame for frame in down if isinstance(frame, LLMContextFrame)]
    assert len(contexts) == 1
    assert contexts[0].speculation
    assert context.messages == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_reconnect_restores_history_and_playback_state():
    fake = FakeFish(
        [
            {"type": "user_turn_taken", "turn_index": 0},
            committed("I need help"),
        ],
        drop_first=True,
    )
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(
            api_key="test",
            url=f"ws://127.0.0.1:{port}",
            reconnect_backoff_min_wait=0.01,
            reconnect_backoff_max_wait=0.02,
        )
        stt._agent_speaking = True
        await run_test(
            stt,
            frames_to_send=[
                SleepFrame(sleep=0.1),
                LLMContextAssistantTurnFrame("Welcome", "2026-10-28T00:00:00Z"),
                audio(),
                SleepFrame(sleep=0.3),
                audio(),
                SleepFrame(sleep=0.1),
            ],
        )
    assert len(fake.configs) == 2
    assert fake.configs[1]["context"]["history"] == [
        {"role": "assistant", "text": "Welcome"},
        {"role": "user", "text": "I need help"},
    ]
    assert (2, {"type": "agent_speaking", "speaking": True}) in fake.messages
    assert not any(
        message["type"] == "context_add" for index, message in fake.messages if index == 2
    )


@pytest.mark.asyncio
async def test_session_uses_auto_mode_and_sends_binary_pcm():
    fake = FakeFish()
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        down, _ = await run_test(
            stt,
            frames_to_send=[
                SleepFrame(sleep=0.1),
                audio(),
                SleepFrame(sleep=0.05),
            ],
            pipeline_params=PipelineParams(enable_usage_metrics=True),
        )
    pcm = next(message for _, message in fake.messages if message["type"] == "audio")
    assert pcm["data"] == b"\x00\x00" * 320
    assert fake.configs[0]["turn_detection"] == {
        "mode": "auto",
        "eager": False,
        "silence_timeout_ms": 5000,
    }
    assert fake.configs[0]["results"]["diarize"] is False
    usage = [
        metric.value.audio_seconds
        for frame in down
        if isinstance(frame, MetricsFrame)
        for metric in frame.data
        if isinstance(metric, STTUsageMetricsData)
    ]
    assert sum(usage) == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_invalid_config_stops_reconnect_attempts():
    connections = 0

    async def reject(websocket):
        nonlocal connections
        connections += 1
        await websocket.recv()
        await websocket.send(
            ormsgpack.packb(
                {
                    "type": "error",
                    "error_type": "invalid_config",
                    "message": "invalid model",
                    "recoverable": False,
                }
            )
        )
        await websocket.close(code=4400)

    async with serve(reject, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(
            api_key="test",
            url=f"ws://127.0.0.1:{port}",
            reconnect_backoff_min_wait=0.01,
        )
        await run_test(stt, frames_to_send=[SleepFrame(sleep=0.1), audio()])
    assert connections == 1
    assert not stt.is_usable


@pytest.mark.asyncio
async def test_handshake_refusal_preserves_provider_reason_and_request_id():
    stt = service()
    stt._websocket_connect = AsyncMock(
        side_effect=InvalidStatus(
            Response(
                401,
                "Unauthorized",
                Headers({"x-request-id": "refused-session"}),
                body=b'{"status":401,"message":"Invalid API key"}',
            )
        )
    )
    await stt._connect_websocket()
    error = next(frame for frame in pushed(stt) if isinstance(frame, ErrorFrame))
    assert "Invalid API key" in error.error
    assert "refused-session" in error.error
    assert error.category == ErrorCategory.AUTHENTICATION
    assert not stt.is_usable
    stt._call_event_handler.assert_any_await("on_connection_error", error.error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        "idle_timeout",
        "max_duration_exceeded",
        "internal",
        "engine_error",
        "invalid_request",
        "server_restarting",
    ],
)
async def test_session_errors_do_not_disable_a_new_session(error_type):
    stt = service()
    stt._request_id = "current-session"
    await stt._handle_message(
        {
            "type": "error",
            "error_type": error_type,
            "code": 500,
            "message": "provider explanation",
            "recoverable": False,
        }
    )
    error = next(frame for frame in pushed(stt) if isinstance(frame, ErrorFrame))
    assert error_type in error.error
    assert "provider explanation" in error.error
    assert "current-session" in error.error
    assert stt.is_usable
    assert all(
        call.args[0] != "on_connection_error" for call in stt._call_event_handler.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["vad", "manual"])
async def test_non_auto_modes_start_on_speech_start_and_commit_without_user_turn_taken(mode):
    stt = service(settings=FishAudioSTTService.Settings(mode=mode))
    await stt._handle_message({"type": "speech_start", "turn_index": 0})
    await stt._handle_message(committed("Hello"))
    types = [type(frame) for frame in pushed(stt)]
    assert types.count(ProposedUserStartedSpeakingFrame) == 2
    assert types.count(ProposedUserStoppedSpeakingFrame) == 2
    assert types.count(TranscriptionFrame) == 1


@pytest.mark.asyncio
async def test_diarization_and_word_times_preserve_transport_identity_and_raw_result():
    stt = service(settings=FishAudioSTTService.Settings(diarize=True, ignore_timestamps=False))
    stt._user_id = "caller"
    words = [{"text": "Hello", "start_ms": 120, "end_ms": 500}]
    result = committed("<|speaker:0|>[laughs] Hello <|speaker:1|> there", words=words)
    await stt._handle_message(result)
    transcript = next(frame for frame in pushed(stt) if isinstance(frame, TranscriptionFrame))
    assert transcript.text == "Hello there"
    assert transcript.user_id == "caller"
    assert transcript.result == result
    config = stt._session_config()["results"]
    assert config["diarize"] is True
    assert config["ignore_timestamps"] is False


@pytest.mark.asyncio
async def test_diagnostic_scores_do_not_start_or_end_turns():
    stt = service(settings=FishAudioSTTService.Settings(vad_events=True, eot_events=True))
    events = [
        {"type": "vad", "audio_ms": 100, "score": 1.0},
        {"type": "eot_prob", "audio_ms": 200, "prob": 1.0},
    ]
    for event in events:
        await stt._handle_message(event)
        stt._call_event_handler.assert_any_await("on_server_event", event)
    assert pushed(stt) == []
    assert stt._session_config()["results"]["vad_events"] is True
    assert stt._session_config()["results"]["eot_events"] is True


@pytest.mark.asyncio
async def test_finalize_is_explicit_and_follows_queued_audio():
    stt = service(settings=FishAudioSTTService.Settings(mode="manual"))
    assert not await stt.finalize()
    stt._websocket = object()
    stt._ready.set()
    await stt.process_audio_frame(audio(), FrameDirection.DOWNSTREAM)
    assert await stt.finalize()
    messages = [stt._send_queue.get_nowait()[1] for _ in range(stt._send_queue.qsize())]
    assert [message["type"] for message in messages] == ["audio", "finalize"]


class ConfigurableFish(FakeFish):
    def __init__(self, *, reject_first=False, lose_first_ack=False):
        super().__init__()
        self.reject_first = reject_first
        self.lose_first_ack = lose_first_ack
        self.updates = 0
        self.on_configure = None

    async def handle(self, websocket):
        config = ormsgpack.unpackb(await websocket.recv())
        self.configs.append(deepcopy(config))
        connection = len(self.configs)
        await websocket.send(ormsgpack.packb({"type": "session_started", "config": config}))
        async for raw in websocket:
            message = ormsgpack.unpackb(raw)
            self.messages.append((connection, message))
            if message["type"] == "configure":
                self.updates += 1
                if self.on_configure:
                    self.on_configure()
                await asyncio.sleep(0.02)
                accepted = not (self.reject_first and self.updates == 1)
                if accepted:
                    config.update({key: value for key, value in message.items() if key != "type"})
                if self.lose_first_ack and self.updates == 1:
                    continue
                await websocket.send(
                    ormsgpack.packb(
                        {
                            "type": "configured",
                            "ok": accepted,
                            "config": config,
                            "message": "invalid settings" if not accepted else "",
                        }
                    )
                )
            elif message["type"] == "audio_done":
                await websocket.send(ormsgpack.packb({"type": "session_done"}))
                return


@pytest.mark.asyncio
async def test_runtime_updates_are_serialized_and_merge_after_acknowledgment():
    fake = ConfigurableFish()
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        modes_before_ack = []
        fake.on_configure = lambda: modes_before_ack.append(stt._settings.mode)
        await run_test(
            stt,
            frames_to_send=[
                SleepFrame(sleep=0.05),
                STTUpdateSettingsFrame(delta=stt.Settings(mode="vad", silence_timeout_ms=900)),
                SleepFrame(sleep=0.005),
                audio(),
                STTUpdateSettingsFrame(delta=stt.Settings(terms=["Ada"], ignore_timestamps=False)),
                SleepFrame(sleep=0.15),
            ],
        )
    updates = [message for _, message in fake.messages if message["type"] == "configure"]
    assert len(updates) == 2
    assert modes_before_ack == ["auto", "vad"]
    assert updates[1]["turn_detection"]["mode"] == "vad"
    assert updates[1]["turn_detection"]["silence_timeout_ms"] == 900
    assert updates[1]["context"] == {"terms": ["Ada"]}
    types = [message["type"] for _, message in fake.messages]
    assert types.index("audio") < len(types) - 1 - types[::-1].index("configure")
    assert stt._settings.mode == "vad"
    assert stt._settings.terms == ["Ada"]
    assert stt._settings.ignore_timestamps is False


@pytest.mark.asyncio
async def test_rejected_configuration_retains_previous_settings_for_the_next_update():
    fake = ConfigurableFish(reject_first=True)
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        down, up = await run_test(
            stt,
            frames_to_send=[
                SleepFrame(sleep=0.05),
                STTUpdateSettingsFrame(delta=stt.Settings(mode="vad")),
                STTUpdateSettingsFrame(delta=stt.Settings(terms=["Ada"])),
                SleepFrame(sleep=0.15),
            ],
        )
    updates = [message for _, message in fake.messages if message["type"] == "configure"]
    assert updates[0]["turn_detection"]["silence_timeout_ms"] == 700
    assert updates[1]["turn_detection"]["mode"] == "auto"
    assert updates[1]["turn_detection"]["silence_timeout_ms"] == 5000
    assert stt._settings.mode == "auto"
    assert stt._settings.terms == ["Ada"]
    assert stt.is_usable
    assert any(isinstance(frame, ErrorFrame) and "rejected" in frame.error for frame in down + up)


@pytest.mark.asyncio
async def test_missing_configured_reconnects_with_last_acknowledged_settings():
    fake = ConfigurableFish(lose_first_ack=True)
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(
            api_key="test", url=f"ws://127.0.0.1:{port}", reconnect_backoff_min_wait=0.01
        )
        stt._CONNECTION_TIMEOUT = 0.08
        await run_test(
            stt,
            frames_to_send=[
                SleepFrame(sleep=0.05),
                STTUpdateSettingsFrame(delta=stt.Settings(mode="manual")),
                SleepFrame(sleep=0.3),
            ],
        )
    assert len(fake.configs) == 2
    assert fake.configs[1]["turn_detection"]["mode"] == "auto"
    assert stt._settings.mode == "auto"


@pytest.mark.asyncio
async def test_context_uses_completed_playback_aggregation_including_an_interrupted_reply():
    fake = FakeFish()
    context = LLMContext(
        [
            {"role": "system", "content": "private instructions"},
            {"role": "tool", "tool_call_id": "lookup", "content": "private result"},
            {"role": "assistant", "content": "generated but unplayed reply"},
        ]
    )
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(api_key="test", url=f"ws://127.0.0.1:{port}")
        await run_test(
            Pipeline([stt, LLMAssistantAggregator(context)]),
            frames_to_send=[
                SleepFrame(sleep=0.05),
                LLMContextFrame(context),
                LLMFullResponseStartFrame(),
                TTSTextFrame("Already played", text_type="sentence"),
                SleepFrame(sleep=0.02),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
            ],
        )
    updates = [message for _, message in fake.messages if message["type"] == "context_add"]
    assert updates == [
        {"type": "context_add", "items": [{"role": "assistant", "text": "Already played"}]}
    ]
    assert stt._session_config()["context"]["history"] == [
        {"role": "assistant", "text": "Already played"}
    ]


@pytest.mark.asyncio
async def test_manual_finalize_completes_a_pipeline_turn_without_auto_classification():
    messages = []

    async def handle(websocket):
        await websocket.recv()
        await websocket.send(ormsgpack.packb({"type": "session_started"}))
        async for raw in websocket:
            message = ormsgpack.unpackb(raw)
            messages.append(message)
            if message["type"] == "audio":
                await websocket.send(ormsgpack.packb({"type": "speech_start", "turn_index": 0}))
            elif message["type"] == "finalize":
                await websocket.send(ormsgpack.packb(committed("Hello", reason="finalize")))
            elif message["type"] == "audio_done":
                await websocket.send(ormsgpack.packb({"type": "session_done"}))
                return

    context = LLMContext()
    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = FishAudioSTTService(
            api_key="test",
            url=f"ws://127.0.0.1:{port}",
            settings=FishAudioSTTService.Settings(mode="manual"),
        )

        @stt.event_handler("on_server_event")
        async def finalize_after_speech_start(service, event):
            if event["type"] == "speech_start":
                assert await service.finalize()

        down, _ = await run_test(
            Pipeline([stt, LLMUserAggregator(context)]),
            frames_to_send=[SleepFrame(sleep=0.05), audio(), SleepFrame(sleep=0.15)],
        )
    assert any(isinstance(frame, LLMContextFrame) for frame in down)
    assert context.messages == [{"role": "user", "content": "Hello"}]
    assert [message["type"] for message in messages] == ["audio", "finalize", "audio_done"]
