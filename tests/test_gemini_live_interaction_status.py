#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for model capability detection and interaction-status turn gating.

Models that perform background reasoning report an ``interaction_status`` on
``server_content``: ``turn_complete`` only closes an output chunk, while the
status says whether the server is still working. These tests drive
``_handle_server_message`` with fabricated server messages and assert on the
frames the service pushes downstream.
"""

import asyncio
import io
from types import SimpleNamespace

import pytest
from loguru import logger

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.google.gemini_live import llm as llm_module
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeServerContent:
    """Stands in for ``LiveServerContent``.

    Only the attributes the receive loop reads are defined. ``interaction_status``
    is omitted entirely when not supplied, mirroring SDK versions predating the
    field.
    """

    def __init__(
        self,
        *,
        turn_complete: bool = False,
        interrupted: bool = False,
        interaction_status: object = None,
        has_status_attr: bool = True,
    ):
        self.model_turn = None
        self.input_transcription = None
        self.output_transcription = None
        self.grounding_metadata = None
        self.turn_complete = turn_complete
        self.interrupted = interrupted
        if has_status_attr:
            self.interaction_status = interaction_status


class _FakeServerMessage:
    """Stands in for ``LiveServerMessage``."""

    def __init__(self, server_content: object = None):
        self.server_content = server_content
        self.tool_call = None
        self.usage_metadata = None
        self.session_resumption_update = None


def _make_service(
    *, model: str = "models/gemini-3.8-live-extended-thinking"
) -> GeminiLiveLLMService:
    """Construct a service with a captured frame sink. ``__init__`` does no I/O."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model=model),
    )

    pushed = []

    async def _capture(frame, direction=None):
        pushed.append(frame)

    service.push_frame = _capture  # type: ignore[method-assign]
    service.pushed_frames = pushed  # type: ignore[attr-defined]
    return service


async def _setup_service(service: GeminiLiveLLMService) -> None:
    """Give the service a task manager so it can start the deferral watchdog."""
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )


def _frame_types(service) -> list[type]:
    return [type(f) for f in service.pushed_frames]


async def _start_bot_turn(service):
    """Put the service into the mid-response state a model turn would leave it in."""
    await _setup_service(service)
    await service._set_bot_is_responding(True)
    await service.push_frame(TTSStartedFrame())
    await service.push_frame(LLMFullResponseStartFrame())
    service.pushed_frames.clear()


# ---------------------------------------------------------------------------
# Model capability detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("models/gemini-2.5-flash-native-audio-preview-12-2025", True),
        ("models/gemini-3-pro-preview", False),
        ("models/gemini-3.8-live", True),
        ("models/gemini-3.8-live-extended-thinking", True),
        ("models/gemini-4-live", True),
        ("models/gemini-2.0-flash-live-001", True),
    ],
)
def test_non_blocking_tool_support_by_model(model, expected):
    """Gemini 3.x models before 3.8 lack NON_BLOCKING; everything else has it."""
    service = _make_service(model=model)
    assert service._supports_non_blocking_tools is expected


@pytest.mark.parametrize(
    "model",
    [
        "models/gemini-3-pro-preview",
        "models/gemini-3.8-live",
        "models/gemini-3.8-live-extended-thinking",
    ],
)
def test_gemini_3_protocol_detection(model):
    """Every Gemini 3.x model, including the 3.8 Live family, uses the 3.x protocol."""
    service = _make_service(model=model)
    assert service._is_gemini_3 is True


@pytest.mark.parametrize(
    "model, expected_level",
    [
        # Live thinking models require a thinking_level; an unset one defaults.
        ("models/gemini-3.8-live-extended-thinking", "LOW"),
        ("models/gemini-3.8-live", None),
        ("models/gemini-2.0-flash-live-001", None),
        ("models/gemini-2.5-flash-exp-native-audio-thinking-dialog", None),
    ],
)
def test_thinking_level_defaults_only_on_live_thinking_models(model, expected_level):
    """Models that require a thinking_level get one; other models get no config."""
    service = _make_service(model=model)

    thinking = service._resolved_thinking_config()

    if expected_level is None:
        assert thinking is None
    else:
        assert thinking is not None
        assert thinking.thinking_level == expected_level


def test_explicit_thinking_level_is_kept():
    """A configured thinking_level is never overridden by the default."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-3.8-live-extended-thinking",
            thinking={"thinking_level": "HIGH"},
        ),
    )

    thinking = service._resolved_thinking_config()

    assert thinking is not None
    assert thinking.thinking_level == "HIGH"


def test_thinking_level_default_merges_with_other_thinking_settings():
    """Defaulting the level keeps the rest of the config and leaves settings untouched."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-3.8-live-extended-thinking",
            thinking={"include_thoughts": True},
        ),
    )

    thinking = service._resolved_thinking_config()

    assert thinking is not None
    assert thinking.include_thoughts is True
    assert thinking.thinking_level == "LOW"
    assert service._settings.thinking == {"include_thoughts": True}


# ---------------------------------------------------------------------------
# interaction_status gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_complete_while_in_progress_holds_end_of_turn():
    """IN_PROGRESS means background work continues, so the bot turn stays open."""
    service = _make_service()
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )

    assert _frame_types(service) == [], "end-of-turn frames must wait for an idle status"
    assert service._bot_is_responding is True


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_status", ["IDLE", "REQUIRES_ACTION"])
async def test_idle_status_closes_a_held_turn(idle_status):
    """Either spelling of the idle status releases the held end-of-turn frames."""
    service = _make_service()
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    assert _frame_types(service) == []

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(interaction_status=idle_status))
    )

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_turn_complete_bundled_with_idle_closes_immediately():
    """A turn_complete that already reports idle needs no deferral."""
    service = _make_service()
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IDLE"))
    )

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_turn_complete_without_interaction_status_closes_turn():
    """Models that never report a status keep the plain turn_complete behavior."""
    service = _make_service(model="models/gemini-2.5-flash-native-audio-preview-12-2025")
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, has_status_attr=False))
    )

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_unspecified_status_does_not_hold_the_turn():
    """An UNSPECIFIED status carries no information and must not defer the turn."""
    service = _make_service()
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(
            _FakeServerContent(
                turn_complete=True, interaction_status="INTERACTION_STATUS_UNSPECIFIED"
            )
        )
    )

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]


@pytest.mark.asyncio
async def test_idle_without_a_held_turn_pushes_nothing():
    """An idle status outside a deferred turn is a no-op."""
    service = _make_service()

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(interaction_status="IDLE"))
    )

    assert _frame_types(service) == []


@pytest.mark.asyncio
async def test_interruption_discards_a_held_turn():
    """After a barge-in the held turn is dropped, so a later idle emits nothing."""
    service = _make_service()
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    await service._handle_interruption()
    service.pushed_frames.clear()

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(interaction_status="IDLE"))
    )

    assert _frame_types(service) == [], "an interrupted turn must not be closed later"


@pytest.mark.asyncio
async def test_held_turn_is_released_when_idle_never_arrives():
    """The watchdog closes the turn if the server never reports going idle."""
    service = _make_service()
    service._DEFERRED_TURN_COMPLETE_TIMEOUT_SECS = 0.05
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    assert _frame_types(service) == []

    await service._deferred_turn_complete_timeout_task

    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


@pytest.mark.asyncio
async def test_streaming_content_keeps_held_turn_open():
    """Server messages restart the watchdog, so a reply still streaming past
    the timeout window is not forced closed; the watchdog fires only once the
    server actually goes silent."""
    service = _make_service()
    service._DEFERRED_TURN_COMPLETE_TIMEOUT_SECS = 0.1
    await _start_bot_turn(service)

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    # Stream transcription chunks across several timeout windows.
    for _ in range(5):
        await asyncio.sleep(0.05)
        content = _FakeServerContent()
        content.output_transcription = SimpleNamespace(text="still talking ")
        await service._handle_server_message(_FakeServerMessage(content))

    assert LLMFullResponseEndFrame not in _frame_types(service), (
        "a held turn must stay open while content is still streaming"
    )

    await service._deferred_turn_complete_timeout_task

    assert _frame_types(service)[-2:] == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False


# ---------------------------------------------------------------------------
# Unsupported-SDK warning
# ---------------------------------------------------------------------------


def _warnings_while(service, *, sdk_has_field: bool, monkeypatch, calls: int = 1) -> str:
    """Run the SDK-support check, returning whatever was logged at WARNING."""
    fields = {"interaction_status": object()} if sdk_has_field else {}
    monkeypatch.setattr(
        llm_module, "LiveServerContent", SimpleNamespace(model_fields=fields), raising=True
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        for _ in range(calls):
            service._warn_if_interaction_status_unsupported()
    finally:
        logger.remove(handler_id)
    return sink.getvalue()


def test_warns_when_sdk_predates_interaction_status(monkeypatch):
    """A thinking model on an SDK without the field can't get the turn gating."""
    service = _make_service(model="models/gemini-3.8-live-extended-thinking")

    output = _warnings_while(service, sdk_has_field=False, monkeypatch=monkeypatch)

    assert "google-genai" in output
    assert service._INTERACTION_STATUS_MIN_SDK in output, (
        "the warning must name the version that fixes it"
    )


def test_no_warning_when_sdk_supports_interaction_status(monkeypatch):
    """With a current SDK there is nothing to warn about."""
    service = _make_service(model="models/gemini-3.8-live-extended-thinking")

    output = _warnings_while(service, sdk_has_field=True, monkeypatch=monkeypatch)

    assert output == ""


@pytest.mark.parametrize(
    "model",
    [
        "models/gemini-3.8-live",
        "models/gemini-2.5-flash-native-audio-preview-12-2025",
        "models/gemini-2.5-flash-exp-native-audio-thinking-dialog",
    ],
)
def test_no_warning_for_models_that_report_no_status(model, monkeypatch):
    """Models that never report a status lose nothing on an older SDK."""
    service = _make_service(model=model)

    output = _warnings_while(service, sdk_has_field=False, monkeypatch=monkeypatch)

    assert output == ""


def test_sdk_warning_is_logged_once(monkeypatch):
    """Reconnects must not repeat the warning."""
    service = _make_service(model="models/gemini-3.8-live-extended-thinking")

    output = _warnings_while(service, sdk_has_field=False, monkeypatch=monkeypatch, calls=3)

    assert output.count("Upgrade to google-genai") == 1


@pytest.mark.asyncio
async def test_turn_complete_bundled_with_idle_supersedes_a_held_turn():
    """A held turn is superseded, not closed twice, by a turn_complete that reports idle."""
    service = _make_service()
    await _start_bot_turn(service)

    closes = []
    original = service._handle_msg_turn_complete

    async def _counting(message):
        closes.append(message)
        await original(message)

    service._handle_msg_turn_complete = _counting  # type: ignore[method-assign]

    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    await service._handle_server_message(
        _FakeServerMessage(
            _FakeServerContent(turn_complete=True, interaction_status="REQUIRES_ACTION")
        )
    )

    assert len(closes) == 1, "the turn must be closed exactly once"
    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]


# ---------------------------------------------------------------------------
# Tool behavior declarations
# ---------------------------------------------------------------------------


def _tagged_declarations(service, monkeypatch) -> list[dict]:
    """Tag one sync and one async declaration, returning the declarations."""
    monkeypatch.setattr(
        service, "_function_is_async", lambda name: name == "async_tool", raising=True
    )
    tools = [{"function_declarations": [{"name": "sync_tool"}, {"name": "async_tool"}]}]
    service._tag_tool_behaviors(tools)
    return tools[0]["function_declarations"]


@pytest.mark.parametrize(
    "model, sync_behavior",
    [
        # BLOCKING is already the default outside the 3.8 Live family.
        ("models/gemini-2.0-flash-live-001", None),
        # The 3.8 Live family defaults to NON_BLOCKING, so blocking is declared.
        ("models/gemini-3.8-live", "BLOCKING"),
        # Live thinking models accept only NON_BLOCKING.
        ("models/gemini-3.8-live-extended-thinking", None),
    ],
)
def test_sync_tools_block_where_the_model_allows_it(model, sync_behavior, monkeypatch):
    """Synchronous tools keep blocking semantics wherever the model can honor them."""
    service = _make_service(model=model)

    declarations = _tagged_declarations(service, monkeypatch)

    assert declarations[0].get("behavior") == sync_behavior
    assert declarations[1].get("behavior") == "NON_BLOCKING"


def test_sync_tool_on_a_strictly_non_blocking_model_warns_once(monkeypatch):
    """Synchronous tools can't block on live thinking models; say so once."""
    service = _make_service(model="models/gemini-3.8-live-extended-thinking")
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")

    try:
        _tagged_declarations(service, monkeypatch)
        _tagged_declarations(service, monkeypatch)
    finally:
        logger.remove(handler_id)

    assert sink.getvalue().count("won't pause") == 1


def test_no_warning_when_sync_tools_can_block(monkeypatch):
    """On gemini-3.8-live the BLOCKING declaration restores sync semantics silently."""
    service = _make_service(model="models/gemini-3.8-live")
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")

    try:
        _tagged_declarations(service, monkeypatch)
    finally:
        logger.remove(handler_id)

    assert sink.getvalue() == ""


# ---------------------------------------------------------------------------
# Context recording across an interruption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_assistant_message_survives_interruption():
    """Text streamed before a barge-in reaches the context even though the
    turn-closing frames are still held waiting for an idle status."""
    # Capture the frames the service pushes while streaming a reply that is
    # cut short: transcription chunks arrive, turn_complete is held because
    # the status is IN_PROGRESS, then the user barges in.
    service = _make_service()
    await _setup_service(service)

    for chunk in ("The answer ", "is 42."):
        content = _FakeServerContent()
        content.output_transcription = SimpleNamespace(text=chunk)
        await service._handle_server_message(_FakeServerMessage(content))
    await service._handle_server_message(
        _FakeServerMessage(_FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"))
    )
    await service._handle_interruption()
    streamed = list(service.pushed_frames)

    # Replay that exact sequence into an assistant aggregator, along with the
    # InterruptionFrame the pipeline broadcasts at the barge-in.
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context)
    await run_test(aggregator, frames_to_send=[*streamed, InterruptionFrame()])

    assistant_messages = [
        m for m in context.get_messages() if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    assert assistant_messages, "the partial reply must be recorded on interruption"
    assert assistant_messages[-1]["content"] == "The answer is 42."


# ---------------------------------------------------------------------------
# Mid-reply disconnect and response cleanup (Issue #5997 Item 4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_reply_disconnect_ends_response_and_releases_end_frame():
    """A mid-reply disconnect must close the open response and must not defer EndFrame for 30s.

    In AUDIO modality:
    - TTSStoppedFrame clears downstream audio transport speaking state.
    - InterruptionFrame signals the downstream LLMAssistantAggregator to commit the
      partial assistant turn as interrupted=True and clear its aggregation buffer,
      preventing stale partial text from merging with the next response after reconnect.
    - LLMFullResponseEndFrame must NOT be emitted: that would falsely mark the
      incomplete response as finished (interrupted=False).
    """
    service = _make_service()
    await _start_bot_turn(service)

    assert service._bot_is_responding is True

    # Simulate connection drop triggering disconnect cleanup
    await service._disconnect()

    assert service._bot_is_responding is False, (
        "_bot_is_responding must be reset to False on disconnect"
    )
    assert _frame_types(service) == [TTSStoppedFrame, InterruptionFrame], (
        "Mid-reply disconnect must push TTSStoppedFrame then InterruptionFrame"
    )

    # A subsequent EndFrame must not be trapped in the 30-second deferral hold
    service.pushed_frames.clear()
    await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert service._end_frame_pending_bot_turn_finished is None, (
        "EndFrame must not remain pending deferral after disconnect"
    )
    assert service._end_frame_deferral_timeout_task is None, (
        "EndFrame deferral watchdog must not be running after disconnect"
    )
    assert _frame_types(service) == [EndFrame], (
        "EndFrame must be processed immediately without entering 30s deferral"
    )


@pytest.mark.asyncio
async def test_disconnect_when_not_responding_emits_no_stop_frames():
    """Disconnect while idle must not invent spurious TTS or LLM stop frames."""
    service = _make_service()
    await _setup_service(service)

    assert service._bot_is_responding is False

    await service._disconnect()

    assert service.pushed_frames == [], "Disconnecting while idle must not push stop frames"


@pytest.mark.asyncio
async def test_normal_turn_complete_does_not_get_duplicate_stop_frames_on_disconnect():
    """A turn that completed normally must not receive duplicate stop frames when disconnecting."""
    service = _make_service()
    await _start_bot_turn(service)

    # Normal turn completion arrives from server
    await service._handle_server_message(_FakeServerMessage(_FakeServerContent(turn_complete=True)))
    assert _frame_types(service) == [TTSStoppedFrame, LLMFullResponseEndFrame]
    assert service._bot_is_responding is False
    service.pushed_frames.clear()

    # Now disconnect runs
    await service._disconnect()

    assert service.pushed_frames == [], "Must not emit duplicate stop frames on disconnect"


@pytest.mark.asyncio
async def test_mid_reply_disconnect_text_modality_emits_only_interruption():
    """In TEXT modality, disconnect mid-reply must push InterruptionFrame but not TTSStoppedFrame."""
    service = _make_service()
    service._settings.modalities = GeminiModalities.TEXT
    await _setup_service(service)

    await service._set_bot_is_responding(True)
    await service.push_frame(LLMFullResponseStartFrame())
    service.pushed_frames.clear()

    await service._disconnect()

    assert service._bot_is_responding is False
    assert _frame_types(service) == [InterruptionFrame], (
        "TEXT modality disconnect must push InterruptionFrame but not TTSStoppedFrame"
    )


@pytest.mark.asyncio
async def test_pending_deferred_end_frame_released_on_disconnect():
    """An EndFrame already held deferred during a response is released when disconnect occurs."""
    service = _make_service()
    await _start_bot_turn(service)

    # EndFrame arrives mid-reply and is deferred
    await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert service._end_frame_pending_bot_turn_finished is not None
    assert service._end_frame_deferral_timeout_task is not None

    # Connection drops
    await service._disconnect()

    assert service._bot_is_responding is False
    assert service._end_frame_deferral_timeout_task is None, (
        "Deferral watchdog timeout must be cancelled on disconnect"
    )
    assert service._end_frame_pending_bot_turn_finished is None, (
        "Pending EndFrame must be cleared/released on disconnect"
    )
    assert _frame_types(service) == [TTSStoppedFrame, InterruptionFrame]


@pytest.mark.asyncio
async def test_mid_reply_cancellation_preserves_interrupted_status_in_aggregator():
    """Omitting LLMFullResponseEndFrame on disconnect ensures downstream aggregator marks cancelled turn as interrupted."""
    service = _make_service()
    await _setup_service(service)
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context)

    stopped_messages = []

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_turn_stopped(agg, message):
        stopped_messages.append(message)

    # Start turn and stream partial reply
    content = _FakeServerContent()
    content.output_transcription = SimpleNamespace(text="Partial reply before cancel")
    await service._handle_server_message(_FakeServerMessage(content))
    streamed = list(service.pushed_frames)
    service.pushed_frames.clear()

    # Disconnect mid-reply (as service.cancel() would invoke)
    await service._disconnect()
    disconnect_frames = list(service.pushed_frames)

    # Replay stream, disconnect frame(s), and CancelFrame into aggregator
    await run_test(
        aggregator, frames_to_send=[*streamed, *disconnect_frames, SleepFrame(), CancelFrame()]
    )

    assert len(stopped_messages) == 1
    assert stopped_messages[0].interrupted is True, (
        "Cancellation after mid-reply disconnect must preserve interrupted=True"
    )


@pytest.mark.asyncio
async def test_connection_error_during_reply_triggers_reconnect_and_releases_end_frame():
    """Real failure path: an unexpected connection drop mid-reply reconnects and clears response state."""
    service = _make_service()
    await _start_bot_turn(service)
    assert service._bot_is_responding is True

    connect_called = False

    async def _mock_connect(**kwargs):
        nonlocal connect_called
        connect_called = True

    service._connect = _mock_connect

    # Connection error occurs while model is responding
    should_reconnect = await service._handle_connection_error(
        ConnectionResetError("Peer reset connection")
    )
    assert should_reconnect is True
    await service._reconnect()

    assert connect_called is True
    assert service._bot_is_responding is False
    assert _frame_types(service) == [TTSStoppedFrame, InterruptionFrame], (
        "Reconnect after provider failure must push TTSStoppedFrame + InterruptionFrame"
    )

    # Subsequent EndFrame is handled immediately without 30s deferral
    service.pushed_frames.clear()
    await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert service._end_frame_pending_bot_turn_finished is None
    assert service._end_frame_deferral_timeout_task is None
    assert _frame_types(service) == [EndFrame]


@pytest.mark.asyncio
async def test_stale_aggregation_does_not_merge_with_next_response():
    """InterruptionFrame on disconnect must clear _aggregation so old partial text cannot
    merge into the next response when the provider reconnects.

    Regression for the stale-merge defect: without InterruptionFrame, the
    LLMAssistantAggregator's internal buffer retains partial text from the first
    response. When the second LLMFullResponseStartFrame arrives it silently overwrites
    only the timestamp, leaving old text in _aggregation. The second response's
    LLMFullResponseEndFrame then commits old+new text as one merged message.
    """
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context)

    old_text = TTSTextFrame("partial old text ", aggregated_by="sentence")
    old_text.append_to_context = True
    new_text = TTSTextFrame("new complete response.", aggregated_by="sentence")
    new_text.append_to_context = True

    # Build the real frame sequence produced by the patched service:
    # (1) first session starts a response
    # (2) _disconnect() fires: TTSStoppedFrame + InterruptionFrame
    # (3) reconnect succeeds; second session starts a fresh response
    frames = [
        LLMFullResponseStartFrame(),  # first session: turn opens
        old_text,  # first session: partial text lands in aggregation
        SleepFrame(0.05),
        TTSStoppedFrame(),  # _disconnect(): AUDIO speaking ends
        InterruptionFrame(),  # _disconnect(): aggregator clears stale state
        SleepFrame(0.05),
        LLMFullResponseStartFrame(),  # second session: new turn opens
        new_text,  # second session: new response text
        SleepFrame(0.05),
        LLMFullResponseEndFrame(),  # second session: normal completion
    ]

    stop_messages = []

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_turn_stopped(agg, msg):
        stop_messages.append(msg)

    await run_test(aggregator, frames_to_send=frames, send_end_frame=True)

    messages = context.get_messages()
    assert len(messages) == 2, (
        "There must be exactly two context messages: the interrupted partial turn "
        "and the completed new-session response"
    )
    # First message: partial text committed with interrupted=True by InterruptionFrame
    assert "partial old text" in messages[0]["content"], (
        "The abandoned partial text must be committed to context (with interrupted=True) "
        "not silently discarded"
    )
    # Second message: only the new response
    assert messages[1]["content"] == "new complete response.", (
        "The second response must be committed as its own clean message"
    )
    assert "partial old text" not in messages[1]["content"], (
        "Old partial text must NOT bleed into the next response"
    )

    # Two on_assistant_turn_stopped events: interrupted + completed
    assert len(stop_messages) == 2
    assert stop_messages[0].interrupted is True, "First turn was abandoned: interrupted=True"
    assert stop_messages[1].interrupted is False, (
        "Second turn completed normally: interrupted=False"
    )


@pytest.mark.asyncio
async def test_true_provider_drop_via_connection_task_handler():
    """True fake-provider-drop test through the real _connection_task_handler path.

    Exercises the exact runtime failure route:
      fake session emits response -> raises network error
      -> _connection_task_handler catches it
      -> _handle_connection_error decides reconnect
      -> _reconnect() -> _disconnect() -> _connect()
      -> second fake session connects successfully
      -> second session emits fresh response and completes normally

    Asserts the full post-reconnect lifecycle state without any 30-second sleep.
    """
    import asyncio
    from types import SimpleNamespace

    service = _make_service()
    await _setup_service(service)

    # ------------------------------------------------------------------ #
    # Build two fake sessions                                              #
    # ------------------------------------------------------------------ #

    class _FakeSession:
        """Minimal fake AsyncSession that drives _handle_server_message calls."""

        def __init__(self, messages, error_after=None):
            self._messages = messages
            self._error_after = error_after
            self.closed = False
            self._stop_event = asyncio.Event()

        def receive(self):
            """Return an async iterator of fake server messages."""
            session_ref = self

            async def _gen():
                for i, msg in enumerate(session_ref._messages):
                    yield msg
                    if session_ref._error_after is not None and i + 1 >= session_ref._error_after:
                        raise ConnectionResetError("Fake provider drop")
                # Wait until closed/cancelled so the receive loop does not busy-spin
                await session_ref._stop_event.wait()

            return _gen()

        async def send_realtime_input(self, **kwargs):
            pass

        async def close(self):
            self.closed = True
            self._stop_event.set()

    # First session: emits one partial transcription message then raises
    partial_content = _FakeServerContent()
    partial_content.output_transcription = SimpleNamespace(text="partial provider text")
    session1 = _FakeSession(
        messages=[_FakeServerMessage(partial_content)],
        error_after=1,  # raise after first message
    )

    # Second session: emits fresh response, then completes turn
    fresh_content = _FakeServerContent(turn_complete=True)
    fresh_content.output_transcription = SimpleNamespace(text="new response after reconnect")
    session2 = _FakeSession(
        messages=[_FakeServerMessage(fresh_content)],
        error_after=None,
    )

    sessions = [session1, session2]
    session_idx = 0

    class _FakeLiveClient:
        class _FakeAio:
            class _FakeLive:
                def connect(self_inner, model, config):
                    nonlocal session_idx
                    s = sessions[session_idx]
                    session_idx += 1
                    return _AsyncCtx(s)

            live = _FakeLive()

        aio = _FakeAio()

    class _AsyncCtx:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, *args):
            pass

    # Patch the client
    service._client = _FakeLiveClient()

    # Wire downstream aggregator to verify context isolation across reconnect
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context)
    await _setup_service(aggregator)

    stop_messages = []

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_turn_stopped(agg, msg):
        stop_messages.append(msg)

    orig_push_frame = service.push_frame

    async def push_and_aggregate(frame, direction=FrameDirection.DOWNSTREAM):
        await orig_push_frame(frame, direction)
        await aggregator.process_frame(frame, direction)

    service.push_frame = push_and_aggregate

    # Run _connect() then let _connection_task_handler run in the background
    await service._connect()

    # Wait until session1 drops and session2 connects
    for _ in range(50):
        if session_idx == 2:
            break
        await asyncio.sleep(0.05)

    assert session_idx == 2, "Both sessions must have been used (first dropped, second reconnect)"

    # Allow session2 to complete processing its turn
    await asyncio.sleep(0.1)

    # ---- Assertions: lifecycle state after provider drop + reconnect ---- #
    assert service._bot_is_responding is False, (
        "_bot_is_responding must be False after second session completes turn"
    )

    frame_type_names = [type(f).__name__ for f in service.pushed_frames]
    assert "TTSStoppedFrame" in frame_type_names, (
        "TTSStoppedFrame must be emitted when AUDIO response is interrupted by provider drop"
    )
    assert "InterruptionFrame" in frame_type_names, (
        "InterruptionFrame must be emitted to clear downstream aggregation state"
    )

    # InterruptionFrame must come AFTER TTSStoppedFrame from the first session's disconnect
    tts_idx = next(i for i, f in enumerate(service.pushed_frames) if isinstance(f, TTSStoppedFrame))
    irq_idx = next(
        i for i, f in enumerate(service.pushed_frames) if isinstance(f, InterruptionFrame)
    )
    assert tts_idx < irq_idx, "TTSStoppedFrame must precede InterruptionFrame"

    # Aggregator context assertions:
    messages = context.get_messages()
    assert len(messages) == 2, (
        "There must be exactly two context messages: interrupted partial turn and fresh response"
    )
    assert messages[0]["content"] == "partial provider text", (
        "The abandoned partial text must be committed as its own message"
    )
    assert messages[1]["content"] == "new response after reconnect", (
        "The new response must be committed cleanly as its own message"
    )
    assert "partial provider text" not in messages[1]["content"], (
        "Old partial text must NOT bleed into the next response"
    )

    assert len(stop_messages) == 2
    assert stop_messages[0].interrupted is True, "First turn was abandoned: interrupted=True"
    assert stop_messages[1].interrupted is False, (
        "Second turn completed normally: interrupted=False"
    )

    # No 30-second deferral watchdog should be running
    assert service._end_frame_deferral_timeout_task is None, (
        "EndFrame deferral watchdog must not be running after reconnect"
    )
    assert service._end_frame_pending_bot_turn_finished is None, (
        "No deferred EndFrame should remain after reconnect"
    )

    # Subsequent EndFrame completes immediately without deferral
    service.pushed_frames.clear()
    await service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert service._end_frame_pending_bot_turn_finished is None
    assert service._end_frame_deferral_timeout_task is None
    assert [type(f) for f in service.pushed_frames] == [EndFrame]

    await service._disconnect()
