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

import io
from types import SimpleNamespace

import pytest
from loguru import logger

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live import llm as llm_module
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.tests.utils import run_test
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
    assert "2.18.0" in output, "the warning must name the version that fixes it"


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
