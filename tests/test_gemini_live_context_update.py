#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mid-session context updates in GeminiLiveLLMService.

Gemini 3.8 Live accepts a ``context_update`` client message that replaces the
system instruction and/or the tool list of a live session, sent with
``AsyncSession.send_context_update`` (google-genai 2.33+). These tests give the
service a fake session and assert on the updates it sends.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live import llm as llm_module
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _SessionWithContextUpdate:
    """Stands in for an ``AsyncSession`` class that has ``send_context_update``."""

    async def send_context_update(self, **update):  # pragma: no cover - never called
        raise NotImplementedError


class _SessionWithoutContextUpdate:
    """Stands in for an ``AsyncSession`` class from an SDK older than 2.33."""


class _FakeSession:
    def __init__(self):
        self.send_context_update = AsyncMock()
        self.send_client_content = AsyncMock()
        self.send_realtime_input = AsyncMock()


@pytest.fixture
def sdk_with_context_update(monkeypatch):
    monkeypatch.setattr(llm_module, "AsyncSession", _SessionWithContextUpdate)


def _schema(name: str) -> FunctionSchema:
    return FunctionSchema(name=name, description=f"{name} tool", properties={}, required=[])


def _tools(*names: str) -> ToolsSchema:
    return ToolsSchema(standard_tools=[_schema(n) for n in names])


def _declared_names(tools) -> list[str]:
    names = []
    for tool in tools or []:
        for decl in tool.get("function_declarations", []):
            names.append(decl["name"])
    return names


def _make_service(
    model: str = "models/gemini-3.8-live", system_instruction: str | None = None, **kwargs
) -> GeminiLiveLLMService:
    return GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model=model, system_instruction=system_instruction),
        **kwargs,
    )


async def _open_session(
    service: GeminiLiveLLMService, context: LLMContext | None = None
) -> _FakeSession:
    """Put the service in the state of a freshly set-up session."""
    service._context = context
    system_instruction, tools = await service._effective_system_instruction_and_tools()
    service._setup_system_instruction = system_instruction
    service._setup_tools = tools
    service._applied_system_instruction = system_instruction
    service._applied_tools = tools
    session = _FakeSession()
    service._session = session
    return session


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("models/gemini-3.8-live", True),
        ("gemini-3.8-live-extended-thinking", True),
        ("models/gemini-3.5-flash-live", False),
        ("models/gemini-2.5-flash-native-audio-preview-12-2025", False),
        ("some-custom-model", False),
    ],
)
def test_context_update_support_by_model(sdk_with_context_update, model, expected):
    assert _make_service(model)._supports_context_update is expected


def test_no_context_update_on_an_older_sdk(monkeypatch):
    monkeypatch.setattr(llm_module, "AsyncSession", _SessionWithoutContextUpdate)
    assert _make_service()._supports_context_update is False


# ---------------------------------------------------------------------------
# Tool updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_tools_frame_sends_only_the_new_tools(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    # The context aggregator stores the new tools before the frame arrives.
    new_tools = _tools("get_weather", "book_table")
    context.set_tools(new_tools)
    await service.process_frame(LLMSetToolsFrame(tools=new_tools), FrameDirection.DOWNSTREAM)

    session.send_context_update.assert_awaited_once()
    update = session.send_context_update.await_args.kwargs
    assert set(update) == {"tools"}
    assert _declared_names(update["tools"]) == ["get_weather", "book_table"]
    assert _declared_names(service._applied_tools) == ["get_weather", "book_table"]


@pytest.mark.asyncio
async def test_removing_every_tool_sends_an_empty_list(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    context.set_tools(ToolsSchema(standard_tools=[]))
    await service.process_frame(LLMSetToolsFrame(tools=[]), FrameDirection.DOWNSTREAM)

    session.send_context_update.assert_awaited_once_with(tools=[])
    assert service._applied_tools is None


@pytest.mark.asyncio
async def test_unchanged_tools_send_nothing(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("get_weather")), FrameDirection.DOWNSTREAM
    )

    session.send_context_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_older_models_keep_their_tools_and_warn_once(sdk_with_context_update, monkeypatch):
    service = _make_service("models/gemini-2.5-flash-native-audio-preview-12-2025")
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)
    synced = []
    monkeypatch.setattr(service, "_sync_registered_tool_handlers", synced.append)

    context.set_tools(_tools("book_table"))
    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )
    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )

    session.send_context_update.assert_not_awaited()
    assert service._warned_context_update_unsupported is True
    # The session still advertises the old tools, so their handlers stay
    # until the next context frame.
    assert synced == []


@pytest.mark.asyncio
async def test_set_tools_frame_is_pushed_on(sdk_with_context_update, monkeypatch):
    service = _make_service()
    await _open_session(service, LLMContext(tools=_tools("get_weather")))
    pushed = []

    async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))

    monkeypatch.setattr(service, "push_frame", push_frame)

    frame = LLMSetToolsFrame(tools=_tools("get_weather"))
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert pushed == [(frame, FrameDirection.DOWNSTREAM)]


@pytest.mark.asyncio
async def test_disconnecting_sends_nothing(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)
    service._disconnecting = True

    context.set_tools(_tools("book_table"))
    await service._apply_context_changes()

    session.send_context_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_updates_diff_against_each_other(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)
    release = asyncio.Event()

    async def slow_send(**update):
        await release.wait()

    session.send_context_update.side_effect = slow_send

    context.set_tools(_tools("book_table"))
    first = asyncio.create_task(service._apply_context_changes())
    await asyncio.sleep(0)
    second = asyncio.create_task(service._apply_context_changes())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    # The second call waits for the first and finds nothing left to send.
    session.send_context_update.assert_awaited_once()
    assert _declared_names(service._applied_tools) == ["book_table"]


@pytest.mark.asyncio
async def test_no_session_sends_nothing(sdk_with_context_update):
    service = _make_service()
    service._context = LLMContext(tools=_tools("get_weather"))

    await service._apply_context_changes()  # must not raise

    assert service._session is None


# ---------------------------------------------------------------------------
# System instruction updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_instruction_setting_updates_the_session(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    session = await _open_session(service, LLMContext())

    await service._update_settings(
        GeminiLiveLLMService.Settings(system_instruction="Speak like a pirate.")
    )

    session.send_context_update.assert_awaited_once_with(system_instruction="Speak like a pirate.")
    # Kept for new sessions too, so a later fresh connect sets it up.
    assert service._system_instruction_from_init == "Speak like a pirate."
    system_instruction, _ = await service._effective_system_instruction_and_tools()
    assert system_instruction == "Speak like a pirate."


@pytest.mark.asyncio
async def test_system_instruction_update_sends_the_prompt_without_addons(
    sdk_with_context_update,
):
    service = _make_service(system_instruction="Be brief.")
    session = await _open_session(service, LLMContext())
    service.append_system_instruction("UI GUIDE TEXT")

    await service._update_settings(GeminiLiveLLMService.Settings(system_instruction="Be warm."))

    session.send_context_update.assert_awaited_once_with(system_instruction="Be warm.")


@pytest.mark.asyncio
async def test_clearing_the_system_instruction_sends_nothing(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    session = await _open_session(service, LLMContext())

    await service._update_settings(GeminiLiveLLMService.Settings(system_instruction=None))

    session.send_context_update.assert_not_awaited()
    assert service._applied_system_instruction == "Be brief."


@pytest.mark.asyncio
async def test_older_models_still_warn_about_system_instruction(sdk_with_context_update):
    service = _make_service("models/gemini-3.5-flash-live", system_instruction="Be brief.")
    session = await _open_session(service, LLMContext())
    warned = []
    service._warn_unhandled_updated_settings = lambda unhandled: warned.append(set(unhandled))

    await service._update_settings(
        GeminiLiveLLMService.Settings(system_instruction="Speak like a pirate.")
    )

    session.send_context_update.assert_not_awaited()
    assert warned == [{"system_instruction"}]
    assert service._system_instruction_from_init == "Be brief."


@pytest.mark.asyncio
async def test_updated_context_sends_its_new_tools_before_the_turn(
    sdk_with_context_update,
):
    service = _make_service(system_instruction="Be brief.")
    session = await _open_session(service, LLMContext(tools=_tools("greet_guest")))
    calls = []
    session.send_context_update.side_effect = lambda **kw: calls.append(("update", kw))
    session.send_client_content.side_effect = lambda **kw: calls.append(("content", kw))
    service._content_awaiting_response = True

    # A Pipecat Flows node transition swaps the context's tools.
    new_context = LLMContext(tools=_tools("take_order"))
    await service._handle_context(new_context, FrameDirection.UPSTREAM)

    assert calls[0][0] == "update"
    assert set(calls[0][1]) == {"tools"}
    assert _declared_names(calls[0][1]["tools"]) == ["take_order"]
    assert calls[1] == ("content", {"turn_complete": True})


# ---------------------------------------------------------------------------
# Reconnects and session resumption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resumed_session_gets_the_current_context_again(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    # Mid-session updates change what the server holds...
    context.set_tools(_tools("book_table"))
    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )
    await service._update_settings(GeminiLiveLLMService.Settings(system_instruction="Be warm."))
    assert session.send_context_update.await_count == 2

    # ...but resuming the session handle restores its original setup.
    service._session = None
    service._session_resumption_handle = "handle-1"
    resumed = _FakeSession()
    await service._handle_session_ready(resumed)

    resumed.send_context_update.assert_awaited_once()
    update = resumed.send_context_update.await_args.kwargs
    assert update["system_instruction"] == "Be warm."
    assert _declared_names(update["tools"]) == ["book_table"]
    assert service._ready_for_realtime_input is True


@pytest.mark.asyncio
async def test_resumed_session_without_updates_sends_nothing(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    await _open_session(service, LLMContext(tools=_tools("get_weather")))

    service._session = None
    service._session_resumption_handle = "handle-1"
    resumed = _FakeSession()
    await service._handle_session_ready(resumed)

    resumed.send_context_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_connect_records_the_setup_only_for_new_sessions(
    sdk_with_context_update, monkeypatch
):
    service = _make_service(system_instruction="Be brief.")
    service._context = LLMContext(tools=_tools("get_weather"))
    monkeypatch.setattr(service, "create_task", lambda coro, *a, **kw: coro.close())

    await service._connect()
    assert service._setup_system_instruction == "Be brief."
    assert _declared_names(service._setup_tools) == ["get_weather"]

    service._context.set_tools(_tools("book_table"))
    await service._connect(session_resumption_handle="handle-1")
    # The resumed session keeps its original setup.
    assert _declared_names(service._setup_tools) == ["get_weather"]


@pytest.mark.asyncio
async def test_failed_update_is_retried_on_the_next_change(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)
    session.send_context_update.side_effect = [RuntimeError("socket closed"), None]
    service._handle_send_error = AsyncMock()

    context.set_tools(_tools("book_table"))
    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )
    assert _declared_names(service._applied_tools) == ["get_weather"]
    service._handle_send_error.assert_awaited_once()

    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )
    assert session.send_context_update.await_count == 2
    assert _declared_names(service._applied_tools) == ["book_table"]


@pytest.mark.asyncio
async def test_initial_context_tools_update_the_session_instead_of_reconnecting(
    sdk_with_context_update,
):
    service = _make_service(system_instruction="Be brief.")
    session = await _open_session(service, None)
    service._reconnect = AsyncMock()
    service._create_initial_response = AsyncMock()

    await service._handle_context(LLMContext(tools=_tools("get_weather")))

    service._reconnect.assert_not_awaited()
    session.send_context_update.assert_awaited_once()
    update = session.send_context_update.await_args.kwargs
    assert _declared_names(update["tools"]) == ["get_weather"]


@pytest.mark.asyncio
async def test_initial_context_tools_still_reconnect_older_models(sdk_with_context_update):
    service = _make_service("models/gemini-3.5-flash-live", system_instruction="Be brief.")
    session = await _open_session(service, None)
    service._reconnect = AsyncMock()
    service._create_initial_response = AsyncMock()

    await service._handle_context(LLMContext(tools=_tools("get_weather")))

    service._reconnect.assert_awaited_once()
    session.send_context_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_that_arrives_before_the_session_is_ready(sdk_with_context_update):
    service = _make_service(system_instruction="Be brief.")
    # The session was set up from the init values, before any context.
    system_instruction, tools = await service._effective_system_instruction_and_tools()
    service._setup_system_instruction = system_instruction
    service._setup_tools = tools
    service._reconnect = AsyncMock()
    service._create_initial_response = AsyncMock()

    await service._handle_context(LLMContext(tools=_tools("get_weather")))
    service._reconnect.assert_not_awaited()

    session = _FakeSession()
    await service._handle_session_ready(session)

    session.send_context_update.assert_awaited_once()
    update = session.send_context_update.await_args.kwargs
    assert set(update) == {"tools"}
    assert _declared_names(update["tools"]) == ["get_weather"]


@pytest.mark.asyncio
async def test_an_update_in_flight_when_the_session_is_replaced(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    old_session = await _open_session(service, context)
    release = asyncio.Event()

    async def slow_send(**update):
        await release.wait()

    old_session.send_context_update.side_effect = slow_send

    context.set_tools(_tools("book_table"))
    in_flight = asyncio.create_task(service._apply_context_changes())
    await asyncio.sleep(0)

    # The connection drops and a new session is set up with the old tools.
    service._session = None
    service._setup_tools = (
        await service.get_llm_adapter().get_llm_invocation_params(
            LLMContext(tools=_tools("get_weather"))
        )
    )["tools"]
    new_session = _FakeSession()
    ready = asyncio.create_task(service._handle_session_ready(new_session))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(in_flight, ready)

    old_session.send_context_update.assert_awaited_once()
    new_session.send_context_update.assert_awaited_once()
    update = new_session.send_context_update.await_args.kwargs
    assert _declared_names(update["tools"]) == ["book_table"]
    assert _declared_names(service._applied_tools) == ["book_table"]


@pytest.mark.asyncio
async def test_set_tools_frame_syncs_tool_handlers(sdk_with_context_update, monkeypatch):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    await _open_session(service, context)
    synced = []
    monkeypatch.setattr(service, "_sync_registered_tool_handlers", synced.append)

    new_tools = _tools("book_table")
    context.set_tools(new_tools)
    await service.process_frame(LLMSetToolsFrame(tools=new_tools), FrameDirection.DOWNSTREAM)

    assert synced == [new_tools]


@pytest.mark.asyncio
async def test_empty_context_tools_fall_back_to_init_tools(sdk_with_context_update):
    service = _make_service(tools=_tools("init_tool"))
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    context.set_tools(ToolsSchema(standard_tools=[]))
    await service.process_frame(LLMSetToolsFrame(tools=[]), FrameDirection.DOWNSTREAM)

    update = session.send_context_update.await_args.kwargs
    assert _declared_names(update["tools"]) == ["init_tool"]


@pytest.mark.asyncio
async def test_tool_updates_do_not_convert_the_conversation(sdk_with_context_update, monkeypatch):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    session = await _open_session(service, context)

    async def fail(*args, **kwargs):
        raise AssertionError("the conversation must not be converted")

    monkeypatch.setattr(service.get_llm_adapter(), "get_llm_invocation_params", fail)

    context.set_tools(_tools("book_table"))
    await service.process_frame(
        LLMSetToolsFrame(tools=_tools("book_table")), FrameDirection.DOWNSTREAM
    )

    update = session.send_context_update.await_args.kwargs
    assert _declared_names(update["tools"]) == ["book_table"]


@pytest.mark.asyncio
async def test_a_failed_send_on_a_replaced_session_is_not_an_error(sdk_with_context_update):
    service = _make_service()
    context = LLMContext(tools=_tools("get_weather"))
    old_session = await _open_session(service, context)
    service._handle_send_error = AsyncMock()

    async def send_then_fail(**update):
        service._session = _FakeSession()  # replaced while the send was out
        raise RuntimeError("socket closed")

    old_session.send_context_update.side_effect = send_then_fail

    context.set_tools(_tools("book_table"))
    await service._apply_context_changes()

    service._handle_send_error.assert_not_awaited()
    assert _declared_names(service._applied_tools) == ["get_weather"]


@pytest.mark.asyncio
async def test_initial_context_system_message_still_reconnects(sdk_with_context_update):
    service = _make_service()  # no system_instruction setting
    session = await _open_session(service, None)
    service._reconnect = AsyncMock()
    service._create_initial_response = AsyncMock()

    context = LLMContext(
        messages=[
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "hi"},
        ]
    )
    with pytest.warns(DeprecationWarning):
        await service._handle_context(context)

    service._reconnect.assert_awaited_once()
    session.send_context_update.assert_not_awaited()
