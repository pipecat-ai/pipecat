#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mid-session tool and system instruction updates on Gemini Live.

Gemini 3.8 Live accepts a context update that replaces the session's system
instruction and/or tools, which the Gen AI SDK sends with
``AsyncSession.send_context_update``. These tests give the service a fake
session and assert on what it is asked to send.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live import llm as llm_module
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.settings import LLMSettings


class _SessionWithContextUpdate:
    """Stands in for an ``AsyncSession`` from google-genai 2.33 or later."""

    async def send_context_update(self, **update):  # what the capability check looks for
        raise NotImplementedError

    def __init__(self):
        self.send_context_update = AsyncMock()


@pytest.fixture
def sdk_with_context_update(monkeypatch):
    """Make the capability check see an SDK that has ``send_context_update``."""
    monkeypatch.setattr(llm_module, "AsyncSession", _SessionWithContextUpdate)


def _make_service(model: str = "models/gemini-3.8-live") -> GeminiLiveLLMService:
    """A service with a fake live session. ``__init__`` does no I/O."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(model=model, system_instruction="Be brief."),
    )
    service._session = _SessionWithContextUpdate()
    return service


def _tools(*names: str) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(name=name, description=f"{name} tool", properties={}, required=[])
            for name in names
        ]
    )


def _declared_names(tools: list) -> list[str]:
    return [d["name"] for tool in tools for d in tool.get("function_declarations", [])]


@pytest.mark.parametrize(
    "model, expected",
    [
        ("models/gemini-2.5-flash-native-audio-preview-12-2025", False),
        ("models/gemini-3-pro-preview", False),
        ("models/gemini-3.8-live", True),
        ("models/gemini-3.8-live-extended-thinking", True),
        ("models/gemini-4-live", True),
    ],
)
def test_context_update_support_by_model(sdk_with_context_update, model, expected):
    assert _make_service(model)._supports_context_update is expected


def test_no_context_update_without_sdk_support():
    """An SDK whose session lacks send_context_update keeps the old behaviour."""
    if hasattr(llm_module.AsyncSession, "send_context_update"):
        pytest.skip("installed google-genai supports context updates")
    assert _make_service()._supports_context_update is False


@pytest.mark.asyncio
async def test_set_tools_frame_updates_the_session_tools(sdk_with_context_update):
    service = _make_service()
    # The context aggregator applies the frame to the context before the service sees it.
    service._context = LLMContext(tools=_tools("check_availability", "book_appointment"))

    await service.process_frame(
        LLMSetToolsFrame(tools=service._context.tools), FrameDirection.DOWNSTREAM
    )

    service._session.send_context_update.assert_awaited_once()
    tools = service._session.send_context_update.await_args.kwargs["tools"]
    assert _declared_names(tools) == ["check_availability", "book_appointment"]


@pytest.mark.asyncio
async def test_removing_every_tool_sends_an_empty_list(sdk_with_context_update):
    service = _make_service()
    service._context = LLMContext(tools=ToolsSchema(standard_tools=[]))

    await service._handle_set_tools()

    service._session.send_context_update.assert_awaited_once_with(tools=[])


@pytest.mark.asyncio
async def test_system_instruction_setting_updates_the_session(sdk_with_context_update):
    service = _make_service()
    service._context = LLMContext()

    await service._update_settings(LLMSettings(system_instruction="Speak German."))

    service._session.send_context_update.assert_awaited_once_with(
        system_instruction="Speak German."
    )
    # A reconnect starts from the new instruction too.
    assert service._system_instruction_from_init == "Speak German."


@pytest.mark.asyncio
async def test_older_models_keep_the_session_unchanged(sdk_with_context_update):
    service = _make_service("models/gemini-2.5-flash-native-audio-preview-12-2025")
    service._context = LLMContext(tools=_tools("lookup"))

    await service._handle_set_tools()
    await service._update_settings(LLMSettings(system_instruction="Speak German."))

    service._session.send_context_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_session_there_is_nothing_to_update(sdk_with_context_update):
    service = _make_service()
    service._session = None
    service._context = LLMContext(tools=_tools("lookup"))

    await service._handle_set_tools()  # must not raise; the next connection uses the context
