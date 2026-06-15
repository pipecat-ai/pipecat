#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Realtime services auto-register handlers from their init-time tools.

Some realtime services advertise the tools passed at construction — flat
(Gemini Live ``tools=``, AWS Nova Sonic ``tools=``, Ultravox
``one_shot_selected_tools=``) or nested (the OpenAI/Grok/Inworld realtime
``session_properties.tools``) — and run with an empty context. The base service
falls back to ``LLMService._init_time_tools()`` when the context advertises no
tools, so a handler bundled on an init-time ``FunctionSchema`` (or a direct
function) registers without a separate ``register_function`` call, mirroring how
the base service syncs from the context's tools on every ``LLMContextFrame``.

Each service is imported with ``pytest.importorskip`` (per service, inside
``_service``) so a provider whose optional dependencies aren't installed is
skipped rather than failing collection for the whole module.
"""

import unittest

import pytest

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.services.llm_service import FunctionCallParams


async def sample_handler(params: FunctionCallParams):
    """A sample handler bundled on a FunctionSchema."""
    await params.result_callback({})


@tool_options(cancel_on_interruption=False)
async def sample_async_handler(params: FunctionCallParams):
    """A sample async (non-interruptible) handler."""
    await params.result_callback({})


def _tools(handler) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="sample",
                description="A sample tool.",
                properties={},
                required=[],
                handler=handler,
            )
        ]
    )


class _InitTimeToolSyncTests:
    """Shared cases for services that auto-register their init-time tools.

    Subclasses provide ``_service(tools)``, building the service with the given
    ``ToolsSchema`` (or ``None``) wired into its init-time tool parameter. They
    use ``pytest.importorskip`` so a service whose optional dependencies aren't
    installed is skipped rather than erroring.
    """

    def _service(self, tools):
        raise NotImplementedError

    async def test_init_time_schema_handler_registers(self):
        service = self._service(_tools(sample_handler))
        # An empty context (NOT_GIVEN tools) falls back to the init-time tools.
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertTrue(service.has_function("sample"))

    async def test_init_time_async_tool_option_honored(self):
        service = self._service(_tools(sample_async_handler))
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertFalse(service._functions["sample"].cancel_on_interruption)

    async def test_no_init_time_tools_is_safe(self):
        service = self._service(None)
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertEqual(list(service._functions), [])


class TestGeminiLiveInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.google.gemini_live.llm")
        return mod.GeminiLiveLLMService(api_key="test-key", tools=tools)

    async def test_raw_dict_init_tools_register_nothing(self):
        # Gemini accepts provider-native dict tools, which carry no handler.
        service = self._service([{"function_declarations": []}])
        service._sync_registered_tool_handlers(NOT_GIVEN)
        self.assertEqual(list(service._functions), [])

    async def test_context_tools_take_precedence(self):
        service = self._service(
            ToolsSchema(
                standard_tools=[FunctionSchema("from_init", "d", {}, [], handler=sample_handler)]
            )
        )
        # When the context advertises tools, the init-time tools aren't used.
        service._sync_registered_tool_handlers(
            ToolsSchema(
                standard_tools=[FunctionSchema("from_context", "d", {}, [], handler=sample_handler)]
            )
        )
        self.assertTrue(service.has_function("from_context"))
        self.assertFalse(service.has_function("from_init"))


class TestAWSNovaSonicInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
        return mod.AWSNovaSonicLLMService(
            secret_access_key="test", access_key_id="test", region="us-east-1", tools=tools
        )


class TestUltravoxInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.ultravox.llm")
        return mod.UltravoxRealtimeLLMService(
            params=mod.OneShotInputParams(api_key="test-key", system_prompt="test"),
            one_shot_selected_tools=tools,
        )


class TestOpenAIRealtimeInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.openai.realtime.llm")
        events = pytest.importorskip("pipecat.services.openai.realtime.events")
        sp = events.SessionProperties(tools=tools)
        return mod.OpenAIRealtimeLLMService(
            api_key="test-key",
            settings=mod.OpenAIRealtimeLLMService.Settings(session_properties=sp),
        )


class TestGrokRealtimeInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.xai.realtime.llm")
        events = pytest.importorskip("pipecat.services.xai.realtime.events")
        sp = events.SessionProperties(tools=tools)
        return mod.GrokRealtimeLLMService(
            api_key="test-key",
            settings=mod.GrokRealtimeLLMService.Settings(session_properties=sp),
        )


class TestInworldRealtimeInitTimeToolSync(_InitTimeToolSyncTests, unittest.IsolatedAsyncioTestCase):
    def _service(self, tools):
        mod = pytest.importorskip("pipecat.services.inworld.realtime.llm")
        events = pytest.importorskip("pipecat.services.inworld.realtime.events")
        sp = events.SessionProperties(tools=tools)
        return mod.InworldRealtimeLLMService(
            api_key="test-key",
            settings=mod.InworldRealtimeLLMService.Settings(session_properties=sp),
        )


if __name__ == "__main__":
    unittest.main()
