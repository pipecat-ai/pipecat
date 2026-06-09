#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for automatic registration of direct functions advertised in a context."""

import types
import unittest

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit, "celsius" or "fahrenheit".
    """
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})


@tool_options(cancel_on_interruption=False, timeout=60)
async def end_call(params: FunctionCallParams, reason: str):
    """End the call.

    Args:
        reason: Why the call is ending.
    """
    await params.result_callback({"status": "ending"})


class TestDirectFunctionDecorator(unittest.TestCase):
    def test_decorator_attaches_options(self):
        self.assertFalse(end_call.cancel_on_interruption)
        self.assertEqual(end_call.timeout, 60)

    def test_undecorated_function_has_no_options(self):
        self.assertFalse(hasattr(get_current_weather, "cancel_on_interruption"))
        self.assertFalse(hasattr(get_current_weather, "timeout"))

    def test_decorator_without_args(self):
        @tool_options
        async def some_tool(params: FunctionCallParams, x: str):
            """Docstring.

            Args:
                x: An argument.
            """
            await params.result_callback({})

        self.assertTrue(some_tool.cancel_on_interruption)
        self.assertIsNone(some_tool.timeout)


class TestToolsSchemaRetainsCallables(unittest.TestCase):
    def test_standard_tools_still_function_schemas(self):
        tools = ToolsSchema(standard_tools=[get_current_weather, get_restaurant_recommendation])
        self.assertTrue(all(isinstance(t, FunctionSchema) for t in tools.standard_tools))
        self.assertEqual(
            {t.name for t in tools.standard_tools},
            {"get_current_weather", "get_restaurant_recommendation"},
        )

    def test_direct_functions_retained(self):
        tools = ToolsSchema(standard_tools=[get_current_weather, end_call])
        self.assertEqual(
            {w.name for w in tools.direct_functions}, {"get_current_weather", "end_call"}
        )

    def test_function_schema_entries_not_in_direct_functions(self):
        schema = FunctionSchema(name="advertise_only", description="d", properties={}, required=[])
        tools = ToolsSchema(standard_tools=[get_current_weather, schema])
        # The schema is advertised but is not a direct function.
        self.assertIn("get_current_weather", {t.name for t in tools.standard_tools})
        self.assertIn("advertise_only", {t.name for t in tools.standard_tools})
        self.assertEqual({w.name for w in tools.direct_functions}, {"get_current_weather"})


class TestLLMContextAcceptsList(unittest.TestCase):
    def test_context_accepts_bare_list(self):
        context = LLMContext(tools=[get_current_weather, get_restaurant_recommendation])
        self.assertIsInstance(context.tools, ToolsSchema)
        self.assertEqual(
            {t.name for t in context.tools.standard_tools},
            {"get_current_weather", "get_restaurant_recommendation"},
        )

    def test_set_tools_accepts_bare_list(self):
        context = LLMContext()
        context.set_tools([get_current_weather])
        self.assertEqual({t.name for t in context.tools.standard_tools}, {"get_current_weather"})

    def test_invalid_tools_type_raises(self):
        with self.assertRaises(TypeError):
            LLMContext(tools="not-a-tools-object")  # type: ignore[arg-type]


class TestAutoRegister(unittest.TestCase):
    def _service(self) -> LLMService:
        return LLMService()

    def test_auto_registers_with_defaults(self):
        service = self._service()
        context = LLMContext(tools=[get_current_weather, get_restaurant_recommendation])
        service._auto_register_direct_functions(context)
        self.assertTrue(service.has_function("get_current_weather"))
        self.assertTrue(service.has_function("get_restaurant_recommendation"))
        item = service._functions["get_current_weather"]
        self.assertTrue(item.cancel_on_interruption)
        self.assertIsNone(item.timeout_secs)

    def test_auto_register_reads_decorator_options(self):
        service = self._service()
        context = LLMContext(tools=[end_call])
        service._auto_register_direct_functions(context)
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_explicit_registration_not_clobbered(self):
        service = self._service()
        # Register explicitly with non-default options first.
        service.register_direct_function(
            get_current_weather, cancel_on_interruption=False, timeout_secs=99
        )
        context = LLMContext(tools=[get_current_weather])
        service._auto_register_direct_functions(context)
        item = service._functions["get_current_weather"]
        # The explicit registration's options survive (no clobber).
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 99)

    def test_function_schema_advertised_not_registered(self):
        service = self._service()
        schema = FunctionSchema(name="advertise_only", description="d", properties={}, required=[])
        context = LLMContext(tools=[get_current_weather, schema])
        service._auto_register_direct_functions(context)
        self.assertTrue(service.has_function("get_current_weather"))
        # 'advertise_only' has no handler; it must not be auto-registered.
        self.assertNotIn("advertise_only", service._functions)

    def test_no_tools_is_safe(self):
        service = self._service()
        service._auto_register_direct_functions(LLMContext())
        self.assertEqual(service._functions, {})

    def test_none_tools_is_safe(self):
        # Some callers (and realtime services in tests) hand the LLM service a
        # context whose ``tools`` is ``None`` rather than NOT_GIVEN. is_given(None)
        # is True, so this must be guarded explicitly.
        service = self._service()
        ctx = types.SimpleNamespace(tools=None)
        service._auto_register_direct_functions(ctx)
        self.assertEqual(service._functions, {})

    def test_idempotent_across_repeated_contexts(self):
        service = self._service()
        context = LLMContext(tools=[end_call])
        service._auto_register_direct_functions(context)
        # A second pass must not change anything (and must not raise).
        service._auto_register_direct_functions(context)
        self.assertEqual(list(service._functions.keys()), ["end_call"])


class TestToolDecoratorSharesAnnotation(unittest.TestCase):
    def test_tool_sets_options_and_marker(self):
        from pipecat.workers.llm.tool_decorator import tool

        @tool(cancel_on_interruption=False, timeout=30)
        async def my_tool(self, params: FunctionCallParams, arg: str):
            """A tool.

            Args:
                arg: An argument.
            """
            await params.result_callback({})

        self.assertTrue(my_tool.is_llm_tool)
        self.assertFalse(my_tool.cancel_on_interruption)
        self.assertEqual(my_tool.timeout, 30)


if __name__ == "__main__":
    unittest.main()
