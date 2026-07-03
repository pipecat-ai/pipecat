#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for automatic registration of tool handlers advertised in a context."""

import io
import types
import unittest

from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.utils.async_tool_cancellation import CANCEL_ASYNC_TOOL_NAME


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


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def end_call(params: FunctionCallParams, reason: str):
    """End the call.

    Args:
        reason: Why the call is ending.
    """
    await params.result_callback({"status": "ending"})


@tool_options(cancel_on_interruption=False)
async def async_task(params: FunctionCallParams, query: str):
    """Run an async task (cancel_on_interruption=False).

    Args:
        query: The task query.
    """
    await params.result_callback({"status": "done"})


@tool_options(cancel_on_interruption=False)
async def async_task_2(params: FunctionCallParams, query: str):
    """Run a second async task (cancel_on_interruption=False).

    Args:
        query: The task query.
    """
    await params.result_callback({"status": "done"})


async def lookup_order_handler(params: FunctionCallParams):
    """A classic (non-direct) handler bundled on a FunctionSchema."""
    await params.result_callback({"status": "shipped"})


def lookup_order_schema() -> FunctionSchema:
    """Build a handler-carrying FunctionSchema for ``lookup_order``."""
    return FunctionSchema(
        name="lookup_order",
        description="Look up an order.",
        properties={"order_id": {"type": "string"}},
        required=["order_id"],
        handler=lookup_order_handler,
    )


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def end_call_handler(params: FunctionCallParams):
    """A schema handler whose call options come from @tool_options."""
    await params.result_callback({"status": "ending"})


def end_call_schema() -> FunctionSchema:
    """Build a handler-carrying FunctionSchema for ``end_call``."""
    return FunctionSchema(
        name="end_call",
        description="End the call.",
        properties={},
        required=[],
        handler=end_call_handler,
    )


class TestToolsSchemaRetainsCallables(unittest.TestCase):
    def test_direct_functions_become_schemas_and_retain_wrappers(self):
        schema = FunctionSchema(name="advertise_only", description="d", properties={}, required=[])
        tools = ToolsSchema(standard_tools=[get_current_weather, schema])
        # Direct functions are advertised as FunctionSchemas...
        self.assertTrue(all(isinstance(t, FunctionSchema) for t in tools.standard_tools))
        self.assertEqual(
            {t.name for t in tools.standard_tools}, {"get_current_weather", "advertise_only"}
        )
        # ...and retained as wrappers so their handlers can be registered. A
        # FunctionSchema-only entry is advertise-only (no wrapper).
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
        service._sync_registered_tool_handlers(context.tools)
        self.assertTrue(service.has_function("get_current_weather"))
        self.assertTrue(service.has_function("get_restaurant_recommendation"))
        item = service._functions["get_current_weather"]
        self.assertTrue(item.cancel_on_interruption)
        self.assertIsNone(item.timeout_secs)

    def test_auto_register_reads_decorator_options(self):
        service = self._service()
        context = LLMContext(tools=[end_call])
        service._sync_registered_tool_handlers(context.tools)
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_explicit_registration_not_clobbered(self):
        service = self._service()
        # Register explicitly with non-default options first.
        service._register_direct_function(
            get_current_weather, cancel_on_interruption=False, timeout_secs=99
        )
        context = LLMContext(tools=[get_current_weather])
        service._sync_registered_tool_handlers(context.tools)
        item = service._functions["get_current_weather"]
        # The explicit registration's options survive (no clobber).
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 99)

    def test_function_schema_advertised_not_registered(self):
        service = self._service()
        schema = FunctionSchema(name="advertise_only", description="d", properties={}, required=[])
        context = LLMContext(tools=[get_current_weather, schema])
        service._sync_registered_tool_handlers(context.tools)
        self.assertTrue(service.has_function("get_current_weather"))
        # 'advertise_only' has no handler; it must not be auto-registered.
        self.assertNotIn("advertise_only", service._functions)

    def test_no_tools_is_safe(self):
        service = self._service()
        service._sync_registered_tool_handlers(LLMContext().tools)
        self.assertEqual(service._functions, {})

    def test_none_tools_is_safe(self):
        # Some callers (and realtime services in tests) hand the LLM service a
        # context whose ``tools`` is ``None`` rather than NOT_GIVEN. is_given(None)
        # is True, so this must be guarded explicitly.
        service = self._service()
        ctx = types.SimpleNamespace(tools=None)
        service._sync_registered_tool_handlers(ctx.tools)
        self.assertEqual(service._functions, {})

    def test_idempotent_across_repeated_contexts(self):
        service = self._service()
        context = LLMContext(tools=[end_call])
        service._sync_registered_tool_handlers(context.tools)
        # A second pass must not change anything (and must not raise).
        service._sync_registered_tool_handlers(context.tools)
        self.assertEqual(list(service._functions.keys()), ["end_call"])


class TestAutoRegisterSchemaHandlers(unittest.TestCase):
    """A FunctionSchema that carries a handler is auto-registered like a direct function."""

    def _service(self) -> LLMService:
        return LLMService()

    def test_schema_handler_is_registered_with_defaults(self):
        service = self._service()
        service._sync_registered_tool_handlers([lookup_order_schema()])
        item = service._functions["lookup_order"]
        self.assertIs(item.handler, lookup_order_handler)
        # Registered from the advertised set, so it's prunable (like a direct function).
        self.assertTrue(item.auto_registered)
        # An undecorated handler gets the classic defaults.
        self.assertTrue(item.cancel_on_interruption)
        self.assertIsNone(item.timeout_secs)

    def test_schema_handler_reads_tool_options_decorator(self):
        service = self._service()
        service._sync_registered_tool_handlers([end_call_schema()])
        item = service._functions["end_call"]
        # Options come from @tool_options on the handler, just like a direct function.
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_handlerless_schema_is_still_advertise_only(self):
        service = self._service()
        advertise_only = FunctionSchema(
            name="advertise_only", description="d", properties={}, required=[]
        )
        service._sync_registered_tool_handlers([advertise_only, lookup_order_schema()])
        self.assertNotIn("advertise_only", service._functions)
        self.assertTrue(service.has_function("lookup_order"))

    def test_deadvertised_schema_handler_is_pruned(self):
        service = self._service()
        service._sync_registered_tool_handlers([lookup_order_schema()])
        self.assertTrue(service.has_function("lookup_order"))
        # A new tool set that no longer advertises it drops the handler.
        service._sync_registered_tool_handlers([])
        self.assertNotIn("lookup_order", service._functions)

    def test_idempotent_across_repeated_contexts(self):
        service = self._service()
        service._sync_registered_tool_handlers([lookup_order_schema()])
        service._sync_registered_tool_handlers([lookup_order_schema()])
        self.assertEqual(list(service._functions.keys()), ["lookup_order"])

    def test_reserved_name_rejected(self):
        service = self._service()
        schema = FunctionSchema(
            name=CANCEL_ASYNC_TOOL_NAME,
            description="d",
            properties={},
            required=[],
            handler=lookup_order_handler,
        )
        with self.assertRaises(ValueError):
            service._sync_registered_tool_handlers([schema])


class TestRedundantManualRegistrationWarning(unittest.TestCase):
    """Manually registering a handler the schema already carries warns, once."""

    def _service(self) -> LLMService:
        return LLMService()

    def _capture_warnings(self):
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        return sink, handler_id

    def test_explicit_registration_wins_and_warns(self):
        service = self._service()

        async def explicit_handler(params: FunctionCallParams):
            await params.result_callback({})

        service.register_function("lookup_order", explicit_handler)
        sink, handler_id = self._capture_warnings()
        try:
            service._sync_registered_tool_handlers([lookup_order_schema()])
        finally:
            logger.remove(handler_id)
        # The explicit handler is left in place (the schema's handler doesn't clobber it).
        self.assertIs(service._functions["lookup_order"].handler, explicit_handler)
        self.assertIn("lookup_order", sink.getvalue())
        self.assertIn("unnecessary", sink.getvalue())

    def test_warning_fires_once_across_repeated_syncs(self):
        service = self._service()

        async def explicit_handler(params: FunctionCallParams):
            await params.result_callback({})

        service.register_function("lookup_order", explicit_handler)
        sink, handler_id = self._capture_warnings()
        try:
            for _ in range(3):
                service._sync_registered_tool_handlers([lookup_order_schema()])
        finally:
            logger.remove(handler_id)
        # Sync runs per inference; the advisory must not repeat every turn.
        self.assertEqual(sink.getvalue().count("lookup_order"), 1)

    def test_no_warning_for_advertise_only_schema(self):
        service = self._service()

        async def explicit_handler(params: FunctionCallParams):
            await params.result_callback({})

        # Classic pattern: register a handler and advertise a handlerless schema.
        service.register_function("advertise_only", explicit_handler)
        advertise_only = FunctionSchema(
            name="advertise_only", description="d", properties={}, required=[]
        )
        sink, handler_id = self._capture_warnings()
        try:
            service._sync_registered_tool_handlers([advertise_only])
        finally:
            logger.remove(handler_id)
        self.assertEqual(sink.getvalue(), "")


class TestUnregisterInteractionWithAutoRegister(unittest.TestCase):
    """A standalone unregister sticks; the advertised set doesn't resurrect it."""

    def _service(self) -> LLMService:
        return LLMService()

    def test_unregister_of_still_advertised_direct_function_sticks(self):
        # Unregistering a direct function that's still advertised must stick: the
        # next context frame must not auto-re-register it (a "zombie"), so calls
        # to it hit the missing-handler recovery path instead.
        service = self._service()
        context = LLMContext(tools=[get_current_weather])
        service._sync_registered_tool_handlers(context.tools)
        service._unregister_direct_function(get_current_weather)
        self.assertNotIn("get_current_weather", service._functions)
        service._sync_registered_tool_handlers(context.tools)
        self.assertNotIn("get_current_weather", service._functions)

    def test_unregister_of_still_advertised_schema_handler_sticks(self):
        # Same guarantee for a FunctionSchema-carried handler removed via
        # unregister_function.
        service = self._service()
        service._sync_registered_tool_handlers([lookup_order_schema()])
        service.unregister_function("lookup_order")
        service._sync_registered_tool_handlers([lookup_order_schema()])
        self.assertNotIn("lookup_order", service._functions)

    def test_explicit_reregister_overrides_suppression(self):
        # Explicitly registering again clears the suppression and survives the
        # next context frame.
        service = self._service()
        context = LLMContext(tools=[get_current_weather])
        service._sync_registered_tool_handlers(context.tools)
        service._unregister_direct_function(get_current_weather)
        service._register_direct_function(get_current_weather)
        self.assertTrue(service.has_function("get_current_weather"))
        service._sync_registered_tool_handlers(context.tools)
        self.assertTrue(service.has_function("get_current_weather"))

    def test_deadvertising_clears_suppression_so_readvertising_restores(self):
        # The suppression only holds while the tool stays advertised. Dropping it
        # from the tool set and advertising it again brings the handler back.
        service = self._service()
        service._sync_registered_tool_handlers([get_current_weather])
        service._unregister_direct_function(get_current_weather)
        service._sync_registered_tool_handlers([get_current_weather])
        self.assertNotIn("get_current_weather", service._functions)
        # Drop it from the advertised set...
        service._sync_registered_tool_handlers([])
        # ...then advertise it again: it auto-registers afresh.
        service._sync_registered_tool_handlers([get_current_weather])
        self.assertTrue(service.has_function("get_current_weather"))


class TestRegisterFunctionOptionPrecedence(unittest.TestCase):
    """Explicit arg > @tool_options/@tool decorator > default.

    register_function and register_direct_function share the resolution
    (_resolve_tool_option), so both registrars are exercised here. The
    timeout-specific edge cases are covered once, via the direct path.
    """

    def _service(self) -> LLMService:
        return LLMService()

    # -- register_function --------------------------------------------------

    def test_register_function_decorator_values_used_when_no_explicit_args(self):
        service = self._service()
        service.register_function("end_call", end_call_handler)  # decorated: False / 60
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_register_function_defaults_used_for_undecorated_handler(self):
        service = self._service()

        async def handler(params: FunctionCallParams):
            await params.result_callback({})

        service.register_function("do_thing", handler)  # undecorated
        item = service._functions["do_thing"]
        self.assertTrue(item.cancel_on_interruption)
        self.assertIsNone(item.timeout_secs)

    def test_register_function_explicit_arg_overrides_decorator(self):
        service = self._service()
        service.register_function("end_call", end_call_handler, cancel_on_interruption=True)
        item = service._functions["end_call"]
        self.assertTrue(item.cancel_on_interruption)  # explicit wins
        self.assertEqual(item.timeout_secs, 60)  # decorator still applies

    # -- register_direct_function -------------------------------------------

    def test_register_direct_function_decorator_values_used_when_no_explicit_args(self):
        service = self._service()
        service._register_direct_function(end_call)  # decorated: False / 60
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_register_direct_function_defaults_used_for_undecorated_function(self):
        service = self._service()
        service._register_direct_function(get_current_weather)  # undecorated
        item = service._functions["get_current_weather"]
        self.assertTrue(item.cancel_on_interruption)
        self.assertIsNone(item.timeout_secs)

    def test_register_direct_function_explicit_arg_overrides_decorator(self):
        service = self._service()
        # Override one option explicitly; the other still comes from the decorator.
        service._register_direct_function(end_call, cancel_on_interruption=True)
        item = service._functions["end_call"]
        self.assertTrue(item.cancel_on_interruption)  # explicit wins
        self.assertEqual(item.timeout_secs, 60)  # decorator still applies

    def test_register_direct_function_explicit_timeout_overrides_decorator(self):
        service = self._service()
        service._register_direct_function(end_call, timeout_secs=5)
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)  # decorator still applies
        self.assertEqual(item.timeout_secs, 5)  # explicit wins

    def test_register_direct_function_explicit_none_timeout_falls_back_to_decorator(self):
        # None means "not provided": it falls back to the decorator value rather
        # than forcing the global default past the decorator.
        service = self._service()
        service._register_direct_function(end_call, timeout_secs=None)
        item = service._functions["end_call"]
        self.assertEqual(item.timeout_secs, 60)


class TestAutoUnregisterFromToolSet(unittest.TestCase):
    """An LLMSetToolsFrame replacing the tool set prunes de-advertised handlers."""

    def _service(self) -> LLMService:
        return LLMService()

    def test_deadvertised_direct_function_is_unregistered(self):
        service = self._service()
        # Auto-register two from the initial context.
        service._sync_registered_tool_handlers(
            LLMContext(tools=[get_current_weather, end_call]).tools
        )
        self.assertTrue(service.has_function("get_current_weather"))
        self.assertTrue(service.has_function("end_call"))
        # A new tool set advertises only one of them.
        service._sync_registered_tool_handlers([get_current_weather])
        self.assertTrue(service.has_function("get_current_weather"))
        self.assertNotIn("end_call", service._functions)

    def test_readvertising_brings_handler_back(self):
        service = self._service()
        service._sync_registered_tool_handlers(LLMContext(tools=[end_call]).tools)
        service._sync_registered_tool_handlers([get_current_weather])
        self.assertNotIn("end_call", service._functions)
        # Re-advertising registers it again (not blocked as "unregistered").
        service._sync_registered_tool_handlers([get_current_weather, end_call])
        self.assertTrue(service.has_function("end_call"))

    def test_empty_tool_set_prunes_all(self):
        service = self._service()
        service._sync_registered_tool_handlers(
            LLMContext(tools=[get_current_weather, end_call]).tools
        )
        service._sync_registered_tool_handlers([])  # clear all tools
        self.assertEqual(service._functions, {})

    def test_explicitly_registered_handler_is_not_pruned(self):
        service = self._service()
        # Explicit registration is outside the advertised-tool-set lifecycle.
        service._register_direct_function(get_current_weather)
        # A tool set that doesn't advertise it must not remove it.
        service._sync_registered_tool_handlers([end_call])
        self.assertTrue(service.has_function("get_current_weather"))
        self.assertTrue(service.has_function("end_call"))

    def test_catch_all_handler_is_not_pruned(self):
        service = self._service()

        async def catch_all(params: FunctionCallParams):
            await params.result_callback({})

        service.register_function(None, catch_all)
        service._sync_registered_tool_handlers(LLMContext(tools=[end_call]).tools)
        service._sync_registered_tool_handlers([get_current_weather])
        # Catch-all survives; the de-advertised direct function is pruned.
        self.assertIn(None, service._functions)
        self.assertNotIn("end_call", service._functions)


class TestDeprecatedExplicitRegistrationApi(unittest.TestCase):
    """register/unregister_direct_function are deprecated but still delegate."""

    def test_register_direct_function_warns_and_registers(self):
        service = LLMService()
        with self.assertWarns(DeprecationWarning):
            service.register_direct_function(end_call)
        # Still works, decorator options included.
        item = service._functions["end_call"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 60)

    def test_unregister_direct_function_warns_and_unregisters(self):
        service = LLMService()
        service._register_direct_function(get_current_weather)
        with self.assertWarns(DeprecationWarning):
            service.unregister_direct_function(get_current_weather)
        self.assertNotIn("get_current_weather", service._functions)


class TestToolDecoratorSharesAnnotation(unittest.TestCase):
    def test_tool_sets_options_and_marker(self):
        from pipecat.workers.llm.tool_decorator import tool

        @tool(cancel_on_interruption=False, timeout_secs=30)
        async def my_tool(self, params: FunctionCallParams, arg: str):
            """A tool.

            Args:
                arg: An argument.
            """
            await params.result_callback({})

        self.assertTrue(my_tool._pipecat_is_llm_tool)
        self.assertFalse(my_tool._pipecat_cancel_on_interruption)
        self.assertEqual(my_tool._pipecat_timeout_secs, 30)

    def test_tool_timeout_alias_is_deprecated_but_works(self):
        from pipecat.workers.llm.tool_decorator import tool

        with self.assertWarns(DeprecationWarning):

            @tool(timeout=30)
            async def my_tool(self, params: FunctionCallParams, arg: str):
                """A tool.

                Args:
                    arg: An argument.
                """
                await params.result_callback({})

        # The deprecated alias still populates the canonical option.
        self.assertEqual(my_tool._pipecat_timeout_secs, 30)


class TestReservedToolName(unittest.TestCase):
    """The built-in cancel tool name can't be registered by user code."""

    def test_register_function_rejects_reserved_name(self):
        service = LLMService()

        async def handler(params: FunctionCallParams):
            await params.result_callback({})

        with self.assertRaises(ValueError):
            service.register_function(CANCEL_ASYNC_TOOL_NAME, handler)

    def test_register_direct_function_rejects_reserved_name(self):
        service = LLMService()

        async def cancel_async_tool_call(params: FunctionCallParams, tool_call_id: str):
            """A direct function whose name collides with the reserved built-in.

            Args:
                tool_call_id: The call to cancel.
            """
            await params.result_callback({})

        with self.assertRaises(ValueError):
            service._register_direct_function(cancel_async_tool_call)


class TestClassicRegisterFunction(unittest.TestCase):
    """register_function (non-direct handlers): registration and removal.

    Option precedence is covered in TestRegisterFunctionOptionPrecedence.
    """

    async def _handler(self, params: FunctionCallParams):
        await params.result_callback({})

    def test_registers_named_handler_with_options(self):
        service = LLMService()
        service.register_function(
            "do_thing", self._handler, cancel_on_interruption=False, timeout_secs=12
        )
        item = service._functions["do_thing"]
        self.assertFalse(item.cancel_on_interruption)
        self.assertEqual(item.timeout_secs, 12)
        # Classic registrations are explicit, never advertised-set-managed.
        self.assertFalse(item.auto_registered)

    def test_unregister_function_removes_handler(self):
        service = LLMService()
        service.register_function("do_thing", self._handler)
        self.assertTrue(service.has_function("do_thing"))
        service.unregister_function("do_thing")
        self.assertNotIn("do_thing", service._functions)


class TestAsyncToolCancellationPruning(unittest.TestCase):
    """Pruning interacts correctly with the built-in async-tool-cancellation tool."""

    def test_pruning_last_async_tool_tears_down_cancellation(self):
        service = LLMService()
        service._sync_registered_tool_handlers(LLMContext(tools=[async_task]).tools)
        service._setup_async_tool_cancellation()
        self.assertIn(CANCEL_ASYNC_TOOL_NAME, service._functions)
        # Drop the only async tool from the advertised set.
        service._sync_registered_tool_handlers([])
        self.assertNotIn("async_task", service._functions)
        # No async tools remain, so the built-in cancel tool is torn down.
        self.assertNotIn(CANCEL_ASYNC_TOOL_NAME, service._functions)

    def test_builtin_cancel_tool_survives_while_async_tools_remain(self):
        service = LLMService()
        service._sync_registered_tool_handlers(LLMContext(tools=[async_task, async_task_2]).tools)
        service._setup_async_tool_cancellation()
        # Drop one async tool; another remains.
        service._sync_registered_tool_handlers([async_task])
        self.assertTrue(service.has_function("async_task"))
        self.assertNotIn("async_task_2", service._functions)
        # The built-in cancel tool isn't auto_registered, so the prune loop
        # leaves it alone (and async tools still remain).
        self.assertIn(CANCEL_ASYNC_TOOL_NAME, service._functions)


if __name__ == "__main__":
    unittest.main()
