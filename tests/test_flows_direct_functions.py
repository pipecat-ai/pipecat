#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from typing import Optional, TypedDict, Union

from pipecat.flows.exceptions import InvalidFunctionError
from pipecat.flows.manager import FlowManager
from pipecat.flows.types import (
    ConsolidatedFunctionResult,
    FlowsDirectFunctionWrapper,
    flows_direct_function,
    flows_tool_options,
)

"""Tests for FlowsDirectFunction class."""


class TestFlowsDirectFunction(unittest.TestCase):
    def test_name_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the name from the function."""

        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function))
        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertEqual(func.name, "my_function")

    def test_description_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the description from the function."""

        async def my_function_short_description(flow_manager: FlowManager):
            """This is a test function."""
            return {"status": "success"}, None

        self.assertIsNone(
            FlowsDirectFunctionWrapper.validate_function(my_function_short_description)
        )
        func = FlowsDirectFunctionWrapper(function=my_function_short_description)
        self.assertEqual(func.description, "This is a test function.")

        async def my_function_long_description(flow_manager: FlowManager):
            """
            This is a test function.

            It does some really cool stuff.

            Trust me, you'll want to use it.
            """
            return {"status": "success"}, None

        self.assertIsNone(
            FlowsDirectFunctionWrapper.validate_function(my_function_long_description)
        )
        func = FlowsDirectFunctionWrapper(function=my_function_long_description)
        self.assertEqual(
            func.description,
            "This is a test function.\n\nIt does some really cool stuff.\n\nTrust me, you'll want to use it.",
        )

    def test_properties_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_no_params))
        func = FlowsDirectFunctionWrapper(function=my_function_no_params)
        self.assertEqual(func.properties, {})

        async def my_function_simple_params(
            flow_manager: FlowManager, name: str, age: int, height: float | None
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_simple_params))
        func = FlowsDirectFunctionWrapper(function=my_function_simple_params)
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "height": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
        )

        async def my_function_complex_params(
            flow_manager: FlowManager,
            address_lines: list[str],
            nickname: str | int | float,
            extra: dict[str, str] | None,
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_complex_params))
        func = FlowsDirectFunctionWrapper(function=my_function_complex_params)
        self.assertEqual(
            func.properties,
            {
                "address_lines": {"type": "array", "items": {"type": "string"}},
                "nickname": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}]
                },
                "extra": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
            },
        )

        class MyInfo1(TypedDict):
            name: str
            age: int

        class MyInfo2(TypedDict, total=False):
            name: str
            age: int

        async def my_function_complex_type_params(
            flow_manager: FlowManager, info1: MyInfo1, info2: MyInfo2
        ):
            return {"status": "success"}, None

        self.assertIsNone(
            FlowsDirectFunctionWrapper.validate_function(my_function_complex_type_params)
        )
        func = FlowsDirectFunctionWrapper(function=my_function_complex_type_params)
        self.assertEqual(
            func.properties,
            {
                "info1": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "info2": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        )

    def test_required_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the required properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_no_params))
        func = FlowsDirectFunctionWrapper(function=my_function_no_params)
        self.assertEqual(func.required, [])

        async def my_function_simple_params(
            flow_manager: FlowManager, name: str, age: int, height: float | None = None
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_simple_params))
        func = FlowsDirectFunctionWrapper(function=my_function_simple_params)
        self.assertEqual(func.required, ["name", "age"])

        async def my_function_complex_params(
            flow_manager: FlowManager,
            address_lines: list[str] | None,
            nickname: str | int = "Bud",
            extra: dict[str, str] | None = None,
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function_complex_params))
        func = FlowsDirectFunctionWrapper(function=my_function_complex_params)
        self.assertEqual(func.required, ["address_lines"])

    def test_property_descriptions_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the property descriptions from the function."""

        async def my_function(flow_manager: FlowManager, name: str, age: int, height: float | None):
            """
            This is a test function.

            Args:
                name (str): The name of the person.
                age (int): The age of the person.
                height (float | None): The height of the person in meters. Defaults to None.
            """
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_function))
        func = FlowsDirectFunctionWrapper(function=my_function)

        # Validate that the function description is still set correctly even with the longer docstring
        self.assertEqual(func.description, "This is a test function.")

        # Validate that the property descriptions are set correctly
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string", "description": "The name of the person."},
                "age": {"type": "integer", "description": "The age of the person."},
                "height": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "description": "The height of the person in meters. Defaults to None.",
                },
            },
        )

    def test_invalid_functions_fail_validation(self):
        """Test that invalid functions fail FlowsDirectFunction validation."""

        def my_function_non_async(flow_manager: FlowManager):
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunctionWrapper.validate_function(my_function_non_async)

        async def my_function_missing_flow_manager():
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunctionWrapper.validate_function(my_function_missing_flow_manager)

        async def my_function_misplaced_flow_manager(foo: str, flow_manager: FlowManager):
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunctionWrapper.validate_function(my_function_misplaced_flow_manager)

    def test_invoke_calls_function_with_args_and_flow_manager(self):
        """Test that FlowsDirectFunction.invoke calls the function with correct args and flow_manager."""

        called = {}

        class DummyFlowManager:
            pass

        async def my_function(flow_manager: DummyFlowManager, name: str, age: int):
            called["flow_manager"] = flow_manager
            called["name"] = name
            called["age"] = age
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        flow_manager = DummyFlowManager()
        args = {"name": "Alice", "age": 30}

        result = asyncio.run(func.invoke(args=args, flow_manager=flow_manager))
        self.assertEqual(result, ({"status": "success"}, None))
        self.assertIs(called["flow_manager"], flow_manager)
        self.assertEqual(called["name"], "Alice")
        self.assertEqual(called["age"], 30)


class TestFlowsDirectFunctionDecorator(unittest.TestCase):
    def test_cancel_on_interruption_defaults_to_false(self):
        """Test that cancel_on_interruption defaults to False for non-decorated functions."""

        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertFalse(func.cancel_on_interruption)

    def test_cancel_on_interruption_can_be_set_to_false(self):
        """Test that cancel_on_interruption can be set to False via decorator."""

        @flows_tool_options(cancel_on_interruption=False)
        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertFalse(func.cancel_on_interruption)

    def test_cancel_on_interruption_can_be_explicitly_set_to_true(self):
        """Test that cancel_on_interruption can be explicitly set to True via decorator."""

        @flows_tool_options(cancel_on_interruption=True)
        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertTrue(func.cancel_on_interruption)

    def test_decorator_preserves_function_metadata(self):
        """Test that the decorator preserves function name and docstring."""

        @flows_tool_options(cancel_on_interruption=False)
        async def my_decorated_function(flow_manager: FlowManager, name: str):
            """This is a decorated function.

            Args:
                name: The name to use.
            """
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_decorated_function)
        self.assertEqual(func.name, "my_decorated_function")
        self.assertEqual(func.description, "This is a decorated function.")
        self.assertEqual(
            func.properties,
            {"name": {"type": "string", "description": "The name to use."}},
        )
        self.assertFalse(func.cancel_on_interruption)

    def test_timeout_secs_defaults_to_none(self):
        """Test that timeout_secs defaults to None for non-decorated functions."""

        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertIsNone(func.timeout_secs)

    def test_timeout_secs_can_be_set(self):
        """Test that timeout_secs can be set via decorator."""

        @flows_tool_options(timeout_secs=30)
        async def my_function(flow_manager: FlowManager):
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertEqual(func.timeout_secs, 30)

    def test_decorator_preserves_function_metadata_with_timeout(self):
        """Test that the decorator preserves function name and docstring with timeout_secs."""

        @flows_tool_options(cancel_on_interruption=False, timeout_secs=15.5)
        async def my_decorated_function(flow_manager: FlowManager, name: str):
            """This is a decorated function.

            Args:
                name: The name to use.
            """
            return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_decorated_function)
        self.assertEqual(func.name, "my_decorated_function")
        self.assertEqual(func.description, "This is a decorated function.")
        self.assertEqual(
            func.properties,
            {"name": {"type": "string", "description": "The name to use."}},
        )
        self.assertFalse(func.cancel_on_interruption)
        self.assertEqual(func.timeout_secs, 15.5)


class TestFlowsDirectFunctionDeprecatedAlias(unittest.TestCase):
    """@flows_direct_function is a deprecated alias of @flows_tool_options."""

    def test_alias_warns_but_still_attaches_options(self):
        with self.assertWarns(DeprecationWarning):

            @flows_direct_function(cancel_on_interruption=True, timeout_secs=42)
            async def my_function(flow_manager: FlowManager):
                return {"status": "success"}, None

        # The deprecated alias still configures the same options.
        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertTrue(func.cancel_on_interruption)
        self.assertEqual(func.timeout_secs, 42)

    def test_tool_options_does_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            @flows_tool_options(cancel_on_interruption=False, timeout_secs=10)
            async def my_function(flow_manager: FlowManager):
                return {"status": "success"}, None

        func = FlowsDirectFunctionWrapper(function=my_function)
        self.assertEqual(func.timeout_secs, 10)


class TestConsolidatedFunctionResult(unittest.TestCase):
    """Regression tests for ``ConsolidatedFunctionResult`` resolvability."""

    def test_get_type_hints_resolves_without_nodeconfig_import(self):
        """Annotating a function with ``ConsolidatedFunctionResult`` should not require importing ``NodeConfig``.

        Previously the alias was defined as ``tuple[Any, "NodeConfig | None"]``
        with a string forward reference, which ``get_type_hints()`` resolves
        against the user's module globals. Users who did not separately import
        ``NodeConfig`` got ``NameError: name 'NodeConfig' is not defined`` —
        which then surfaced from ``FlowManager._set_node`` and stalled the
        flow. See https://github.com/pipecat-ai/pipecat-flows/issues/271.
        """
        from typing import get_type_hints

        async def my_tool(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
            """Do something and optionally transition."""
            return None, None

        hints = get_type_hints(my_tool)
        self.assertIn("return", hints)

    def test_direct_function_wrapper_accepts_consolidated_return_type(self):
        """``FlowsDirectFunctionWrapper`` introspects via ``get_type_hints`` internally."""

        async def my_tool(flow_manager: FlowManager) -> ConsolidatedFunctionResult:
            """Do something and optionally transition."""
            return None, None

        self.assertIsNone(FlowsDirectFunctionWrapper.validate_function(my_tool))
        FlowsDirectFunctionWrapper(function=my_tool)


if __name__ == "__main__":
    unittest.main()
