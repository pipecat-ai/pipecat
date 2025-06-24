import asyncio
import unittest
from typing import Optional, TypedDict, Union

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams

# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


class TestDirectFunction(unittest.TestCase):
    def test_name_is_set_from_function(self):
        async def my_function(params: FunctionCallParams):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)
        self.assertEqual(func.name, "my_function")

    def test_description_is_set_from_function(self):
        async def my_function_short_description(params: FunctionCallParams):
            """This is a test function."""
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_short_description)
        self.assertEqual(func.description, "This is a test function.")

        async def my_function_long_description(params: FunctionCallParams):
            """
            This is a test function.

            It does some really cool stuff.

            Trust me, you'll want to use it.
            """
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_long_description)
        self.assertEqual(
            func.description,
            "This is a test function.\n\nIt does some really cool stuff.\n\nTrust me, you'll want to use it.",
        )

    def test_properties_are_set_from_function(self):
        async def my_function_no_params(params: FunctionCallParams):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_no_params)
        self.assertEqual(func.properties, {})

        async def my_function_simple_params(
            params: FunctionCallParams, name: str, age: int, height: Union[float, None]
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_simple_params)
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "height": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
        )

        async def my_function_complex_params(
            params: FunctionCallParams,
            address_lines: list[str],
            nickname: str | int | float,
            extra: Optional[dict[str, str]],
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_complex_params)
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
            params: FunctionCallParams, info1: MyInfo1, info2: MyInfo2
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_complex_type_params)
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
        async def my_function_no_params(params: FunctionCallParams):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_no_params)
        self.assertEqual(func.required, [])

        async def my_function_simple_params(
            params: FunctionCallParams, name: str, age: int, height: Union[float, None] = None
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_simple_params)
        self.assertEqual(func.required, ["name", "age"])

        async def my_function_complex_params(
            params: FunctionCallParams,
            address_lines: Optional[list[str]],
            nickname: str | int = "Bud",
            extra: Optional[dict[str, str]] = None,
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_complex_params)
        self.assertEqual(func.required, ["address_lines"])

    def test_property_descriptions_are_set_from_function(self):
        async def my_function(
            params: FunctionCallParams, name: str, age: int, height: Union[float, None]
        ):
            """
            This is a test function.

            Args:
                name (str): The name of the person.
                age (int): The age of the person.
                height (float | None): The height of the person in meters. Defaults to None.
            """
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)

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
        def my_function_non_async(params: FunctionCallParams):
            return {"status": "success"}, None

        with self.assertRaises(Exception):
            DirectFunctionWrapper(function=my_function_non_async)

        async def my_function_missing_params():
            return {"status": "success"}, None

        with self.assertRaises(Exception):
            DirectFunctionWrapper(my_function_missing_params)

        async def my_function_misplaced_params(foo: str, params: FunctionCallParams):
            return {"status": "success"}, None

        with self.assertRaises(Exception):
            DirectFunctionWrapper(my_function_misplaced_params)

    def test_invoke_calls_function_with_args_and_params_object(self):
        called = {}

        class DummyParams:
            pass

        async def my_function(params: DummyParams, name: str, age: int):
            called["params"] = params
            called["name"] = name
            called["age"] = age
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)
        params = DummyParams()
        args = {"name": "Alice", "age": 30}

        result = asyncio.run(func.invoke(args=args, params=params))
        self.assertEqual(result, ({"status": "success"}, None))
        self.assertIs(called["params"], params)
        self.assertEqual(called["name"], "Alice")
        self.assertEqual(called["age"], 30)


if __name__ == "__main__":
    unittest.main()
