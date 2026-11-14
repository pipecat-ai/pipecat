#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError

from pipecat.adapters.llm.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams


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
            params: FunctionCallParams, name: str, age: int, height: float | None
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
            extra: dict[str, str] | None,
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

    def test_any_params_accept_any_value(self):
        async def my_function_any_params(
            params: FunctionCallParams,
            value: Any,
            tags: list[Any],
            metadata: dict[str, Any],
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_any_params)
        self.assertEqual(
            func.properties,
            {
                "value": {},
                "tags": {"type": "array", "items": {}},
                "metadata": {"type": "object", "additionalProperties": {}},
            },
        )

    def test_required_is_set_from_function(self):
        async def my_function_no_params(params: FunctionCallParams):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_no_params)
        self.assertEqual(func.required, [])

        async def my_function_simple_params(
            params: FunctionCallParams, name: str, age: int, height: float | None = None
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_simple_params)
        self.assertEqual(func.required, ["name", "age"])

        async def my_function_complex_params(
            params: FunctionCallParams,
            address_lines: list[str] | None,
            nickname: str | int = "Bud",
            extra: dict[str, str] | None = None,
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function_complex_params)
        self.assertEqual(func.required, ["address_lines"])

    def test_property_descriptions_are_set_from_function(self):
        async def my_function(
            params: FunctionCallParams, name: str, age: int, height: float | None
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

        with self.assertRaisesRegex(Exception, "must be async"):
            DirectFunctionWrapper(function=my_function_non_async)

        async def my_function_missing_params():
            return {"status": "success"}, None

        with self.assertRaisesRegex(Exception, "must have at least one parameter"):
            DirectFunctionWrapper(my_function_missing_params)

        async def my_function_misplaced_params(foo: str, params: FunctionCallParams):
            return {"status": "success"}, None

        with self.assertRaisesRegex(Exception, "first parameter must be named"):
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

    def test_enum_and_literal_params_list_their_allowed_values(self):
        class Unit(Enum):
            CELSIUS = "celsius"
            FAHRENHEIT = "fahrenheit"

        class Priority(IntEnum):
            LOW = 1
            HIGH = 2

        async def my_function(
            params: FunctionCallParams,
            unit: Unit,
            priority: Priority,
            mode: Literal["fast", "slow"],
            mixed: Literal["auto", 0],
            maybe_unit: Unit | None = None,
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)
        self.assertEqual(
            func.properties,
            {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "priority": {"type": "integer", "enum": [1, 2]},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "mixed": {"enum": ["auto", 0]},
                "maybe_unit": {
                    "anyOf": [
                        {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        {"type": "null"},
                    ]
                },
            },
        )

    def test_pydantic_and_dataclass_fields_with_defaults_are_optional(self):
        class Address(BaseModel):
            city: str
            zip_code: str | None = None
            country: str = Field(default="IN", alias="countryCode", description="ISO code")

        @dataclass
        class Window:
            start: str
            days: int = 1
            tags: list[str] = field(default_factory=list)

        async def my_function(params: FunctionCallParams, address: Address, window: Window):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)
        self.assertEqual(
            func.properties,
            {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "countryCode": {"type": "string", "description": "ISO code"},
                    },
                    "required": ["city"],
                },
                "window": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "days": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["start"],
                },
            },
        )

    def test_fixed_length_tuple_allows_every_position(self):
        async def my_function(
            params: FunctionCallParams, pair: tuple[int, str], same: tuple[int, int]
        ):
            return {"status": "success"}, None

        func = DirectFunctionWrapper(function=my_function)
        self.assertEqual(
            func.properties["pair"],
            {
                "type": "array",
                "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "minItems": 2,
                "maxItems": 2,
            },
        )
        self.assertEqual(
            func.properties["same"],
            {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        )

    def test_invoke_converts_args_to_declared_types(self):
        class Unit(Enum):
            CELSIUS = "celsius"

        class Address(BaseModel):
            city: str

        @dataclass
        class Window:
            start: str
            days: int = 1

        called = {}

        async def my_function(
            params: Any,
            unit: Unit,
            address: Address,
            windows: list[Window],
            note: str,
            raw: dict,
            maybe: Address | None = None,
        ):
            called.update(
                unit=unit, address=address, windows=windows, note=note, raw=raw, maybe=maybe
            )

        func = DirectFunctionWrapper(function=my_function)
        raw = {"keep": "as is"}
        asyncio.run(
            func.invoke(
                args={
                    "unit": "celsius",
                    "address": {"city": "Pune"},
                    "windows": [{"start": "mon"}],
                    "note": "hi",
                    "raw": raw,
                    "maybe": None,
                },
                params=None,
            )
        )
        self.assertIs(called["unit"], Unit.CELSIUS)
        self.assertEqual(called["address"], Address(city="Pune"))
        self.assertEqual(called["windows"], [Window(start="mon", days=1)])
        self.assertEqual(called["note"], "hi")
        self.assertIs(called["raw"], raw)
        self.assertIsNone(called["maybe"])

    def test_invoke_rejects_args_that_dont_match_declared_type(self):
        class Unit(Enum):
            CELSIUS = "celsius"

        async def my_function(params: Any, unit: Unit):
            return unit

        func = DirectFunctionWrapper(function=my_function)
        with self.assertRaises(ValidationError):
            asyncio.run(func.invoke(args={"unit": "kelvin"}, params=None))


if __name__ == "__main__":
    unittest.main()
