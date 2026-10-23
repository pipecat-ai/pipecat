#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import Annotated, NotRequired, Required, TypedDict

from pipecat.adapters.llm.schemas.direct_function import DirectFunctionWrapper


class TestDirectFunctionTypedDict(unittest.TestCase):
    def _assert_typeddict_required(self, payload_type, required):
        async def lookup(params, payload: payload_type):
            return payload

        schema = DirectFunctionWrapper(lookup).to_function_schema().to_default_dict()
        payload = schema["parameters"]["properties"]["payload"]
        self.assertEqual(payload.get("required", []), required)
        self.assertEqual(payload["properties"]["name"], {"type": "string"})
        self.assertEqual(payload["properties"]["limit"], {"type": "integer"})

    def test_typeddict_inherited_required_fields(self):
        class RequiredBase(TypedDict):
            name: str

        class OptionalChild(RequiredBase, total=False):
            limit: int

        class OptionalBase(TypedDict, total=False):
            limit: int

        class RequiredChild(OptionalBase):
            name: str

        for payload_type in (OptionalChild, RequiredChild):
            with self.subTest(payload_type=payload_type):
                self._assert_typeddict_required(payload_type, ["name"])

    def test_typeddict_required_qualifiers(self):
        class OptionalLimit(TypedDict):
            name: str
            limit: NotRequired[int]

        class RequiredName(TypedDict, total=False):
            name: Required[str]
            limit: int

        for payload_type in (OptionalLimit, RequiredName):
            with self.subTest(payload_type=payload_type):
                self._assert_typeddict_required(payload_type, ["name"])

    def test_typeddict_annotated_required_qualifiers(self):
        class Payload(TypedDict):
            name: Annotated[Required[str], "display name"]
            limit: Annotated[NotRequired[int], "result limit"]

        self._assert_typeddict_required(Payload, ["name"])

    def test_typeddict_uniform_total(self):
        class AllRequired(TypedDict):
            name: str
            limit: int

        class AllOptional(TypedDict, total=False):
            name: str
            limit: int

        self._assert_typeddict_required(AllRequired, ["name", "limit"])
        self._assert_typeddict_required(AllOptional, [])

    def test_typeddict_deferred_required_qualifiers(self):
        namespace = {"__name__": __name__}
        exec(
            "from __future__ import annotations\n"
            "from typing import NotRequired, Required, TypedDict\n"
            "class OptionalLimit(TypedDict):\n"
            "    name: str\n"
            "    limit: NotRequired[int]\n"
            "class RequiredName(TypedDict, total=False):\n"
            "    name: Required[str]\n"
            "    limit: int\n",
            namespace,
        )
        for name in ("OptionalLimit", "RequiredName"):
            with self.subTest(payload_type=name):
                self._assert_typeddict_required(namespace[name], ["name"])
