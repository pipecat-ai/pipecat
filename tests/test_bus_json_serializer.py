#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for JSON bus messages and their type adapters."""

import json
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.bus.messages import BusDataMessage, BusFrameMessage
from pipecat.bus.serializers.json import JSONMessageSerializer
from pipecat.frames.frames import LLMUpdateSettingsFrame, STTUpdateSettingsFrame, TextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import LLMSettings, STTSettings
from pipecat.utils.types import NOT_GIVEN


@dataclass
class NullableMessage(BusDataMessage):
    """Message with nullable fields whose defaults are not null."""

    value: str | None = "default"
    values: list[str] | None = field(default_factory=list)
    callback: Callable[[], None] | None = None


class NullableModel(BaseModel):
    """Metadata model with a nullable field and a non-null default."""

    value: str | None = "default"


class UnsetModel(BaseModel):
    """Metadata model with a nullable field that defaults to null."""

    value: str | None = None


class TestExplicitNullRoundTrip(unittest.TestCase):
    """Explicit nulls remain distinct from omitted settings across a network bus."""

    def setUp(self):
        self.serializer = JSONMessageSerializer()

    def round_trip(self, message):
        return self.serializer.deserialize(self.serializer.serialize(message))

    def test_nullable_dataclass_fields(self):
        message = NullableMessage(source="controller", value=None, values=None)

        restored = self.round_trip(message)

        self.assertIsNone(restored.value)
        self.assertIsNone(restored.values)
        self.assertIsNone(restored.target)
        self.assertEqual(restored.id, message.id)

    def test_nullable_pydantic_metadata(self):
        frame = TextFrame(text="hello")
        frame.metadata["config"] = NullableModel(value=None)
        message = BusFrameMessage(
            source="controller", frame=frame, direction=FrameDirection.DOWNSTREAM
        )

        restored = self.round_trip(message)

        self.assertIsInstance(restored.frame.metadata["config"], NullableModel)
        self.assertIsNone(restored.frame.metadata["config"].value)

    def test_unset_pydantic_fields_stay_unset(self):
        frame = TextFrame(text="hello")
        frame.metadata["unset"] = UnsetModel()
        frame.metadata["explicit"] = UnsetModel(value=None)
        message = BusFrameMessage(
            source="controller", frame=frame, direction=FrameDirection.DOWNSTREAM
        )

        restored = self.round_trip(message)

        self.assertEqual(restored.frame.metadata["unset"].model_fields_set, set())
        self.assertEqual(restored.frame.metadata["explicit"].model_fields_set, {"value"})

    def test_llm_delta_clears_settings(self):
        delta = LLMSettings(temperature=None, max_tokens=None)
        message = BusFrameMessage(
            source="controller",
            frame=LLMUpdateSettingsFrame(delta=delta),
            direction=FrameDirection.DOWNSTREAM,
        )

        restored = self.round_trip(message)
        received_delta = restored.frame.delta
        self.assertEqual(received_delta.given_fields(), delta.given_fields())
        self.assertIs(received_delta.model, NOT_GIVEN)

        current = LLMSettings(model="example-model", temperature=0.7, max_tokens=256)
        current.apply_update(received_delta)
        self.assertIsNone(current.temperature)
        self.assertIsNone(current.max_tokens)
        self.assertEqual(current.model, "example-model")

    def test_stt_delta_clears_language(self):
        delta = STTSettings(language=None)
        message = BusFrameMessage(
            source="controller",
            frame=STTUpdateSettingsFrame(delta=delta),
            direction=FrameDirection.DOWNSTREAM,
        )

        restored = self.round_trip(message)
        self.assertEqual(restored.frame.delta.given_fields(), {"language": None})
        current = STTSettings(model="example-stt", language="en")
        current.apply_update(restored.frame.delta)
        self.assertIsNone(current.language)
        self.assertEqual(current.model, "example-stt")

    def test_falsy_and_unspecified_settings(self):
        delta = LLMSettings(temperature=0.0, max_tokens=0, use_provider_thinking_default=False)
        message = BusFrameMessage(
            source="controller",
            frame=LLMUpdateSettingsFrame(delta=delta),
            direction=FrameDirection.DOWNSTREAM,
        )

        restored = self.round_trip(message)

        self.assertEqual(restored.frame.delta.given_fields(), delta.given_fields())
        self.assertIs(restored.frame.delta.model, NOT_GIVEN)

    def test_unsupported_callable_remains_omitted(self):
        message = NullableMessage(source="controller", callback=lambda: None)

        encoded = self.serializer.serialize(message)
        restored = self.serializer.deserialize(encoded)

        self.assertNotIn("callback", json.loads(encoded)["__data__"])
        self.assertIsNone(restored.callback)


class TestToolsSchemaCustomToolsRoundTrip(unittest.TestCase):
    """Bus adapters must preserve `ToolsSchema.custom_tools` (e.g. Gemini's
    `google_search`) when a frame crosses a network bus (Redis/pgmq), not
    just the in-process case.
    """

    def setUp(self):
        self.serializer = JSONMessageSerializer()
        self.standard_tool = FunctionSchema(
            name="get_weather",
            description="Get the weather",
            properties={"location": {"type": "string"}},
            required=["location"],
        )

    def test_tools_schema_preserves_custom_tools(self):
        tools = ToolsSchema(
            standard_tools=[self.standard_tool],
            custom_tools={AdapterType.GEMINI: [{"google_search": {}}]},
        )

        data = self.serializer._serialize_value(tools)
        restored = self.serializer._deserialize_value(data)

        self.assertIsInstance(restored, ToolsSchema)
        self.assertEqual(len(restored.standard_tools), 1)
        self.assertEqual(restored.standard_tools[0].name, "get_weather")
        self.assertEqual(restored.custom_tools, {AdapterType.GEMINI: [{"google_search": {}}]})

    def test_tools_schema_without_custom_tools_round_trips_to_none(self):
        tools = ToolsSchema(standard_tools=[self.standard_tool])

        data = self.serializer._serialize_value(tools)
        restored = self.serializer._deserialize_value(data)

        self.assertIsNone(restored.custom_tools)

    def test_llm_context_tools_preserve_custom_tools(self):
        tools = ToolsSchema(
            standard_tools=[self.standard_tool],
            custom_tools={AdapterType.OPENAI: [{"type": "web_search"}]},
        )
        context = LLMContext(messages=[{"role": "user", "content": "hi"}], tools=tools)

        data = self.serializer._serialize_value(context)
        restored = self.serializer._deserialize_value(data)

        self.assertIsInstance(restored, LLMContext)
        self.assertEqual(
            restored.tools.custom_tools, {AdapterType.OPENAI: [{"type": "web_search"}]}
        )

    def test_bytes_round_trip_through_serialize_deserialize(self):
        """End-to-end sanity check that the full JSON encode/decode cycle preserves custom_tools.

        Mirrors an `LLMContextFrame` crossing a network bus (Redis/pgmq).
        """
        from pipecat.bus.messages import BusFrameMessage
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.frame_processor import FrameDirection

        tools = ToolsSchema(
            standard_tools=[self.standard_tool],
            custom_tools={AdapterType.GEMINI: [{"google_search": {}}]},
        )
        context = LLMContext(messages=[{"role": "user", "content": "hi"}], tools=tools)
        message = BusFrameMessage(
            source="worker-a",
            frame=LLMContextFrame(context=context),
            direction=FrameDirection.DOWNSTREAM,
        )

        raw = self.serializer.serialize(message)
        restored_message = self.serializer.deserialize(raw)

        self.assertEqual(
            restored_message.frame.context.tools.custom_tools,
            {AdapterType.GEMINI: [{"google_search": {}}]},
        )


if __name__ == "__main__":
    unittest.main()
