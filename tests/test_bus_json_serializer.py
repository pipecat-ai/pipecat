#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for JSONMessageSerializer's ToolsSchema/LLMContext adapters."""

import unittest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.bus.serializers.json import JSONMessageSerializer
from pipecat.processors.aggregators.llm_context import LLMContext


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
