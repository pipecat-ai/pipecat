#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.bus.messages import BusFrameMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.frames.frames import LLMContextFrame, TextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.processors.frame_processor import FrameDirection


class TestTextFrameRoundTrip(unittest.TestCase):
    def setUp(self):
        self.serializer = JSONMessageSerializer()

    def test_round_trip(self):
        msg = BusFrameMessage(
            source="a",
            frame=TextFrame(text="hello world"),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello world")


class TestTranscriptionFrameRoundTrip(unittest.TestCase):
    def setUp(self):
        self.serializer = JSONMessageSerializer()

    def test_round_trip_basic(self):
        msg = BusFrameMessage(
            source="a",
            frame=TranscriptionFrame(
                text="hello",
                user_id="user-1",
                timestamp="2026-03-17T00:00:00Z",
            ),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored.frame, TranscriptionFrame)
        self.assertEqual(restored.frame.text, "hello")
        self.assertEqual(restored.frame.user_id, "user-1")
        self.assertEqual(restored.frame.timestamp, "2026-03-17T00:00:00Z")
        self.assertIsNone(restored.frame.language)
        self.assertFalse(restored.frame.finalized)

    def test_round_trip_with_language(self):
        from pipecat.transcriptions.language import Language

        msg = BusFrameMessage(
            source="a",
            frame=TranscriptionFrame(
                text="hola",
                user_id="user-1",
                timestamp="2026-03-17T00:00:00Z",
                language=Language.ES,
                finalized=True,
            ),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertEqual(restored.frame.language, Language.ES)
        self.assertTrue(restored.frame.finalized)


class TestLLMContextFrameRoundTrip(unittest.TestCase):
    def setUp(self):
        self.serializer = JSONMessageSerializer()

    def _round_trip_context(self, ctx):
        msg = BusFrameMessage(
            source="a",
            frame=LLMContextFrame(context=ctx),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)
        self.assertIsInstance(restored.frame, LLMContextFrame)
        return restored.frame.context

    def test_round_trip_messages_only(self):
        ctx = LLMContext(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
            ]
        )
        restored = self._round_trip_context(ctx)

        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.messages[0]["role"], "system")
        self.assertEqual(restored.messages[1]["content"], "hello")
        self.assertIsInstance(restored.tools, NotGiven)
        self.assertIsInstance(restored.tool_choice, NotGiven)

    def test_round_trip_with_specific_message(self):
        ctx = LLMContext(
            messages=[
                {"role": "user", "content": "hi"},
                LLMSpecificMessage(llm="anthropic", message={"custom": "data"}),
            ]
        )
        restored = self._round_trip_context(ctx)

        self.assertEqual(len(restored.messages), 2)
        self.assertIsInstance(restored.messages[0], dict)
        self.assertIsInstance(restored.messages[1], LLMSpecificMessage)
        self.assertEqual(restored.messages[1].llm, "anthropic")
        self.assertEqual(restored.messages[1].message, {"custom": "data"})

    def test_round_trip_with_tools(self):
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="get_weather",
                    description="Get the weather",
                    properties={"location": {"type": "string"}},
                    required=["location"],
                ),
            ]
        )
        ctx = LLMContext(
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
        )
        restored = self._round_trip_context(ctx)

        self.assertNotIsInstance(restored.tools, NotGiven)
        self.assertEqual(len(restored.tools.standard_tools), 1)
        self.assertEqual(restored.tools.standard_tools[0].name, "get_weather")
        self.assertEqual(restored.tools.standard_tools[0].required, ["location"])

    def test_round_trip_with_tool_choice(self):
        ctx = LLMContext(
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
        restored = self._round_trip_context(ctx)

        self.assertEqual(restored.tool_choice, "auto")


if __name__ == "__main__":
    unittest.main()
