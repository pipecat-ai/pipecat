#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from dataclasses import dataclass, field

from pydantic import BaseModel

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.bus.adapters import ToolsSchemaAdapter
from pipecat.bus.messages import (
    BusActivateWorkerMessage,
    BusCancelMessage,
    BusDataMessage,
    BusEndMessage,
    BusFrameMessage,
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusMessage,
)
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.frames.frames import LLMContextFrame, TextFrame
from pipecat.pipeline.job_context import JobStatus
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


class _Address(BaseModel):
    city: str
    zip_code: str


class _UserInfo(BaseModel):
    name: str
    age: int
    address: _Address | None = None


@dataclass(kw_only=True)
class _MessageWithNonInit(BusDataMessage):
    tag: str = field(init=False, default="default")


class TestJSONMessageSerializer(unittest.TestCase):
    def setUp(self):
        self.serializer = JSONMessageSerializer()

    def test_round_trip_simple_message(self):
        """BusMessage serializes and deserializes correctly."""
        msg = BusDataMessage(source="task_a", target="task_b")
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusMessage)
        self.assertEqual(restored.source, "task_a")
        self.assertEqual(restored.target, "task_b")

    def test_round_trip_broadcast_message(self):
        """Broadcast message (no target) round-trips."""
        msg = BusDataMessage(source="task_a")
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusMessage)
        self.assertEqual(restored.source, "task_a")
        self.assertIsNone(restored.target)

    def test_round_trip_activate_message(self):
        """BusActivateWorkerMessage with args round-trips."""
        msg = BusActivateWorkerMessage(
            source="parent",
            target="child",
            args={"messages": [{"role": "user", "content": "hello"}]},
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusActivateWorkerMessage)
        self.assertEqual(restored.source, "parent")
        self.assertEqual(restored.target, "child")
        self.assertEqual(restored.args["messages"][0]["content"], "hello")

    def test_round_trip_end_message(self):
        """BusEndMessage round-trips."""
        msg = BusEndMessage(source="task_a", reason="done")
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusEndMessage)
        self.assertEqual(restored.reason, "done")

    def test_round_trip_cancel_message(self):
        """BusCancelMessage round-trips."""
        msg = BusCancelMessage(source="task_a", reason="abort")
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusCancelMessage)
        self.assertEqual(restored.reason, "abort")

    def test_round_trip_job_request(self):
        """BusJobRequestMessage with payload round-trips."""
        msg = BusJobRequestMessage(
            source="parent",
            target="worker",
            job_id="t-123",
            payload={"key": "value"},
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusJobRequestMessage)
        self.assertEqual(restored.job_id, "t-123")
        self.assertEqual(restored.payload, {"key": "value"})

    def test_round_trip_job_response(self):
        """BusJobResponseMessage round-trips."""
        msg = BusJobResponseMessage(
            source="worker",
            target="parent",
            job_id="t-123",
            status=JobStatus.COMPLETED,
            response={"result": 42},
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusJobResponseMessage)
        self.assertEqual(restored.job_id, "t-123")
        self.assertEqual(restored.response, {"result": 42})
        self.assertEqual(restored.status, JobStatus.COMPLETED)

    def test_round_trip_frame_message(self):
        """BusFrameMessage with TextFrame round-trips via adapter."""
        msg = BusFrameMessage(
            source="task_a",
            frame=TextFrame(text="hello world"),
            direction=FrameDirection.DOWNSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusFrameMessage)
        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello world")
        self.assertEqual(restored.direction, FrameDirection.DOWNSTREAM)
        self.assertEqual(restored.source, "task_a")

    def test_frame_message_upstream_direction(self):
        """UPSTREAM direction preserved in round-trip."""
        msg = BusFrameMessage(
            source="task_a",
            frame=TextFrame(text="up"),
            direction=FrameDirection.UPSTREAM,
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertEqual(restored.direction, FrameDirection.UPSTREAM)

    def test_unregistered_frame_warns_and_skips(self):
        """Serializing a frame with no adapter warns and skips the field."""
        serializer = JSONMessageSerializer()  # no adapters registered

        msg = BusFrameMessage(
            source="task_a",
            frame=TextFrame(text="hello"),
            direction=FrameDirection.DOWNSTREAM,
        )
        # Should not raise — unserializable field is skipped with a warning
        data = serializer.serialize(msg)
        self.assertIsInstance(data, bytes)

    def test_unknown_message_type_returns_none(self):
        """Deserializing an unknown message type returns None."""
        bad_data = b'{"__type__":"bogus.BogusMessage","__data__":{"source":"a"}}'
        result = self.serializer.deserialize(bad_data)
        self.assertIsNone(result)

    def test_serialized_is_bytes(self):
        """serialize() returns bytes."""
        msg = BusDataMessage(source="a")
        data = self.serializer.serialize(msg)
        self.assertIsInstance(data, bytes)

    def test_adapter_mro_lookup(self):
        """Adapter registered for a parent class handles subclasses."""

        class CustomTextFrame(TextFrame):
            pass

        msg = BusFrameMessage(
            source="a",
            frame=CustomTextFrame(text="sub"),
            direction=FrameDirection.DOWNSTREAM,
        )
        # TextTypeAdapter is registered for TextFrame, should handle subclass
        data = self.serializer.serialize(msg)
        self.assertIsInstance(data, bytes)

    def test_round_trip_pydantic_base_model(self):
        """Pydantic BaseModel round-trips preserving the type."""
        msg = BusJobResponseMessage(
            source="worker",
            target="parent",
            job_id="t-456",
            status=JobStatus.COMPLETED,
            response={"user": _UserInfo(name="Alice", age=30)},
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusJobResponseMessage)
        user = restored.response["user"]
        self.assertIsInstance(user, _UserInfo)
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.age, 30)

    def test_round_trip_nested_pydantic_base_model(self):
        """Nested Pydantic BaseModels round-trip preserving types."""
        user = _UserInfo(name="Alice", age=30, address=_Address(city="NYC", zip_code="10001"))
        msg = BusJobResponseMessage(
            source="worker",
            target="parent",
            job_id="t-789",
            status=JobStatus.COMPLETED,
            response={"user": user},
        )
        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertIsInstance(restored, BusJobResponseMessage)
        restored_user = restored.response["user"]
        self.assertIsInstance(restored_user, _UserInfo)
        self.assertIsInstance(restored_user.address, _Address)
        self.assertEqual(restored_user.address.city, "NYC")
        self.assertEqual(restored_user.address.zip_code, "10001")

    def test_non_init_fields_preserved(self):
        """Non-init dataclass fields survive round-trip via setattr."""
        msg = _MessageWithNonInit(source="worker_a", target="worker_b")
        msg.tag = "custom_tag"

        data = self.serializer.serialize(msg)
        restored = self.serializer.deserialize(data)

        self.assertEqual(restored.tag, "custom_tag")


def _tools_with_custom():
    """A ToolsSchema with both a standard tool and provider-specific custom tools."""
    standard = FunctionSchema(
        name="get_weather",
        description="Get the weather",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    custom = {AdapterType.GEMINI: [{"google_search": {}}]}
    return ToolsSchema(standard_tools=[standard], custom_tools=custom)


class TestCustomToolsRoundTrip(unittest.TestCase):
    """Provider-specific custom_tools must survive bus serialization (see #4833)."""

    def test_tools_schema_adapter_preserves_custom_tools(self):
        """ToolsSchemaAdapter round-trips custom_tools, not just standard_tools."""
        adapter = ToolsSchemaAdapter()
        ident = lambda x: x  # noqa: E731  (callback unused for tools)

        restored = adapter.deserialize(adapter.serialize(_tools_with_custom(), ident), ident)

        self.assertEqual([t.name for t in restored.standard_tools], ["get_weather"])
        self.assertEqual(restored.custom_tools, {AdapterType.GEMINI: [{"google_search": {}}]})

    def test_llm_context_frame_preserves_custom_tools_over_bus(self):
        """An LLMContextFrame crossing the bus keeps the context's custom_tools."""
        serializer = JSONMessageSerializer()
        ctx = LLMContext(messages=[{"role": "user", "content": "hi"}], tools=_tools_with_custom())
        msg = BusFrameMessage(
            source="task_a",
            frame=LLMContextFrame(context=ctx),
            direction=FrameDirection.DOWNSTREAM,
        )

        restored = serializer.deserialize(serializer.serialize(msg))

        self.assertIsInstance(restored.frame, LLMContextFrame)
        tools = restored.frame.context.tools
        self.assertEqual([t.name for t in tools.standard_tools], ["get_weather"])
        self.assertEqual(tools.custom_tools, {AdapterType.GEMINI: [{"google_search": {}}]})


if __name__ == "__main__":
    unittest.main()
