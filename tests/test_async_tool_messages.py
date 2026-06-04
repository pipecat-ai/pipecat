#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from pipecat.processors.aggregators import async_tool_messages

# The parser tests intentionally exercise the parser via the canonical
# builders, so a drift between the two sides will surface as a parse failure
# in CI rather than as a silent contract break in production.


def _started_message(tool_call_id: str = "call_123") -> dict:
    return async_tool_messages.build_started_message(tool_call_id)


def _intermediate_message(
    tool_call_id: str = "call_123",
    result: str = '"intermediate-1"',
) -> dict:
    return async_tool_messages.build_intermediate_result_message(tool_call_id, result)


def _final_message(
    tool_call_id: str = "call_123",
    result: str = '"final-result"',
) -> dict:
    return async_tool_messages.build_final_result_message(tool_call_id, result)


class TestParseMessage(unittest.TestCase):
    def test_parses_started(self):
        info = async_tool_messages.parse_message(_started_message("abc"))
        assert info is not None
        assert info.kind == "started"
        assert info.tool_call_id == "abc"
        assert info.status == "running"
        assert info.result is None
        assert "asynchronous task" in info.description

    def test_parses_intermediate(self):
        info = async_tool_messages.parse_message(_intermediate_message("abc", '"hello"'))
        assert info is not None
        assert info.kind == "intermediate"
        assert info.tool_call_id == "abc"
        assert info.status == "running"
        assert info.result == '"hello"'

    def test_parses_final(self):
        info = async_tool_messages.parse_message(_final_message("abc", '"done"'))
        assert info is not None
        assert info.kind == "final"
        assert info.tool_call_id == "abc"
        assert info.status == "finished"
        assert info.result == '"done"'

    def test_parses_completed_sentinel_result(self):
        # When a function returns no value, the aggregator sets the result to
        # the literal "COMPLETED" — same convention used for synchronous tool
        # calls. The parser doesn't treat it specially; it's just a string.
        info = async_tool_messages.parse_message(_final_message("abc", "COMPLETED"))
        assert info is not None
        assert info.kind == "final"
        assert info.result == "COMPLETED"

    def test_returns_none_for_regular_user_message(self):
        assert async_tool_messages.parse_message({"role": "user", "content": "hello"}) is None

    def test_returns_none_for_regular_assistant_message(self):
        assert async_tool_messages.parse_message({"role": "assistant", "content": "hi"}) is None

    def test_returns_none_for_regular_tool_message(self):
        # IN_PROGRESS / regular tool result string content.
        assert (
            async_tool_messages.parse_message(
                {"role": "tool", "tool_call_id": "x", "content": "IN_PROGRESS"}
            )
            is None
        )
        assert (
            async_tool_messages.parse_message(
                {"role": "tool", "tool_call_id": "x", "content": "weather: sunny"}
            )
            is None
        )

    def test_returns_none_for_developer_message_without_payload(self):
        # role=developer is also used for non-async-tool things (potentially).
        assert (
            async_tool_messages.parse_message(
                {"role": "developer", "content": "some other developer note"}
            )
            is None
        )

    def test_returns_none_for_invalid_json_content(self):
        assert async_tool_messages.parse_message({"role": "tool", "content": "{not json"}) is None

    def test_returns_none_for_non_dict_json(self):
        assert async_tool_messages.parse_message({"role": "tool", "content": "[1, 2, 3]"}) is None

    def test_returns_none_for_wrong_payload_type(self):
        assert (
            async_tool_messages.parse_message(
                {
                    "role": "tool",
                    "content": json.dumps({"type": "something_else", "tool_call_id": "x"}),
                }
            )
            is None
        )

    def test_returns_none_when_tool_call_id_missing(self):
        assert (
            async_tool_messages.parse_message(
                {
                    "role": "tool",
                    "content": json.dumps({"type": "async_tool", "status": "running"}),
                }
            )
            is None
        )

    def test_returns_none_when_status_invalid(self):
        assert (
            async_tool_messages.parse_message(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {"type": "async_tool", "tool_call_id": "x", "status": "weird"}
                    ),
                }
            )
            is None
        )

    def test_returns_none_for_non_string_content(self):
        # A multimodal message with content as a list would not be an async-tool message.
        assert (
            async_tool_messages.parse_message(
                {"role": "tool", "content": [{"type": "text", "text": "hi"}]}
            )
            is None
        )

    def test_returns_none_for_missing_role(self):
        assert async_tool_messages.parse_message({"content": "{}"}) is None


class TestBuilders(unittest.TestCase):
    """Verify the builders produce the canonical payload shape and round-trip cleanly."""

    def test_started_message_shape(self):
        msg = async_tool_messages.build_started_message("call_42")
        # Top-level: role=tool plus the tool_call_id (so the message can sit
        # alongside other regular tool messages in the context).
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_42"
        payload = json.loads(msg["content"])
        assert payload["type"] == "async_tool"
        assert payload["status"] == "running"
        assert payload["tool_call_id"] == "call_42"
        assert "result" not in payload
        assert isinstance(payload["description"], str) and payload["description"]

    def test_intermediate_message_shape(self):
        msg = async_tool_messages.build_intermediate_result_message("call_99", '"step-1"')
        # Intermediate/final use role=developer and don't carry tool_call_id at
        # the top level (that's only inside the payload).
        assert msg["role"] == "developer"
        assert "tool_call_id" not in msg
        payload = json.loads(msg["content"])
        assert payload["type"] == "async_tool"
        assert payload["status"] == "running"
        assert payload["tool_call_id"] == "call_99"
        assert payload["result"] == '"step-1"'
        assert isinstance(payload["description"], str) and payload["description"]

    def test_final_message_shape(self):
        msg = async_tool_messages.build_final_result_message("call_7", '"all-done"')
        assert msg["role"] == "developer"
        assert "tool_call_id" not in msg
        payload = json.loads(msg["content"])
        assert payload["type"] == "async_tool"
        assert payload["status"] == "finished"
        assert payload["tool_call_id"] == "call_7"
        assert payload["result"] == '"all-done"'
        assert isinstance(payload["description"], str) and payload["description"]

    def test_final_message_with_completed_sentinel(self):
        # The aggregator passes the literal "COMPLETED" string when the
        # function returned no value (same convention as for synchronous
        # tool calls). The builder doesn't treat it specially; it just
        # round-trips as the result.
        msg = async_tool_messages.build_final_result_message("call_1", "COMPLETED")
        payload = json.loads(msg["content"])
        assert payload["result"] == "COMPLETED"
        info = async_tool_messages.parse_message(msg)
        assert info is not None
        assert info.kind == "final"
        assert info.result == "COMPLETED"

    def test_started_round_trip(self):
        msg = async_tool_messages.build_started_message("call_x")
        info = async_tool_messages.parse_message(msg)
        assert info is not None
        assert info.kind == "started"
        assert info.tool_call_id == "call_x"
        assert info.status == "running"
        assert info.result is None

    def test_intermediate_round_trip(self):
        msg = async_tool_messages.build_intermediate_result_message("call_x", '{"step": 1}')
        info = async_tool_messages.parse_message(msg)
        assert info is not None
        assert info.kind == "intermediate"
        assert info.tool_call_id == "call_x"
        assert info.status == "running"
        assert info.result == '{"step": 1}'

    def test_final_round_trip(self):
        msg = async_tool_messages.build_final_result_message("call_x", '{"answer": 42}')
        info = async_tool_messages.parse_message(msg)
        assert info is not None
        assert info.kind == "final"
        assert info.tool_call_id == "call_x"
        assert info.status == "finished"
        assert info.result == '{"answer": 42}'


if __name__ == "__main__":
    unittest.main()
