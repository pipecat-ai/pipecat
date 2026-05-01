#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from pipecat.processors.aggregators.async_tool_messages import (
    AsyncToolMessage,
    format_async_tool_text_for_provider,
    is_async_tool_message,
    parse_async_tool_message,
)


def _placeholder_message(tool_call_id: str = "call_123") -> dict:
    """Build a placeholder async-tool message matching the aggregator's output."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(
            {
                "type": "async_tool",
                "status": "running",
                "tool_call_id": tool_call_id,
                "description": (
                    "An asynchronous task associated with this tool_call_id has started "
                    "running. Expect results to arrive later as developer messages..."
                ),
            }
        ),
    }


def _intermediate_message(
    tool_call_id: str = "call_123",
    result: str = '"intermediate-1"',
) -> dict:
    """Build an intermediate async-tool message matching the aggregator's output."""
    return {
        "role": "developer",
        "content": json.dumps(
            {
                "type": "async_tool",
                "tool_call_id": tool_call_id,
                "status": "running",
                "description": "This is an intermediate result for the asynchronous task...",
                "result": result,
            }
        ),
    }


def _final_message(
    tool_call_id: str = "call_123",
    result: str = '"final-result"',
) -> dict:
    """Build a final async-tool message matching the aggregator's output."""
    return {
        "role": "developer",
        "content": json.dumps(
            {
                "type": "async_tool",
                "tool_call_id": tool_call_id,
                "status": "finished",
                "description": "This is the final result for the asynchronous task...",
                "result": result,
            }
        ),
    }


class TestParseAsyncToolMessage(unittest.TestCase):
    def test_parses_placeholder(self):
        info = parse_async_tool_message(_placeholder_message("abc"))
        assert info is not None
        assert info.kind == "placeholder"
        assert info.tool_call_id == "abc"
        assert info.status == "running"
        assert info.result is None
        assert "asynchronous task" in info.description

    def test_parses_intermediate(self):
        info = parse_async_tool_message(_intermediate_message("abc", '"hello"'))
        assert info is not None
        assert info.kind == "intermediate"
        assert info.tool_call_id == "abc"
        assert info.status == "running"
        assert info.result == '"hello"'

    def test_parses_final(self):
        info = parse_async_tool_message(_final_message("abc", '"done"'))
        assert info is not None
        assert info.kind == "final"
        assert info.tool_call_id == "abc"
        assert info.status == "finished"
        assert info.result == '"done"'

    def test_raw_content_preserves_original_envelope(self):
        # raw_content should round-trip the source message's `content` field so
        # services can forward the full envelope to providers.
        msg = _final_message("abc", '"done"')
        info = parse_async_tool_message(msg)
        assert info is not None
        assert info.raw_content == msg["content"]
        # Sanity: it should parse back to the original envelope dict.
        envelope = json.loads(info.raw_content)
        assert envelope["type"] == "async_tool"
        assert envelope["tool_call_id"] == "abc"
        assert envelope["status"] == "finished"
        assert envelope["result"] == '"done"'

    def test_parses_completed_sentinel_result(self):
        # When the async function returns no value, the aggregator sets result to "COMPLETED".
        info = parse_async_tool_message(_final_message("abc", "COMPLETED"))
        assert info is not None
        assert info.kind == "final"
        assert info.result == "COMPLETED"

    def test_returns_none_for_regular_user_message(self):
        assert parse_async_tool_message({"role": "user", "content": "hello"}) is None

    def test_returns_none_for_regular_assistant_message(self):
        assert parse_async_tool_message({"role": "assistant", "content": "hi"}) is None

    def test_returns_none_for_regular_tool_message(self):
        # IN_PROGRESS / regular tool result string content.
        assert (
            parse_async_tool_message(
                {"role": "tool", "tool_call_id": "x", "content": "IN_PROGRESS"}
            )
            is None
        )
        assert (
            parse_async_tool_message(
                {"role": "tool", "tool_call_id": "x", "content": "weather: sunny"}
            )
            is None
        )

    def test_returns_none_for_developer_message_without_envelope(self):
        # role=developer is also used for non-async-tool things (potentially).
        assert (
            parse_async_tool_message({"role": "developer", "content": "some other developer note"})
            is None
        )

    def test_returns_none_for_invalid_json_content(self):
        assert parse_async_tool_message({"role": "tool", "content": "{not json"}) is None

    def test_returns_none_for_non_dict_json(self):
        assert parse_async_tool_message({"role": "tool", "content": "[1, 2, 3]"}) is None

    def test_returns_none_for_wrong_envelope_type(self):
        assert (
            parse_async_tool_message(
                {
                    "role": "tool",
                    "content": json.dumps({"type": "something_else", "tool_call_id": "x"}),
                }
            )
            is None
        )

    def test_returns_none_when_tool_call_id_missing(self):
        assert (
            parse_async_tool_message(
                {
                    "role": "tool",
                    "content": json.dumps({"type": "async_tool", "status": "running"}),
                }
            )
            is None
        )

    def test_returns_none_when_status_invalid(self):
        assert (
            parse_async_tool_message(
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
            parse_async_tool_message({"role": "tool", "content": [{"type": "text", "text": "hi"}]})
            is None
        )

    def test_returns_none_for_missing_role(self):
        assert parse_async_tool_message({"content": "{}"}) is None


class TestIsAsyncToolMessage(unittest.TestCase):
    def test_true_for_placeholder(self):
        assert is_async_tool_message(_placeholder_message()) is True

    def test_true_for_intermediate(self):
        assert is_async_tool_message(_intermediate_message()) is True

    def test_true_for_final(self):
        assert is_async_tool_message(_final_message()) is True

    def test_false_for_regular_message(self):
        assert is_async_tool_message({"role": "user", "content": "hi"}) is False
        assert (
            is_async_tool_message({"role": "tool", "tool_call_id": "x", "content": "IN_PROGRESS"})
            is False
        )


class TestFormatAsyncToolTextForProvider(unittest.TestCase):
    def test_default_template_includes_id_status_and_result(self):
        info = AsyncToolMessage(
            kind="final",
            tool_call_id="call_42",
            status="finished",
            description="done",
            result='"the answer"',
            raw_content="{}",
        )
        text = format_async_tool_text_for_provider(info)
        assert "call_42" in text
        assert "finished" in text
        assert '"the answer"' in text

    def test_custom_template(self):
        info = AsyncToolMessage(
            kind="intermediate",
            tool_call_id="call_99",
            status="running",
            description="",
            result='"step-1"',
            raw_content="{}",
        )
        text = format_async_tool_text_for_provider(
            info,
            template="tool {tool_call_id} -> {result} ({status})",
        )
        assert text == 'tool call_99 -> "step-1" (running)'

    def test_template_can_use_description_field(self):
        info = AsyncToolMessage(
            kind="final",
            tool_call_id="call_1",
            status="finished",
            description="my description",
            result='"x"',
            raw_content="{}",
        )
        text = format_async_tool_text_for_provider(info, template="{description}: {result}")
        assert text == 'my description: "x"'

    def test_raises_on_placeholder(self):
        info = AsyncToolMessage(
            kind="placeholder",
            tool_call_id="x",
            status="running",
            description="",
            result=None,
            raw_content="{}",
        )
        with self.assertRaises(ValueError):
            format_async_tool_text_for_provider(info)

    def test_handles_none_result_in_intermediate_via_default_template(self):
        # Defensive: if result somehow ends up None on a non-placeholder, the
        # formatter substitutes empty string rather than raising.
        info = AsyncToolMessage(
            kind="intermediate",
            tool_call_id="x",
            status="running",
            description="",
            result=None,
            raw_content="{}",
        )
        text = format_async_tool_text_for_provider(info)
        assert "x" in text
        assert "running" in text


if __name__ == "__main__":
    unittest.main()
