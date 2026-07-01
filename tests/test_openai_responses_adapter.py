#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the OpenAI Responses adapter's context -> input conversion."""

from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage


def _adapter():
    return OpenAIResponsesLLMAdapter()


def _convert(messages):
    return _adapter()._convert_messages_to_input(messages)


class TestReasoningInputItems:
    def test_reasoning_message_becomes_reasoning_item(self):
        """A persisted reasoning message converts to a Responses reasoning item."""
        items = _convert(
            [
                LLMSpecificMessage(
                    llm="openai_responses",
                    message={
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "Let me think."}],
                        "encrypted_content": "ENCRYPTED",
                    },
                )
            ]
        )

        assert items[0] == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Let me think."}],
            "encrypted_content": "ENCRYPTED",
        }

    def test_reasoning_precedes_assistant_message(self):
        """Reasoning round-trips positioned before the assistant reply it produced.

        The service persists the reasoning item mid-stream, before the assistant
        turn is finalized, so append order already yields the ordering OpenAI
        expects (reasoning -> message).
        """
        context = LLMContext()
        context.add_message(
            LLMSpecificMessage(
                llm="openai_responses",
                message={
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "ENCRYPTED",
                },
            )
        )
        context.add_message({"role": "assistant", "content": "Hello!"})

        items = _convert(_adapter().get_messages(context))

        assert items[0]["type"] == "reasoning"
        assert items[1]["role"] == "assistant"

    def test_reasoning_without_encrypted_content_omits_field(self):
        """`encrypted_content` is optional; omit it rather than send null."""
        items = _convert(
            [
                LLMSpecificMessage(
                    llm="openai_responses",
                    message={"type": "reasoning", "id": "rs_2", "summary": []},
                )
            ]
        )

        assert items[0]["type"] == "reasoning"
        assert items[0]["id"] == "rs_2"
        assert "encrypted_content" not in items[0]

    def test_non_reasoning_specific_message_passes_through(self):
        """Unknown LLM-specific payloads are assumed already in input shape."""
        raw = {"type": "something_else", "foo": "bar"}
        items = _convert([LLMSpecificMessage(llm="openai_responses", message=raw)])

        assert items[0] == raw
