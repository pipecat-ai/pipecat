#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Responses adapter message rendering for logging.

Reasoning items round-trip through the universal context as
``LLMSpecificMessage``s. ``get_messages_for_logging()`` must render them as
plain JSON-serializable dicts (the tracing decorator ``json.dumps``es the
result) with the ``encrypted_content`` payload elided.
"""

import json

from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext

REASONING_MESSAGE = {
    "type": "reasoning",
    "id": "rs_123",
    "summary": [{"type": "summary_text", "text": "thinking about the weather"}],
    "encrypted_content": "gAAAA" * 300,
}


def _adapter_and_context():
    adapter = OpenAIResponsesLLMAdapter()
    context = LLMContext(
        messages=[
            {"role": "user", "content": "hello"},
            adapter.create_llm_specific_message(dict(REASONING_MESSAGE)),
            {"role": "assistant", "content": "hi"},
        ]
    )
    return adapter, context


def test_reasoning_encrypted_content_is_elided():
    adapter, context = _adapter_and_context()
    messages = adapter.get_messages_for_logging(context)
    assert len(messages) == 3
    reasoning = messages[1]
    assert reasoning["type"] == "reasoning"
    assert reasoning["id"] == "rs_123"
    assert reasoning["summary"] == REASONING_MESSAGE["summary"]
    assert reasoning["encrypted_content"] == "..."


def test_messages_for_logging_are_json_serializable():
    """The tracing decorator ``json.dumps``es the result for the LLM span's
    input attribute, so every rendered message must be JSON-serializable."""
    adapter, context = _adapter_and_context()
    json.dumps(adapter.get_messages_for_logging(context))


def test_standard_messages_pass_through():
    adapter, context = _adapter_and_context()
    messages = adapter.get_messages_for_logging(context)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi"}


def test_context_not_mutated_by_elision():
    adapter, context = _adapter_and_context()
    adapter.get_messages_for_logging(context)
    stored = context.get_messages("openai_responses")[1]
    assert stored.message["encrypted_content"] == REASONING_MESSAGE["encrypted_content"]


def test_reasoning_without_encrypted_content_is_untouched():
    adapter = OpenAIResponsesLLMAdapter()
    context = LLMContext(
        messages=[
            adapter.create_llm_specific_message({"type": "reasoning", "id": "rs_9", "summary": []}),
        ]
    )
    messages = adapter.get_messages_for_logging(context)
    assert messages[0] == {"type": "reasoning", "id": "rs_9", "summary": []}
    json.dumps(messages)
