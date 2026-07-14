#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for _convert_tool_content_for_seeding in GeminiLiveLLMService (#4926)."""

import pytest

from google.genai.types import Content, FunctionCall, FunctionResponse, Part

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class TestConvertToolContentForSeeding:
    def test_text_content_passes_through(self):
        messages = [
            Content(role="user", parts=[Part(text="Hello")]),
            Content(role="model", parts=[Part(text="Hi there")]),
        ]
        result = GeminiLiveLLMService._convert_tool_content_for_seeding(messages)
        assert result == messages

    def test_function_call_converted_to_text_summary(self):
        messages = [
            Content(
                role="model",
                parts=[Part(function_call=FunctionCall(name="transfer_to_agent", args={"agent": "support"}))],
            )
        ]
        result = GeminiLiveLLMService._convert_tool_content_for_seeding(messages)
        assert len(result) == 1
        assert result[0].role == "model"
        assert "transfer_to_agent" in result[0].parts[0].text

    def test_function_response_converted_to_text_summary(self):
        messages = [
            Content(
                role="user",
                parts=[Part(function_response=FunctionResponse(name="transfer_to_agent", response={"status": "done"}))],
            )
        ]
        result = GeminiLiveLLMService._convert_tool_content_for_seeding(messages)
        assert len(result) == 1
        assert result[0].role == "user"
        assert "transfer_to_agent" in result[0].parts[0].text
        assert "done" in result[0].parts[0].text
