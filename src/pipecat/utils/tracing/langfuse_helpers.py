#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for Langfuse-oriented trace shaping in the Dograh Pipecat fork."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from opentelemetry.trace import Span


_trace_public_resolver: Callable[[], bool] | None = None


def set_trace_public_resolver(resolver: Callable[[], bool] | None) -> None:
    """Install a per-span policy for Langfuse trace visibility.

    Pipecat on its own exports to a single Langfuse project, so one env flag
    describes the whole deployment. An embedding app that fans spans out to
    several projects — one per tenant — has no single answer, and registers a
    resolver here to decide per span instead.

    Args:
        resolver: Called with no arguments as each traced span starts; returns
            True to mark that span's trace public. Pass None to fall back to
            ``LANGFUSE_TRACES_PUBLIC``.
    """
    global _trace_public_resolver
    _trace_public_resolver = resolver


def traces_public_from_env() -> bool:
    """Read the deployment-wide ``LANGFUSE_TRACES_PUBLIC`` opt-in.

    Returns:
        True when the env var opts in to public traces.
    """
    return os.getenv("LANGFUSE_TRACES_PUBLIC", "").strip().lower() in ("1", "true", "yes")


def mark_trace_public(span: Span) -> None:
    """Mark the current trace as public for Langfuse sharing.

    Public traces are readable by anyone holding the URL, with no login, so
    this is opt-in: through the resolver from :func:`set_trace_public_resolver`
    when one is installed, otherwise through ``LANGFUSE_TRACES_PUBLIC``. A
    resolver that raises leaves the trace private — visibility fails closed.

    Args:
        span: The span to mark. Every span of a trace carries the attribute,
            because whichever one reaches Langfuse first creates the trace.
    """
    resolver = _trace_public_resolver
    if resolver is None:
        should_mark = traces_public_from_env()
    else:
        try:
            should_mark = resolver()
        except Exception as e:
            logger.warning(f"Trace visibility resolver failed, keeping trace private: {e}")
            should_mark = False

    if not should_mark:
        return
    span.set_attribute("langfuse.trace.public", True)


def standardize_messages_to_chatml(messages: Any) -> Any:
    """Normalize provider-native messages into ChatML-like structures."""
    if not messages or not isinstance(messages, list):
        return messages

    chatml_messages = []
    for message in messages:
        if not isinstance(message, dict) or "parts" not in message:
            chatml_messages.append(message)
            continue

        role = message.get("role", "user")
        parts = message.get("parts", [])
        chatml_role = "assistant" if role == "model" else role

        text_parts = []
        tool_calls = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            if "text" in part:
                text_parts.append(part["text"])
            elif "function_call" in part:
                function_call = part["function_call"]
                tool_calls.append(
                    {
                        "id": function_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", ""),
                            "arguments": json.dumps(function_call.get("args", {})),
                        },
                    }
                )
            elif "function_response" in part:
                function_response = part["function_response"]
                chatml_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_response.get("id", ""),
                        "content": json.dumps(function_response.get("response", {})),
                    }
                )
            elif "inline_data" in part:
                text_parts.append("[inline_data]")
            elif "file_data" in part:
                text_parts.append("[file_data]")

        if tool_calls:
            chatml_message: dict[str, Any] = {"role": chatml_role, "tool_calls": tool_calls}
            if text_parts:
                chatml_message["content"] = " ".join(text_parts)
            chatml_messages.append(chatml_message)
        elif text_parts:
            content = text_parts[0] if len(text_parts) == 1 else " ".join(text_parts)
            chatml_messages.append({"role": chatml_role, "content": content})

    return chatml_messages


def standardize_tools_to_chatml(tools: Any) -> Any:
    """Normalize provider-native tool definitions into ChatML-like structures."""
    if not tools or not isinstance(tools, list):
        return tools

    chatml_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function":
            chatml_tools.append(tool)
            continue

        for declaration in tool.get("function_declarations", []):
            chatml_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.get("name", ""),
                        "description": declaration.get("description", ""),
                        "parameters": declaration.get("parameters", {}),
                    },
                }
            )

    return chatml_tools


def set_tts_input_attributes(span: Span, text: str | None) -> None:
    """Attach TTS input text using both upstream and Langfuse-friendly keys."""
    if not text:
        return
    span.set_attribute("text", text)
    span.set_attribute("input", text)
    span.set_attribute("metrics.character_count", len(text))


def set_stt_output_attributes(span: Span, transcript: str | None) -> None:
    """Attach STT transcript using both upstream and Langfuse-friendly keys."""
    if not transcript:
        return
    span.set_attribute("transcript", transcript)
    span.set_attribute("output", transcript)


def build_llm_output_payload(
    text_output: str,
    function_calls: list[dict[str, Any]],
) -> str | None:
    """Build a single output payload that preserves text and tool calls."""
    if function_calls:
        payload: dict[str, Any] = {"tool_calls": function_calls}
        if text_output:
            payload["content"] = text_output
        return json.dumps(payload, default=str)

    if text_output:
        return text_output

    return None
