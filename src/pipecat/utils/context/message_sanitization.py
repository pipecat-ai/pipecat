#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utilities for sanitizing context messages before export."""

from __future__ import annotations

from typing import Any


def strip_thought_from_id(value: Any) -> Any:
    """Strip ``__thought__`` suffixes from tool call identifiers.

    Args:
        value: Identifier value to sanitize.

    Returns:
        The sanitized identifier, or the original value when no suffix is present.
    """
    if not value or not isinstance(value, str) or "__thought__" not in value:
        return value
    return value.split("__thought__", 1)[0]


def strip_thought_ids_from_messages(messages: Any) -> Any:
    """Strip ``__thought__`` suffixes from tool call identifiers in messages.

    Args:
        messages: Conversation messages to sanitize.

    Returns:
        A sanitized message list, or the original value when it is not a list.
    """
    if not messages or not isinstance(messages, list):
        return messages

    cleaned_messages = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned_messages.append(message)
            continue

        cleaned_message = message
        if "tool_call_id" in cleaned_message:
            cleaned_message = {
                **cleaned_message,
                "tool_call_id": strip_thought_from_id(cleaned_message["tool_call_id"]),
            }

        tool_calls = cleaned_message.get("tool_calls")
        if isinstance(tool_calls, list):
            cleaned_message = {
                **cleaned_message,
                "tool_calls": [
                    {**tool_call, "id": strip_thought_from_id(tool_call.get("id", ""))}
                    if isinstance(tool_call, dict) and "id" in tool_call
                    else tool_call
                    for tool_call in tool_calls
                ],
            }

        cleaned_messages.append(cleaned_message)

    return cleaned_messages
