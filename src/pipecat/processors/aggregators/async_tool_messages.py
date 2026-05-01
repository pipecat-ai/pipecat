#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for detecting and decoding async-tool messages in an LLM context.

When a function is registered with ``cancel_on_interruption=False``, the
``LLMUserContextAggregator`` / ``LLMAssistantContextAggregator`` pair appends
async-tool messages to the conversation context as the underlying task
progresses:

- A ``placeholder`` message (``role="tool"``) is appended immediately when the
  tool starts running.
- An ``intermediate`` message (``role="developer"``) is appended each time an
  intermediate result is reported via
  ``result_callback(..., FunctionCallResultProperties(is_final=False))``.
- A ``final`` message (``role="developer"``) is appended when the task
  finishes.

Realtime LLM services need to recognize these messages to forward async results
to their providers. This module exposes the detection and decoding primitives.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

AsyncToolMessageKind = Literal["placeholder", "intermediate", "final"]


@dataclass(frozen=True)
class AsyncToolMessage:
    """A parsed async-tool message extracted from an LLM context entry.

    Parameters:
        kind: Which of the three async-tool message stages this is.
        tool_call_id: The id of the tool invocation this message relates to.
        status: ``"running"`` for placeholder/intermediate, ``"finished"`` for
            the final message.
        description: Human-readable description from the message envelope. May
            be empty.
        result: For ``intermediate`` and ``final`` messages, the JSON-encoded
            result string (or the literal ``"COMPLETED"`` if the function
            returned no value). ``None`` for ``placeholder`` messages.
        raw_content: The original JSON-encoded envelope string (i.e. the
            ``content`` field of the source LLM context message). Use this
            when forwarding the message to a provider as a formal tool
            result, so the provider receives the complete envelope (with
            ``type``, ``status``, ``tool_call_id``, ``description``, and any
            ``result``) rather than just a sub-field.
    """

    kind: AsyncToolMessageKind
    tool_call_id: str
    status: Literal["running", "finished"]
    description: str
    result: str | None
    raw_content: str


def is_async_tool_message(message: dict[str, Any]) -> bool:
    """Return True if ``message`` is an async-tool message envelope."""
    return parse_async_tool_message(message) is not None


def parse_async_tool_message(message: dict[str, Any]) -> AsyncToolMessage | None:
    """Decode an async-tool message envelope, or return None if not async-tool.

    Args:
        message: A message dict from the LLM context.

    Returns:
        An ``AsyncToolMessage`` if the message is a recognized async-tool
        envelope, otherwise ``None``.
    """
    role = message.get("role")
    if role not in ("tool", "developer"):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        envelope = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(envelope, dict) or envelope.get("type") != "async_tool":
        return None
    tool_call_id = envelope.get("tool_call_id")
    status = envelope.get("status")
    if not isinstance(tool_call_id, str) or status not in ("running", "finished"):
        return None
    description = envelope.get("description", "")
    if not isinstance(description, str):
        description = ""
    result = envelope.get("result")
    if result is not None and not isinstance(result, str):
        result = None
    if result is None:
        kind: AsyncToolMessageKind = "placeholder"
    elif status == "finished":
        kind = "final"
    else:
        kind = "intermediate"
    return AsyncToolMessage(
        kind=kind,
        tool_call_id=tool_call_id,
        status=status,
        description=description,
        result=result,
        raw_content=content,
    )


_DEFAULT_PROVIDER_TEXT_TEMPLATE = (
    "[Async tool update for tool_call_id={tool_call_id}, status={status}] {result}"
)


def format_async_tool_text_for_provider(
    info: AsyncToolMessage,
    *,
    template: str | None = None,
) -> str:
    """Render an async-tool result as text for mid-conversation provider injection.

    Realtime services that fully honor the async-tool mechanism send the
    ``placeholder`` message via the formal tool-result channel and then send
    subsequent ``intermediate`` / ``final`` results as text injected
    mid-conversation. This function produces the text payload for that
    injection.

    Args:
        info: The parsed async-tool message. Must not be a ``placeholder``.
        template: Optional override format string. Available substitution
            keys are ``tool_call_id``, ``status``, ``result``, and
            ``description``.

    Returns:
        The rendered text.

    Raises:
        ValueError: If ``info.kind == "placeholder"``. Placeholders should be
            sent via the formal tool-result channel, not as text.
    """
    if info.kind == "placeholder":
        raise ValueError(
            "format_async_tool_text_for_provider should not be called on placeholder "
            "messages; placeholders should be sent via the formal tool-result channel."
        )
    fmt = template if template is not None else _DEFAULT_PROVIDER_TEXT_TEMPLATE
    return fmt.format(
        tool_call_id=info.tool_call_id,
        status=info.status,
        result=info.result or "",
        description=info.description,
    )
