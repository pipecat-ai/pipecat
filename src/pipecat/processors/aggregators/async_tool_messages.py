#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for the async-tool message protocol used in LLM contexts.

When a function is registered with ``cancel_on_interruption=False``, the
``LLMUserContextAggregator`` / ``LLMAssistantContextAggregator`` pair appends
async-tool messages to the conversation context as the underlying task
progresses:

- A ``started`` message (``role="tool"``) is appended immediately when the
  tool starts running.
- An ``intermediate`` message (``role="developer"``) is appended each time an
  intermediate result is reported via
  ``result_callback(..., FunctionCallResultProperties(is_final=False))``.
- A ``final`` message (``role="developer"``) is appended when the task
  finishes.

This module is the single source of truth for the on-the-wire payload shape:

- The aggregator uses the ``build_*_message`` functions when injecting messages.
- Realtime LLM services use ``parse_message`` to detect async-tool messages
  while iterating the context, then read ``payload.result`` and deliver it via
  their formal tool-result channel.

Internally, ``AsyncToolMessagePayload`` is the canonical structured form;
the on-the-wire JSON string is always derived from it (never stored) so the
two representations can't drift.

Consumers are expected to import the module rather than its individual
functions, e.g.::

    from pipecat.processors.aggregators import async_tool_messages
    ...
    async_tool_messages.build_started_message(tool_call_id)
    async_tool_messages.parse_message(msg)
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.processors.aggregators.llm_context import LLMStandardMessage

AsyncToolMessageKind = Literal["started", "intermediate", "final"]

# --- Payload shape (private; canonical source of truth) ---------------------

# The ``type`` field that identifies an async-tool message payload. Both the
# builders and the parser use this constant; do not duplicate the literal.
_PAYLOAD_TYPE = "async_tool"

# Status value for started / intermediate messages (task still running).
_STATUS_RUNNING = "running"

# Status value for the final message (task complete).
_STATUS_FINISHED = "finished"

# Description shipped on the started message. The text is intentionally
# self-explanatory so a model reading the context can tell what's about to
# happen even without out-of-band knowledge of the protocol.
_STARTED_DESCRIPTION = (
    "An asynchronous task associated with this tool_call_id has started "
    "running. Expect results to arrive later as developer messages that look "
    "roughly like this one (with 'type=async_tool' and a matching tool_call_id) "
    "but with a 'result' field. Note that there *may* be more than one result "
    "(i.e., a stream of results), but there doesn't have to be (there may be "
    "only one). The last result will come in a message with 'status=finished'."
)

# Description shipped on each intermediate-result message.
_INTERMEDIATE_DESCRIPTION = (
    "This is an intermediate result for the asynchronous task associated with "
    "this tool_call_id. The task is still running. More intermediate results "
    "may follow, or the next result may be the final one with "
    "'status=finished'."
)

# Description shipped on the final-result message.
_FINAL_DESCRIPTION = (
    "This is the final result for the asynchronous task associated with this "
    "tool_call_id. The task has completed. No further results will arrive for "
    "this tool_call_id."
)


@dataclass(frozen=True)
class AsyncToolMessagePayload:
    """The structured contents of an async-tool message in an LLM context.

    Parameters:
        kind: Which of the three async-tool message stages this is.
        tool_call_id: The id of the tool invocation this payload relates to.
        status: ``"running"`` for started/intermediate, ``"finished"`` for
            the final message.
        description: Human-readable description from the payload. May be empty.
        result: For ``intermediate`` and ``final`` messages, the JSON-encoded
            result string (or the literal ``"COMPLETED"`` if the function
            returned no value). ``None`` for ``started`` messages.
    """

    kind: AsyncToolMessageKind
    tool_call_id: str
    status: Literal["running", "finished"]
    description: str
    result: str | None


# --- Internal: payload ↔ on-the-wire forms -----------------------------------


def _payload_to_json(payload: AsyncToolMessagePayload) -> str:
    """Serialize a payload to its on-the-wire JSON string form.

    Fields that don't apply to the payload's kind are omitted (notably
    ``result`` is left out of ``started`` payloads, since the task hasn't
    produced a result yet).
    """
    obj: dict[str, Any] = {
        "type": _PAYLOAD_TYPE,
        "status": payload.status,
        "tool_call_id": payload.tool_call_id,
        "description": payload.description,
    }
    if payload.result is not None:
        obj["result"] = payload.result
    return json.dumps(obj)


def _payload_to_message(payload: AsyncToolMessagePayload) -> LLMStandardMessage:
    """Wrap a payload in the LLM context message shape that matches its kind.

    - ``started``: ``role="tool"`` plus ``tool_call_id`` at the top level
      (so the message can sit alongside other regular tool-result messages).
    - ``intermediate`` / ``final``: ``role="developer"``; ``tool_call_id``
      lives only inside the JSON payload.
    """
    content = _payload_to_json(payload)
    if payload.kind == "started":
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": payload.tool_call_id,
        }
    return {
        "role": "developer",
        "content": content,
    }


# --- Builders ----------------------------------------------------------------


def build_started_message(tool_call_id: str) -> LLMStandardMessage:
    """Build a ``started`` async-tool message for an LLM context.

    Append the returned message to the LLM context immediately when an async
    function call (registered with ``cancel_on_interruption=False``) starts
    running. The message lets the model know a task is in flight and that its
    results will arrive later in subsequent ``developer``-role messages.

    Args:
        tool_call_id: The id of the tool invocation this message is for.

    Returns:
        A message ready to pass to ``LLMContext.add_message``.
    """
    return _payload_to_message(
        AsyncToolMessagePayload(
            kind="started",
            tool_call_id=tool_call_id,
            status=_STATUS_RUNNING,
            description=_STARTED_DESCRIPTION,
            result=None,
        )
    )


def build_intermediate_result_message(tool_call_id: str, result: str) -> LLMStandardMessage:
    """Build an intermediate-result async-tool message for an LLM context.

    Append the returned message to the LLM context each time the running async
    function reports a non-final result via
    ``result_callback(..., FunctionCallResultProperties(is_final=False))``.

    Args:
        tool_call_id: The id of the tool invocation the result is for.
        result: The JSON-encoded result string (caller is responsible for
            encoding the function's actual return value, typically via
            ``json.dumps``).

    Returns:
        A message ready to pass to ``LLMContext.add_message``.
    """
    return _payload_to_message(
        AsyncToolMessagePayload(
            kind="intermediate",
            tool_call_id=tool_call_id,
            status=_STATUS_RUNNING,
            description=_INTERMEDIATE_DESCRIPTION,
            result=result,
        )
    )


def build_final_result_message(tool_call_id: str, result: str) -> LLMStandardMessage:
    """Build a final-result async-tool message for an LLM context.

    Append the returned message to the LLM context when the async function
    finishes. After this message no further async-tool messages will arrive
    for this ``tool_call_id``.

    Args:
        tool_call_id: The id of the tool invocation the result is for.
        result: The JSON-encoded result string, or the literal ``"COMPLETED"``
            sentinel when the function returned ``None`` (matching the same
            convention used for synchronous tool calls).

    Returns:
        A message ready to pass to ``LLMContext.add_message``.
    """
    return _payload_to_message(
        AsyncToolMessagePayload(
            kind="final",
            tool_call_id=tool_call_id,
            status=_STATUS_FINISHED,
            description=_FINAL_DESCRIPTION,
            result=result,
        )
    )


# --- Parsing -----------------------------------------------------------------


def parse_message(message: LLMStandardMessage) -> AsyncToolMessagePayload | None:
    """Decode an async-tool message payload, or return None if not async-tool.

    Args:
        message: A standard message from the LLM context. Callers iterating
            over ``LLMContext.get_messages()`` should filter out
            ``LLMSpecificMessage`` entries first; only ``LLMStandardMessage``
            values can carry async-tool payloads.

    Returns:
        An ``AsyncToolMessagePayload`` if the message is a recognized
        async-tool payload, otherwise ``None``.
    """
    role = message.get("role")
    if role not in ("tool", "developer"):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != _PAYLOAD_TYPE:
        return None
    tool_call_id = payload.get("tool_call_id")
    status = payload.get("status")
    if not isinstance(tool_call_id, str) or status not in (_STATUS_RUNNING, _STATUS_FINISHED):
        return None
    description = payload.get("description", "")
    if not isinstance(description, str):
        description = ""
    result = payload.get("result")
    if result is not None and not isinstance(result, str):
        result = None
    if result is None:
        kind: AsyncToolMessageKind = "started"
    elif status == _STATUS_FINISHED:
        kind = "final"
    else:
        kind = "intermediate"
    return AsyncToolMessagePayload(
        kind=kind,
        tool_call_id=tool_call_id,
        status=status,
        description=description,
        result=result,
    )
