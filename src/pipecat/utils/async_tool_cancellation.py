#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Constants for the built-in async tool cancellation feature.

When an ``LLMService`` has functions registered with
``cancel_on_interruption=False`` (async tools), it automatically injects the
``cancel_async_tool_call`` tool and the instructions below into every inference
request so the LLM can cancel stale in-progress calls.
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema

CANCEL_ASYNC_TOOL_NAME = "cancel_async_tool_call"

ASYNC_TOOL_CANCELLATION_INSTRUCTIONS = """ASYNC TOOL CANCELLATION:
Some tool calls run asynchronously in the background. When one starts, a tool response \
is added to the conversation whose content is a JSON object with \
"type": "tool", "status": "started", and a "tool_call_id" field containing the \
exact ID of that call (e.g. {"type": "tool", "status": "started", "tool_call_id": "..."}).

If the user changes topic, explicitly says they no longer need the result, or the pending \
result would clearly be stale, call cancel_async_tool_call. \
To find the correct tool_call_id: locate the most recent tool response in the conversation \
whose content has "status": "started" and whose call has NOT already been cancelled, \
then copy the "tool_call_id" value from that content exactly as-is. \
Never invent or guess a tool_call_id."""

CANCEL_ASYNC_TOOL_SCHEMA = FunctionSchema(
    name=CANCEL_ASYNC_TOOL_NAME,
    description=(
        "Cancel a single async tool call that is no longer needed. "
        "Use this when the user changes topic, indicates a pending result is "
        "no longer relevant, or when processing the result would produce a "
        "stale or confusing response. "
        "The tool_call_id must be the exact 'id' value from the assistant's "
        "tool call which we wish to cancel, visible in the conversation history."
    ),
    properties={
        "tool_call_id": {
            "type": "string",
            "description": ("The exact id of the async call to cancel."),
        }
    },
    required=["tool_call_id"],
)
