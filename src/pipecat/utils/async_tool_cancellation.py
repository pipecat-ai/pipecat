#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Constants for the built-in async tool cancellation feature.

A function registered with ``cancellable_by_llm=True`` is advertised
alongside a cancel tool named for it, built here, and the instructions below are
composed into the system instruction — so the LLM can drop work whose result is
no longer wanted. A tool that doesn't opt in has no cancel tool at all.

A single running call is stopped without arguments. Telling several calls of one
tool apart takes the ``tool_call_id`` the conversation already carries, on the
message that reported each call running.
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema

ASYNC_TOOL_CANCELLATION_INSTRUCTIONS = """ASYNC TOOL CANCELLATION:
Some of your tools keep running in the background after you have replied, and some of \
those can be stopped early.

Work that can be stopped early has its own cancel tool, named for it: a running \
write_report call is stopped by cancel_write_report. Work with no such tool cannot be \
stopped and will finish on its own.

When the user no longer wants a result you are still waiting on, call the corresponding \
cancel tool. Only the call stops the work: saying you cancelled it, or that you'll skip it, \
leaves it running and its result will still arrive and contradict you. So when the same turn \
also asks for something else, make the call and answer — don't just answer.

Call the cancel tool with no arguments and it stops the one call that is running. If \
several calls of that same tool are running, it needs a tool_call_id to say which. Each \
call's id is already in the conversation: find the tool message that reported it running, \
which carries "status": "running", its "tool_call_id", and the arguments that call was \
given. Copy that id exactly as written; never invent or guess one."""

CANCEL_TOOL_PREFIX = "cancel_"


def cancel_tool_name(function_name: str) -> str:
    """Name of the tool that cancels calls of ``function_name``.

    Args:
        function_name: The cancellable tool.

    Returns:
        The built-in cancel tool's name, e.g. ``cancel_write_report``.
    """
    return f"{CANCEL_TOOL_PREFIX}{function_name}"


def build_cancel_tool_schema(function_name: str) -> FunctionSchema:
    """Build the cancel tool for one cancellable tool.

    Which work to stop is carried by the tool the model picks, so a single
    running call is stopped without arguments. ``tool_call_id`` only has to be
    given to choose between several calls of the same tool, and the handler
    refuses with the ids when it is needed and missing.

    Args:
        function_name: The tool whose calls this one cancels.

    Returns:
        The schema to advertise alongside ``function_name``.
    """
    return FunctionSchema(
        name=cancel_tool_name(function_name),
        description=(
            f"Stop a running {function_name} call whose result is no longer "
            "needed — the user says to drop it, asks for something that replaces "
            "it, or says something that makes the pending result stale. Call it "
            "with no arguments to stop the one running call; only when several "
            f"{function_name} calls are running does it need a tool_call_id to "
            "say which."
        ),
        properties={
            "tool_call_id": {
                "type": "string",
                "description": (
                    f"Which {function_name} call to stop. Needed only when "
                    "several are running, and carried by the message in the "
                    "conversation that reported each one running."
                ),
            },
        },
        required=[],
    )
