#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turning ACP frames into something a person can hear.

A renderer is a processor downstream of
:class:`~pipecat.services.acp.service.ACPService` that consumes ACP frames and
emits whatever the rest of the pipeline should say. ACP's output is written to
be read: file paths, diffs, token counts, and a reasoning stream that can run
for thousands of words. Speaking it verbatim does not work, so the renderer is
where a voice client decides what is worth saying.

:func:`describe` is the smallest possible version, one line per frame with no
speech at all. :class:`~pipecat.observers.loggers.acp_log_observer.ACPLogObserver`
logs it, which is a useful starting point: run a bot, watch the whole agent
stream go by, and decide what a real renderer should do with it.
"""

from pipecat.frames.frames import Frame
from pipecat.services.acp.frames import (
    ACPAgentMessageFrame,
    ACPAgentThoughtFrame,
    ACPClientRequestFrame,
    ACPPlanFrame,
    ACPSessionStartedFrame,
    ACPToolCallFrame,
    ACPToolCallUpdateFrame,
    ACPTurnEndedFrame,
    ACPTurnStartedFrame,
)


def describe(frame: Frame) -> str | None:
    """Summarize an ACP frame in one line.

    Args:
        frame: The frame to summarize.

    Returns:
        A one-line description, or None if the frame is not an ACP frame.
    """
    if isinstance(frame, ACPSessionStartedFrame):
        return f"session {frame.session_id} open (mode: {frame.current_mode_id})"
    if isinstance(frame, ACPTurnStartedFrame):
        return "turn started"
    if isinstance(frame, ACPTurnEndedFrame):
        return f"turn ended ({frame.stop_reason.value})"
    if isinstance(frame, ACPAgentMessageFrame):
        return f"message: {frame.content.text or f'<{frame.content.type}>'}"
    if isinstance(frame, ACPAgentThoughtFrame):
        return f"thought: {frame.content.text or f'<{frame.content.type}>'}"
    if isinstance(frame, ACPToolCallFrame):
        call = frame.tool_call
        return f"tool {call.kind.value} {call.status.value}: {call.title}"
    if isinstance(frame, ACPToolCallUpdateFrame):
        call = frame.tool_call
        return f"tool {call.kind.value} {call.status.value}: {call.title}"
    if isinstance(frame, ACPPlanFrame):
        steps = ", ".join(f"{e.content} [{e.status.value}]" for e in frame.entries)
        return f"plan: {steps}"
    if isinstance(frame, ACPClientRequestFrame):
        return f"agent asked for {frame.method}"
    return None
