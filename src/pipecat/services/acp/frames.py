#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frames for the Agent Client Protocol.

Every ACP message that crosses the pipeline boundary has a frame here. The
mapping is deliberately flat: one frame per ``session/update`` variant, one per
lifecycle event, and a single request/response pair covering everything the
agent asks the client to do.

Frame classes are chosen for interruption behavior. Session and turn boundaries
are control frames, streamed content is data frames, and the request/response
pair are system frames so a barge-in cannot strand the agent waiting on an
answer that was dropped from a queue.
"""

from dataclasses import dataclass, field

from pipecat.frames.frames import ControlFrame, DataFrame, SystemFrame
from pipecat.services.acp.types import (
    ACPErrorData,
    AgentCapabilities,
    AvailableCommand,
    ContentBlock,
    PlanEntry,
    ReadTextFileParams,
    RequestPermissionParams,
    SessionMode,
    StopReason,
    TerminalParams,
    ToolCall,
    ToolCallUpdate,
    WriteTextFileParams,
)

#
# Session lifecycle
#


@dataclass
class ACPSessionStartedFrame(ControlFrame):
    """A session is open and ready for prompts.

    Parameters:
        session_id: Identifier of the new session.
        agent_capabilities: What the agent supports.
        current_mode_id: The mode the agent is operating in.
        available_modes: Every mode the agent offers.
        commands: Slash commands the agent exposes.
    """

    session_id: str = ""
    agent_capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    current_mode_id: str | None = None
    available_modes: list[SessionMode] = field(default_factory=list)
    commands: list[AvailableCommand] = field(default_factory=list)


@dataclass
class ACPSessionEndedFrame(ControlFrame):
    """The agent process is gone and the session is closed.

    Parameters:
        session_id: Identifier of the closed session.
        reason: Why the session ended.
    """

    session_id: str = ""
    reason: str | None = None


#
# Turn lifecycle
#


@dataclass
class ACPTurnStartedFrame(ControlFrame):
    """A prompt has been sent and the agent is working.

    Parameters:
        session_id: Session the turn belongs to.
    """

    session_id: str = ""


@dataclass
class ACPTurnEndedFrame(ControlFrame):
    """The agent finished acting on a prompt.

    Parameters:
        session_id: Session the turn belongs to.
        stop_reason: Why the turn ended.
    """

    session_id: str = ""
    stop_reason: StopReason = StopReason.END_TURN


#
# session/update variants
#


@dataclass
class ACPAgentMessageFrame(DataFrame):
    """A chunk of the agent's reply to the user.

    Parameters:
        session_id: Session the chunk belongs to.
        content: The chunk's content block.
    """

    session_id: str = ""
    content: ContentBlock = field(default_factory=lambda: ContentBlock(type="text", text=""))


@dataclass
class ACPAgentThoughtFrame(DataFrame):
    """A chunk of the agent's reasoning.

    Parameters:
        session_id: Session the chunk belongs to.
        content: The chunk's content block.
    """

    session_id: str = ""
    content: ContentBlock = field(default_factory=lambda: ContentBlock(type="text", text=""))


@dataclass
class ACPUserMessageFrame(DataFrame):
    """A user turn echoed back by the agent, during session replay.

    Parameters:
        session_id: Session the chunk belongs to.
        content: The chunk's content block.
    """

    session_id: str = ""
    content: ContentBlock = field(default_factory=lambda: ContentBlock(type="text", text=""))


@dataclass
class ACPToolCallFrame(DataFrame):
    """The agent started a tool call.

    Parameters:
        session_id: Session the call belongs to.
        tool_call: The call, including its title and kind.
    """

    session_id: str = ""
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(tool_call_id=""))


@dataclass
class ACPToolCallUpdateFrame(DataFrame):
    """A tool call changed status or produced output.

    The update carries only what changed. :attr:`tool_call` is the service's
    merged view of the call, so a consumer can render the update without having
    tracked the original.

    Parameters:
        session_id: Session the call belongs to.
        update: The fields the agent reported as changed.
        tool_call: The call with the update already applied.
    """

    session_id: str = ""
    update: ToolCallUpdate = field(default_factory=lambda: ToolCallUpdate(tool_call_id=""))
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(tool_call_id=""))


@dataclass
class ACPPlanFrame(DataFrame):
    """The agent's plan for the turn, revised as it works.

    Parameters:
        session_id: Session the plan belongs to.
        entries: The plan's steps, in order.
    """

    session_id: str = ""
    entries: list[PlanEntry] = field(default_factory=list)


@dataclass
class ACPModeUpdatedFrame(ControlFrame):
    """The agent switched operating mode.

    Parameters:
        session_id: Session that changed mode.
        current_mode_id: The mode now in effect.
    """

    session_id: str = ""
    current_mode_id: str | None = None


@dataclass
class ACPCommandsUpdatedFrame(ControlFrame):
    """The agent's available slash commands changed.

    Parameters:
        session_id: Session whose commands changed.
        commands: The commands now available.
    """

    session_id: str = ""
    commands: list[AvailableCommand] = field(default_factory=list)


#
# Agent-initiated requests
#


@dataclass
class ACPClientRequestFrame(SystemFrame):
    """The agent is asking the client to do something and is blocked until it answers.

    The service broadcasts these both upstream and downstream, so a processor
    that answers them works on either side of the service. Answer by pushing an
    :class:`ACPClientResponseFrame` carrying the same ``request_id``. An
    unanswered request fails after the service's request timeout.

    Parameters:
        request_id: Correlates the request with its response.
        method: The ACP method the agent called.
    """

    request_id: str = ""
    method: str = ""


@dataclass
class ACPPermissionRequestFrame(ACPClientRequestFrame):
    """The agent needs the user to approve a tool call.

    Parameters:
        params: The call awaiting approval and the options to choose from.
    """

    params: RequestPermissionParams | None = None


@dataclass
class ACPReadFileRequestFrame(ACPClientRequestFrame):
    """The agent wants the client to read a file.

    Only sent when the client advertised the ``fs.readTextFile`` capability.

    Parameters:
        params: The path and optional line range to read.
    """

    params: ReadTextFileParams | None = None


@dataclass
class ACPWriteFileRequestFrame(ACPClientRequestFrame):
    """The agent wants the client to write a file.

    Only sent when the client advertised the ``fs.writeTextFile`` capability.

    Parameters:
        params: The path and the file's full new contents.
    """

    params: WriteTextFileParams | None = None


@dataclass
class ACPTerminalRequestFrame(ACPClientRequestFrame):
    """The agent wants the client to operate a terminal.

    Covers every ``terminal/*`` method; :attr:`~ACPClientRequestFrame.method`
    says which. Only sent when the client advertised the ``terminal``
    capability.

    Parameters:
        params: Parameters of the terminal operation.
    """

    params: TerminalParams | None = None


@dataclass
class ACPClientResponseFrame(SystemFrame):
    """An answer to an :class:`ACPClientRequestFrame`.

    Push either ``result`` or ``error``. The service unblocks the agent with
    whichever is set, preferring ``error``.

    Parameters:
        request_id: The request being answered.
        result: The method's result, as camelCase-ready fields.
        error: A JSON-RPC error to return instead of a result.
    """

    request_id: str = ""
    result: dict | None = None
    error: ACPErrorData | None = None


#
# Pipeline to agent
#


@dataclass
class ACPPromptFrame(DataFrame):
    """A user turn to send to the agent.

    Parameters:
        blocks: The turn's content.
    """

    blocks: list[ContentBlock] = field(default_factory=list)


@dataclass
class ACPCancelTurnFrame(SystemFrame):
    """Interrupt the agent's in-flight turn."""

    pass


@dataclass
class ACPSetModeFrame(ControlFrame):
    """Switch the agent's operating mode.

    Parameters:
        mode_id: The mode to switch to.
    """

    mode_id: str = ""
